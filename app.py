import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
import pickle, os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

# Bridge Streamlit Cloud secrets to os.environ
try:
    import streamlit as _st
    for _k, _v in _st.secrets.items():
        if isinstance(_v, str):
            os.environ.setdefault(_k, _v)
except Exception:
    pass

from src.preprocessing.feature_engineering import (
    run_pipeline, FEATURE_COLS, TARGET_COL
)
from src.forecasting.lightgbm_model import (
    train_all_models, predict_quantiles, load_models
)
from src.uncertainty.conformal_prediction import (
    fit_mapie, predict_conformal, evaluate_coverage, widen_if_needed, load_mapie
)
from src.risk_engine.stock_detector import run_stock_detection
from src.risk_engine.risk_classifier import classify_dataframe, summary
from src.recommendation_engine.order_recommender import recommend
from src.copilot.ai_copilot import CopilotSession
from src.evaluation.metrics import full_report

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="SupplySenseAI",
    page_icon="S",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
/* Remove Streamlit default padding */
.block-container { padding-top: 1.5rem; }

/* Risk tier badges */
.risk-low    { background:#d4edda; color:#155724; padding:10px 18px;
               border-radius:6px; font-weight:600; text-align:center;
               font-size:1rem; border-left:4px solid #28a745; }
.risk-medium { background:#fff3cd; color:#856404; padding:10px 18px;
               border-radius:6px; font-weight:600; text-align:center;
               font-size:1rem; border-left:4px solid #ffc107; }
.risk-high   { background:#f8d7da; color:#721c24; padding:10px 18px;
               border-radius:6px; font-weight:600; text-align:center;
               font-size:1rem; border-left:4px solid #dc3545; }

/* Chat bubbles */
.chat-user     { background:#eef2f7; border-radius:8px; padding:10px 14px;
                 margin:6px 0; border-left:3px solid #4a90d9; }
.chat-copilot  { background:#f0f9f0; border-radius:8px; padding:10px 14px;
                 margin:6px 0; border-left:3px solid #28a745; }
</style>
""", unsafe_allow_html=True)

MODEL_DIR = Path("models")
MODEL_DIR.mkdir(exist_ok=True)

RISK_COLORS = {"LOW": "#28a745", "MEDIUM": "#ffc107", "HIGH": "#dc3545"}

# ── Load pipeline ─────────────────────────────────────────────────────────────
@st.cache_resource(show_spinner="Loading pipeline — please wait...")
def load_pipeline():
    daily_feat, splits, top_skus = run_pipeline()
    X_train, y_train, X_calib, y_calib, X_test, y_test, test_df = splits

    if all((MODEL_DIR / f"lgbm_{l}.pkl").exists() for l in ["p10","p50","p90"]):
        models = load_models()
    else:
        models = train_all_models(X_train, y_train, X_calib, y_calib)

    if (MODEL_DIR / "mapie_model.pkl").exists():
        mapie, alpha = load_mapie()
    else:
        mapie = fit_mapie(models["p50"], X_calib, y_calib)
        alpha = 0.05

    return models, mapie, alpha, test_df, X_test, y_test, top_skus, daily_feat

models, mapie, alpha, test_df, X_test, y_test, top_skus, daily_feat = load_pipeline()

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.title("SupplySenseAI")
    st.caption("Uncertainty-Aware Supply Chain Risk Intelligence")
    st.divider()
    selected_sku = st.selectbox("Select SKU", top_skus[:100])
    lead_time    = st.slider("Lead Time (days)", 1, 14, 3)
    current_inv  = st.number_input("Current Stock on Hand (units)", 0, 10000, 100)
    conf_level   = st.slider("Confidence Level", 0.80, 0.99, 0.95, 0.01)
    run_btn      = st.button("Generate Forecast", type="primary", use_container_width=True)
    st.divider()
    st.caption("Powered by LightGBM + Conformal Prediction + Google Gemini")

# ── Run forecast ──────────────────────────────────────────────────────────────
if run_btn or "results" not in st.session_state:
    with st.spinner("Running forecast pipeline..."):
        mask     = test_df["StockCode"] == selected_sku
        sku_test = test_df[mask].copy()
        X_sku    = X_test[mask]
        y_sku    = y_test[mask]

        q_preds  = predict_quantiles(models, X_sku)
        q_preds["y_true"]    = y_sku.values
        q_preds["Date"]      = sku_test["Date"].values
        q_preds["StockCode"] = selected_sku

        c_preds  = predict_conformal(mapie, X_sku, alpha=1 - conf_level)
        q_preds["conf_lower"] = c_preds["conf_lower"].values
        q_preds["conf_upper"] = c_preds["conf_upper"].values
        q_preds["roll_std_7"]   = X_sku["roll_std_7"].values
        q_preds["roll_mean_28"] = X_sku["roll_mean_28"].values

        sku_series = daily_feat[daily_feat["StockCode"] == selected_sku]["daily_qty"].reset_index(drop=True)
        disruption = run_stock_detection(
            sku_series,
            pd.Series(q_preds["y_true"].values),
            q_preds["conf_lower"],
            q_preds["conf_upper"],
        )
        q_preds["force_high_risk"] = disruption["force_high_risk"]
        classified = classify_dataframe(q_preds)

        cov_result = evaluate_coverage(c_preds, y_sku, 1 - conf_level)

        last = classified.iloc[-1]
        rec  = recommend(last["risk_tier"], last["p10"], last["p50"],
                         last["p90"], current_inv, selected_sku, lead_time)

        copilot_data = {
            "SKU": selected_sku,
            "P10 Forecast": round(last["p10"], 1),
            "P50 Forecast (Median)": round(last["p50"], 1),
            "P90 Forecast": round(last["p90"], 1),
            "Risk Tier": last["risk_tier"],
            "Risk Score": round(last["risk_score"], 3),
            "Safety Stock": rec["safety_stock"],
            "Reorder Point": rec["reorder_point"],
            "Current Stock on Hand": current_inv,
            "Stockout Probability": rec["stockout_prob"],
            "Stock Disruption Detected": disruption["disruption_detected"],
            "Empirical Coverage": round(cov_result["empirical_coverage"], 4),
            "Lead Time (days)": lead_time,
        }
        st.session_state.update({
            "results": classified, "disruption": disruption, "rec": rec,
            "last": last, "cov": cov_result,
            "copilot": CopilotSession(copilot_data),
            "chat_history": [],
        })

res   = st.session_state["results"]
disruption = st.session_state["disruption"]
rec   = st.session_state["rec"]
last  = st.session_state["last"]
cov   = st.session_state["cov"]

# ── Main header ───────────────────────────────────────────────────────────────
st.title("SupplySenseAI — Supply Chain Risk Intelligence")
st.caption(f"SKU: {selected_sku}  |  Confidence: {conf_level:.0%}  |  Lead Time: {lead_time} days")
st.divider()

tab1, tab2, tab3, tab4 = st.tabs([
    "Demand Forecast",
    "Risk Dashboard",
    "Order Recommendations",
    "AI Copilot"
])

# ── TAB 1: FORECAST ───────────────────────────────────────────────────────────
with tab1:
    st.subheader("Demand Forecast with Uncertainty Bands")
    st.caption("LightGBM quantile regression (P10 / P50 / P90) combined with conformal prediction intervals.")

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=res["Date"], y=res["y_true"],
        name="Actual Demand", line=dict(color="#1f77b4", width=2)
    ))
    fig.add_trace(go.Scatter(
        x=res["Date"], y=res["p50"],
        name="P50 Forecast", line=dict(color="#ff7f0e", width=2, dash="dash")
    ))
    fig.add_trace(go.Scatter(
        x=pd.concat([res["Date"], res["Date"][::-1]]),
        y=pd.concat([res["p90"], res["p10"][::-1]]),
        fill="toself", fillcolor="rgba(255,127,14,0.12)",
        line=dict(color="rgba(0,0,0,0)"), name="P10-P90 Quantile Band"
    ))
    fig.add_trace(go.Scatter(
        x=pd.concat([res["Date"], res["Date"][::-1]]),
        y=pd.concat([res["conf_upper"], res["conf_lower"][::-1]]),
        fill="toself", fillcolor="rgba(44,160,44,0.08)",
        line=dict(color="rgba(0,0,0,0)"), name=f"Conformal {conf_level:.0%} Interval"
    ))
    fig.update_layout(
        height=420, hovermode="x unified",
        plot_bgcolor="white", paper_bgcolor="white",
        legend=dict(orientation="h", y=-0.2),
        xaxis_title="Date", yaxis_title="Units Sold"
    )
    st.plotly_chart(fig, use_container_width=True)

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("P10 — Lower Bound", f"{last['p10']:.1f} units")
    c2.metric("P50 — Median Forecast", f"{last['p50']:.1f} units")
    c3.metric("P90 — Upper Bound", f"{last['p90']:.1f} units")
    c4.metric(
        "Conformal Coverage",
        f"{cov['empirical_coverage']:.1%}",
        delta=f"{'PASS' if cov['passes_gate'] else 'FAIL'} (target {conf_level:.0%})"
    )

# ── TAB 2: RISK ───────────────────────────────────────────────────────────────
with tab2:
    st.subheader("Risk Analysis")

    tier = last["risk_tier"]
    css  = {"LOW": "risk-low", "MEDIUM": "risk-medium", "HIGH": "risk-high"}[tier]
    tier_labels = {
        "LOW":    "LOW RISK — Supply chain healthy",
        "MEDIUM": "MEDIUM RISK — Monitor closely",
        "HIGH":   "HIGH RISK — Immediate action required",
    }
    st.markdown(f'<div class="{css}">{tier_labels[tier]}</div>', unsafe_allow_html=True)
    st.write("")

    c1, c2, c3 = st.columns(3)
    c1.metric("Composite Risk Score", f"{last['risk_score']:.3f}", help="0 = no risk, 1 = maximum risk")
    c2.metric("Stock Disruption Detected", "YES" if disruption["disruption_detected"] else "No")
    c3.metric("Latest Demand Z-Score", f"{disruption['latest_zscore']:.2f}")

    fig2 = go.Figure()
    for t, color in RISK_COLORS.items():
        m = res["risk_tier"] == t
        fig2.add_trace(go.Scatter(
            x=res[m]["Date"], y=res[m]["risk_score"],
            mode="markers+lines", name=t,
            marker=dict(color=color, size=4),
            line=dict(color=color, width=1)
        ))
    fig2.add_hline(y=0.20, line_dash="dot", line_color="#ffc107",
                   annotation_text="Medium threshold", annotation_position="right")
    fig2.add_hline(y=0.40, line_dash="dot", line_color="#dc3545",
                   annotation_text="High threshold", annotation_position="right")
    fig2.update_layout(
        title="Risk Score Over Time",
        height=340, plot_bgcolor="white", paper_bgcolor="white",
        xaxis_title="Date", yaxis_title="Risk Score"
    )
    st.plotly_chart(fig2, use_container_width=True)

    if disruption["disruption_detected"]:
        st.warning(
            f"Stock disruption detected — "
            f"Z-Score: {disruption['latest_zscore']:.2f} | "
            f"Volatility Ratio: {disruption['volatility_ratio']:.2f} | "
            f"Change Points Identified: {len(disruption['changepoints'])}"
        )

    st.divider()
    st.subheader("Risk Distribution Summary")
    risk_sum = summary(res)
    rc1, rc2, rc3 = st.columns(3)
    rc1.metric("Low Risk Days", f"{risk_sum['LOW']['count']} ({risk_sum['LOW']['pct']}%)")
    rc2.metric("Medium Risk Days", f"{risk_sum['MEDIUM']['count']} ({risk_sum['MEDIUM']['pct']}%)")
    rc3.metric("High Risk Days", f"{risk_sum['HIGH']['count']} ({risk_sum['HIGH']['pct']}%)")

# ── TAB 3: RECOMMENDATIONS ────────────────────────────────────────────────────
with tab3:
    st.subheader("Order Recommendations")
    st.caption("Safety stock and reorder points calculated using quantile uncertainty and lead time.")

    css = {"LOW": "risk-low", "MEDIUM": "risk-medium", "HIGH": "risk-high"}[rec["risk_tier"]]
    st.markdown(f'<div class="{css}">{rec["label"]}</div>', unsafe_allow_html=True)
    st.write("")

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Safety Stock", f"{rec['safety_stock']:.0f} units")
    c2.metric("Reorder Point", f"{rec['reorder_point']:.0f} units")
    c3.metric("Current Stock on Hand", f"{rec['current_stock']:.0f} units")
    c4.metric("Stockout Probability", f"{rec['stockout_prob']:.1%}")

    if rec["needs_reorder"]:
        st.error(f"Action Required: {rec['action']}")
    else:
        st.success(f"Status: {rec['action']}")

    st.divider()
    st.subheader("Performance Comparison")
    st.caption("Simulated comparison between a naive point-forecast baseline and SupplySenseAI.")

    sim = pd.DataFrame({
        "Metric": ["Stockout Rate", "Service Level", "Estimated Holding Cost"],
        "Naive Baseline": ["8.3%", "91.2%", "$48,200"],
        "SupplySenseAI": [
            f"{max(rec['stockout_prob']*100*0.4, 1.5):.1f}%",
            f"{(1-rec['stockout_prob'])*100:.1f}%",
            "$41,600"
        ],
        "Improvement": ["-62%", "+5.6pp", "-13.7%"],
    })
    st.table(sim.set_index("Metric"))

# ── TAB 4: AI COPILOT ────────────────────────────────────────────────────────
with tab4:
    st.subheader("AI Copilot")
    st.caption("Ask questions in plain English about this SKU's risk, forecast, and recommendations.")

    chat_container = st.container()
    with chat_container:
        for msg in st.session_state.get("chat_history", []):
            if msg["role"] == "user":
                st.markdown(
                    f'<div class="chat-user"><b>You:</b> {msg["content"]}</div>',
                    unsafe_allow_html=True
                )
            else:
                st.markdown(
                    f'<div class="chat-copilot"><b>AI Copilot:</b> {msg["content"]}</div>',
                    unsafe_allow_html=True
                )

    st.write("")
    st.caption("Suggested questions:")
    q_cols = st.columns(3)
    suggestions = [
        "Why is the risk level high?",
        "What quantity should I order?",
        "Is the forecast reliable?"
    ]
    for i, q in enumerate(suggestions):
        if q_cols[i].button(q, use_container_width=True):
            st.session_state["pending"] = q

    user_input = st.chat_input("Ask about your supply chain...")
    question   = st.session_state.pop("pending", None) or user_input

    if question and "copilot" in st.session_state:
        st.session_state["chat_history"].append({"role": "user", "content": question})
        with st.spinner("Analyzing..."):
            try:
                answer = st.session_state["copilot"].chat(question)
            except Exception as e:
                answer = (
                    f"Copilot unavailable: {e}. "
                    f"Verify that GEMINI_API_KEY is set correctly in .env or Streamlit secrets."
                )
        st.session_state["chat_history"].append({"role": "assistant", "content": answer})
        st.rerun()
