# SupplySenseAI

**Uncertainty-Aware Supply Chain Risk Intelligence**

SupplySenseAI is a production-ready supply chain analytics platform that combines quantile regression, conformal prediction, and AI-powered analysis to help operations teams forecast demand, quantify risk, and make confident procurement decisions.

---

## Live Demo

Deploy directly to Streamlit Cloud — no local setup required for judges.

---

## Features

- **Demand Forecasting** — LightGBM quantile regression (P10 / P50 / P90) trained on real retail transaction data
- **Conformal Prediction** — Statistically valid uncertainty intervals with guaranteed coverage properties
- **Risk Classification** — Automatic LOW / MEDIUM / HIGH tier assignment based on composite risk scoring
- **Stock Disruption Detection** — Rolling Z-score, volatility ratio, and PELT change-point analysis
- **Order Recommendations** — Safety stock and reorder point calculations calibrated to risk tier and lead time
- **AI Copilot** — Conversational interface powered by Google Gemini 2.0 Flash for natural-language supply chain Q&A

---

## Project Structure

```
SupplySenseAI/
├── app.py                          # Main Streamlit application
├── requirements.txt                # Python dependencies
├── .env                            # API key (local development)
├── .streamlit/
│   └── secrets.toml                # Streamlit Cloud secrets template
├── models/                         # Pre-trained LightGBM + MAPIE model files
├── data/                           # Place online_retail_II.csv here
├── notebooks/
│   └── SupplySenseAI_Complete_Project.ipynb
└── src/
    ├── preprocessing/
    │   └── feature_engineering.py  # Data loading, cleaning, feature construction
    ├── forecasting/
    │   └── lightgbm_model.py       # Quantile regression training and inference
    ├── uncertainty/
    │   └── conformal_prediction.py # MAPIE conformal calibration and prediction
    ├── risk_engine/
    │   ├── risk_classifier.py      # Composite risk scoring and tier assignment
    │   └── stock_detector.py       # Demand anomaly and disruption detection
    ├── recommendation_engine/
    │   └── order_recommender.py    # Safety stock, reorder point, order sizing
    ├── copilot/
    │   └── ai_copilot.py           # Google Gemini 2.0 Flash integration
    └── evaluation/
        └── metrics.py              # Coverage and accuracy reporting
```

---

## Setup

### 1. Dataset

Download the **Online Retail II** dataset from UCI Machine Learning Repository and place it at:

```
data/online_retail_II.csv
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Configure API key

Edit `.env`:

```
GEMINI_API_KEY=your_key_here
```

### 4. Run locally

```bash
streamlit run app.py
```

---

## Streamlit Cloud Deployment

1. Push this repository to GitHub (ensure `data/online_retail_II.csv` is included or modify the download path in `feature_engineering.py`).
2. Go to [share.streamlit.io](https://share.streamlit.io) and connect your repository.
3. In **Settings → Secrets**, paste the contents of `.streamlit/secrets.toml`.
4. Click **Deploy**.

---

## Technical Approach

| Component | Method |
|---|---|
| Forecasting | LightGBM quantile regression at P10, P50, P90 |
| Uncertainty Quantification | MAPIE conformal prediction (split conformal) |
| Anomaly Detection | Rolling Z-score + volatility ratio + PELT change points |
| Risk Scoring | Composite of interval width ratio and volatility score |
| AI Copilot | Google Gemini 2.0 Flash via REST API |
| Frontend | Streamlit with Plotly interactive charts |

---

## Dataset

UCI Online Retail II — real UK-based e-commerce transactions (2009–2011).  
Top 50 SKUs by total sales volume are selected for analysis.

