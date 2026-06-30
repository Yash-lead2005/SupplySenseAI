import pandas as pd

LOW_MED  = 0.20
MED_HIGH = 0.40


def risk_score(p10, p50, p90, roll_std_7, roll_mean_28,
               w_iv=0.6, w_vs=0.4) -> float:
    iv = (p90 - p10) / max(float(p50), 1.0)
    vs = float(roll_std_7) / max(float(roll_mean_28), 1.0)
    return round(min((w_iv * iv + w_vs * vs) / 2.0, 1.0), 4)


def classify(score: float, force_high=False) -> str:
    if force_high:        return "HIGH"
    if score < LOW_MED:   return "LOW"
    if score < MED_HIGH:  return "MEDIUM"
    return "HIGH"


def classify_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["risk_score"] = df.apply(
        lambda r: risk_score(r["p10"], r["p50"], r["p90"],
                             r.get("roll_std_7", 0), r.get("roll_mean_28", 1)),
        axis=1)
    force_col = "force_high_risk" if "force_high_risk" in df.columns else None
    df["risk_tier"] = df.apply(
        lambda r: classify(r["risk_score"],
                           r[force_col] if force_col else False), axis=1)
    return df


def summary(df: pd.DataFrame) -> dict:
    counts = df["risk_tier"].value_counts()
    total  = len(df)
    return {t: {"count": counts.get(t, 0),
                "pct":   round(counts.get(t, 0) / total * 100, 1)}
            for t in ["LOW", "MEDIUM", "HIGH"]}
