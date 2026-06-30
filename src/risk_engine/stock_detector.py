"""
Stock disruption detector.
Identifies demand anomalies, volatility spikes, and structural change points
that may indicate supply chain disruptions requiring elevated risk classification.
"""
import numpy as np
import pandas as pd
import ruptures as rpt
from typing import List, Dict


def rolling_zscore(series: pd.Series, window: int = 28,
                    threshold: float = 3.0) -> pd.DataFrame:
    """Compute rolling Z-score and flag anomalies exceeding the threshold."""
    mean = series.rolling(window, min_periods=7).mean()
    std  = series.rolling(window, min_periods=7).std().replace(0, 1e-6)
    z    = (series - mean) / std
    return pd.DataFrame({
        "z_score":    z,
        "is_anomaly": (z.abs() > threshold).astype(int),
    })


def volatility_ratio(series: pd.Series, short: int = 7, long: int = 28,
                      threshold: float = 2.0) -> pd.DataFrame:
    """Compute short/long volatility ratio and flag high-volatility periods."""
    s_short = series.rolling(short, min_periods=3).std().replace(0, 1e-6)
    s_long  = series.rolling(long,  min_periods=7).std().replace(0, 1e-6)
    ratio   = s_short / s_long
    return pd.DataFrame({
        "vol_ratio":   ratio,
        "is_volatile": (ratio > threshold).astype(int),
    })


def detect_changepoints(series: pd.Series, pen: float = 10.0) -> List[int]:
    """Detect structural change points in the demand series using PELT algorithm."""
    signal = series.values.astype(float)
    if len(signal) < 20:
        return []
    try:
        algo = rpt.Pelt(model="rbf", min_size=5, jump=1)
        bps  = algo.fit(signal).predict(pen=pen)
        return [b for b in bps if b < len(signal)]
    except Exception:
        return []


def underestimation_audit(y_true, conf_lower, conf_upper,
                           window: int = 30,
                           min_coverage: float = 0.90) -> Dict:
    """
    Audit whether recent predictions are systematically underestimating demand.
    If empirical coverage falls below the minimum, force HIGH risk classification.
    """
    in_interval = (
        (y_true.iloc[-window:] >= conf_lower.iloc[-window:]) &
        (y_true.iloc[-window:] <= conf_upper.iloc[-window:])
    )
    coverage = in_interval.mean()
    failed   = coverage < min_coverage
    return {
        "recent_coverage":  round(coverage, 4),
        "coverage_failed":  failed,
        "force_high_risk":  failed,
    }


def run_stock_detection(series: pd.Series,
                         y_true=None,
                         conf_lower=None,
                         conf_upper=None) -> Dict:
    """
    Run the full stock disruption detection pipeline.

    Returns a dictionary with detection flags, Z-scores, volatility metrics,
    and change point locations.
    """
    z_df   = rolling_zscore(series)
    vol_df = volatility_ratio(series)
    bps    = detect_changepoints(series)

    recent_anomalies = int(z_df["is_anomaly"].tail(14).sum())
    is_volatile      = bool(vol_df["is_volatile"].tail(7).any())
    disruption       = (recent_anomalies > 2) or is_volatile or (len(bps) > 0)

    audit = {}
    if disruption and y_true is not None:
        audit = underestimation_audit(y_true, conf_lower, conf_upper)

    return {
        "disruption_detected": disruption,
        "recent_anomalies": recent_anomalies,
        "is_volatile":      is_volatile,
        "changepoints":     bps,
        "force_high_risk":  audit.get("force_high_risk", False),
        "recent_coverage":  audit.get("recent_coverage", None),
        "latest_zscore":    float(z_df["z_score"].iloc[-1]),
        "volatility_ratio": float(vol_df["vol_ratio"].iloc[-1]),
    }
