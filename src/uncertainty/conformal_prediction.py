"""
Conformal prediction wrapper — compatible with both:
  mapie 0.9.x  (MapieRegressor)
  mapie 1.x    (SplitConformalRegressor)
"""
import numpy as np
import pandas as pd
import pickle
from pathlib import Path

MODEL_DIR = Path("models")

# ── Detect installed mapie version ────────────────────────────────────────────
import mapie as _mapie
_MAPIE_MAJOR = int(_mapie.__version__.split(".")[0])

if _MAPIE_MAJOR >= 1:
    from mapie.regression import SplitConformalRegressor as _Backend
    _NEW_API = True
else:
    from mapie.regression import MapieRegressor as _Backend          # type: ignore
    _NEW_API = False


# ── Public API ────────────────────────────────────────────────────────────────

def fit_mapie(base_model, X_calib, y_calib, alpha=0.05):
    """
    Calibrate a conformal predictor on the held-out calibration set.
    Guarantees: P(y ∈ [lower, upper]) ≥ 1 − alpha
    """
    confidence = 1.0 - alpha

    if _NEW_API:
        # mapie >= 1.0: SplitConformalRegressor
        # 'prefit=True' means the base_model is already trained;
        # conformalize() calibrates the residual scores on X_calib / y_calib.
        mapie = _Backend(
            estimator=base_model,
            confidence_level=confidence,
            conformity_score="absolute",
            prefit=True,
        )
        mapie.conformalize(X_calib, y_calib)
    else:
        # mapie 0.9.x: MapieRegressor
        mapie = _Backend(
            estimator=base_model,
            method="base",
            cv="prefit",
            random_state=42,
        )
        mapie.fit(X_calib, y_calib)

    MODEL_DIR.mkdir(exist_ok=True)
    with open(MODEL_DIR / "mapie_model.pkl", "wb") as f:
        pickle.dump({"mapie": mapie, "alpha": alpha, "new_api": _NEW_API}, f)

    print(f"MAPIE calibrated (mapie {_mapie.__version__}). "
          f"Target coverage: {confidence:.0%}")
    return mapie


def predict_conformal(mapie, X, alpha=0.05):
    """Return a DataFrame with columns: y_pred, conf_lower, conf_upper."""
    confidence = 1.0 - alpha

    if _NEW_API:
        # predict_interval returns (y_pred [n,], intervals [n, 2, 1])
        # where intervals[:, 0, 0] = lower, intervals[:, 1, 0] = upper
        y_pred, intervals = mapie.predict_interval(X)
        lower = np.clip(intervals[:, 0, 0], 0, None)
        upper = intervals[:, 1, 0]
    else:
        # predict returns (y_pred [n,], pis [n, 2, n_alpha])
        y_pred, y_pis = mapie.predict(X, alpha=alpha)
        lower = np.clip(y_pis[:, 0, 0], 0, None)
        upper = y_pis[:, 1, 0]

    return pd.DataFrame(
        {
            "y_pred":     np.clip(y_pred, 0, None),
            "conf_lower": lower,
            "conf_upper": upper,
        },
        index=X.index,
    )


def evaluate_coverage(conf_preds, y_true, declared_alpha=0.05):
    """Check empirical coverage vs declared level. Gate: within ±3%."""
    in_interval = (
        (y_true.values >= conf_preds["conf_lower"].values) &
        (y_true.values <= conf_preds["conf_upper"].values)
    )
    empirical = in_interval.mean()
    declared  = 1.0 - declared_alpha
    error     = abs(empirical - declared)
    passes    = error <= 0.03

    print(
        f"Declared: {declared:.0%} | Empirical: {empirical:.2%} | "
        f"Error: {error:.4f} | Gate: {'PASS' if passes else 'FAIL'}"
    )
    return {
        "empirical_coverage": empirical,
        "calibration_error":  error,
        "passes_gate":        passes,
    }


def widen_if_needed(conf_preds, coverage_result, multiplier=1.3):
    """Auto-widen intervals if coverage gate fails (e.g. during a demand disruption)."""
    if not coverage_result["passes_gate"]:
        center = conf_preds["y_pred"]
        half   = (conf_preds["conf_upper"] - conf_preds["conf_lower"]) / 2
        conf_preds = conf_preds.copy()
        conf_preds["conf_lower"] = (center - half * multiplier).clip(lower=0)
        conf_preds["conf_upper"] =  center + half * multiplier
        conf_preds["widened"]    = True
        print("Intervals widened due to coverage failure.")
    else:
        conf_preds["widened"] = False
    return conf_preds


def load_mapie():
    with open(MODEL_DIR / "mapie_model.pkl", "rb") as f:
        obj = pickle.load(f)
    return obj["mapie"], obj["alpha"]
