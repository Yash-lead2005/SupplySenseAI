import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error


def mae(y, yhat):   return round(mean_absolute_error(y, yhat), 4)
def rmse(y, yhat):  return round(np.sqrt(mean_squared_error(y, yhat)), 4)
def mape(y, yhat):
    mask = np.array(y) > 0
    return round(np.mean(np.abs((np.array(y)[mask] - np.array(yhat)[mask])
                                 / np.array(y)[mask])) * 100, 2)

def coverage(y, lo, hi):
    return round(((np.array(y) >= np.array(lo)) &
                   (np.array(y) <= np.array(hi))).mean(), 4)

def calibration_error(y, lo, hi, declared):
    return round(abs(coverage(y, lo, hi) - declared), 4)

def pinball(y, yhat, q):
    e = np.array(y) - np.array(yhat)
    return round(np.where(e >= 0, q * e, (q - 1) * e).mean(), 4)

def winkler(y, lo, hi, alpha=0.05):
    y, lo, hi = np.array(y), np.array(lo), np.array(hi)
    w = hi - lo
    p = (2/alpha) * (np.maximum(lo - y, 0) + np.maximum(y - hi, 0))
    return round((w + p).mean(), 4)

def full_report(y_true, p10, p50, p90, conf_lo, conf_hi, declared=0.95):
    r = {
        "MAE (P50)":           mae(y_true, p50),
        "RMSE (P50)":          rmse(y_true, p50),
        "MAPE (P50) %":        mape(y_true, p50),
        "Pinball P10":         pinball(y_true, p10, 0.10),
        "Pinball P50":         pinball(y_true, p50, 0.50),
        "Pinball P90":         pinball(y_true, p90, 0.90),
        "Coverage P10-P90":    coverage(y_true, p10, p90),
        "Coverage Conformal":  coverage(y_true, conf_lo, conf_hi),
        "Calibration Error":   calibration_error(y_true, conf_lo, conf_hi, declared),
        "Winkler Score":       winkler(y_true, conf_lo, conf_hi),
        "Passes 3% Gate":      calibration_error(y_true, conf_lo, conf_hi, declared) <= 0.03,
    }
    print("\n=== EVALUATION REPORT ===")
    for k, v in r.items():
        print(f"  {k:<25}: {v}")
    return r
