"""
Order recommendation engine.
Calculates safety stock, reorder points, and stockout probability
using quantile forecast uncertainty and lead time.
"""
import numpy as np
from scipy.stats import norm

LEAD_TIME_DEFAULT = 3
Z_SCORE_LOOKUP    = {
    "LOW":    norm.ppf(0.90),
    "MEDIUM": norm.ppf(0.95),
    "HIGH":   norm.ppf(0.99),
}


def _demand_sigma(p10: float, p90: float) -> float:
    """Estimate demand standard deviation from the quantile spread."""
    return max((p90 - p10) / 2.5631, 0.01)


def compute_safety_stock(risk_tier: str, p10: float, p90: float,
                          lead_time: int = LEAD_TIME_DEFAULT) -> float:
    """Calculate safety stock based on risk tier and demand uncertainty."""
    z   = Z_SCORE_LOOKUP[risk_tier]
    sig = _demand_sigma(p10, p90)
    return max(z * sig * np.sqrt(lead_time), 0.0)


def compute_reorder_point(p50: float, safety_stock: float,
                           lead_time: int = LEAD_TIME_DEFAULT) -> float:
    """Calculate the reorder point: expected demand over lead time + safety stock."""
    return p50 * lead_time + safety_stock


def compute_stockout_probability(current_units: float, reorder_point: float,
                                  p10: float, p90: float,
                                  lead_time: int = LEAD_TIME_DEFAULT) -> float:
    """Estimate probability of a stockout given current units on hand."""
    sig = _demand_sigma(p10, p90) * np.sqrt(lead_time)
    if sig < 1e-6:
        return 0.0 if current_units >= reorder_point else 1.0
    return round(float(1.0 - norm.cdf((current_units - reorder_point) / sig)), 4)


def recommend(risk_tier: str, p10: float, p50: float, p90: float,
              current_units: float, sku_id: str = "",
              lead_time: int = LEAD_TIME_DEFAULT) -> dict:
    """
    Generate a procurement recommendation for a given SKU.

    Returns a dictionary containing safety stock, reorder point,
    stockout probability, and a plain-English action statement.
    """
    ss   = compute_safety_stock(risk_tier, p10, p90, lead_time)
    rop  = compute_reorder_point(p50, ss, lead_time)
    sp   = compute_stockout_probability(current_units, rop, p10, p90, lead_time)
    needs_reorder = current_units < rop
    order_qty     = max(rop - current_units + ss, 0.0) if needs_reorder else 0.0

    tier_labels = {
        "LOW":    "LOW RISK — Supply chain healthy",
        "MEDIUM": "MEDIUM RISK — Review procurement schedule",
        "HIGH":   "HIGH RISK — Immediate procurement action required",
    }

    if needs_reorder:
        action = (
            f"Place order for {order_qty:.0f} units immediately. "
            f"Current stock ({current_units:.0f} units) is below the reorder point "
            f"({rop:.0f} units)."
        )
    else:
        buffer = current_units - rop
        action = (
            f"No order required at this time. "
            f"Current stock is {buffer:.0f} units above the reorder point ({rop:.0f} units)."
        )

    return {
        "sku_id":            sku_id,
        "risk_tier":         risk_tier,
        "label":             tier_labels[risk_tier],
        "safety_stock":      round(ss, 1),
        "reorder_point":     round(rop, 1),
        "current_stock": round(current_units, 1),
        "stockout_prob":     sp,
        "needs_reorder":     needs_reorder,
        "order_qty":         round(order_qty, 0),
        "action":            action,
    }
