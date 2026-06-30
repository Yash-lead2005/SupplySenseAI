import pandas as pd
import numpy as np
from pathlib import Path
import urllib.request
import os

DATA_DIR = Path("data")
DATA_DIR.mkdir(exist_ok=True)

FEATURE_COLS = [
    "sku_code", "day_of_week", "day_of_month", "week_of_year",
    "month", "quarter", "is_weekend", "is_month_end", "is_month_start",
    "lag_1", "lag_7", "lag_14", "lag_28",
    "roll_mean_7", "roll_std_7", "roll_max_7",
    "roll_mean_14", "roll_std_14",
    "roll_mean_28", "roll_std_28", "roll_max_28",
]
TARGET_COL = "daily_qty"


def download_dataset():
    path = DATA_DIR / "online_retail_II.csv"

    if not path.exists():
        raise FileNotFoundError(f"{path} not found!")

    print("Using local dataset:", path)
    return path

def load_and_clean(path):
    """Load and clean the raw dataset."""
    df = pd.read_csv(path, dtype={"CustomerID": str}, encoding="latin1")
    df.rename(columns={
    "Invoice": "InvoiceNo",
    "Price": "UnitPrice",
    "Customer ID": "CustomerID"
}, inplace=True)
    df = df[~df["InvoiceNo"].astype(str).str.startswith("C")]
    df = df[df["Quantity"] > 0]
    df = df[df["UnitPrice"] > 0]
    df.dropna(subset=["Description", "CustomerID"], inplace=True)
    df["InvoiceDate"] = pd.to_datetime(df["InvoiceDate"])
    df["Date"] = pd.to_datetime(df["InvoiceDate"].dt.date)
    df["Revenue"] = df["Quantity"] * df["UnitPrice"]
    return df


def build_daily_demand(df, n_skus=50):
    """Aggregate to daily demand per SKU and select top N SKUs."""

    # Aggregate quantity sold per day
    daily = (
        df.groupby(["StockCode", "Description", "Date"])
          .agg(daily_qty=("Quantity", "sum"))
          .reset_index()
    )

    # Select top selling SKUs
    top_skus = (
        daily.groupby("StockCode")["daily_qty"]
             .sum()
             .sort_values(ascending=False)
             .head(n_skus)
             .index
             .tolist()
    )

    daily = daily[daily["StockCode"].isin(top_skus)].copy()

    # Keep only necessary columns
    daily = daily[["StockCode", "Date", "daily_qty"]]

    # Remove duplicate StockCode-Date combinations
    daily = (
        daily.groupby(["StockCode", "Date"], as_index=True)
             .sum(numeric_only=True)
    )

    print("Index unique:", daily.index.is_unique)

    duplicates = daily.index[daily.index.duplicated()]
    print("Duplicate count:", len(duplicates))

    # Create full date range
    all_dates = pd.date_range(
        daily.index.get_level_values("Date").min(),
        daily.index.get_level_values("Date").max(),
        freq="D"
    )

    # Full SKU-Date index
    full_idx = pd.MultiIndex.from_product(
        [top_skus, all_dates],
        names=["StockCode", "Date"]
    )

    # Fill missing dates with zero demand
    daily = (
        daily.reindex(full_idx, fill_value=0)
             .reset_index()
    )

    return daily, top_skus

def engineer_features(df):
    """Add all time-series features."""
    df = df.sort_values(["StockCode", "Date"]).reset_index(drop=True)

    df["day_of_week"]    = df["Date"].dt.dayofweek
    df["day_of_month"]   = df["Date"].dt.day
    df["week_of_year"]   = df["Date"].dt.isocalendar().week.astype(int)
    df["month"]          = df["Date"].dt.month
    df["quarter"]        = df["Date"].dt.quarter
    df["is_weekend"]     = (df["day_of_week"] >= 5).astype(int)
    df["is_month_end"]   = df["Date"].dt.is_month_end.astype(int)
    df["is_month_start"] = df["Date"].dt.is_month_start.astype(int)
    df["sku_code"]       = df["StockCode"].astype("category").cat.codes

    grp = df.groupby("StockCode")["daily_qty"]
    for lag in [1, 7, 14, 28]:
        df[f"lag_{lag}"] = grp.shift(lag)

    for w in [7, 14, 28]:
        shifted = grp.shift(1)
        df[f"roll_mean_{w}"] = shifted.transform(lambda x: x.rolling(w, min_periods=1).mean())
        df[f"roll_std_{w}"]  = shifted.transform(lambda x: x.rolling(w, min_periods=1).std().fillna(0))
        df[f"roll_max_{w}"]  = shifted.transform(lambda x: x.rolling(w, min_periods=1).max())

    df.dropna(subset=["lag_1", "lag_7", "lag_14", "lag_28"], inplace=True)
    return df


def split_data(df):
    """Temporal 70/15/15 train/calibration/test split."""
    dates = sorted(df["Date"].unique())
    n = len(dates)
    train_end = dates[int(n * 0.70)]
    calib_end = dates[int(n * 0.85)]

    train = df[df["Date"] <= train_end]
    calib = df[(df["Date"] > train_end) & (df["Date"] <= calib_end)]
    test  = df[df["Date"] > calib_end]

    return (train[FEATURE_COLS], train[TARGET_COL],
            calib[FEATURE_COLS], calib[TARGET_COL],
            test[FEATURE_COLS],  test[TARGET_COL], test)


def run_pipeline():
    path = download_dataset()
    print("Loading and cleaning...")
    df_raw = load_and_clean(path)
    print("Building daily demand matrix...")
    daily, top_skus = build_daily_demand(df_raw)
    print("Engineering features...")
    daily_feat = engineer_features(daily)
    print(f"Dataset shape: {daily_feat.shape}")
    splits = split_data(daily_feat)
    print("Pipeline complete.")
    return daily_feat, splits, top_skus


if __name__ == "__main__":
    run_pipeline()
