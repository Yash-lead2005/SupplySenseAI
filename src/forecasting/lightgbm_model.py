import lightgbm as lgb
import numpy as np
import pandas as pd
import pickle
from pathlib import Path
from sklearn.metrics import mean_absolute_error, mean_squared_error

MODEL_DIR = Path("models")
MODEL_DIR.mkdir(exist_ok=True)

BASE_PARAMS = {
    "n_estimators":     800,
    "learning_rate":    0.05,
    "num_leaves":       63,
    "min_child_samples":20,
    "feature_fraction": 0.8,
    "bagging_fraction": 0.8,
    "bagging_freq":     5,
    "reg_alpha":        0.1,
    "reg_lambda":       0.1,
    "n_jobs":           -1,
    "verbosity":        -1,
    "random_state":     42,
    "objective":        "quantile",
}

CALLBACKS = [
    lgb.early_stopping(stopping_rounds=30, verbose=False),
    lgb.log_evaluation(period=200),
]


def train_quantile_model(X_train, y_train, X_val, y_val, quantile):
    params = {**BASE_PARAMS, "alpha": quantile}
    model = lgb.LGBMRegressor(**params)
    model.fit(X_train, y_train,
              eval_set=[(X_val, y_val)],
              eval_metric="quantile",
              callbacks=CALLBACKS)
    return model


def train_all_models(X_train, y_train, X_val, y_val):
    models = {}
    for q, label in zip([0.10, 0.50, 0.90], ["p10", "p50", "p90"]):
        print(f"Training {label.upper()} model...")
        models[label] = train_quantile_model(X_train, y_train, X_val, y_val, q)
        with open(MODEL_DIR / f"lgbm_{label}.pkl", "wb") as f:
            pickle.dump(models[label], f)
        print(f"  Saved lgbm_{label}.pkl")
    return models


def predict_quantiles(models, X):
    preds = {label: models[label].predict(X).clip(min=0)
             for label in ["p10", "p50", "p90"]}
    df = pd.DataFrame(preds, index=X.index)
    df["p10"] = df[["p10", "p50"]].min(axis=1)
    df["p90"] = df[["p50", "p90"]].max(axis=1)
    return df


def load_models():
    models = {}
    for label in ["p10", "p50", "p90"]:
        with open(MODEL_DIR / f"lgbm_{label}.pkl", "rb") as f:
            models[label] = pickle.load(f)
    return models


def evaluate(models, X_test, y_test):
    preds = predict_quantiles(models, X_test)
    mae  = mean_absolute_error(y_test, preds["p50"])
    rmse = np.sqrt(mean_squared_error(y_test, preds["p50"]))
    cov  = ((y_test.values >= preds["p10"].values) &
             (y_test.values <= preds["p90"].values)).mean()
    print(f"MAE  : {mae:.4f}")
    print(f"RMSE : {rmse:.4f}")
    print(f"Coverage P10-P90 : {cov:.2%}")
    return {"mae": mae, "rmse": rmse, "coverage_p10_p90": cov}
