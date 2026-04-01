"""
module_2_xgboost_leadtime.py
------------------------------
ML Module 2: Lead-Time Prediction  (XGBoost)

Predicts actual lead time (days) for each upcoming PO given:
  supplier behaviour features, order characteristics, seasonal context.

Why XGBoost here vs Monte Carlo:
  - Learns non-linear interactions (e.g. large PO qty in Q4 = longer LT)
  - Captures supplier-specific patterns from 5 years of GR history
  - Outputs P50 / P90 via quantile regression — proper uncertainty bounds
  - Feature importance reveals *why* LT is high (actionable for planners)

Models trained:
  1. XGBRegressor  (objective=reg:squarederror) → point prediction
  2. XGBRegressor  (objective=reg:quantileerror, quantile_alpha=0.9) → P90

Run:  python module_2_xgboost_leadtime.py
"""

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_absolute_error, mean_squared_error
import joblib
from pathlib import Path

MODELS_DIR = Path("models")
MODELS_DIR.mkdir(exist_ok=True)

# ── Features used to predict actual lead time ─────────────────────────────────
LT_FEATURES = [
    "nominal_lead_time_days",   # SAP-stored baseline
    "transit_risk_level",       # 0=Low, 1=Med, 2=High
    "po_slippage_days",         # recent PO confirmation slippage
    "qi_rejection",             # last receipt had quality issue
    "month",                    # seasonality
    "quarter",                  # Q4 congestion effect
    "day_of_year",              # precise seasonal position
    "month_sin",                # smooth cyclic encoding
    "month_cos",
    "stock_cover_days",         # urgency signal — low cover → expedite pressure
    "demand_rolling_slope",     # demand trend context
    "daily_demand",             # absolute volume pressure
]

XGB_PARAMS_MEAN = {
    "n_estimators":     500,
    "max_depth":        6,
    "learning_rate":    0.05,
    "subsample":        0.8,
    "colsample_bytree": 0.8,
    "min_child_weight": 5,
    "objective":        "reg:squarederror",
    "random_state":     42,
    "n_jobs":           -1,
    "early_stopping_rounds": 30,
}

XGB_PARAMS_P90 = {
    **XGB_PARAMS_MEAN,
    "objective":       "reg:quantileerror",
    "quantile_alpha":   0.90,
}


# ── Train ──────────────────────────────────────────────────────────────────────
def train_leadtime_models(df: pd.DataFrame, mat_id: str) -> dict:
    print(f"\n  Training XGBoost lead-time models for {mat_id}...")
    mat_df = df[df["material_id"] == mat_id].sort_values("date").reset_index(drop=True)

    X = mat_df[LT_FEATURES].values
    y = mat_df["actual_lead_time_days"].values

    # Time-series split — never train on future data
    tss    = TimeSeriesSplit(n_splits=5)
    splits = list(tss.split(X))
    tr_idx, val_idx = splits[-1]   # use last fold for final eval

    X_train, y_train = X[tr_idx],  y[tr_idx]
    X_val,   y_val   = X[val_idx], y[val_idx]

    # ── Mean model ──
    mean_model = xgb.XGBRegressor(**XGB_PARAMS_MEAN)
    mean_model.fit(
        X_train, y_train,
        eval_set=[(X_val, y_val)],
        verbose=False,
    )

    # ── P90 quantile model ──
    p90_model = xgb.XGBRegressor(**XGB_PARAMS_P90)
    p90_model.fit(
        X_train, y_train,
        eval_set=[(X_val, y_val)],
        verbose=False,
    )

    # ── Eval ──
    y_pred_mean = mean_model.predict(X_val)
    y_pred_p90  = p90_model.predict(X_val)
    mae  = mean_absolute_error(y_val, y_pred_mean)
    rmse = np.sqrt(mean_squared_error(y_val, y_pred_mean))
    p90_coverage = np.mean(y_val <= y_pred_p90)

    print(f"    MAE={mae:.2f}d  RMSE={rmse:.2f}d  P90 coverage={p90_coverage:.1%} (target ≥90%)")

    # ── Feature importance ──
    importance = dict(zip(LT_FEATURES, mean_model.feature_importances_))
    top_feats  = sorted(importance.items(), key=lambda x: -x[1])[:5]
    print(f"    Top features: " + "  |  ".join(f"{k}={v:.3f}" for k, v in top_feats))

    # Save
    safe_id = mat_id.replace("-", "_")
    joblib.dump(mean_model, MODELS_DIR / f"xgb_lt_mean_{safe_id}.pkl")
    joblib.dump(p90_model,  MODELS_DIR / f"xgb_lt_p90_{safe_id}.pkl")

    return {
        "mean_model": mean_model, "p90_model": p90_model,
        "mae": mae, "rmse": rmse, "p90_coverage": p90_coverage,
        "top_features": top_feats,
    }


# ── Inference ──────────────────────────────────────────────────────────────────
def predict_leadtime(mat_id: str, feature_row: dict, models: dict = None) -> dict:
    """
    Predict lead time for one upcoming PO.
    feature_row: dict of {feature_name: value} — all LT_FEATURES required.
    """
    if models is None:
        safe_id    = mat_id.replace("-", "_")
        mean_model = joblib.load(MODELS_DIR / f"xgb_lt_mean_{safe_id}.pkl")
        p90_model  = joblib.load(MODELS_DIR / f"xgb_lt_p90_{safe_id}.pkl")
    else:
        mean_model = models["mean_model"]
        p90_model  = models["p90_model"]

    row = np.array([[feature_row[f] for f in LT_FEATURES]])
    p50 = float(mean_model.predict(row)[0])
    p90 = float(p90_model.predict(row)[0])
    p90 = max(p90, p50)   # P90 must be ≥ P50

    sap_nominal = feature_row["nominal_lead_time_days"]
    gap_vs_sap  = p90 - sap_nominal

    if gap_vs_sap > 7:
        risk_flag = "🔴  HIGH — P90 exceeds SAP nominal by >7 days. Reorder NOW."
    elif gap_vs_sap > 3:
        risk_flag = "🟠  MEDIUM — P90 materially above SAP nominal. Pull in reorder."
    else:
        risk_flag = "🟢  LOW — P90 close to SAP nominal. Normal planning."

    return {
        "material_id":       mat_id,
        "sap_nominal_lt":    sap_nominal,
        "predicted_lt_p50":  round(p50, 1),
        "predicted_lt_p90":  round(p90, 1),
        "gap_p90_vs_sap":    round(gap_vs_sap, 1),
        "risk_flag":         risk_flag,
    }


# ── Main ──────────────────────────────────────────────────────────────────────
def run_xgboost_leadtime(data_path="data/training_data.csv", retrain=True):
    print("=" * 65)
    print("  MODULE 2: XGBOOST LEAD-TIME PREDICTION")
    print("=" * 65)
    print("  Models: XGBRegressor (mean) + XGBRegressor (P90 quantile)")

    df = pd.read_csv(data_path, parse_dates=["date"])
    all_models  = {}
    predictions = {}

    for mat_id in df["material_id"].unique():
        if retrain:
            result = train_leadtime_models(df, mat_id)
            all_models[mat_id] = result
        else:
            safe_id = mat_id.replace("-", "_")
            all_models[mat_id] = {
                "mean_model": joblib.load(MODELS_DIR / f"xgb_lt_mean_{safe_id}.pkl"),
                "p90_model":  joblib.load(MODELS_DIR / f"xgb_lt_p90_{safe_id}.pkl"),
            }

        # Predict using the most recent row as "current PO context"
        mat_df = df[df["material_id"] == mat_id].sort_values("date")
        latest = mat_df.iloc[-1][LT_FEATURES].to_dict()
        pred   = predict_leadtime(mat_id, latest, models=all_models[mat_id])
        predictions[mat_id] = pred

    # ── Print results ──
    print("\n" + "=" * 65)
    print("  LEAD-TIME PREDICTIONS FOR CURRENT OPEN POs")
    print("=" * 65)
    for mat_id, pred in predictions.items():
        print(f"\n  {mat_id}")
        print(f"  SAP Nominal : {pred['sap_nominal_lt']} days")
        print(f"  XGB P50     : {pred['predicted_lt_p50']} days")
        print(f"  XGB P90     : {pred['predicted_lt_p90']} days  (gap vs SAP: {pred['gap_p90_vs_sap']:+.1f}d)")
        print(f"  Assessment  : {pred['risk_flag']}")

    return predictions


if __name__ == "__main__":
    run_xgboost_leadtime(retrain=True)
