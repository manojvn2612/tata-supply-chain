"""
module_2_global_xgboost_leadtime.py
-------------------------------------
ML Module 2 (Global): Lead-Time Prediction  (XGBoost)

ONE global XGBoost model trained on ALL materials simultaneously.
Material identity is encoded as a feature (label-encoded integer),
so the model learns per-supplier behaviour while sharing cross-material
patterns (e.g. Q4 congestion affects all suppliers, but Vendor B more).

Why global > per-material:
  - 5,475 rows vs 1,825 — better quantile calibration for P90
  - Learns: "transit_risk=2 ALWAYS means +7d regardless of material"
  - Learns: "Q4 + high demand_slope = delay for ALL suppliers"
  - Adding new material: just add rows, no new model file

Models:
  1. XGBRegressor (mean / P50) — point estimate
  2. XGBRegressor (P90 quantile) — upper bound for planning

Run:
  python module_2_global_xgboost_leadtime.py
  python module_2_global_xgboost_leadtime.py --no-retrain
"""

import sys
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.model_selection import TimeSeriesSplit
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_absolute_error, mean_squared_error
import joblib
from pathlib import Path

MODELS_DIR = Path("models")
MODELS_DIR.mkdir(exist_ok=True)

MODEL_MEAN_PATH = MODELS_DIR / "global_xgb_lt_mean.pkl"
MODEL_P90_PATH  = MODELS_DIR / "global_xgb_lt_p90.pkl"
MAT_ENC_PATH    = MODELS_DIR / "global_xgb_lt_mat_encoder.pkl"

# material_id encoded as integer is added automatically
LT_FEATURES = [
    "material_id_enc",          # ← global: learned per-material bias
    "nominal_lead_time_days",
    "transit_risk_level",
    "po_slippage_days",
    "qi_rejection",
    "month",
    "quarter",
    "day_of_year",
    "month_sin",
    "month_cos",
    "stock_cover_days",
    "demand_rolling_slope",
    "daily_demand",
]

XGB_BASE = dict(
    n_estimators          = 800,
    max_depth             = 6,
    learning_rate         = 0.03,
    subsample             = 0.8,
    colsample_bytree      = 0.8,
    min_child_weight      = 5,
    n_jobs                = -1,
    random_state          = 42,
    early_stopping_rounds = 40,
)


def _prepare(df: pd.DataFrame, mat_enc: LabelEncoder = None, fit: bool = False):
    df = df.copy().sort_values(["material_id", "date"]).reset_index(drop=True)

    if fit:
        mat_enc = LabelEncoder()
        df["material_id_enc"] = mat_enc.fit_transform(df["material_id"])
    else:
        df["material_id_enc"] = mat_enc.transform(df["material_id"])

    X = df[LT_FEATURES].values
    y = df["actual_lead_time_days"].values
    return X, y, mat_enc


def train_global_leadtime(data_path: str = "data/training_data.csv") -> dict:
    print("\n  Training GLOBAL XGBoost lead-time model (P50 + P90)...")
    df = pd.read_csv(data_path, parse_dates=["date"])

    X, y, mat_enc = _prepare(df, fit=True)

    # Time-series split — use last fold for evaluation
    tss    = TimeSeriesSplit(n_splits=5)
    splits = list(tss.split(X))
    tr_idx, val_idx = splits[-1]
    X_tr, y_tr = X[tr_idx], y[tr_idx]
    X_va, y_va = X[val_idx], y[val_idx]

    print(f"  Materials: {list(mat_enc.classes_)}")
    print(f"  Train rows: {len(X_tr):,}  |  Val rows: {len(X_va):,}")

    # ── P50 mean model ──
    mean_model = xgb.XGBRegressor(**XGB_BASE, objective="reg:squarederror")
    mean_model.fit(X_tr, y_tr, eval_set=[(X_va, y_va)], verbose=False)

    # ── P90 quantile model ──
    p90_model = xgb.XGBRegressor(**XGB_BASE,
                                  objective="reg:quantileerror",
                                  quantile_alpha=0.90)
    p90_model.fit(X_tr, y_tr, eval_set=[(X_va, y_va)], verbose=False)

    # ── Metrics ──
    y_mean = mean_model.predict(X_va)
    y_p90  = p90_model.predict(X_va)
    mae          = mean_absolute_error(y_va, y_mean)
    rmse         = np.sqrt(mean_squared_error(y_va, y_mean))
    p90_coverage = np.mean(y_va <= y_p90)

    print(f"  Global MAE={mae:.2f}d  RMSE={rmse:.2f}d  P90 coverage={p90_coverage:.1%}")

    # ── Per-material breakdown ──
    df_val = df.iloc[val_idx].copy()
    df_val["y_pred_mean"] = y_mean
    df_val["y_pred_p90"]  = y_p90
    print("  Per-material val performance:")
    for mat_id in sorted(df_val["material_id"].unique()):
        sub  = df_val[df_val["material_id"] == mat_id]
        m    = mean_absolute_error(sub["actual_lead_time_days"], sub["y_pred_mean"])
        cov  = (sub["actual_lead_time_days"] <= sub["y_pred_p90"]).mean()
        print(f"    {mat_id:<14}  MAE={m:.2f}d  P90 cov={cov:.1%}")

    # ── Feature importance ──
    imp    = dict(zip(LT_FEATURES, mean_model.feature_importances_))
    top5   = sorted(imp.items(), key=lambda x: -x[1])[:5]
    print("  Top features: " + "  |  ".join(f"{k}={v:.3f}" for k, v in top5))

    # Save
    joblib.dump(mean_model, MODEL_MEAN_PATH)
    joblib.dump(p90_model,  MODEL_P90_PATH)
    joblib.dump(mat_enc,    MAT_ENC_PATH)
    print(f"  Saved → {MODEL_MEAN_PATH}, {MODEL_P90_PATH}")

    return {"mean_model": mean_model, "p90_model": p90_model,
            "mat_enc": mat_enc, "mae": mae, "p90_coverage": p90_coverage}


def predict_leadtime_global(mat_id: str, feature_row: dict,
                             mean_model=None, p90_model=None,
                             mat_enc=None) -> dict:
    """
    Predict lead time for one upcoming PO using the global model.
    feature_row: dict of all LT_FEATURES except material_id_enc (added here).
    """
    if mean_model is None:
        mean_model = joblib.load(MODEL_MEAN_PATH)
        p90_model  = joblib.load(MODEL_P90_PATH)
        mat_enc    = joblib.load(MAT_ENC_PATH)

    row = feature_row.copy()
    row["material_id_enc"] = int(mat_enc.transform([mat_id])[0])

    X    = np.array([[row[f] for f in LT_FEATURES]])
    p50  = float(mean_model.predict(X)[0])
    p90  = float(p90_model.predict(X)[0])
    p90  = max(p90, p50)

    nominal  = feature_row.get("nominal_lead_time_days", p50)
    gap      = p90 - nominal

    flag = ("🔴  HIGH — P90 >7d above SAP nominal. Reorder immediately."
            if gap > 7 else
            "🟠  MEDIUM — P90 materially above SAP. Pull in reorder."
            if gap > 3 else
            "🟢  LOW — P90 close to SAP nominal. Normal planning.")

    return {
        "material_id":      mat_id,
        "sap_nominal_lt":   nominal,
        "predicted_lt_p50": round(p50, 1),
        "predicted_lt_p90": round(p90, 1),
        "gap_p90_vs_sap":   round(gap, 1),
        "risk_flag":        flag,
    }


def run_global_xgboost_leadtime(data_path: str = "data/training_data.csv",
                                 retrain: bool = True) -> dict:
    print("=" * 65)
    print("  MODULE 2 (Global): XGBOOST LEAD-TIME PREDICTION")
    print("=" * 65)

    if retrain or not MODEL_MEAN_PATH.exists():
        result     = train_global_leadtime(data_path)
        mean_model = result["mean_model"]
        p90_model  = result["p90_model"]
        mat_enc    = result["mat_enc"]
    else:
        print("  Loading saved global XGBoost models...")
        mean_model = joblib.load(MODEL_MEAN_PATH)
        p90_model  = joblib.load(MODEL_P90_PATH)
        mat_enc    = joblib.load(MAT_ENC_PATH)

    df = pd.read_csv(data_path, parse_dates=["date"])
    predictions = {}

    print("\n  LEAD-TIME PREDICTIONS (current PO context)")
    print("=" * 65)
    for mat_id in sorted(df["material_id"].unique()):
        latest = (df[df["material_id"] == mat_id]
                  .sort_values("date").iloc[-1])
        row = {f: latest[f] for f in LT_FEATURES if f != "material_id_enc"}
        pred = predict_leadtime_global(mat_id, row, mean_model, p90_model, mat_enc)
        predictions[mat_id] = pred

        print(f"\n  {mat_id}")
        print(f"  SAP Nominal : {pred['sap_nominal_lt']} days")
        print(f"  XGB P50     : {pred['predicted_lt_p50']} days")
        print(f"  XGB P90     : {pred['predicted_lt_p90']} days  "
              f"(gap: {pred['gap_p90_vs_sap']:+.1f}d)")
        print(f"  Assessment  : {pred['risk_flag']}")

    return predictions


if __name__ == "__main__":
    retrain = "--no-retrain" not in sys.argv
    run_global_xgboost_leadtime(retrain=retrain)
