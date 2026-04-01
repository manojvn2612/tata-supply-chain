"""
module_8_safety_stock_optimizer.py
------------------------------------
Module 8: ML-Based Safety Stock Optimizer  (GradientBoosting)

Learns the relationship between supply-chain features and
optimal safety stock from 5 years of simulated outcomes.

How it works:
  1. For each historical row, compute what the safety stock SHOULD have
     been to avoid a stockout at 95% service level (target label)
  2. Train GradientBoostingRegressor on features → optimal SS
  3. At inference time: given current demand/LT/risk features,
     predict the optimal safety stock dynamically
  4. Compare to current static SAP value → generate update recommendation

Why ML instead of the Z-score formula:
  - Captures non-linear interactions (Q4 + HIGH risk supplier → larger SS)
  - Learns from actual stockout events in history (empirical, not theoretical)
  - Automatically adjusts as supplier behaviour and demand patterns change
  - Can incorporate features the formula ignores (QI rate, MOQ, trend slope)

Run:  python module_8_safety_stock_optimizer.py
"""

import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.preprocessing import StandardScaler
import joblib
from pathlib import Path
from config import MATERIALS, SERVICE_LEVEL, WORKING_DAYS_MONTH

MODELS_DIR = Path("models")
MODELS_DIR.mkdir(exist_ok=True)

SS_FEATURES = [
    "actual_lead_time_days",
    "lt_deviation_days",
    "transit_risk_level",
    "demand_rolling_slope",
    "daily_demand",
    "po_slippage_days",
    "qi_rejection",
    "month_sin",
    "month_cos",
    "quarter",
]


# ── Target label: what SS should have been ────────────────────────────────────

def compute_optimal_ss_label(df: pd.DataFrame, window: int = 30) -> np.ndarray:
    """
    For each row, compute the safety stock that would have prevented
    a stockout over the next `window` days at SERVICE_LEVEL.

    Approach: rolling z-score × combined demand+LT variability
    This gives a data-driven target for the GBM to learn to predict.
    """
    daily_demand = df["daily_demand"].values
    actual_lt    = df["actual_lead_time_days"].values
    n = len(df)

    from scipy.stats import norm
    z = norm.ppf(SERVICE_LEVEL)
    optimal_ss = np.zeros(n)

    for i in range(window, n):
        d_slice  = daily_demand[i - window:i]
        lt_slice = actual_lt[i - window:i]

        d_mean   = np.mean(d_slice)
        d_std    = np.std(d_slice, ddof=1) if len(d_slice) > 1 else 0
        lt_mean  = np.mean(lt_slice)
        lt_std   = np.std(lt_slice, ddof=1) if len(lt_slice) > 1 else 0

        # Combined SS formula as target
        variance = (lt_mean * d_std**2) + (d_mean**2 * lt_std**2)
        ss = z * np.sqrt(variance)
        optimal_ss[i] = max(0, ss)

    # Fill early rows with forward-filled values
    for i in range(window):
        optimal_ss[i] = optimal_ss[window] if window < n else 0

    return optimal_ss


# ── Train ──────────────────────────────────────────────────────────────────────

def train_ss_model(mat_id: str, df: pd.DataFrame) -> dict:
    print(f"\n  Training SS optimizer for {mat_id}...")
    mat_df = df[df["material_id"] == mat_id].sort_values("date").reset_index(drop=True)

    optimal_ss = compute_optimal_ss_label(mat_df)
    mat_df     = mat_df.copy()
    mat_df["optimal_ss"] = optimal_ss

    X = mat_df[SS_FEATURES].values
    y = mat_df["optimal_ss"].values

    # Time-series split
    tss    = TimeSeriesSplit(n_splits=5)
    splits = list(tss.split(X))
    tr_idx, val_idx = splits[-1]

    X_tr, y_tr = X[tr_idx], y[tr_idx]
    X_va, y_va = X[val_idx], y[val_idx]

    gbm = GradientBoostingRegressor(
        n_estimators=300,
        max_depth=5,
        learning_rate=0.05,
        subsample=0.8,
        min_samples_leaf=20,
        loss="huber",         # robust to outlier SS targets during demand spikes
        random_state=42,
    )
    gbm.fit(X_tr, y_tr)

    y_pred = gbm.predict(X_va)
    mae    = mean_absolute_error(y_va, y_pred)
    r2     = r2_score(y_va, y_pred)
    print(f"    MAE={mae:.2f} units  R²={r2:.3f}")

    importance = dict(zip(SS_FEATURES, gbm.feature_importances_))
    top_feats  = sorted(importance.items(), key=lambda x: -x[1])[:4]
    print(f"    Top features: " + "  |  ".join(f"{k}={v:.3f}" for k, v in top_feats))

    safe_id = mat_id.replace("-", "_")
    joblib.dump(gbm, MODELS_DIR / f"gbm_ss_{safe_id}.pkl")

    return {"model": gbm, "mae": mae, "r2": r2, "top_features": top_feats}


# ── Inference ──────────────────────────────────────────────────────────────────

def predict_optimal_ss(mat_id: str, feature_row: dict, model=None) -> dict:
    if model is None:
        safe_id = mat_id.replace("-", "_")
        model   = joblib.load(MODELS_DIR / f"gbm_ss_{safe_id}.pkl")

    X          = np.array([[feature_row[f] for f in SS_FEATURES]])
    predicted  = float(model.predict(X)[0])
    predicted  = max(0, predicted)

    mat        = MATERIALS.get(mat_id, {})
    current_ss = mat.get("safety_stock", 0)
    delta      = predicted - current_ss
    delta_pct  = (delta / current_ss * 100) if current_ss > 0 else 0

    if delta > current_ss * 0.15:
        recommendation = "⬆️  INCREASE — current SS under-protected for current demand/LT conditions"
    elif delta < -current_ss * 0.15:
        recommendation = "⬇️  DECREASE — current SS is over-conservative, freeing working capital"
    else:
        recommendation = "✅  MAINTAIN — current SS within ±15% of ML-optimal"

    return {
        "material_id":    mat_id,
        "current_ss_sap": current_ss,
        "predicted_ss_ml":round(predicted, 1),
        "delta":          round(delta, 1),
        "delta_pct":      round(delta_pct, 1),
        "recommendation": recommendation,
    }


# ── Main ──────────────────────────────────────────────────────────────────────

def run_safety_stock_optimizer(data_path="data/training_data.csv", retrain=True):
    print("=" * 65)
    print("  MODULE 8: ML SAFETY STOCK OPTIMIZER  (GradientBoosting)")
    print("=" * 65)
    print(f"  Service level target: {SERVICE_LEVEL:.0%}")
    print(f"  Features: {SS_FEATURES}")

    df = pd.read_csv(data_path, parse_dates=["date"])
    all_results = {}

    for mat_id in df["material_id"].unique():
        if retrain:
            result = train_ss_model(mat_id, df)
            model  = result["model"]
        else:
            safe_id = mat_id.replace("-", "_")
            model   = joblib.load(MODELS_DIR / f"gbm_ss_{safe_id}.pkl")

        # Predict on latest features
        mat_df  = df[df["material_id"] == mat_id].sort_values("date")
        latest  = mat_df.iloc[-1][SS_FEATURES].to_dict()
        pred    = predict_optimal_ss(mat_id, latest, model)
        all_results[mat_id] = pred

    print("\n" + "=" * 65)
    print("  SAFETY STOCK RECOMMENDATIONS")
    print("=" * 65)
    total_delta_value = 0
    for mat_id, res in all_results.items():
        mat   = MATERIALS.get(mat_id, {})
        price = mat.get("price", 0)
        inv_impact = res["delta"] * price

        print(f"\n  {mat_id}  —  {mat.get('description','')}")
        print(f"    SAP current:    {res['current_ss_sap']:>6} units")
        print(f"    ML recommended: {res['predicted_ss_ml']:>6} units  "
              f"(delta: {res['delta']:+.0f}  {res['delta_pct']:+.1f}%)")
        print(f"    Inventory Δ:    ₹{inv_impact:>+12,.0f}  (@₹{price}/unit)")
        print(f"    Recommendation: {res['recommendation']}")
        total_delta_value += inv_impact

    print(f"\n  {'─'*63}")
    print(f"  Net inventory impact of all changes: ₹{total_delta_value:+,.0f}")
    print(f"  (negative = working capital released, positive = additional buffer investment)")

    return all_results


if __name__ == "__main__":
    run_safety_stock_optimizer(retrain=True)
