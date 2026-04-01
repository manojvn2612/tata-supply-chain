"""
module_8_global_safety_stock_optimizer.py
-------------------------------------------
ML Module 8 (Global): Safety Stock Optimizer  (GradientBoosting)

ONE global GBM trained across ALL materials simultaneously.
material_id encoded as integer so the model learns per-material SS
behaviour while sharing cross-supply-chain signal.

Why global beats per-material:
  - 5,475 rows vs 1,825 → better generalisation
  - Learns "transit_risk=2 always needs larger SS" across all materials
  - New material gets SS recommendation without retraining
  - 1 model file instead of 3

Run:
  python module_8_global_safety_stock_optimizer.py
  python module_8_global_safety_stock_optimizer.py --no-retrain
"""

import sys
import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.preprocessing import LabelEncoder
from scipy.stats import norm
import joblib
from pathlib import Path

try:
    from config import MATERIALS, SERVICE_LEVEL
except ImportError:
    MATERIALS      = {}
    SERVICE_LEVEL  = 0.95

MODELS_DIR   = Path("models"); MODELS_DIR.mkdir(exist_ok=True)
MODEL_PATH   = MODELS_DIR / "global_gbm_ss.pkl"
MAT_ENC_PATH = MODELS_DIR / "global_ss_mat_encoder.pkl"

SS_FEATURES = [
    "material_id_enc",
    "actual_lead_time_days", "lt_deviation_days", "transit_risk_level",
    "demand_rolling_slope",  "daily_demand",       "po_slippage_days",
    "qi_rejection",          "month_sin",           "month_cos", "quarter",
]


def _compute_optimal_ss(df: pd.DataFrame, window: int = 30) -> np.ndarray:
    """
    Compute the safety stock that would have prevented a stockout at
    SERVICE_LEVEL confidence for every historical row.
    Uses rolling demand + LT variability via the combined SS formula.
    """
    z      = norm.ppf(SERVICE_LEVEL)
    n      = len(df)
    demand = df["daily_demand"].values
    lt     = df["actual_lead_time_days"].values
    ss     = np.zeros(n)
    for i in range(window, n):
        d_sl  = demand[i - window:i]
        lt_sl = lt[i - window:i]
        var   = (lt_sl.mean() * d_sl.std(ddof=1)**2
                 + d_sl.mean()**2 * lt_sl.std(ddof=1)**2)
        ss[i] = max(0, z * np.sqrt(max(0, var)))
    ss[:window] = ss[window] if window < n else 0
    return ss


def train_global_ss_optimizer(data_path: str = "data/training_data.csv") -> dict:
    print("\n  Training GLOBAL GBM Safety Stock Optimizer  (all materials)...")
    df = pd.read_csv(data_path, parse_dates=["date"])

    # Encode material IDs
    mat_enc = LabelEncoder()
    df["material_id_enc"] = mat_enc.fit_transform(df["material_id"])

    # Build per-material optimal SS labels, then concat
    frames = []
    for mat_id, mat_df in df.groupby("material_id"):
        mat_df = mat_df.sort_values("date").reset_index(drop=True).copy()
        mat_df["optimal_ss"] = _compute_optimal_ss(mat_df)
        frames.append(mat_df)
    df = pd.concat(frames).sort_values(["material_id", "date"]).reset_index(drop=True)

    X = df[SS_FEATURES].values
    y = df["optimal_ss"].values

    print(f"  Materials: {list(mat_enc.classes_)}")
    print(f"  Total rows: {len(X):,}  |  Service level: {SERVICE_LEVEL:.0%}")
    print(f"  Target SS range: [{y.min():.1f}, {y.max():.1f}]")

    # Time-series split — no future leakage
    tss = TimeSeriesSplit(n_splits=5)
    tr_idx, val_idx = list(tss.split(X))[-1]

    gbm = GradientBoostingRegressor(
        n_estimators=400, max_depth=5, learning_rate=0.04,
        subsample=0.8, min_samples_leaf=20,
        loss="huber",        # robust to SS spikes during demand surges
        random_state=42,
    )
    gbm.fit(X[tr_idx], y[tr_idx])

    y_pred = gbm.predict(X[val_idx])
    mae    = mean_absolute_error(y[val_idx], y_pred)
    r2     = r2_score(y[val_idx], y_pred)
    print(f"  Global val MAE={mae:.2f} units  R²={r2:.3f}")

    # Per-material breakdown
    df_val = df.iloc[val_idx].copy()
    df_val["pred"] = y_pred
    for mat_id in sorted(df["material_id"].unique()):
        sub     = df_val[df_val["material_id"] == mat_id]
        m_mae   = mean_absolute_error(sub["optimal_ss"], sub["pred"])
        sap_ss  = MATERIALS.get(mat_id, {}).get("safety_stock", sub["optimal_ss"].mean())
        sap_mae = mean_absolute_error(sub["optimal_ss"], [sap_ss] * len(sub))
        imp     = (sap_mae - m_mae) / sap_mae * 100 if sap_mae > 0 else 0
        print(f"    {mat_id:<14}  MAE={m_mae:.1f}  vs static SAP: {imp:+.1f}% improvement")

    imp_feats = dict(zip(SS_FEATURES, gbm.feature_importances_))
    top5 = sorted(imp_feats.items(), key=lambda x: -x[1])[:5]
    print("  Top features: " + "  |  ".join(f"{k}={v:.3f}" for k, v in top5))

    joblib.dump(gbm,     MODEL_PATH)
    joblib.dump(mat_enc, MAT_ENC_PATH)
    print(f"  Saved → {MODEL_PATH}  |  {MAT_ENC_PATH}")
    return {"model": gbm, "mat_enc": mat_enc, "mae": mae, "r2": r2}


def predict_optimal_ss(mat_id: str, feature_row: dict,
                       model=None, mat_enc=None) -> dict:
    """
    Predict optimal safety stock for one material given current features.
    feature_row: dict with all SS_FEATURES except material_id_enc.
    """
    if model is None:
        model   = joblib.load(MODEL_PATH)
        mat_enc = joblib.load(MAT_ENC_PATH)

    row = feature_row.copy()
    row["material_id_enc"] = int(mat_enc.transform([mat_id])[0])
    X          = np.array([[row[f] for f in SS_FEATURES]])
    predicted  = float(max(0, model.predict(X)[0]))

    current_ss = MATERIALS.get(mat_id, {}).get("safety_stock", 0)
    delta      = predicted - current_ss
    delta_pct  = (delta / current_ss * 100) if current_ss > 0 else 0

    rec = (
        "⬆️  INCREASE — current SS under-protected for current conditions"
        if delta > current_ss * 0.15 else
        "⬇️  DECREASE — over-conservative, release working capital"
        if delta < -current_ss * 0.15 else
        "✅  MAINTAIN — within ±15% of ML-optimal"
    )
    return {
        "material_id":     mat_id,
        "current_ss_sap":  current_ss,
        "predicted_ss_ml": round(predicted, 1),
        "delta":           round(delta, 1),
        "delta_pct":       round(delta_pct, 1),
        "recommendation":  rec,
    }


def run_global_safety_stock_optimizer(data_path: str = "data/training_data.csv",
                                       retrain: bool = True) -> dict:
    print("=" * 65)
    print("  MODULE 8 (Global): SAFETY STOCK OPTIMIZER  (GBM)")
    print("=" * 65)
    print(f"  Service level: {SERVICE_LEVEL:.0%}  |  Mode: {'TRAIN' if retrain else 'LOAD'}")

    if retrain or not MODEL_PATH.exists():
        result  = train_global_ss_optimizer(data_path)
        model   = result["model"]
        mat_enc = result["mat_enc"]
    else:
        print("  Loading saved global GBM model...")
        model   = joblib.load(MODEL_PATH)
        mat_enc = joblib.load(MAT_ENC_PATH)

    df = pd.read_csv(data_path, parse_dates=["date"])
    all_results = {}

    print("\n" + "=" * 65)
    print("  SAFETY STOCK RECOMMENDATIONS  (one global model)")
    print("=" * 65)

    total_delta_value = 0
    for mat_id in sorted(df["material_id"].unique()):
        mat_df = df[df["material_id"] == mat_id].sort_values("date")
        latest = mat_df.iloc[-1]
        row    = {f: latest[f] for f in SS_FEATURES if f != "material_id_enc"}
        res    = predict_optimal_ss(mat_id, row, model, mat_enc)
        all_results[mat_id] = res

        price         = MATERIALS.get(mat_id, {}).get("price", 0)
        inv_impact    = res["delta"] * price
        total_delta_value += inv_impact

        print(f"\n  {mat_id}  —  {MATERIALS.get(mat_id,{}).get('description','')}")
        print(f"    SAP current:    {res['current_ss_sap']:>6} units")
        print(f"    ML recommended: {res['predicted_ss_ml']:>6} units  "
              f"(Δ {res['delta']:+.0f}  /  {res['delta_pct']:+.1f}%)")
        print(f"    Inventory Δ:    ₹{inv_impact:>+12,.0f}  (@₹{price:,}/unit)")
        print(f"    Recommendation: {res['recommendation']}")

    print(f"\n  {'─'*63}")
    print(f"  Net inventory impact of all SS changes: ₹{total_delta_value:+,.0f}")
    print(f"  (negative = working capital released, positive = buffer investment)")

    return all_results


if __name__ == "__main__":
    retrain = "--no-retrain" not in sys.argv
    run_global_safety_stock_optimizer(retrain=retrain)
