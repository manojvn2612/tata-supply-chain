"""
module_2_leadtime_prediction.py
---------------------------------
ML Module 2: Lead-Time Prediction (XGBoost P50/P90)

Trains two XGBoost models per material:
  - P50 model: point prediction (mean)
  - P90 model: 90th percentile (planning buffer)

Confidence score: P90 coverage gap — narrow gap = confident, wide gap = uncertain
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

LT_FEATURES = [
    "nominal_lead_time_days", "transit_risk_level", "po_slippage_days",
    "qi_rejection", "month", "quarter", "day_of_year",
    "month_sin", "month_cos", "stock_cover_days",
    "demand_rolling_slope", "daily_demand",
]

XGB_BASE = dict(n_estimators=500, max_depth=6, learning_rate=0.05,
                subsample=0.8, colsample_bytree=0.8, min_child_weight=5,
                n_jobs=-1, random_state=42, early_stopping_rounds=30)


def _compute_scores(mean_model, p90_model, X_val, y_val) -> dict:
    """
    Full scoring suite for lead-time XGBoost.
    Metrics: MAE, RMSE, MAPE, R², P90 coverage, mean absolute % error,
             P90-P50 gap (uncertainty band width), bias, feature importance top-5.
    """
    from sklearn.metrics import r2_score as sk_r2

    y_p50 = mean_model.predict(X_val)
    y_p90 = p90_model.predict(X_val)

    mae       = float(mean_absolute_error(y_val, y_p50))
    rmse      = float(np.sqrt(mean_squared_error(y_val, y_p50)))
    r2        = float(sk_r2(y_val, y_p50))
    bias      = float(np.mean(y_p50 - y_val))

    # MAPE
    mask = y_val > 0.01
    mape = float(np.mean(np.abs((y_val[mask] - y_p50[mask]) / y_val[mask])) * 100) if mask.sum() else 0.0

    # P90 metrics
    p90_cov     = float(np.mean(y_val <= y_p90))
    avg_gap     = float(np.mean(np.maximum(0, y_p90 - y_p50)))   # uncertainty band width
    p90_mae     = float(mean_absolute_error(y_val, y_p90))        # how far P90 is from actuals

    # Pinball loss for P90 (quantile loss — lower = better)
    alpha       = 0.90
    errors      = y_val - y_p90
    pinball     = float(np.mean(np.where(errors >= 0, alpha * errors, (alpha - 1) * errors)))

    # Feature importances (top 5)
    fi   = dict(zip(LT_FEATURES, mean_model.feature_importances_))
    top5 = sorted(fi.items(), key=lambda x: -x[1])[:5]

    gap_penalty = min(avg_gap * 4, 40)
    cov_bonus   = max(0, (p90_cov - 0.85) * 100)
    conf        = round(min(99, max(50, 80 - gap_penalty + cov_bonus)), 1)

    return {
        # Regression metrics (P50 model)
        "mae_days":           round(mae, 4),
        "rmse_days":          round(rmse, 4),
        "mape_pct":           round(mape, 2),
        "r2_score":           round(r2, 4),
        "bias_days":          round(bias, 4),
        # P90 / quantile metrics
        "p90_coverage":       round(p90_cov, 4),
        "p90_mae_days":       round(p90_mae, 4),
        "avg_p50_p90_gap":    round(avg_gap, 4),
        "pinball_loss_p90":   round(pinball, 4),
        # Feature importance
        "top5_features":      [(k, round(v, 4)) for k, v in top5],
        # Confidence
        "confidence_pct":     conf,
        "n_val_samples":      int(len(y_val)),
        "grade": ("A" if p90_cov >= 0.92 and mae <= 2.0 else
                  "B" if p90_cov >= 0.88 and mae <= 4.0 else
                  "C" if p90_cov >= 0.80 else "D"),
        "note": ("✅ Good" if p90_cov >= 0.90 else
                 "⚠️ P90 under-covering" if p90_cov < 0.80 else "🟡 Acceptable"),
    }


# alias for backward compat
_compute_confidence = _compute_scores


def train_leadtime_models(df: pd.DataFrame, mat_id: str) -> dict:
    print(f"    Training XGBoost LT for {mat_id}...", end=" ", flush=True)
    mat_df = df[df["material_id"] == mat_id].sort_values("date").reset_index(drop=True)
    X = mat_df[LT_FEATURES].values
    y = mat_df["actual_lead_time_days"].values
    splits = list(TimeSeriesSplit(n_splits=5).split(X))
    tr_idx, vl_idx = splits[-1]

    from sklearn.metrics import r2_score as sk_r2

    # ── Train with eval tracking to get loss curves ──
    mean_m = xgb.XGBRegressor(**XGB_BASE, objective="reg:squarederror")
    mean_m.fit(X[tr_idx], y[tr_idx], eval_set=[(X[tr_idx], y[tr_idx]), (X[vl_idx], y[vl_idx])], verbose=False)

    p90_m = xgb.XGBRegressor(**XGB_BASE, objective="reg:quantileerror", quantile_alpha=0.90)
    p90_m.fit(X[tr_idx], y[tr_idx], eval_set=[(X[tr_idx], y[tr_idx]), (X[vl_idx], y[vl_idx])], verbose=False)

    # ── Validation scores ──
    scores = _compute_scores(mean_m, p90_m, X[vl_idx], y[vl_idx])

    # ── Training loss curves (sampled every 20 rounds) ──
    mean_er = mean_m.evals_result()
    p90_er  = p90_m.evals_result()
    mean_train_curve = list(list(mean_er.get("validation_0", {}).values() or [[]])[0])[::20]
    mean_val_curve   = list(list(mean_er.get("validation_1", {}).values() or [[]])[0])[::20]
    p90_val_curve    = list(list(p90_er.get("validation_1",  {}).values() or [[]])[0])[::20]
    best_iter_mean   = int(mean_m.best_iteration) if hasattr(mean_m, "best_iteration") and mean_m.best_iteration else 0
    best_iter_p90    = int(p90_m.best_iteration)  if hasattr(p90_m,  "best_iteration") and p90_m.best_iteration  else 0

    # ── 5-fold cross-val R², MAE, P90 coverage ──
    cv_r2, cv_mae, cv_p90cov = [], [], []
    for tr_i, vl_i in TimeSeriesSplit(n_splits=5).split(X):
        m_cv = xgb.XGBRegressor(**{**XGB_BASE, "early_stopping_rounds": None},
                                 objective="reg:squarederror", n_estimators=best_iter_mean or 300)
        q_cv = xgb.XGBRegressor(**{**XGB_BASE, "early_stopping_rounds": None},
                                 objective="reg:quantileerror", quantile_alpha=0.90,
                                 n_estimators=best_iter_p90 or 300)
        m_cv.fit(X[tr_i], y[tr_i], verbose=False)
        q_cv.fit(X[tr_i], y[tr_i], verbose=False)
        yh   = m_cv.predict(X[vl_i])
        yp90 = q_cv.predict(X[vl_i])
        cv_r2.append(round(float(sk_r2(y[vl_i], yh)), 4))
        cv_mae.append(round(float(mean_absolute_error(y[vl_i], yh)), 4))
        cv_p90cov.append(round(float(np.mean(y[vl_i] <= yp90)), 4))

    scores["training_loss_curve_mean"] = [round(v, 4) for v in mean_train_curve]
    scores["val_loss_curve_mean"]      = [round(v, 4) for v in mean_val_curve]
    scores["val_loss_curve_p90"]       = [round(v, 4) for v in p90_val_curve]
    scores["best_iteration_mean"]      = best_iter_mean
    scores["best_iteration_p90"]       = best_iter_p90
    scores["cv_r2_per_fold"]           = cv_r2
    scores["cv_r2_mean"]               = round(float(np.mean(cv_r2)), 4)
    scores["cv_r2_std"]                = round(float(np.std(cv_r2)), 4)
    scores["cv_mae_per_fold"]          = cv_mae
    scores["cv_mae_mean"]              = round(float(np.mean(cv_mae)), 4)
    scores["cv_p90cov_per_fold"]       = cv_p90cov
    scores["cv_p90cov_mean"]           = round(float(np.mean(cv_p90cov)), 4)

    safe = mat_id.replace("-", "_")
    joblib.dump(mean_m, MODELS_DIR / f"xgb_lt_mean_{safe}.pkl")
    joblib.dump(p90_m,  MODELS_DIR / f"xgb_lt_p90_{safe}.pkl")
    print(f"\n    ── {mat_id} Lead-Time Scores ──")
    print(f"    MAE={scores['mae_days']}d  RMSE={scores['rmse_days']}d  MAPE={scores['mape_pct']}%  R²={scores['r2_score']}")
    print(f"    Bias={scores['bias_days']}d  P90_cov={scores['p90_coverage']:.0%}  P90_MAE={scores['p90_mae_days']}d")
    print(f"    PinballLoss(P90)={scores['pinball_loss_p90']}  P50-P90 gap={scores['avg_p50_p90_gap']}d")
    print(f"    Best iteration: mean={best_iter_mean}  p90={best_iter_p90}")
    print(f"    CV R²  (5-fold): {cv_r2}  → mean={scores['cv_r2_mean']} ±{scores['cv_r2_std']}")
    print(f"    CV MAE (5-fold): {cv_mae}  → mean={scores['cv_mae_mean']}")
    print(f"    CV P90 coverage: {cv_p90cov}  → mean={scores['cv_p90cov_mean']:.0%}")
    print(f"    Grade={scores['grade']}  Confidence={scores['confidence_pct']}%  {scores['note']}")
    print(f"    Top P50 features: {', '.join(f'{k}({v})' for k,v in scores['top5_features'][:3])}")
    return {"mean_model": mean_m, "p90_model": p90_m, "scores": scores}


def predict_leadtime(mat_id: str, feature_row: dict, models: dict = None) -> dict:
    if models is None:
        safe = mat_id.replace("-", "_")
        models = {
            "mean_model": joblib.load(MODELS_DIR / f"xgb_lt_mean_{safe}.pkl"),
            "p90_model":  joblib.load(MODELS_DIR / f"xgb_lt_p90_{safe}.pkl"),
        }
    row = np.array([[feature_row[f] for f in LT_FEATURES]])
    p50 = float(models["mean_model"].predict(row)[0])
    p90 = max(float(models["p90_model"].predict(row)[0]), p50)
    gap = p90 - feature_row["nominal_lead_time_days"]
    risk = ("🔴 HIGH — >7d above SAP nominal. Reorder NOW."   if gap > 7  else
            "🟠 MEDIUM — materially above SAP. Pull in reorder." if gap > 3 else
            "🟢 LOW — close to SAP nominal. Normal planning.")
    return {
        "material_id":      mat_id,
        "sap_nominal_lt":   feature_row["nominal_lead_time_days"],
        "predicted_lt_p50": round(p50, 1),
        "predicted_lt_p90": round(p90, 1),
        "gap_p90_vs_sap":   round(gap, 1),
        "risk_flag":        risk,
    }


def run_leadtime_prediction(data_path="data/training_data.csv", retrain=True) -> dict:
    print("=" * 65)
    print("  MODULE 2: XGBOOST LEAD-TIME PREDICTION")
    print("=" * 65)
    df = pd.read_csv(data_path, parse_dates=["date"])
    all_models, predictions = {}, {}
    for mat_id in sorted(df["material_id"].unique()):
        if retrain:
            r = train_leadtime_models(df, mat_id)
            all_models[mat_id] = r
        else:
            safe = mat_id.replace("-", "_")
            all_models[mat_id] = {
                "mean_model": joblib.load(MODELS_DIR / f"xgb_lt_mean_{safe}.pkl"),
                "p90_model":  joblib.load(MODELS_DIR / f"xgb_lt_p90_{safe}.pkl"),
                "confidence": {"confidence_pct": 85.0},
            }
        mat_df  = df[df["material_id"] == mat_id].sort_values("date")
        latest  = mat_df.iloc[-1][LT_FEATURES].to_dict()
        pred    = predict_leadtime(mat_id, latest, all_models[mat_id])
        pred["scores"] = all_models[mat_id].get("scores", all_models[mat_id].get("confidence", {}))
        predictions[mat_id] = pred
        s = pred["scores"]
        print(f"  {mat_id}: P50={pred['predicted_lt_p50']}d  P90={pred['predicted_lt_p90']}d  "
              f"gap={pred['gap_p90_vs_sap']:+.1f}d  {pred['risk_flag']}")
    return predictions


if __name__ == "__main__":
    run_leadtime_prediction(retrain=True)
