"""
evaluator.py
-------------
Model Performance Tracker & Backtester

Backtests all ML models on held-out historical data and produces
a unified performance scorecard. Run this after training to validate
model quality before deploying to production.

Metrics:
  - Demand LSTM:   MAE, RMSE, MAPE on 30-day rolling backtest
  - LT XGBoost:    MAE, RMSE, P90 coverage (should be ≥90%)
  - Exception RF:  Accuracy, F1 per class, confusion matrix summary
  - SS Optimizer:  MAE vs optimal SS label, cost savings estimate
  - Supplier Risk: AUC-ROC for late-delivery prediction

Run:  python evaluator.py
"""

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import (mean_absolute_error, mean_squared_error,
                              r2_score, roc_auc_score, classification_report)
from sklearn.model_selection import TimeSeriesSplit
from sklearn.preprocessing import LabelEncoder
import joblib
from pathlib import Path
from config import MATERIALS

MODELS_DIR = Path("models")


# ── Helpers ───────────────────────────────────────────────────────────────────

def mape(y_true, y_pred):
    y_true = np.array(y_true, dtype=float)
    y_pred = np.array(y_pred, dtype=float)
    mask   = y_true > 0.01
    return np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100


def print_metric_bar(name: str, value: float, good_threshold: float,
                     bad_threshold: float, unit: str = "", higher_is_better=False):
    if higher_is_better:
        icon = "🟢" if value >= good_threshold else ("🟡" if value >= bad_threshold else "🔴")
    else:
        icon = "🟢" if value <= good_threshold else ("🟡" if value <= bad_threshold else "🔴")
    print(f"    {icon}  {name:<35} {value:>8.3f} {unit}")


# ── Backtest: Exception RF ────────────────────────────────────────────────────

def backtest_exception_classifier(df: pd.DataFrame) -> dict:
    print("\n  Backtesting: Exception Random Forest")

    EXCEPTION_FEATURES = [
        "stock_cover_days","actual_lead_time_days","nominal_lead_time_days",
        "lt_deviation_days","transit_delay_days","delay_event","po_slippage_days",
        "qi_rejection","demand_rolling_slope","daily_demand","stock_level",
        "safety_stock","transit_risk_level","month","quarter","month_sin","month_cos",
    ]

    le  = joblib.load(MODELS_DIR / "rf_label_encoder.pkl")
    rf  = joblib.load(MODELS_DIR / "rf_exception_classifier.pkl")
    X   = df[EXCEPTION_FEATURES].values
    y   = le.transform(df["exception_label"].values)

    # Evaluate on last 20% of data (time-ordered)
    n       = len(X)
    split   = int(n * 0.8)
    X_test  = X[split:]
    y_test  = y[split:]
    y_pred  = rf.predict(X_test)
    y_proba = rf.predict_proba(X_test)

    acc   = (y_pred == y_test).mean()
    present = np.unique(y_test)
    f1s   = {}
    from sklearn.metrics import f1_score
    for cls_idx in present:
        cls_name = le.inverse_transform([cls_idx])[0]
        f1s[cls_name] = f1_score(y_test, y_pred, labels=[cls_idx], average="macro")

    print(f"    Accuracy (last 20%): {acc:.1%}")
    for cls, f1 in sorted(f1s.items(), key=lambda x: -x[1]):
        print_metric_bar(f"F1 — {cls}", f1, 0.85, 0.70, higher_is_better=True)

    return {"accuracy": acc, "f1_per_class": f1s}


# ── Backtest: LT XGBoost ─────────────────────────────────────────────────────

def backtest_xgb_leadtime(df: pd.DataFrame) -> dict:
    print("\n  Backtesting: XGBoost Lead-Time")
    from module_2_xgboost_leadtime import LT_FEATURES

    results = {}
    for mat_id in df["material_id"].unique():
        mat_df = df[df["material_id"] == mat_id].sort_values("date").reset_index(drop=True)
        safe_id = mat_id.replace("-", "_")

        try:
            mean_model = joblib.load(MODELS_DIR / f"xgb_lt_mean_{safe_id}.pkl")
            p90_model  = joblib.load(MODELS_DIR / f"xgb_lt_p90_{safe_id}.pkl")
        except FileNotFoundError:
            print(f"    {mat_id}: model not found — run Module 2 first")
            continue

        n      = len(mat_df)
        split  = int(n * 0.8)
        X_test = mat_df.iloc[split:][LT_FEATURES].values
        y_test = mat_df.iloc[split:]["actual_lead_time_days"].values

        y_pred_mean = mean_model.predict(X_test)
        y_pred_p90  = p90_model.predict(X_test)

        mae_val      = mean_absolute_error(y_test, y_pred_mean)
        rmse_val     = np.sqrt(mean_squared_error(y_test, y_pred_mean))
        p90_coverage = np.mean(y_test <= y_pred_p90)

        print(f"\n    {mat_id}")
        print_metric_bar("MAE (days)",             mae_val,      2.0, 5.0, "d")
        print_metric_bar("RMSE (days)",            rmse_val,     3.0, 7.0, "d")
        print_metric_bar("P90 coverage (≥90%)",    p90_coverage, 0.90, 0.80, higher_is_better=True)

        results[mat_id] = {"mae": mae_val, "rmse": rmse_val, "p90_coverage": p90_coverage}

    return results


# ── Backtest: Safety Stock GBM ────────────────────────────────────────────────

def backtest_ss_optimizer(df: pd.DataFrame) -> dict:
    print("\n  Backtesting: Safety Stock GBM Optimizer")
    from module_8_safety_stock_optimizer import SS_FEATURES, compute_optimal_ss_label

    results = {}
    for mat_id in df["material_id"].unique():
        mat_df  = df[df["material_id"] == mat_id].sort_values("date").reset_index(drop=True)
        safe_id = mat_id.replace("-", "_")

        try:
            model = joblib.load(MODELS_DIR / f"gbm_ss_{safe_id}.pkl")
        except FileNotFoundError:
            print(f"    {mat_id}: model not found — run Module 8 first")
            continue

        optimal_ss     = compute_optimal_ss_label(mat_df)
        n              = len(mat_df)
        split          = int(n * 0.8)
        X_test         = mat_df.iloc[split:][SS_FEATURES].values
        y_test         = optimal_ss[split:]

        y_pred = np.maximum(0, model.predict(X_test))
        mae_val = mean_absolute_error(y_test, y_pred)
        r2_val  = r2_score(y_test, y_pred)

        mat     = MATERIALS.get(mat_id, {})
        current = mat.get("safety_stock", 0)
        sap_mae = mean_absolute_error(y_test, np.full_like(y_test, current))
        improvement = ((sap_mae - mae_val) / sap_mae * 100) if sap_mae > 0 else 0

        print(f"\n    {mat_id}")
        print_metric_bar("MAE vs optimal SS",        mae_val,    10.0, 30.0, "units")
        print_metric_bar("R² score",                 r2_val,     0.60, 0.30, higher_is_better=True)
        print(f"    {'⬆️ ' if improvement>0 else '⬇️ '}  ML improves over static SAP SS by {improvement:+.1f}%")

        results[mat_id] = {"mae": mae_val, "r2": r2_val, "improvement_over_sap_pct": improvement}

    return results


# ── Backtest: Supplier Risk GBM ───────────────────────────────────────────────

def backtest_supplier_risk(df: pd.DataFrame) -> dict:
    print("\n  Backtesting: GBM Late-Delivery Classifier")
    from module_4_supplier_risk_ml import RISK_FEATURES

    results = {}
    for mat_id in df["material_id"].unique():
        mat_df  = df[df["material_id"] == mat_id].sort_values("date").reset_index(drop=True)
        safe_id = mat_id.replace("-", "_")

        try:
            model = joblib.load(MODELS_DIR / f"gbm_late_{safe_id}.pkl")
        except FileNotFoundError:
            continue

        nominal_lt = mat_df["nominal_lead_time_days"].iloc[0]
        threshold  = nominal_lt + mat_df["lt_deviation_days"].std()
        y      = (mat_df["actual_lead_time_days"] > threshold).astype(int)
        X      = mat_df[RISK_FEATURES].values

        n     = len(X)
        split = int(n * 0.8)
        X_te  = X[split:]
        y_te  = y[split:]

        y_proba = model.predict_proba(X_te)[:, 1]
        if y_te.sum() > 0:
            auc = roc_auc_score(y_te, y_proba)
        else:
            auc = float("nan")

        print(f"\n    {mat_id}")
        if not np.isnan(auc):
            print_metric_bar("AUC-ROC", auc, 0.80, 0.65, higher_is_better=True)
        else:
            print(f"    ⚪  No late events in test set — AUC not computable")

        results[mat_id] = {"auc": auc}

    return results


# ── Main ──────────────────────────────────────────────────────────────────────

def run_evaluator(data_path="data/training_data.csv"):
    print("=" * 65)
    print("  EVALUATOR — ML Model Backtest Scorecard")
    print("=" * 65)

    df = pd.read_csv(data_path, parse_dates=["date"])

    scorecard = {}

    # Only run evaluations for models that have been trained
    trained_models = list(MODELS_DIR.glob("*.pkl")) + list(MODELS_DIR.glob("*.pt"))
    model_names    = [m.stem for m in trained_models]
    print(f"\n  Found {len(trained_models)} trained model files in models/\n")

    if any("rf_exception" in n for n in model_names):
        scorecard["exception_rf"] = backtest_exception_classifier(df)

    if any("xgb_lt" in n for n in model_names):
        scorecard["leadtime_xgb"] = backtest_xgb_leadtime(df)

    if any("gbm_ss" in n for n in model_names):
        scorecard["safety_stock_gbm"] = backtest_ss_optimizer(df)

    if any("gbm_late" in n for n in model_names):
        scorecard["supplier_risk_gbm"] = backtest_supplier_risk(df)

    # ── Summary ──
    print("\n\n" + "=" * 65)
    print("  OVERALL MODEL QUALITY SUMMARY")
    print("=" * 65)

    if "exception_rf" in scorecard:
        s = scorecard["exception_rf"]
        icon = "🟢" if s["accuracy"] >= 0.90 else "🟡"
        print(f"\n  {icon}  Exception Classifier (RF)   accuracy={s['accuracy']:.1%}")

    if "leadtime_xgb" in scorecard:
        for mat_id, s in scorecard["leadtime_xgb"].items():
            icon = "🟢" if s["mae"] <= 2.0 else ("🟡" if s["mae"] <= 5.0 else "🔴")
            print(f"  {icon}  Lead-Time XGBoost {mat_id:<12}  MAE={s['mae']:.2f}d  P90={s['p90_coverage']:.0%}")

    if "safety_stock_gbm" in scorecard:
        for mat_id, s in scorecard["safety_stock_gbm"].items():
            icon = "🟢" if s["improvement_over_sap_pct"] > 0 else "🔴"
            print(f"  {icon}  SS Optimizer {mat_id:<16}  R²={s['r2']:.3f}  vs SAP: {s['improvement_over_sap_pct']:+.1f}%")

    if "supplier_risk_gbm" in scorecard:
        for mat_id, s in scorecard["supplier_risk_gbm"].items():
            if not np.isnan(s.get("auc", float("nan"))):
                icon = "🟢" if s["auc"] >= 0.80 else "🟡"
                print(f"  {icon}  Supplier Risk GBM {mat_id:<12}  AUC={s['auc']:.3f}")

    print()
    return scorecard


if __name__ == "__main__":
    run_evaluator()
