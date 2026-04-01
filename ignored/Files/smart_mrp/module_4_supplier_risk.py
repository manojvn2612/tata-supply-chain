"""
module_4_supplier_risk.py
---------------------------
ML Module 4: Supplier Risk Scoring (IsolationForest + GBM)

Model 1: IsolationForest — anomaly detection, no labels needed
Model 2: GradientBoostingClassifier — P(late delivery)

Confidence: AUC-ROC + model quality score (0-100)
"""

import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest, GradientBoostingClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, log_loss, classification_report
import joblib
from pathlib import Path

MODELS_DIR = Path("models")
MODELS_DIR.mkdir(exist_ok=True)

ANOMALY_FEATURES = [
    "actual_lead_time_days", "lt_deviation_days", "transit_delay_days",
    "po_slippage_days", "qi_rejection", "stock_cover_days",
]
RISK_FEATURES = [
    "nominal_lead_time_days", "transit_risk_level", "po_slippage_days",
    "qi_rejection", "month", "quarter", "month_sin", "month_cos",
    "demand_rolling_slope", "daily_demand", "stock_cover_days", "lt_deviation_days",
]
RISK_THRESHOLDS = {"RED": 0.55, "AMBER": 0.30}


def _compute_iso_scores(iso, scaler, X) -> dict:
    """
    IsolationForest scoring.
    Metrics: anomaly rate, decision score stats, silhouette-like separation,
             score distribution (mean/std/min/max of decision_function).
    """
    X_sc    = scaler.transform(X)
    scores  = iso.decision_function(X_sc)   # higher = more normal
    labels  = iso.predict(X_sc)             # -1 = anomaly, 1 = normal
    n_anom  = int((labels == -1).sum())
    n_total = len(labels)

    normal_scores  = scores[labels == 1]
    anomaly_scores = scores[labels == -1]

    # Separation: larger gap between normal and anomaly score means = cleaner separation
    sep = float(normal_scores.mean() - anomaly_scores.mean()) if len(anomaly_scores) > 0 else 0.0

    # Normalise decision scores to 0-100 anomaly risk (higher = more anomalous)
    s_min, s_max = scores.min(), scores.max()
    if s_max > s_min:
        risk_norm = 100 * (1 - (scores - s_min) / (s_max - s_min))
    else:
        risk_norm = np.zeros_like(scores)

    conf = round(float(100 - risk_norm.mean()), 1)

    return {
        "n_anomalies":             n_anom,
        "n_total":                 n_total,
        "anomaly_rate":            round(n_anom / n_total, 4),
        "avg_decision_score":      round(float(scores.mean()), 4),
        "std_decision_score":      round(float(scores.std()), 4),
        "min_decision_score":      round(float(scores.min()), 4),
        "max_decision_score":      round(float(scores.max()), 4),
        "normal_vs_anomaly_sep":   round(sep, 4),   # separation quality
        "avg_anomaly_risk_score":  round(float(risk_norm.mean()), 2),
        "confidence_pct":          conf,
        "grade": ("A" if n_anom / n_total <= 0.06 and sep > 0.1 else
                  "B" if n_anom / n_total <= 0.10 else "C"),
    }


def _compute_gbm_scores(gbm, X_test, y_test) -> dict:
    """
    GBM late-delivery classifier scoring.
    Metrics: AUC-ROC, log loss, F1, precision, recall, Brier,
             Matthews CC, Cohen's kappa, feature importance top-5.
    """
    from sklearn.metrics import (
        f1_score, precision_score, recall_score, log_loss,
        roc_auc_score, brier_score_loss, matthews_corrcoef, cohen_kappa_score
    )

    y_proba = gbm.predict_proba(X_test)[:, 1]
    y_pred  = gbm.predict(X_test)

    if y_test.sum() > 0 and (y_test == 0).sum() > 0:
        auc   = float(roc_auc_score(y_test, y_proba))
        ll    = float(log_loss(y_test, y_proba))
        brier = float(brier_score_loss(y_test, y_proba))
    else:
        auc, ll, brier = float("nan"), float("nan"), float("nan")

    f1    = float(f1_score(y_test, y_pred, zero_division=0))
    prec  = float(precision_score(y_test, y_pred, zero_division=0))
    rec   = float(recall_score(y_test, y_pred, zero_division=0))

    try:    mcc   = float(matthews_corrcoef(y_test, y_pred))
    except: mcc   = float("nan")
    try:    kappa = float(cohen_kappa_score(y_test, y_pred))
    except: kappa = float("nan")

    # Feature importance top 5
    fi   = dict(zip(RISK_FEATURES, gbm.feature_importances_))
    top5 = sorted(fi.items(), key=lambda x: -x[1])[:5]

    quality = round(max(0, (auc - 0.5) * 200), 1) if not np.isnan(auc) else 50.0

    def _r(v): return round(v, 4) if not np.isnan(v) else None

    return {
        "auc_roc":          _r(auc),
        "log_loss":         _r(ll),
        "brier_score":      _r(brier),
        "f1_score":         round(f1, 4),
        "precision":        round(prec, 4),
        "recall":           round(rec, 4),
        "matthews_corrcoef":_r(mcc),
        "cohen_kappa":      _r(kappa),
        "model_quality":    quality,
        "confidence_pct":   round(min(99, quality), 1),
        "top5_features":    [(k, round(v, 4)) for k, v in top5],
        "n_test_samples":   int(len(y_test)),
        "n_late_events":    int(y_test.sum()),
        "grade": ("A" if not np.isnan(auc) and auc >= 0.85 else
                  "B" if not np.isnan(auc) and auc >= 0.70 else "C"),
    }


# aliases for backward compat
_compute_iso_confidence = _compute_iso_scores
_compute_gbm_confidence = _compute_gbm_scores


def train_iso_forest(mat_df: pd.DataFrame, mat_id: str):
    print(f"    IsoForest {mat_id}...", end=" ", flush=True)
    X      = mat_df[ANOMALY_FEATURES].values
    scaler = StandardScaler()
    X_sc   = scaler.fit_transform(X)
    iso    = IsolationForest(n_estimators=200, contamination=0.05, n_jobs=-1, random_state=42)
    iso.fit(X_sc)
    conf = _compute_iso_scores(iso, scaler, X)
    safe = mat_id.replace("-", "_")
    joblib.dump(iso,    MODELS_DIR / f"iso_forest_{safe}.pkl")
    joblib.dump(scaler, MODELS_DIR / f"iso_scaler_{safe}.pkl")
    print(f"\n    ── {mat_id} IsoForest Scores ──")
    print(f"    Anomalies={conf['n_anomalies']}/{conf['n_total']} ({conf['anomaly_rate']:.1%})  "
          f"Separation={conf['normal_vs_anomaly_sep']}  Grade={conf['grade']}")
    print(f"    DecisionScore: mean={conf['avg_decision_score']}  std={conf['std_decision_score']}  "
          f"min={conf['min_decision_score']}  max={conf['max_decision_score']}")
    print(f"    AvgRisk={conf['avg_anomaly_risk_score']}/100  Confidence={conf['confidence_pct']}%")
    return iso, scaler, conf


def train_gbm_classifier(mat_df: pd.DataFrame, mat_id: str):
    print(f"    GBM late-classifier {mat_id}...", end=" ", flush=True)
    threshold = mat_df["nominal_lead_time_days"].iloc[0] + mat_df["lt_deviation_days"].std()
    y = (mat_df["actual_lead_time_days"] > threshold).astype(int)
    X = mat_df[RISK_FEATURES].values
    if y.sum() < 20:
        print("insufficient late events — skipping")
        return None, None

    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, random_state=42,
                                               stratify=y if y.sum() > 1 else None)
    gbm = GradientBoostingClassifier(n_estimators=200, max_depth=4, learning_rate=0.05,
                                     subsample=0.8, random_state=42)
    gbm.fit(X_tr, y_tr)
    scores = _compute_gbm_scores(gbm, X_te, y_te)

    # ── Training deviance curve (train vs test, sampled every 20 estimators) ──
    from sklearn.metrics import log_loss as _log_loss
    train_dev, test_dev = [], []
    for i, yhat_tr in enumerate(gbm.staged_predict_proba(X_tr)):
        if i % 20 == 0:
            train_dev.append(round(float(_log_loss(y_tr, yhat_tr)), 4))
    for i, yhat_te in enumerate(gbm.staged_predict_proba(X_te)):
        if i % 20 == 0:
            test_dev.append(round(float(_log_loss(y_te, yhat_te)), 4))
    scores["train_deviance_curve"] = train_dev
    scores["test_deviance_curve"]  = test_dev
    scores["best_iter_by_deviance"] = int(np.argmin(test_dev) * 20) if test_dev else 0

    # ── 5-fold cross-val AUC ──
    from sklearn.model_selection import StratifiedKFold
    cv_auc = []
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    if y.sum() >= 5:
        for tr_i, vl_i in skf.split(X, y):
            g = GradientBoostingClassifier(n_estimators=100, max_depth=4,
                                           learning_rate=0.05, subsample=0.8, random_state=42)
            g.fit(X[tr_i], y[tr_i])
            yp = g.predict_proba(X[vl_i])[:, 1]
            try:
                cv_auc.append(round(float(roc_auc_score(y[vl_i], yp)), 4))
            except Exception:
                pass
    scores["cv_auc_per_fold"] = cv_auc
    scores["cv_auc_mean"]     = round(float(np.mean(cv_auc)), 4) if cv_auc else None
    scores["cv_auc_std"]      = round(float(np.std(cv_auc)),  4) if cv_auc else None

    safe = mat_id.replace("-", "_")
    joblib.dump(gbm, MODELS_DIR / f"gbm_late_{safe}.pkl")
    print(f"\n    ── {mat_id} GBM Late-Delivery Scores ──")
    print(f"    AUC-ROC={scores['auc_roc']}  Log_Loss={scores['log_loss']}  Brier={scores['brier_score']}")
    print(f"    F1={scores['f1_score']}  Precision={scores['precision']}  Recall={scores['recall']}")
    print(f"    MCC={scores['matthews_corrcoef']}  Kappa={scores['cohen_kappa']}")
    print(f"    CV AUC (5-fold): {cv_auc}  → mean={scores['cv_auc_mean']} ±{scores['cv_auc_std']}")
    print(f"    Best iter by deviance: {scores['best_iter_by_deviance']}")
    print(f"    Quality={scores['model_quality']}/100  Grade={scores['grade']}  Confidence={scores['confidence_pct']}%")
    print(f"    Late events in test: {scores['n_late_events']}/{scores['n_test_samples']}")
    print(f"    Top features: {', '.join(f'{k}({v})' for k,v in scores['top5_features'][:3])}")
    return gbm, scores


def score_risk(mat_id: str, mat_df: pd.DataFrame,
               iso=None, scaler=None, gbm=None) -> dict:
    safe = mat_id.replace("-", "_")
    if iso is None:
        iso    = joblib.load(MODELS_DIR / f"iso_forest_{safe}.pkl")
        scaler = joblib.load(MODELS_DIR / f"iso_scaler_{safe}.pkl")

    latest    = mat_df.iloc[-1][ANOMALY_FEATURES].values.reshape(1, -1)
    X_sc      = scaler.transform(latest)
    iso_score = float(iso.decision_function(X_sc)[0])
    iso_label = int(iso.predict(X_sc)[0])

    # Normalise to 0-100 risk
    s_min, s_max = -0.5, 0.5
    anomaly_risk = round(100 * (1 - (iso_score - s_min) / (s_max - s_min + 1e-9)), 1)
    anomaly_risk = max(0, min(100, anomaly_risk))

    p_late = None
    if gbm is None:
        try:   gbm = joblib.load(MODELS_DIR / f"gbm_late_{safe}.pkl")
        except FileNotFoundError: gbm = None
    if gbm is not None:
        X_risk = mat_df.iloc[-1][RISK_FEATURES].values.reshape(1, -1)
        p_late = round(float(gbm.predict_proba(X_risk)[0, 1]), 3)

    tier = ("🔴 RED"   if (p_late or 0) >= RISK_THRESHOLDS["RED"]   else
            "🟡 AMBER" if (p_late or 0) >= RISK_THRESHOLDS["AMBER"] else "🟢 GREEN")

    return {
        "material_id":          mat_id,
        "anomaly_risk_score":   anomaly_risk,
        "iso_flag":             iso_label == -1,
        "p_late_next_delivery": p_late,
        "risk_tier":            tier,
    }


def run_supplier_risk(data_path="data/training_data.csv", retrain=True) -> dict:
    print("=" * 65)
    print("  MODULE 4: SUPPLIER RISK (IsoForest + GBM)")
    print("=" * 65)
    df = pd.read_csv(data_path, parse_dates=["date"])
    results  = {}
    iso_conf_all, gbm_conf_all = {}, {}

    for mat_id in sorted(df["material_id"].unique()):
        mat_df = df[df["material_id"] == mat_id].sort_values("date").reset_index(drop=True)
        safe   = mat_id.replace("-", "_")
        if retrain:
            iso, scaler, i_scores = train_iso_forest(mat_df, mat_id)
            gbm, g_scores         = train_gbm_classifier(mat_df, mat_id)
            iso_conf_all[mat_id] = i_scores
            gbm_conf_all[mat_id] = g_scores
        else:
            iso    = joblib.load(MODELS_DIR / f"iso_forest_{safe}.pkl")
            scaler = joblib.load(MODELS_DIR / f"iso_scaler_{safe}.pkl")
            gbm    = None
            try: gbm = joblib.load(MODELS_DIR / f"gbm_late_{safe}.pkl")
            except FileNotFoundError: pass

        report = score_risk(mat_id, mat_df, iso, scaler, gbm)
        report["iso_confidence"] = iso_conf_all.get(mat_id, {})
        report["gbm_confidence"] = gbm_conf_all.get(mat_id, {})
        results[mat_id] = report

        p_str = f"{report['p_late_next_delivery']:.0%}" if report["p_late_next_delivery"] is not None else "N/A"
        print(f"  {mat_id}: anomaly={report['anomaly_risk_score']}/100  "
              f"P(late)={p_str}  {report['risk_tier']}")

    return results


if __name__ == "__main__":
    run_supplier_risk(retrain=True)
