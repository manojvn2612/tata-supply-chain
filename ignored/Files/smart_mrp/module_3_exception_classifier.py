"""
module_3_exception_classifier.py
-----------------------------------
ML Module 3: Exception Classification (Random Forest)

Predicts MRP exception type from stock/demand/LT features.
Outputs: class, confidence %, Brier score calibration, per-class probabilities.
"""

import os, json
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, brier_score_loss
import joblib
from pathlib import Path

MODELS_DIR = Path("models")
MODELS_DIR.mkdir(exist_ok=True)

EXCEPTION_FEATURES = [
    "stock_cover_days", "actual_lead_time_days", "nominal_lead_time_days",
    "lt_deviation_days", "transit_delay_days", "delay_event", "po_slippage_days",
    "qi_rejection", "demand_rolling_slope", "daily_demand", "stock_level",
    "safety_stock", "transit_risk_level", "month", "quarter", "month_sin", "month_cos",
]

IMPACT_RANK = {"SHORTAGE": 0, "EXPEDITE": 1, "DEMAND_SPIKE": 2, "RESCHEDULE_IN": 3, "NO_EXCEPTION": 4}
IMPACT_ICON = {"SHORTAGE": "🔴", "EXPEDITE": "🟠", "DEMAND_SPIKE": "🟡", "RESCHEDULE_IN": "🔵", "NO_EXCEPTION": "🟢"}

SYSTEM_PROMPT = """You are a senior SAP MRP expert. Given an ML-classified exception, produce a JSON action card.
Respond ONLY with JSON — no markdown, no preamble:
{"plain_english":"2-3 sentence explanation","business_impact":"CRITICAL|HIGH|MEDIUM|LOW",
 "cascade_risk":"BOM cascade impact","recommended_action":"specific action with SAP transaction",
 "urgency_days":<integer>}"""


def _compute_scores(rf, le, X_test, y_test) -> dict:
    """
    Full scoring suite for RF exception classifier.
    Metrics: accuracy, F1 per class, precision, recall, log loss,
             Brier score, ROC-AUC (OvR), Matthews correlation coefficient,
             Cohen's kappa, top feature importances.
    """
    from sklearn.metrics import (
        f1_score, precision_score, recall_score, log_loss,
        roc_auc_score, matthews_corrcoef, cohen_kappa_score,
        confusion_matrix, classification_report
    )
    from sklearn.calibration import calibration_curve

    y_pred  = rf.predict(X_test)
    y_proba = rf.predict_proba(X_test)

    acc       = float((y_pred == y_test).mean())
    mean_conf = float(y_proba.max(axis=1).mean())

    # Macro-averaged metrics across all classes
    f1_macro  = float(f1_score(y_test, y_pred, average="macro",  zero_division=0))
    f1_weighted = float(f1_score(y_test, y_pred, average="weighted", zero_division=0))
    prec_macro = float(precision_score(y_test, y_pred, average="macro",  zero_division=0))
    rec_macro  = float(recall_score(y_test,  y_pred, average="macro",  zero_division=0))

    # Per-class F1, precision, recall
    classes      = le.classes_
    present      = np.unique(y_test)
    per_class    = {}
    for idx in present:
        cls_name = le.inverse_transform([idx])[0]
        binary   = (y_test == idx).astype(int)
        p_col    = y_proba[:, idx]
        per_class[cls_name] = {
            "f1":        round(float(f1_score(y_test, y_pred, labels=[idx], average="macro", zero_division=0)), 4),
            "precision": round(float(precision_score(y_test, y_pred, labels=[idx], average="macro", zero_division=0)), 4),
            "recall":    round(float(recall_score(y_test, y_pred, labels=[idx], average="macro", zero_division=0)), 4),
            "support":   int((y_test == idx).sum()),
        }

    # Multi-class log loss
    try:    ll = float(log_loss(y_test, y_proba))
    except: ll = float("nan")

    # Multi-class AUC (OvR, macro)
    try:    auc = float(roc_auc_score(y_test, y_proba, multi_class="ovr", average="macro"))
    except: auc = float("nan")

    # Brier score on dominant class
    dominant = int(np.bincount(y_test).argmax())
    y_bin    = (y_test == dominant).astype(int)
    p_class  = y_proba[:, dominant]
    try:    brier = float(brier_score_loss(y_bin, p_class))
    except: brier = 0.25

    # Matthews Correlation Coefficient (good for imbalanced classes)
    try:    mcc = float(matthews_corrcoef(y_test, y_pred))
    except: mcc = float("nan")

    # Cohen's Kappa
    try:    kappa = float(cohen_kappa_score(y_test, y_pred))
    except: kappa = float("nan")

    # Top 10 feature importances
    fi    = dict(zip(EXCEPTION_FEATURES, rf.feature_importances_))
    top10 = sorted(fi.items(), key=lambda x: -x[1])[:10]

    cal = ("WELL_CALIBRATED" if brier < 0.10 else
           "MODERATE"        if brier < 0.20 else "OVERCONFIDENT")

    return {
        # Overall
        "accuracy":           round(acc, 4),
        "f1_macro":           round(f1_macro, 4),
        "f1_weighted":        round(f1_weighted, 4),
        "precision_macro":    round(prec_macro, 4),
        "recall_macro":       round(rec_macro, 4),
        # Probability quality
        "log_loss":           round(ll, 4)    if not np.isnan(ll)    else None,
        "auc_roc_macro_ovr":  round(auc, 4)  if not np.isnan(auc)   else None,
        "brier_score":        round(brier, 4),
        "calibration_label":  cal,
        # Agreement
        "matthews_corrcoef":  round(mcc, 4)   if not np.isnan(mcc)   else None,
        "cohen_kappa":        round(kappa, 4) if not np.isnan(kappa) else None,
        # Per class
        "per_class":          per_class,
        # Feature importance
        "top10_features":     [(k, round(v, 4)) for k, v in top10],
        # Confidence
        "mean_confidence":    round(mean_conf, 4),
        "confidence_pct":     round(mean_conf * 100, 1),
        "n_test_samples":     int(len(y_test)),
        "grade": ("A" if acc >= 0.95 and f1_macro >= 0.90 else
                  "B" if acc >= 0.85 and f1_macro >= 0.75 else
                  "C" if acc >= 0.75 else "D"),
    }


# alias
_compute_confidence = _compute_scores


def train_exception_classifier(df: pd.DataFrame) -> dict:
    print("    Training Random Forest exception classifier...", end=" ", flush=True)
    X  = df[EXCEPTION_FEATURES].values
    le = LabelEncoder()
    y  = le.fit_transform(df["exception_label"].values)
    cc = np.bincount(y)
    strat = y if cc.min() >= 2 else None
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.20, random_state=42, stratify=strat)

    rf = RandomForestClassifier(n_estimators=300, max_depth=12, min_samples_leaf=10,
                                class_weight="balanced", n_jobs=-1, random_state=42,
                                oob_score=True)
    rf.fit(X_tr, y_tr)
    scores = _compute_scores(rf, le, X_te, y_te)

    # ── OOB score (free generalization estimate — no held-out set needed) ──
    scores["oob_score"] = round(float(rf.oob_score_), 4)

    # ── Confusion matrix ──
    from sklearn.metrics import confusion_matrix
    cm = confusion_matrix(y_te, rf.predict(X_te)).tolist()
    scores["confusion_matrix"]        = cm
    scores["confusion_matrix_labels"] = list(le.classes_)

    joblib.dump(rf, MODELS_DIR / "rf_exception_classifier.pkl")
    joblib.dump(le, MODELS_DIR / "rf_label_encoder.pkl")
    print(f"\n    ── Exception Classifier Scores ──")
    print(f"    Accuracy={scores['accuracy']:.1%}  OOB={scores['oob_score']:.1%}  "
          f"F1_macro={scores['f1_macro']:.4f}  F1_weighted={scores['f1_weighted']:.4f}")
    print(f"    Precision={scores['precision_macro']:.4f}  Recall={scores['recall_macro']:.4f}  "
          f"AUC_OvR={scores.get('auc_roc_macro_ovr','N/A')}  Log_Loss={scores.get('log_loss','N/A')}")
    print(f"    Brier={scores['brier_score']}  Calibration={scores['calibration_label']}  "
          f"MCC={scores.get('matthews_corrcoef','N/A')}  Kappa={scores.get('cohen_kappa','N/A')}")
    print(f"    Grade={scores['grade']}  Confidence={scores['confidence_pct']}%")
    print(f"    Per-class F1: " + "  ".join(
          f"{cls}={v['f1']}" for cls, v in scores['per_class'].items()))
    print(f"    Top features: {', '.join(f'{k}({v})' for k,v in scores['top10_features'][:5])}")
    print(f"    Confusion matrix ({scores['confusion_matrix_labels']}):")
    for lbl, row in zip(scores["confusion_matrix_labels"], scores["confusion_matrix"]):
        print(f"      {lbl:<22} {row}")
    return {"model": rf, "encoder": le, "scores": scores}


def predict_exception(feature_row: dict, rf=None, le=None) -> dict:
    if rf is None:
        rf = joblib.load(MODELS_DIR / "rf_exception_classifier.pkl")
        le = joblib.load(MODELS_DIR / "rf_label_encoder.pkl")
    X    = np.array([[feature_row[f] for f in EXCEPTION_FEATURES]])
    idx  = rf.predict(X)[0]
    prob = rf.predict_proba(X)[0]
    cls  = le.inverse_transform([idx])[0]
    return {
        "predicted_exception": cls,
        "confidence":          round(float(prob.max()), 3),
        "confidence_pct":      round(float(prob.max()) * 100, 1),
        "all_probabilities":   {le.classes_[i]: round(float(p), 3) for i, p in enumerate(prob)},
    }


def llm_explain(exception_type: str, confidence: float, feature_row: dict, mat_id: str) -> dict:
    api_key = os.getenv("ANTHROPIC_API_KEY")
    if api_key:
        try:
            import anthropic
            client = anthropic.Anthropic(api_key=api_key)
            user_msg = (f"Material: {mat_id}\nException: {exception_type} (confidence: {confidence:.0%})\n"
                        f"stock_cover_days={feature_row.get('stock_cover_days','N/A')}\n"
                        f"actual_lead_time_days={feature_row.get('actual_lead_time_days','N/A')}\n"
                        f"lt_deviation_days={feature_row.get('lt_deviation_days','N/A')}\n"
                        f"transit_risk_level={feature_row.get('transit_risk_level','N/A')} (0=Low,1=Med,2=High)\n"
                        f"safety_stock={feature_row.get('safety_stock','N/A')}\n"
                        f"BOM: FG-100001 <- SFG-150001(x1), RM-200010(x2), RM-200020(x1)")
            r = client.messages.create(model="claude-opus-4-5", max_tokens=400,
                                       system=SYSTEM_PROMPT,
                                       messages=[{"role": "user", "content": user_msg}])
            raw = r.content[0].text.strip().lstrip("```json").rstrip("```").strip()
            result = json.loads(raw)
            result["_source"] = "claude_api"
            return result
        except Exception as e:
            print(f"      ⚠️ Claude API: {e}")

    # Rule fallback
    urgency = {"SHORTAGE": 3, "EXPEDITE": 7, "DEMAND_SPIKE": 14, "RESCHEDULE_IN": 14, "NO_EXCEPTION": 99}
    impact  = {"SHORTAGE": "CRITICAL", "EXPEDITE": "HIGH", "DEMAND_SPIKE": "MEDIUM",
               "RESCHEDULE_IN": "LOW", "NO_EXCEPTION": "LOW"}
    cascade = {"SHORTAGE": "Direct BOM cascade: shortage → SFG halt → FG assembly stop.",
               "EXPEDITE": "Delay cascades to FG if not expedited within urgency window.",
               "DEMAND_SPIKE": "Rising demand will deplete safety stock ahead of next receipt.",
               "RESCHEDULE_IN": "Excess stock — push PO out to free working capital.",
               "NO_EXCEPTION": "No cascade risk at this time."}
    action  = {"SHORTAGE": f"Open MD04 for {mat_id}. Place emergency PO ≥ MOQ via ME21N.",
               "EXPEDITE": f"Open ME22N for open POs on {mat_id}. Request earlier delivery.",
               "DEMAND_SPIKE": f"Review MM02 for {mat_id}. Increase safety stock and reorder point.",
               "RESCHEDULE_IN": f"Open ME22N for {mat_id}. Request supplier to push delivery 1-2 weeks.",
               "NO_EXCEPTION": "No action required. Monitor via MD06."}
    return {
        "plain_english":      f"RF model classified {mat_id} as {exception_type} ({confidence:.0%} confidence).",
        "business_impact":    impact.get(exception_type, "MEDIUM"),
        "cascade_risk":       cascade.get(exception_type, "Unknown"),
        "recommended_action": action.get(exception_type, "Review in MD04."),
        "urgency_days":       urgency.get(exception_type, 30),
        "_source":            "rule_fallback",
    }


def run_exception_classifier(data_path="data/training_data.csv", retrain=True) -> list:
    print("=" * 65)
    print("  MODULE 3: EXCEPTION CLASSIFIER (Random Forest)")
    print("=" * 65)
    df = pd.read_csv(data_path, parse_dates=["date"])
    if retrain:
        r       = train_exception_classifier(df)
        rf, le  = r["model"], r["encoder"]
        model_scores = r["scores"]
    else:
        rf = joblib.load(MODELS_DIR / "rf_exception_classifier.pkl")
        le = joblib.load(MODELS_DIR / "rf_label_encoder.pkl")
        model_scores = {"confidence_pct": 85.0, "calibration_label": "UNKNOWN"}

    results = []
    for mat_id in sorted(df["material_id"].unique()):
        mat_df  = df[df["material_id"] == mat_id].sort_values("date")
        latest  = mat_df.iloc[-1][EXCEPTION_FEATURES].to_dict()
        pred    = predict_exception(latest, rf, le)
        exp     = llm_explain(pred["predicted_exception"], pred["confidence"], latest, mat_id)
        results.append({"material_id": mat_id, "prediction": pred, "explanation": exp,
                         "model_scores": model_scores})
        icon = IMPACT_ICON.get(pred["predicted_exception"], "⚪")
        print(f"  {icon} {mat_id}: {pred['predicted_exception']}  "
              f"conf={pred['confidence_pct']}%  impact={exp['business_impact']}  "
              f"urgency={exp['urgency_days']}d  [{exp['_source']}]")

    results.sort(key=lambda x: IMPACT_RANK.get(x["prediction"]["predicted_exception"], 99))
    return results


if __name__ == "__main__":
    run_exception_classifier(retrain=True)
