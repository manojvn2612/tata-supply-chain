"""
module_3_exception_classifier.py
-----------------------------------
ML Module 3: Exception Classification  (Random Forest + Claude API)

Two-stage pipeline:
  Stage 1 — Random Forest multi-class classifier
    Trained on 5-year history, predicts exception type from current
    stock/demand/LT features. Outputs class + probability scores.

  Stage 2 — Claude API (LLM)
    Takes the RF prediction + feature context → generates a plain-language
    explanation, business impact rating, and specific planner action.
    Falls back to rule-based text if no API key is set.

Why RF for classification (not statistics):
  - Learns interaction effects: HIGH risk + low stock + Q4 → SHORTAGE probability
  - Probability calibration tells us how confident the model is
  - Feature importances show planners *what's driving* each exception
  - Handles class imbalance via class_weight='balanced'

Run:  python module_3_exception_classifier.py
"""

import os
import json
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import joblib
from pathlib import Path

MODELS_DIR = Path("models")
MODELS_DIR.mkdir(exist_ok=True)

EXCEPTION_FEATURES = [
    "stock_cover_days",
    "actual_lead_time_days",
    "nominal_lead_time_days",
    "lt_deviation_days",
    "transit_delay_days",
    "delay_event",
    "po_slippage_days",
    "qi_rejection",
    "demand_rolling_slope",
    "daily_demand",
    "stock_level",
    "safety_stock",
    "transit_risk_level",
    "month",
    "quarter",
    "month_sin",
    "month_cos",
]

IMPACT_RANK = {"SHORTAGE": 0, "EXPEDITE": 1, "DEMAND_SPIKE": 2, "RESCHEDULE_IN": 3, "NO_EXCEPTION": 4}
IMPACT_ICON = {"SHORTAGE": "🔴", "EXPEDITE": "🟠", "DEMAND_SPIKE": "🟡", "RESCHEDULE_IN": "🔵", "NO_EXCEPTION": "🟢"}
CLAUDE_MODEL = "claude-opus-4-5"


# ── Stage 1: Random Forest ────────────────────────────────────────────────────

def train_exception_classifier(df: pd.DataFrame) -> dict:
    print("\n  Training Random Forest exception classifier...")

    X = df[EXCEPTION_FEATURES].values
    le = LabelEncoder()
    y = le.fit_transform(df["exception_label"].values)

    # Only stratify if every class has ≥ 2 members
    class_counts = np.bincount(y)
    use_stratify  = y if class_counts.min() >= 2 else None
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=use_stratify
    )

    rf = RandomForestClassifier(
        n_estimators=300,
        max_depth=12,
        min_samples_leaf=10,
        class_weight="balanced",   # handles imbalanced exception classes
        n_jobs=-1,
        random_state=42,
    )
    rf.fit(X_train, y_train)

    y_pred = rf.predict(X_test)
    report = classification_report(y_test, y_pred, labels=np.unique(y_test),
                                   target_names=le.inverse_transform(np.unique(y_test)),
                                   output_dict=True)

    # Overall accuracy
    accuracy = report["accuracy"]
    print(f"    Accuracy: {accuracy:.1%}")
    print(f"    Per-class F1:")
    present_classes = le.inverse_transform(np.unique(y_test))
    for cls in present_classes:
        f1 = report.get(cls, {}).get("f1-score", 0.0)
        print(f"      {IMPACT_ICON.get(cls,'⚪')} {cls:<20} F1={f1:.2f}")

    # Feature importances
    importance = dict(zip(EXCEPTION_FEATURES, rf.feature_importances_))
    top_feats  = sorted(importance.items(), key=lambda x: -x[1])[:5]
    print(f"    Top features: " + "  |  ".join(f"{k}={v:.3f}" for k, v in top_feats))

    joblib.dump(rf, MODELS_DIR / "rf_exception_classifier.pkl")
    joblib.dump(le, MODELS_DIR / "rf_label_encoder.pkl")

    return {"model": rf, "encoder": le, "accuracy": accuracy, "top_features": top_feats}


def predict_exception(feature_row: dict, rf_model=None, le=None) -> dict:
    """Classify one observation and return class + all probabilities."""
    if rf_model is None:
        rf_model = joblib.load(MODELS_DIR / "rf_exception_classifier.pkl")
        le       = joblib.load(MODELS_DIR / "rf_label_encoder.pkl")

    X = np.array([[feature_row[f] for f in EXCEPTION_FEATURES]])
    pred_idx  = rf_model.predict(X)[0]
    proba     = rf_model.predict_proba(X)[0]
    pred_class = le.inverse_transform([pred_idx])[0]

    class_probs = {le.classes_[i]: round(float(p), 3) for i, p in enumerate(proba)}
    confidence  = round(float(proba.max()), 3)

    return {
        "predicted_exception": pred_class,
        "confidence":          confidence,
        "all_probabilities":   class_probs,
    }


# ── Stage 2: Claude LLM Explanation ──────────────────────────────────────────

SYSTEM_PROMPT = """
You are a senior SAP MRP expert. You receive an ML-classified MRP exception and
produce a structured JSON action card for the planner. Be specific — include PO
numbers, quantities, and dates where available.

Respond ONLY with JSON in this exact format (no markdown, no explanation):
{
  "plain_english": "2-3 sentence explanation of what is happening",
  "business_impact": "CRITICAL | HIGH | MEDIUM | LOW",
  "cascade_risk": "How this cascades through the BOM to FG production",
  "recommended_action": "Specific action with quantities and SAP transaction",
  "urgency_days": <integer — days before situation worsens>
}
""".strip()


def llm_explain(exception_type: str, confidence: float, feature_row: dict, mat_id: str) -> dict:
    """Call Claude API for a plain-language explanation. Falls back to rules if no key."""
    api_key = os.getenv("ANTHROPIC_API_KEY")

    if api_key:
        try:
            import anthropic
            client = anthropic.Anthropic(api_key=api_key)
            user_msg = f"""
ML Model Prediction:
  Material     : {mat_id}
  Exception    : {exception_type}  (confidence: {confidence:.0%})

Current Features:
  stock_cover_days      = {feature_row.get('stock_cover_days', 'N/A')}
  actual_lead_time_days = {feature_row.get('actual_lead_time_days', 'N/A')}
  lt_deviation_days     = {feature_row.get('lt_deviation_days', 'N/A')}
  transit_delay_days    = {feature_row.get('transit_delay_days', 'N/A')}
  demand_rolling_slope  = {feature_row.get('demand_rolling_slope', 'N/A')}
  safety_stock          = {feature_row.get('safety_stock', 'N/A')}
  transit_risk_level    = {feature_row.get('transit_risk_level', 'N/A')} (0=Low,1=Med,2=High)

BOM context: FG-100001 uses SFG-150001 (×1) and RM-200010 (×2).
""".strip()
            response = client.messages.create(
                model=CLAUDE_MODEL, max_tokens=500,
                system=SYSTEM_PROMPT,
                messages=[{"role": "user", "content": user_msg}],
            )
            raw = response.content[0].text.strip().lstrip("```json").rstrip("```").strip()
            result = json.loads(raw)
            result["_source"] = "claude_api"
            return result
        except Exception as e:
            print(f"      ⚠️  Claude API error: {e}. Using rule fallback.")

    # ── Rule-based fallback ──
    urgency = {"SHORTAGE": 5, "EXPEDITE": 7, "DEMAND_SPIKE": 14, "RESCHEDULE_IN": 14, "NO_EXCEPTION": 99}
    impact  = {"SHORTAGE": "CRITICAL", "EXPEDITE": "HIGH", "DEMAND_SPIKE": "MEDIUM",
               "RESCHEDULE_IN": "LOW", "NO_EXCEPTION": "LOW"}
    cascade = {
        "SHORTAGE":     "Direct BOM cascade: RM shortage → SFG production halt → FG assembly stop.",
        "EXPEDITE":     "Delay risk cascades to FG if not expedited within urgency window.",
        "DEMAND_SPIKE": "Rising demand will deplete safety stock ahead of next planned receipt.",
        "RESCHEDULE_IN":"Excess stock incoming — consider pushing PO out to free working capital.",
        "NO_EXCEPTION": "No cascade risk identified at this time.",
    }
    action = {
        "SHORTAGE":     f"Open MD04 for {mat_id}. Place emergency PO ≥ MOQ immediately. Contact supplier for earliest possible delivery.",
        "EXPEDITE":     f"Open ME22N for open POs on {mat_id}. Request earlier delivery confirmation from supplier.",
        "DEMAND_SPIKE": f"Review MRP parameters in MM02 for {mat_id}. Consider increasing safety stock and reorder point.",
        "RESCHEDULE_IN":f"Open ME22N for {mat_id}. Request supplier to push delivery date by 1–2 weeks.",
        "NO_EXCEPTION": f"No action required. Continue monitoring via MD06.",
    }
    return {
        "plain_english": f"ML model classified {mat_id} as {exception_type} with {confidence:.0%} confidence.",
        "business_impact": impact.get(exception_type, "MEDIUM"),
        "cascade_risk": cascade.get(exception_type, "Unknown"),
        "recommended_action": action.get(exception_type, "Review in MD04."),
        "urgency_days": urgency.get(exception_type, 30),
        "_source": "rule_fallback",
    }


# ── Main ──────────────────────────────────────────────────────────────────────

def run_exception_classifier(data_path="data/training_data.csv", retrain=True):
    print("=" * 65)
    print("  MODULE 3: EXCEPTION CLASSIFIER  (Random Forest + Claude API)")
    print("=" * 65)

    df = pd.read_csv(data_path, parse_dates=["date"])

    if retrain:
        result = train_exception_classifier(df)
        rf_model = result["model"]
        le       = result["encoder"]
    else:
        rf_model = joblib.load(MODELS_DIR / "rf_exception_classifier.pkl")
        le       = joblib.load(MODELS_DIR / "rf_label_encoder.pkl")

    # Run inference on latest row for each material
    print("\n  Classifying current state for all materials...")
    all_results = []

    for mat_id in df["material_id"].unique():
        mat_df  = df[df["material_id"] == mat_id].sort_values("date")
        latest  = mat_df.iloc[-1][EXCEPTION_FEATURES].to_dict()

        rf_pred = predict_exception(latest, rf_model, le)
        exc     = rf_pred["predicted_exception"]
        conf    = rf_pred["confidence"]

        print(f"    {mat_id}: {IMPACT_ICON.get(exc,'⚪')} {exc}  (confidence: {conf:.0%})", end=" ")
        explanation = llm_explain(exc, conf, latest, mat_id)
        print(f"[{explanation['_source']}]")

        all_results.append({
            "material_id":   mat_id,
            "rf_prediction": rf_pred,
            "explanation":   explanation,
        })

    # Sort by impact severity
    all_results.sort(key=lambda x: IMPACT_RANK.get(x["rf_prediction"]["predicted_exception"], 99))

    # ── Print ──
    print("\n" + "=" * 65)
    print("  PRIORITISED EXCEPTION REPORT")
    print("=" * 65)

    for item in all_results:
        mat   = item["material_id"]
        rf    = item["rf_prediction"]
        exp   = item["explanation"]
        exc   = rf["predicted_exception"]
        icon  = IMPACT_ICON.get(exc, "⚪")

        if exc == "NO_EXCEPTION":
            continue

        print(f"\n  {icon} {exc:<20} {mat}  (RF confidence: {rf['confidence']:.0%})")
        print(f"  {'─'*61}")
        print(f"  📋 {exp['plain_english']}")
        print(f"  ⚡  Impact:   {exp['business_impact']}")
        print(f"  🔗 Cascade:  {exp['cascade_risk']}")
        print(f"  ✅ Action:   {exp['recommended_action']}")
        print(f"  ⏱️  Urgency:  within {exp['urgency_days']} days")
        print(f"  📊 All probabilities: " +
              "  ".join(f"{k}={v:.0%}" for k, v in rf["all_probabilities"].items()))

    return all_results


if __name__ == "__main__":
    run_exception_classifier(retrain=True)
