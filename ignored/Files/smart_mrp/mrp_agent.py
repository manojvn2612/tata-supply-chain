"""
mrp_agent.py
=============
Smart MRP AI Agent — Plant 1000, E-Motor Assembly Line

This is the CENTRAL ORCHESTRATOR. It:
  1. Receives a user request (from CLI or API)
  2. Decides which models to run
  3. Calls each model module and collects results + confidence scores
  4. Feeds everything to Claude API which formats the response
  5. Returns a structured planning brief to the user

Architecture:
  User → Agent → [Model 1..8] → Claude API → Formatted Output

Models called by Agent:
  ┌─────────────────────────────────────────────────────────────┐
  │ Module 1  LSTM Demand Forecast        (demand_forecast)      │
  │ Module 2  XGBoost Lead-Time P50/P90   (leadtime_prediction)  │
  │ Module 3  RF Exception Classifier     (exception_classifier) │
  │ Module 4  IsoForest + GBM Supplier    (supplier_risk)        │
  │ Module 5  BOM Exploder + Stock Netting (bom_exploder)        │
  │ Module 6  Reorder Proposal Engine     (reorder_proposals)    │
  │ Module 7  What-If Scenario Simulator  (whatif_simulator)     │
  │ Module 8  GBM Safety Stock Optimizer  (safety_stock)         │
  └─────────────────────────────────────────────────────────────┘

Run:  python mrp_agent.py
      python mrp_agent.py --retrain        # retrain all models first
      python mrp_agent.py --module demand  # run one specific model
      python mrp_agent.py --query "why is RM-200010 high risk?"
"""

import os
import sys
import json
import time
import argparse
import importlib
from datetime import datetime
from pathlib import Path

DATA_PATH  = "data/training_data.csv"
MODELS_DIR = Path("models")

# ── Agent Identity ─────────────────────────────────────────────────────────────
AGENT_NAME    = "Smart MRP Copilot"
AGENT_VERSION = "3.0"
PLANT         = "Plant 1000 — E-Motor Assembly Line"

BANNER = f"""
╔══════════════════════════════════════════════════════════════╗
║   {AGENT_NAME} v{AGENT_VERSION}                              ║
║   {PLANT}                        ║
║                                                              ║
║   Models available:                                          ║
║   [1] LSTM Demand Forecast      [5] BOM Exploder            ║
║   [2] XGBoost Lead-Time P50/P90 [6] Reorder Proposals       ║
║   [3] RF Exception Classifier   [7] What-If Simulator        ║
║   [4] IsoForest + GBM Risk      [8] Safety Stock Optimizer   ║
║                                                              ║
║   Commands: run · retrain · scores · brief · help · quit    ║
╚══════════════════════════════════════════════════════════════╝
"""

# ── Model registry — agent uses this to know which functions to call ──────────
MODEL_REGISTRY = {
    "demand": {
        "module":   "module_1_demand_forecast",
        "function": "run_demand_forecast",
        "label":    "LSTM Demand Forecast",
        "needs_retrain": True,
    },
    "leadtime": {
        "module":   "module_2_leadtime_prediction",
        "function": "run_leadtime_prediction",
        "label":    "XGBoost Lead-Time",
        "needs_retrain": True,
    },
    "exception": {
        "module":   "module_3_exception_classifier",
        "function": "run_exception_classifier",
        "label":    "RF Exception Classifier",
        "needs_retrain": True,
    },
    "supplier": {
        "module":   "module_4_supplier_risk",
        "function": "run_supplier_risk",
        "label":    "Supplier Risk (IsoForest + GBM)",
        "needs_retrain": True,
    },
    "bom": {
        "module":   "modules_5678",
        "function": "run_bom_exploder",
        "label":    "BOM Exploder",
        "needs_retrain": False,
    },
    "proposals": {
        "module":   "modules_5678",
        "function": "run_reorder_proposals",
        "label":    "Reorder Proposals",
        "needs_retrain": False,
    },
    "whatif": {
        "module":   "modules_5678",
        "function": "run_whatif_simulator",
        "label":    "What-If Simulator",
        "needs_retrain": False,
    },
    "safety_stock": {
        "module":   "modules_5678",
        "function": "run_safety_stock_optimizer",
        "label":    "Safety Stock Optimizer (GBM)",
        "needs_retrain": True,
    },
}

RUN_ORDER = ["demand", "leadtime", "exception", "supplier",
             "bom", "proposals", "whatif", "safety_stock"]

# ── Claude API system prompt ───────────────────────────────────────────────────
AGENT_SYSTEM_PROMPT = """
You are the Smart MRP AI Copilot — a senior supply chain planning assistant
for Plant 1000, E-Motor assembly line (SAP ERP environment).

You have just executed a set of ML models and received their outputs
with calibrated confidence scores. Your job:

1. Start with the PRIORITY ACTION BOARD — most urgent items first (SHORTAGE > EXPEDITE > HIGH RISK)
2. For EVERY ML prediction, show the confidence score prominently and explain it plainly
3. Translate ML metrics into business language:
   - Brier score → "model probabilities are reliable / unreliable"
   - AUC-ROC → "model is X% better than random at spotting late deliveries"
   - MAE → "prediction is off by ±X units on average"
4. Always include: prediction | confidence | recommended SAP action | urgency
5. End with a DECISION TABLE — simple matrix of material / action / deadline
6. Be direct and specific — planners need decisions, not hedging

Confidence display rules:
  ≥85%  → "HIGH confidence — act on this"
  65-85% → "MODERATE confidence — verify before acting"
  <65%  → "⚠️ LOW confidence — treat as directional only"

Always reference SAP transactions: MD04, MD06, ME21N, ME22N, MM02, MB51
""".strip()


# ═══════════════════════════════════════════════════════════════════════════════
#  STEP 1 — DATA GENERATION
# ═══════════════════════════════════════════════════════════════════════════════

def ensure_training_data(force_regenerate=False):
    path = Path(DATA_PATH)
    if force_regenerate or not path.exists() or path.stat().st_size < 1000:
        print("\n  [Agent] Generating training data...")
        from data_generator import generate_all
        generate_all(DATA_PATH)
    else:
        print(f"  [Agent] Training data found: {path} ({path.stat().st_size // 1024}KB)")


# ═══════════════════════════════════════════════════════════════════════════════
#  STEP 2 — MODEL EXECUTION
# ═══════════════════════════════════════════════════════════════════════════════

def _call_model(model_key: str, retrain: bool = False) -> dict:
    """
    Agent calls a single model module by key.
    Returns: {"status": "ok"|"error", "result": ..., "elapsed": float}
    """
    reg = MODEL_REGISTRY[model_key]
    t0  = time.time()
    try:
        mod = importlib.import_module(reg["module"])
        fn  = getattr(mod, reg["function"])
        if reg["needs_retrain"]:
            if reg["function"] in ("run_bom_exploder", "run_whatif_simulator",
                                   "run_reorder_proposals"):
                result = fn()
            else:
                result = fn(DATA_PATH, retrain=retrain)
        else:
            result = fn()
        return {"status": "ok", "result": result, "elapsed": round(time.time() - t0, 1)}
    except Exception as e:
        import traceback
        return {"status": "error", "error": str(e),
                "traceback": traceback.format_exc()[-500:],
                "elapsed": round(time.time() - t0, 1)}


def run_all_models(retrain: bool = False) -> dict:
    """Agent runs all 8 models in sequence and collects results."""
    print(f"\n  [Agent] Running all models  (retrain={retrain})\n")
    all_results = {}
    for key in RUN_ORDER:
        label = MODEL_REGISTRY[key]["label"]
        print(f"  ▶ [{key}] {label}...", flush=True)
        r = _call_model(key, retrain=retrain)
        all_results[key] = r
        if r["status"] == "ok":
            print(f"      ✓ done in {r['elapsed']}s")
        else:
            print(f"      ✗ ERROR: {r['error'][:80]}")
    print(f"\n  [Agent] All models complete.\n")
    return all_results


def run_selected_models(keys: list, retrain: bool = False) -> dict:
    """Agent runs only the specified models."""
    results = {}
    for key in keys:
        if key not in MODEL_REGISTRY:
            results[key] = {"status": "error", "error": f"Unknown model key: {key}"}
            continue
        print(f"  ▶ [{key}] {MODEL_REGISTRY[key]['label']}...", flush=True)
        r = _call_model(key, retrain=retrain)
        results[key] = r
        print(f"      {'✓' if r['status']=='ok' else '✗'}")
    return results


# ═══════════════════════════════════════════════════════════════════════════════
#  STEP 3 — RESULT FORMATTING FOR LLM
# ═══════════════════════════════════════════════════════════════════════════════

def _conf_bar(pct: float, width: int = 10) -> str:
    filled = round((pct or 0) / 100 * width)
    filled = max(0, min(width, filled))
    label  = ("HIGH" if pct >= 85 else "MODERATE" if pct >= 65 else "⚠️LOW")
    return f"{'█' * filled}{'░' * (width - filled)} {pct:.0f}% [{label}]"


def format_results_for_llm(results: dict) -> str:
    """Converts model outputs into a structured text context for Claude."""
    lines = [f"=== SMART MRP MODEL RESULTS — {datetime.now().strftime('%Y-%m-%d %H:%M')} ===\n"]

    # Demand forecasts
    if "demand" in results and results["demand"]["status"] == "ok":
        lines.append("── MODULE 1: LSTM DEMAND FORECAST ──")
        for mat_id, fc in results["demand"]["result"].items():
            s = fc.get("scores", {})
            lines.append(f"  {mat_id}: 14d_total={fc['14d_total']}  avg={fc['daily_avg']}/day  "
                         f"peak={fc['peak_demand']} on {fc['peak_day']}")
            lines.append(f"    MAE={s.get('mae','N/A')}  RMSE={s.get('rmse','N/A')}  "
                         f"MAPE={s.get('mape_pct','N/A')}%  R²={s.get('r2_score','N/A')}")
            lines.append(f"    HuberLoss={s.get('huber_loss','N/A')}  Bias={s.get('bias','N/A')}  "
                         f"DirAcc={s.get('directional_acc','N/A')}  Grade={s.get('grade','N/A')}")
            lines.append(f"    Confidence={_conf_bar(s.get('confidence_pct', 70))}  "
                         f"Uncertainty={s.get('uncertainty_band','N/A')}")
        lines.append("")

    # Lead time
    if "leadtime" in results and results["leadtime"]["status"] == "ok":
        lines.append("── MODULE 2: LEAD-TIME PREDICTION (XGBoost P50/P90) ──")
        for mat_id, pred in results["leadtime"]["result"].items():
            s = pred.get("scores", pred.get("confidence", {}))
            lines.append(f"  {mat_id}: SAP={pred['sap_nominal_lt']}d  "
                         f"P50={pred['predicted_lt_p50']}d  P90={pred['predicted_lt_p90']}d  "
                         f"gap={pred['gap_p90_vs_sap']:+.1f}d")
            lines.append(f"    {pred['risk_flag']}")
            lines.append(f"    MAE={s.get('mae_days','N/A')}d  RMSE={s.get('rmse_days','N/A')}d  "
                         f"MAPE={s.get('mape_pct','N/A')}%  R²={s.get('r2_score','N/A')}  "
                         f"Bias={s.get('bias_days','N/A')}d")
            lines.append(f"    P90_cov={s.get('p90_coverage','N/A')}  P90_MAE={s.get('p90_mae_days','N/A')}d  "
                         f"PinballLoss={s.get('pinball_loss_p90','N/A')}  P50-P90gap={s.get('avg_p50_p90_gap','N/A')}d")
            lines.append(f"    Grade={s.get('grade','N/A')}  Confidence={_conf_bar(s.get('confidence_pct', 75))}")
            if s.get('top5_features'):
                lines.append(f"    Top features: {', '.join(f'{k}({v})' for k,v in s['top5_features'][:3])}")
        lines.append("")

    # Exception classifier
    if "exception" in results and results["exception"]["status"] == "ok":
        lines.append("── MODULE 3: EXCEPTION CLASSIFIER (Random Forest) ──")
        ex_list = results["exception"]["result"]
        if ex_list:
            ms = ex_list[0].get("model_scores", ex_list[0].get("model_confidence", {}))
            lines.append(f"  Model: Acc={ms.get('accuracy','N/A')}  F1_macro={ms.get('f1_macro','N/A')}  "
                         f"F1_weighted={ms.get('f1_weighted','N/A')}")
            lines.append(f"    Precision={ms.get('precision_macro','N/A')}  Recall={ms.get('recall_macro','N/A')}  "
                         f"AUC_OvR={ms.get('auc_roc_macro_ovr','N/A')}  LogLoss={ms.get('log_loss','N/A')}")
            lines.append(f"    Brier={ms.get('brier_score','N/A')}  Calibration={ms.get('calibration_label','N/A')}  "
                         f"MCC={ms.get('matthews_corrcoef','N/A')}  Kappa={ms.get('cohen_kappa','N/A')}")
            lines.append(f"    Grade={ms.get('grade','N/A')}  Confidence={_conf_bar(ms.get('confidence_pct', 85))}")
            if ms.get('per_class'):
                lines.append(f"    Per-class F1: " + "  ".join(
                    f"{cls}={v['f1']}" for cls, v in ms['per_class'].items()))
            if ms.get('top10_features'):
                lines.append(f"    Top features: {', '.join(f'{k}({v})' for k,v in ms['top10_features'][:5])}")
        for item in ex_list:
            p   = item["prediction"]
            exp = item["explanation"]
            lines.append(f"\n  {item['material_id']}: {p['predicted_exception']}  "
                         f"conf={_conf_bar(p['confidence_pct'])}")
            lines.append(f"    All probs: " +
                         "  ".join(f"{k}={v:.0%}" for k, v in p["all_probabilities"].items()))
            lines.append(f"    Impact: {exp['business_impact']}  |  Urgency: {exp['urgency_days']}d")
            lines.append(f"    → {exp['recommended_action']}")
            lines.append(f"    Cascade: {exp['cascade_risk']}")
        lines.append("")

    # Supplier risk
    if "supplier" in results and results["supplier"]["status"] == "ok":
        lines.append("── MODULE 4: SUPPLIER RISK (IsoForest + GBM) ──")
        for mat_id, rep in results["supplier"]["result"].items():
            gs = rep.get("gbm_confidence", {})
            is_ = rep.get("iso_confidence", {})
            p_str = f"{rep['p_late_next_delivery']:.0%}" if rep["p_late_next_delivery"] is not None else "N/A"
            lines.append(f"  {mat_id}: anomaly={rep['anomaly_risk_score']}/100  P(late)={p_str}  {rep['risk_tier']}")
            # IsoForest scores
            lines.append(f"    IsoForest: anomaly_rate={is_.get('anomaly_rate','N/A'):.1%}  "
                         f"sep={is_.get('normal_vs_anomaly_sep','N/A')}  "
                         f"grade={is_.get('grade','N/A')}  conf={_conf_bar(is_.get('confidence_pct', 85))}"
                         if is_ else f"    IsoForest: conf={_conf_bar(85)}")
            # GBM scores
            if gs:
                lines.append(f"    GBM: AUC={gs.get('auc_roc','N/A')}  LogLoss={gs.get('log_loss','N/A')}  "
                             f"Brier={gs.get('brier_score','N/A')}  F1={gs.get('f1_score','N/A')}")
                lines.append(f"         Prec={gs.get('precision','N/A')}  Recall={gs.get('recall','N/A')}  "
                             f"MCC={gs.get('matthews_corrcoef','N/A')}  Kappa={gs.get('cohen_kappa','N/A')}")
                lines.append(f"         Grade={gs.get('grade','N/A')}  Quality={gs.get('model_quality','N/A')}/100  "
                             f"conf={_conf_bar(gs.get('confidence_pct', 80))}")
        lines.append("")

    # BOM / Net requirements
    if "bom" in results and results["bom"]["status"] == "ok":
        bom_df = results["bom"]["result"]
        if hasattr(bom_df, "iterrows"):
            lines.append("── MODULE 5: BOM NET REQUIREMENTS ──")
            for _, row in bom_df.iterrows():
                flag = "⚠️ GAP" if row["net_req_total"] > 0 else "✅ OK"
                lines.append(f"  {row['material_id']:<14} Level={row['level']}  "
                             f"net_req_total={row['net_req_total']:.1f}  {flag}")
        lines.append("")

    # Reorder proposals
    if "proposals" in results and results["proposals"]["status"] == "ok":
        props = [p for p in results["proposals"]["result"] if p["action"] != "NO_ACTION"]
        total = sum(p["price_total"] for p in props)
        lines.append(f"── MODULE 6: REORDER PROPOSALS ({len(props)} POs pending) ──")
        for p in props:
            lines.append(f"  {p['action']:<22} {p['material_id']:<14} "
                         f"qty={p['order_qty']}  due={p['due_date']}  ₹{p['price_total']:,.0f}  "
                         f"{p['priority']}")
        lines.append(f"  Total value: ₹{total:,.0f}")
        lines.append("")

    # What-if scenarios
    if "whatif" in results and results["whatif"]["status"] == "ok":
        lines.append("── MODULE 7: WHAT-IF SCENARIOS ──")
        for r in results["whatif"]["result"]:
            sev_short = r["severity"].split("—")[0].strip()
            n_crit    = sum(1 for m in r["affected_materials"] if not m["can_cover_in_time"])
            cost_str  = f"₹{r['total_stockout_cost']:,.0f}" if r["total_stockout_cost"] else "₹0"
            lines.append(f"  {r['scenario_name']:<42} {sev_short}  "
                         f"at_risk={n_crit}  cost={cost_str}")
        lines.append("")

    # Safety stock
    if "safety_stock" in results and results["safety_stock"]["status"] == "ok":
        lines.append("── MODULE 8: SAFETY STOCK OPTIMIZER (GBM) ──")
        for mat_id, res in results["safety_stock"]["result"].items():
            s = res.get("scores", res.get("confidence", {}))
            lines.append(f"  {mat_id}: SAP={res['current_ss_sap']} → ML={res['predicted_ss_ml']}  "
                         f"Δ{res['delta']:+.0f} ({res['delta_pct']:+.0f}%)  {res['recommendation']}")
            lines.append(f"    MAE={s.get('mae_units','N/A')}  RMSE={s.get('rmse_units','N/A')}  "
                         f"MAPE={s.get('mape_pct','N/A')}%  R²={s.get('r2_score','N/A')}")
            lines.append(f"    ExplVar={s.get('explained_variance','N/A')}  MedianAE={s.get('median_abs_error','N/A')}  "
                         f"Huber={s.get('huber_loss','N/A')}  Bias={s.get('bias_units','N/A')}")
            lines.append(f"    Grade={s.get('grade','N/A')}  Confidence={_conf_bar(s.get('confidence_pct', 70))}  "
                         f"{s.get('note','')}")
        lines.append("")

    return "\n".join(lines)


# ═══════════════════════════════════════════════════════════════════════════════
#  STEP 4 — CLAUDE API CALL
# ═══════════════════════════════════════════════════════════════════════════════

def call_claude_agent(user_message: str, model_context: str,
                      history: list) -> str:
    """Send model results + user question to Claude and return response."""
    api_key = os.getenv("ANTHROPIC_API_KEY")
    if not api_key:
        return _fallback_format(model_context, user_message)

    try:
        import anthropic
        client = anthropic.Anthropic(api_key=api_key)
        messages = history.copy()
        messages.append({
            "role": "user",
            "content": f"MODEL OUTPUTS:\n{model_context}\n\nUSER QUERY: {user_message}"
        })
        response = client.messages.create(
            model="claude-opus-4-5",
            max_tokens=2000,
            system=AGENT_SYSTEM_PROMPT,
            messages=messages[-20:],   # keep last 10 turns
        )
        return response.content[0].text
    except Exception as e:
        return (f"[Claude API error: {e}]\n\n"
                + _fallback_format(model_context, user_message))


def _fallback_format(context: str, query: str) -> str:
    """Plain-text summary when Claude API is unavailable."""
    lines = ["\n" + "═" * 60, "  SMART MRP PLANNING BRIEF (no-API mode)", "═" * 60 + "\n"]
    lines.append(context)
    lines.append("\n  [Set ANTHROPIC_API_KEY for AI-enhanced explanations]")
    return "\n".join(lines)


# ═══════════════════════════════════════════════════════════════════════════════
#  STEP 5 — SCORES DISPLAY
# ═══════════════════════════════════════════════════════════════════════════════

def display_scores(results: dict):
    """Print a clean confidence scorecard for all models."""
    print("\n" + "═" * 65)
    print("  MODEL CONFIDENCE SCORECARD")
    print("═" * 65)

    score_map = {
        "demand":      ("LSTM Demand",         "confidence"),
        "leadtime":    ("XGBoost Lead-Time",    "confidence"),
        "exception":   ("RF Exception",         "model_confidence"),
        "supplier":    ("Supplier IsoForest",    "iso_confidence"),
        "safety_stock":("GBM Safety Stock",     "confidence"),
    }

    for key, (label, conf_key) in score_map.items():
        if key not in results or results[key]["status"] != "ok":
            print(f"  {label:<28}  NOT RUN")
            continue
        r = results[key]["result"]

        # Extract confidence from various result shapes
        conf = None
        if key == "demand":
            # dict of mat_id -> forecast
            confs = [v.get("confidence", {}).get("confidence_pct", 0)
                     for v in r.values() if isinstance(v, dict)]
            conf = {"confidence_pct": round(sum(confs) / len(confs), 1) if confs else 0}
        elif key == "leadtime":
            confs = [v.get("confidence", {}).get("confidence_pct", 0)
                     for v in r.values() if isinstance(v, dict)]
            conf = {"confidence_pct": round(sum(confs) / len(confs), 1) if confs else 0}
        elif key == "exception":
            conf = r[0].get("model_confidence", {}) if r else {}
        elif key == "supplier":
            # average across materials
            confs = [v.get("gbm_confidence", {}).get("confidence_pct", 0)
                     for v in r.values() if isinstance(v, dict)]
            conf = {"confidence_pct": round(sum(confs) / len(confs), 1) if confs else 0}
        elif key == "safety_stock":
            confs = [v.get("confidence", {}).get("confidence_pct", 0)
                     for v in r.values() if isinstance(v, dict)]
            conf = {"confidence_pct": round(sum(confs) / len(confs), 1) if confs else 0}

        if conf:
            pct = conf.get("confidence_pct", 0)
            bar = _conf_bar(pct)
            extra = ""
            if key == "exception":
                extra = (f"  acc={conf.get('accuracy', ''):.1%}  "
                         f"brier={conf.get('brier_score', '')}  "
                         f"cal={conf.get('calibration_label', '')}")
            print(f"  {label:<28}  {bar}{extra}")

    print()


# ═══════════════════════════════════════════════════════════════════════════════
#  MAIN — Interactive CLI Agent Loop
# ═══════════════════════════════════════════════════════════════════════════════

def run_agent(retrain_flag=False, module_key=None, query=None):
    print(BANNER)
    results       = {}
    model_context = ""
    history       = []

    # Non-interactive mode: run specific module then exit
    if module_key:
        ensure_training_data()
        if module_key in MODEL_REGISTRY:
            results[module_key] = _call_model(module_key, retrain=retrain_flag)
            model_context = format_results_for_llm(results)
            if query:
                response = call_claude_agent(query, model_context, history)
                print(response)
        else:
            print(f"  Unknown module: {module_key}")
            print(f"  Available: {', '.join(MODEL_REGISTRY.keys())}")
        return

    if query and not module_key:
        ensure_training_data()
        print("  [Agent] Running all models to answer your query...\n")
        results       = run_all_models(retrain=retrain_flag)
        model_context = format_results_for_llm(results)
        response      = call_claude_agent(query, model_context, history)
        print("\n" + "─" * 65)
        print(response)
        return

    # Interactive loop
    print("  Type 'help' for commands.\n")
    while True:
        try:
            user_input = input("  You: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n  [Agent] Session ended.\n")
            break

        if not user_input:
            continue

        cmd = user_input.lower()

        if cmd in ("quit", "exit", "q"):
            print("  [Agent] Goodbye. All model outputs saved in models/.\n")
            break

        elif cmd == "help":
            print("""
  Commands:
    run          — run all 8 models and get full planning brief
    retrain      — retrain all models then run (takes longer)
    scores       — show confidence scorecard for all models
    brief        — re-display last planning brief
    demand       — run only LSTM demand forecast
    leadtime     — run only XGBoost lead-time prediction
    exception    — run only RF exception classifier
    supplier     — run only supplier risk models
    bom          — run only BOM exploder
    proposals    — run only reorder proposals
    whatif       — run only what-if scenarios
    safety_stock — run only safety stock optimizer
    data         — regenerate training data
    quit         — exit

  Or ask any question in plain English after running models, e.g.:
    "Which material needs action TODAY?"
    "Why is RM-200010 flagged as RESCHEDULE_IN?"
    "What is the confidence for the shortage prediction?"
    "Should I trust the safety stock recommendations?"
    "What happens if Vendor B delays another 7 days?"
""")

        elif cmd == "run":
            ensure_training_data()
            results       = run_all_models(retrain=False)
            model_context = format_results_for_llm(results)
            response      = call_claude_agent("Give me the full planning brief.", model_context, history)
            print(f"\n  [Agent]:\n{response}\n")
            history.append({"role": "user",      "content": "Give me the full planning brief."})
            history.append({"role": "assistant", "content": response})

        elif cmd == "retrain":
            ensure_training_data()
            results       = run_all_models(retrain=True)
            model_context = format_results_for_llm(results)
            response      = call_claude_agent("Give me the full planning brief.", model_context, history)
            print(f"\n  [Agent]:\n{response}\n")
            history.append({"role": "user",      "content": "Full planning brief after retraining."})
            history.append({"role": "assistant", "content": response})

        elif cmd == "scores":
            if not results:
                print("  [Agent] No models run yet. Type 'run' first.\n")
            else:
                display_scores(results)

        elif cmd == "brief":
            if not model_context:
                print("  [Agent] No results yet. Type 'run' first.\n")
            else:
                response = call_claude_agent("Show the full planning brief.", model_context, history)
                print(f"\n  [Agent]:\n{response}\n")

        elif cmd == "data":
            ensure_training_data(force_regenerate=True)
            print("  [Agent] Training data regenerated.\n")

        elif cmd in MODEL_REGISTRY:
            ensure_training_data()
            print(f"  [Agent] Running [{cmd}] {MODEL_REGISTRY[cmd]['label']}...")
            r = _call_model(cmd, retrain=False)
            results[cmd]  = r
            model_context = format_results_for_llm(results)
            if r["status"] == "ok":
                response = call_claude_agent(
                    f"Show me results for {MODEL_REGISTRY[cmd]['label']}.", model_context, history)
                print(f"\n  [Agent]:\n{response}\n")
            else:
                print(f"  [Agent] Error: {r['error']}\n")

        else:
            # Free-form question
            if not results:
                print("  [Agent] Run the models first with 'run', then ask your question.\n")
            else:
                response = call_claude_agent(user_input, model_context, history)
                print(f"\n  [Agent]:\n{response}\n")
                history.append({"role": "user",      "content": user_input})
                history.append({"role": "assistant", "content": response})
                if len(history) > 20:
                    history = history[-20:]


# ═══════════════════════════════════════════════════════════════════════════════
#  Entry point
# ═══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Smart MRP AI Agent")
    parser.add_argument("--retrain", action="store_true", help="Retrain all models before running")
    parser.add_argument("--module",  type=str, default=None,
                        help=f"Run single model: {', '.join(MODEL_REGISTRY.keys())}")
    parser.add_argument("--query",   type=str, default=None,
                        help="One-shot query (non-interactive)")
    args = parser.parse_args()

    run_agent(
        retrain_flag=args.retrain,
        module_key=args.module,
        query=args.query,
    )
