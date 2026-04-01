# """
# run_pipeline.py
# ----------------
# Master runner — Smart MRP ML Pipeline  (all 9 modules)

# Modules:
#   1  LSTM Demand Forecast        (PyTorch)
#   2  XGBoost Lead-Time P50/P90   (XGBoost)
#   3  RF Exception Classifier     (sklearn RF + Claude API)
#   4  Supplier Risk ML            (IsolationForest + GBM)
#   5  BOM Exploder + Stock Netting
#   6  Reorder Proposal Engine
#   7  What-If Scenario Simulator
#   8  Safety Stock Optimizer      (GradientBoosting)
#   9  Prophet Forecast Baseline

# Usage:
#   python run_pipeline.py               # all modules, retrain
#   python run_pipeline.py --no-retrain  # skip retraining
#   python run_pipeline.py --module 5    # single module
#   python run_pipeline.py --eval        # backtest only
# """

# import sys, time, importlib
# from datetime import date

# BANNER = """
# ╔══════════════════════════════════════════════════════════════════╗
# ║    SMART MRP — FULL ML PIPELINE  v2                             ║
# ║    LSTM · XGBoost · RandomForest · IsolationForest · GBM        ║
# ║    BOM Explosion · Reorder Proposals · Scenarios · SS Opt       ║
# ╚══════════════════════════════════════════════════════════════════╝
# """

# DATA_PATH = "data/training_data.csv"

# MODULE_MAP = {
#     1: ("LSTM Demand Forecast",      "module_1_lstm_demand_forecast",   "run_lstm_forecast",        True),
#     2: ("XGBoost Lead-Time",         "module_2_xgboost_leadtime",       "run_xgboost_leadtime",     True),
#     3: ("RF Exception Classifier",   "module_3_exception_classifier",   "run_exception_classifier", True),
#     4: ("Supplier Risk ML",          "module_4_supplier_risk_ml",       "run_supplier_risk_ml",     True),
#     5: ("BOM Exploder",              "module_5_bom_exploder",           "run_bom_exploder",         False),
#     6: ("Reorder Proposals",         "module_6_reorder_proposals",      "run_reorder_proposals",    False),
#     7: ("What-If Simulator",         "module_7_whatif_simulator",       "run_whatif_simulator",     False),
#     8: ("Safety Stock Optimizer",    "module_8_safety_stock_optimizer", "run_safety_stock_optimizer",True),
#     9: ("Prophet Forecast Baseline", "module_9_prophet_forecast",       "run_prophet_forecast",     False),
# }


# def run_module(num, retrain=True):
#     name, mod_name, fn_name, takes_retrain = MODULE_MAP[num]
#     mod = importlib.import_module(mod_name)
#     fn  = getattr(mod, fn_name)
#     if takes_retrain:
#         return fn(DATA_PATH, retrain=retrain)
#     elif fn_name in ("run_prophet_forecast",):
#         return fn(DATA_PATH)
#     else:
#         return fn()


# def run_all(retrain=True):
#     print(BANNER)
#     print(f"  Date: {date.today()}  |  Retrain: {retrain}\n")

#     from data_generator import generate_all
#     print("  Generating 5-year training data...")
#     generate_all(DATA_PATH)

#     t0 = time.time()
#     results = {}

#     for num in range(1, 10):
#         name = MODULE_MAP[num][0]
#         print(f"\n{'='*65}\n  [{num}] {name}\n{'='*65}")
#         try:
#             results[num] = run_module(num, retrain=retrain)
#         except Exception as e:
#             print(f"  ⚠️  Module {num} skipped: {e}")
#             print("      Check requirements.txt and install missing packages.")
#             results[num] = None

#     print(f"\n{'='*65}\n  [E] EVALUATOR\n{'='*65}")
#     from evaluator import run_evaluator
#     results["eval"] = run_evaluator(DATA_PATH)

#     _brief(results)
#     print(f"  Total runtime: {round(time.time()-t0,1)}s\n")
#     return results


# def _brief(results):
#     print("\n" + "═"*65)
#     print("  ★  DAILY PLANNING BRIEF  ★")
#     print("═"*65)

#     if results.get(1):
#         print("\n  LSTM FORECASTS (14d ahead)")
#         for m, fc in results[1].items():
#             print(f"    {m}: avg={fc['daily_avg']}/day  total={fc['14d_total']}  peak={fc['peak_demand']} on {fc['peak_day']}")

#     if results.get(2):
#         print("\n  LEAD-TIME RISK (XGBoost P90)")
#         for m, p in results[2].items():
#             gap  = p.get("gap_p90_vs_sap", 0)
#             icon = "🔴" if gap > 7 else ("🟡" if gap > 3 else "🟢")
#             print(f"    {icon} {m}: SAP={p['sap_nominal_lt']}d P90={p['predicted_lt_p90']}d ({gap:+.1f}d)")

#     if results.get(3):
#         icons = {"SHORTAGE":"🔴","EXPEDITE":"🟠","DEMAND_SPIKE":"🟡","RESCHEDULE_IN":"🔵","NO_EXCEPTION":"🟢"}
#         print("\n  EXCEPTIONS (Random Forest)")
#         for item in results[3]:
#             exc  = item["rf_prediction"]["predicted_exception"]
#             conf = item["rf_prediction"]["confidence"]
#             if exc != "NO_EXCEPTION":
#                 action = item["explanation"]["recommended_action"][:55]
#                 print(f"    {icons.get(exc,'⚪')} {item['material_id']}: {exc} ({conf:.0%}) → {action}...")

#     if results.get(8):
#         print("\n  SAFETY STOCK (GBM Optimizer)")
#         for m, r in results[8].items():
#             print(f"    {m}: {r['current_ss_sap']} → {r['predicted_ss_ml']}  ({r['delta']:+.0f})")

#     if results.get(7):
#         print("\n  WHAT-IF SCENARIOS")
#         for r in results[7]:
#             print(f"    {r['severity'].split(chr(8212))[0].strip():<20} {r['scenario_name']}")

#     print("\n  All changes require planner approval before SAP write-back.")
#     print("═"*65 + "\n")


# if __name__ == "__main__":
#     args    = sys.argv[1:]
#     retrain = "--no-retrain" not in args
#     if "--eval" in args:
#         from evaluator import run_evaluator
#         run_evaluator(DATA_PATH)
#     elif "--module" in args:
#         idx = args.index("--module")
#         run_module(int(args[idx+1]), retrain=retrain)
#     else:
#         run_all(retrain=retrain)
"""run_pipeline_global.py — runs all 9 modules using global models"""
import sys, time, importlib
from datetime import date

DATA_PATH = "data/training_data.csv"

MODULE_MAP = {
    1: ("module_1_global_lstm_forecast",          "run_global_lstm_forecast",         True),
    2: ("module_2_global_xgboost_leadtime",       "run_global_xgboost_leadtime",      True),
    3: ("module_3_exception_classifier",          "run_exception_classifier",         True),
    4: ("module_4_global_supplier_risk",          "run_global_supplier_risk",         True),
    5: ("module_5_bom_exploder",                  "run_bom_exploder",                 False),
    6: ("module_6_reorder_proposals",             "run_reorder_proposals",            False),
    7: ("module_7_whatif_simulator",              "run_whatif_simulator",             False),
    8: ("module_8_global_safety_stock_optimizer", "run_global_safety_stock_optimizer",True),
    9: ("module_9_prophet_forecast",              "run_prophet_forecast",             False),
}

def run_module(num, retrain=True):
    mod_name, fn_name, takes_retrain = MODULE_MAP[num]
    fn = getattr(importlib.import_module(mod_name), fn_name)
    return fn(DATA_PATH, retrain=retrain) if takes_retrain else fn(DATA_PATH) if fn_name=="run_prophet_forecast" else fn()

def run_all(retrain=True):
    from data_generator import generate_all
    generate_all(DATA_PATH)
    t0 = time.time(); results = {}
    for num in range(1, 10):
        try: results[num] = run_module(num, retrain)
        except Exception as e: print(f"  ⚠️  Module {num}: {e}"); results[num] = None
    from evaluator import run_evaluator
    results["eval"] = run_evaluator(DATA_PATH)
    print(f"\n  Done in {round(time.time()-t0,1)}s")
    return results

if __name__ == "__main__":
    args = sys.argv[1:]
    if "--eval" in args:
        from evaluator import run_evaluator; run_evaluator(DATA_PATH)
    elif "--module" in args:
        run_module(int(args[args.index("--module")+1]), "--no-retrain" not in args)
    else:
        run_all("--no-retrain" not in args)