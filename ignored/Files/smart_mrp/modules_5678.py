"""
module_5_bom_exploder.py  +  module_6_reorder_proposals.py  +
module_7_whatif_simulator.py  +  module_8_safety_stock_optimizer.py

Combined into one file for cleaner agent imports.
Each section has its own run_* function called by the agent.
"""

import numpy as np
import pandas as pd
from copy import deepcopy
from datetime import date, timedelta
from pathlib import Path
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_absolute_error, r2_score
from scipy.stats import norm
import joblib
import sys, os
sys.path.insert(0, os.path.dirname(__file__))
from config import MATERIALS, BOM, CONSUMPTION_HISTORY, MONTHS

MODELS_DIR = Path("models")
MODELS_DIR.mkdir(exist_ok=True)

PLANNING_DATE          = date(2026, 2, 3)
WORKING_DAYS_MONTH     = 22
STOCKOUT_COST_PER_DAY  = 500_000   # ₹ revenue at risk per day

# ═══════════════════════════════════════════════════════════════════════════════
#  MODULE 5 — BOM EXPLODER + STOCK NETTING
# ═══════════════════════════════════════════════════════════════════════════════

def flatten_bom(fg_id, qty=1.0, level=0, _visited=None):
    if _visited is None: _visited = set()
    if fg_id in _visited: return []
    _visited.add(fg_id)
    rows = []
    for comp, comp_qty in BOM.get(fg_id, {}).items():
        total = qty * comp_qty
        rows.append({"parent": fg_id, "component": comp,
                     "qty_per_parent": comp_qty, "total_qty_per_fg": total, "level": level + 1})
        rows.extend(flatten_bom(comp, total, level + 1, _visited))
    return rows


def get_confirmed_po_qty(mat_id):
    mat = MATERIALS.get(mat_id, {})
    return sum(po["qty"] for po in mat.get("open_pos", []))


def compute_net_requirements(fg_id, fg_demand_monthly):
    flat = flatten_bom(fg_id)
    if not flat:
        return pd.DataFrame()
    bom_df = (pd.DataFrame(flat)
                .groupby(["component", "level"])
                .agg(total_qty_per_fg=("total_qty_per_fg", "sum"))
                .reset_index().sort_values("level"))
    # Include the FG itself
    rows = [{"material_id": fg_id, "level": 0,
             "gross_req": fg_demand_monthly,
             "net_req":   [max(0, r - MATERIALS[fg_id]["stock"] - get_confirmed_po_qty(fg_id))
                          for r in fg_demand_monthly],
             "net_req_total": max(0, sum(fg_demand_monthly)
                                 - MATERIALS[fg_id]["stock"] - get_confirmed_po_qty(fg_id))}]
    for _, row in bom_df.iterrows():
        comp    = row["component"]
        qty_pgf = row["total_qty_per_fg"]
        gross   = [d * qty_pgf for d in fg_demand_monthly]
        stock   = MATERIALS.get(comp, {}).get("stock", 0)
        po_qty  = get_confirmed_po_qty(comp)
        net     = [max(0, g - stock - po_qty) for g in gross]
        rows.append({"material_id": comp, "level": int(row["level"]),
                     "gross_req": gross, "net_req": net,
                     "net_req_total": round(sum(net), 1)})
    return pd.DataFrame(rows)


def run_bom_exploder() -> pd.DataFrame:
    print("=" * 65)
    print("  MODULE 5: BOM EXPLODER + STOCK NETTING")
    print("=" * 65)
    fg_demand = CONSUMPTION_HISTORY["FG-100001"]
    net_df    = compute_net_requirements("FG-100001", fg_demand)
    print(f"  {'Material':<14} {'Level':<8} {'Net Req Total':>15}")
    print(f"  {'─'*40}")
    for _, r in net_df.iterrows():
        flag = " ⚠️" if r["net_req_total"] > 0 else " ✅"
        print(f"  {r['material_id']:<14} {r['level']:<8} {r['net_req_total']:>15.1f}{flag}")
    return net_df


# ═══════════════════════════════════════════════════════════════════════════════
#  MODULE 6 — REORDER PROPOSALS
# ═══════════════════════════════════════════════════════════════════════════════

PRIORITY_RANK = {"🔴 CRITICAL": 0, "🟠 HIGH": 1, "🟡 MEDIUM": 2, "🟢 COVERED": 3}


def apply_lot_size(net_req, mat_id):
    if net_req <= 0: return 0
    mat  = MATERIALS[mat_id]
    proc = mat["lot_size"]
    moq  = mat["moq"]
    lsv  = mat["lot_size_value"]
    if proc == "EX":
        qty = net_req
    elif proc == "FX":
        qty = np.ceil(net_req / lsv) * lsv
    elif proc == "HB":
        qty = max(net_req, mat["max_stock"] - mat["stock"])
    else:
        qty = net_req
    return max(qty, moq)


def generate_proposals(net_df: pd.DataFrame) -> list:
    proposals = []
    for _, row in net_df.iterrows():
        mat_id = row["material_id"]
        if mat_id not in MATERIALS or row["net_req_total"] <= 0:
            proposals.append({"material_id": mat_id, "action": "NO_ACTION",
                               "order_qty": 0, "due_date": None,
                               "priority": "🟢 COVERED", "price_total": 0})
            continue
        mat         = MATERIALS[mat_id]
        order_qty   = round(apply_lot_size(row["net_req_total"], mat_id))
        effective_lt = mat["lead_time_days"] + mat["po_proc_days"] + mat["gr_proc_days"]
        due_date    = PLANNING_DATE + timedelta(days=effective_lt)
        net_total   = row["net_req_total"]
        stock_cover_lt = mat["stock"] / max(1, net_total / WORKING_DAYS_MONTH)

        if stock_cover_lt < effective_lt * 0.5:   priority = "🔴 CRITICAL"
        elif stock_cover_lt < effective_lt:        priority = "🟠 HIGH"
        elif net_total > 0:                        priority = "🟡 MEDIUM"
        else:                                      priority = "🟢 COVERED"

        action = ("PLACE_IMMEDIATELY" if priority == "🔴 CRITICAL" else
                  "PLACE_URGENT"     if priority == "🟠 HIGH"     else "PLAN_STANDARD")
        proposals.append({
            "material_id":   mat_id,
            "description":   mat["description"],
            "action":        action,
            "order_qty":     int(order_qty),
            "lot_size_proc": mat["lot_size"],
            "due_date":      due_date.isoformat(),
            "effective_lt":  effective_lt,
            "net_req_total": round(row["net_req_total"], 1),
            "priority":      priority,
            "price_total":   round(order_qty * mat["price"]),
            "supplier":      mat["supplier"],
        })
    proposals.sort(key=lambda x: PRIORITY_RANK.get(x["priority"], 99))
    return proposals


def run_reorder_proposals(net_df=None) -> list:
    print("=" * 65)
    print("  MODULE 6: REORDER PROPOSALS")
    print("=" * 65)
    if net_df is None:
        net_df = run_bom_exploder()
    proposals = generate_proposals(net_df)
    total_val  = sum(p["price_total"] for p in proposals)
    print(f"\n  {'Material':<14} {'Action':<22} {'Qty':>6} {'Due Date':<12} {'₹ Value':>12}  Priority")
    print(f"  {'─'*75}")
    for p in proposals:
        if p["action"] == "NO_ACTION": continue
        print(f"  {p['material_id']:<14} {p['action']:<22} {p['order_qty']:>6} "
              f"{str(p['due_date']):<12} {p['price_total']:>12,.0f}  {p['priority']}")
    print(f"\n  Total procurement value: ₹{total_val:,.0f}")
    return proposals


# ═══════════════════════════════════════════════════════════════════════════════
#  MODULE 7 — WHAT-IF SIMULATOR
# ═══════════════════════════════════════════════════════════════════════════════

DEFAULT_SCENARIOS = [
    {"name": "Baseline (no change)",
     "description": "Current state — no disruptions",
     "demand_delta_pct": 0, "lt_delay_days": {}, "stock_reduction": {}},
    {"name": "Vendor B port delay +7d",
     "description": "RM-200010 delayed 7 days (Excel: High transit scenario)",
     "demand_delta_pct": 0, "lt_delay_days": {"RM-200010": 7}, "stock_reduction": {}},
    {"name": "Demand spike +10%",
     "description": "FG demand increases 10% across all months",
     "demand_delta_pct": 10, "lt_delay_days": {}, "stock_reduction": {}},
    {"name": "Compound: delay +7d AND demand +10%",
     "description": "Port delay AND demand increase simultaneously",
     "demand_delta_pct": 10, "lt_delay_days": {"RM-200010": 7}, "stock_reduction": {}},
    {"name": "Demand spike +20%",
     "description": "Aggressive demand growth scenario",
     "demand_delta_pct": 20, "lt_delay_days": {}, "stock_reduction": {}},
    {"name": "QI hold: 30% of RM-200010 stock blocked",
     "description": "Quality inspection blocks 30% of Copper Kit stock",
     "demand_delta_pct": 0, "lt_delay_days": {}, "stock_reduction": {"RM-200010": 0.30}},
]


def run_scenario(scenario, fg_demand_base):
    delta    = scenario.get("demand_delta_pct", 0) / 100
    fg_dem   = [d * (1 + delta) for d in fg_demand_base]
    mats     = deepcopy(MATERIALS)
    for mid, pct in scenario.get("stock_reduction", {}).items():
        if mid in mats:
            mats[mid]["stock"] = max(0, mats[mid]["stock"] - mats[mid]["stock"] * pct)
    lt_delays = scenario.get("lt_delay_days", {})
    net_df    = compute_net_requirements("FG-100001", fg_dem)

    critical = []
    for _, row in net_df.iterrows():
        mid = row["material_id"]
        if mid not in MATERIALS or row["net_req_total"] <= 0: continue
        mat      = MATERIALS[mid]
        extra_lt = lt_delays.get(mid, 0)
        eff_lt   = mat["lead_time_days"] + mat["po_proc_days"] + mat["gr_proc_days"] + extra_lt
        net_req  = row["net_req"]
        first_gap_idx = next((i for i, r in enumerate(net_req) if r > 0), None)
        first_gap_qty = net_req[first_gap_idx] if first_gap_idx is not None else 0
        stockout_date = (PLANNING_DATE + timedelta(days=first_gap_idx * WORKING_DAYS_MONTH)
                         if first_gap_idx is not None else None)
        po_arrival    = PLANNING_DATE + timedelta(days=int(eff_lt))
        can_cover     = stockout_date is None or po_arrival <= stockout_date
        min_exp       = max(first_gap_qty, float(mat.get("moq", 0))) if first_gap_qty > 0 else 0
        gap_days      = max(0, eff_lt - (first_gap_idx * WORKING_DAYS_MONTH)) if first_gap_idx is not None else 0
        critical.append({
            "material_id": mid, "description": mat.get("description", mid),
            "net_req_total": round(row["net_req_total"], 1),
            "first_gap_month": MONTHS[first_gap_idx] if first_gap_idx is not None else "None",
            "first_gap_qty": round(first_gap_qty, 1),
            "stockout_date": stockout_date.isoformat() if stockout_date else None,
            "effective_lt_days": eff_lt, "po_arrives": po_arrival.isoformat(),
            "can_cover_in_time": can_cover, "min_expedite_qty": round(min_exp, 0),
            "stockout_cost_est": round(gap_days * STOCKOUT_COST_PER_DAY),
            "expedite_cost_est": round(min_exp * mat.get("price", 0) * 0.10),
        })

    has_critical = any(not m["can_cover_in_time"] for m in critical)
    severity     = ("🔴 SEVERE — stockout likely without immediate action" if has_critical else
                    "🟠 WARNING — gaps exist but can be covered if ordered now" if critical else
                    "🟢 SAFE — current stock + POs cover all demand")
    total_s_cost = sum(m["stockout_cost_est"] for m in critical)
    total_e_cost = sum(m["expedite_cost_est"]  for m in critical)
    return {
        "scenario_name": scenario["name"], "description": scenario["description"],
        "demand_delta_pct": scenario.get("demand_delta_pct", 0), "lt_delays": lt_delays,
        "severity": severity, "affected_materials": critical,
        "total_stockout_cost": total_s_cost, "total_expedite_cost": total_e_cost,
        "better_to_expedite": total_e_cost < total_s_cost,
    }


def run_whatif_simulator() -> list:
    print("=" * 65)
    print("  MODULE 7: WHAT-IF SCENARIO SIMULATOR")
    print("=" * 65)
    fg_base  = CONSUMPTION_HISTORY["FG-100001"]
    results  = []
    for i, sc in enumerate(DEFAULT_SCENARIOS, 1):
        print(f"  [{i}] {sc['name']}...", end=" ", flush=True)
        r = run_scenario(sc, fg_base)
        results.append(r)
        print(r["severity"].split("—")[0].strip())
    print(f"\n  {'Scenario':<42} {'Severity':<10} {'Materials':>10} {'Cost Impact':>15}")
    print(f"  {'─'*80}")
    for r in results:
        n = sum(1 for m in r["affected_materials"] if not m["can_cover_in_time"])
        sev = r["severity"].split("—")[0].strip()
        cost = f"₹{r['total_stockout_cost']:,.0f}" if r["total_stockout_cost"] else "₹0"
        print(f"  {r['scenario_name']:<42} {sev:<10} {n:>10} {cost:>15}")
    return results


# ═══════════════════════════════════════════════════════════════════════════════
#  MODULE 8 — SAFETY STOCK OPTIMIZER (GBM)
# ═══════════════════════════════════════════════════════════════════════════════

SERVICE_LEVEL = 0.95
SS_FEATURES   = [
    "actual_lead_time_days", "lt_deviation_days", "transit_risk_level",
    "demand_rolling_slope", "daily_demand", "po_slippage_days",
    "qi_rejection", "month_sin", "month_cos", "quarter",
]


def compute_optimal_ss(df: pd.DataFrame, window=30) -> np.ndarray:
    demand = df["daily_demand"].values
    lt     = df["actual_lead_time_days"].values
    z      = norm.ppf(SERVICE_LEVEL)
    n      = len(df)
    ss     = np.zeros(n)
    for i in range(window, n):
        d_sl = demand[i - window:i]
        l_sl = lt[i - window:i]
        d_m, d_s = np.mean(d_sl), np.std(d_sl, ddof=1)
        l_m, l_s = np.mean(l_sl), np.std(l_sl, ddof=1)
        ss[i] = z * np.sqrt(l_m * d_s**2 + d_m**2 * l_s**2)
    return np.maximum(0, ss)


def _compute_ss_scores(gbm, X_test, y_test) -> dict:
    """
    Full scoring suite for GBM Safety Stock Optimizer.
    Metrics: MAE, RMSE, MAPE, R², Huber loss, bias, explained variance,
             median absolute error, feature importance top-5.
    """
    from sklearn.metrics import (
        mean_absolute_error, mean_squared_error, r2_score,
        explained_variance_score, median_absolute_error
    )

    y_pred = np.maximum(0, gbm.predict(X_test))

    mae     = float(mean_absolute_error(y_test, y_pred))
    rmse    = float(np.sqrt(mean_squared_error(y_test, y_pred)))
    r2      = float(r2_score(y_test, y_pred))
    ev      = float(explained_variance_score(y_test, y_pred))
    med_ae  = float(median_absolute_error(y_test, y_pred))
    bias    = float(np.mean(y_pred - y_test))

    # MAPE
    mask = y_test > 0.01
    mape = float(np.mean(np.abs((y_test[mask] - y_pred[mask]) / y_test[mask])) * 100) if mask.sum() else 0.0

    # Huber loss
    delta   = 1.0
    abs_err = np.abs(y_test - y_pred)
    huber   = float(np.mean(np.where(abs_err <= delta, 0.5 * abs_err**2,
                                     delta * (abs_err - 0.5 * delta))))

    # Prediction IQR (uncertainty width)
    iqr = float(np.percentile(y_pred, 75) - np.percentile(y_pred, 25))

    # Feature importance top 5
    fi   = dict(zip(SS_FEATURES, gbm.feature_importances_))
    top5 = sorted(fi.items(), key=lambda x: -x[1])[:5]

    mean_p = float(y_pred.mean()) or 1.0
    conf   = round(max(40, min(99, 100 - (mae / mean_p * 100))), 1)

    return {
        "mae_units":           round(mae, 4),
        "rmse_units":          round(rmse, 4),
        "mape_pct":            round(mape, 2),
        "r2_score":            round(r2, 4),
        "explained_variance":  round(ev, 4),
        "median_abs_error":    round(med_ae, 4),
        "huber_loss":          round(huber, 4),
        "bias_units":          round(bias, 4),
        "prediction_iqr":      round(iqr, 2),
        "top5_features":       [(k, round(v, 4)) for k, v in top5],
        "confidence_pct":      conf,
        "n_test_samples":      int(len(y_test)),
        "grade": ("A" if r2 >= 0.80 else "B" if r2 >= 0.60 else "C" if r2 >= 0.40 else "D"),
        "note": ("✅ Good" if conf >= 80 else
                 "⚠️ Moderate — directional guidance" if conf >= 60 else "🔴 Low"),
    }


# alias
_compute_ss_confidence = _compute_ss_scores


def train_ss_optimizer(df: pd.DataFrame, mat_id: str) -> dict:
    print(f"    GBM SS optimizer {mat_id}...", end=" ", flush=True)
    mat_df   = df[df["material_id"] == mat_id].sort_values("date").reset_index(drop=True)
    optimal  = compute_optimal_ss(mat_df)
    X        = mat_df[SS_FEATURES].values
    y        = optimal
    splits   = list(TimeSeriesSplit(n_splits=5).split(X))
    tr_idx, vl_idx = splits[-1]
    gbm = GradientBoostingRegressor(n_estimators=400, max_depth=4, learning_rate=0.05,
                                    subsample=0.8, loss="huber", random_state=42)
    gbm.fit(X[tr_idx], y[tr_idx])
    scores = _compute_ss_scores(gbm, X[vl_idx], y[vl_idx])

    # ── Training deviance curve (staged predictions, sampled every 40 estimators) ──
    train_dev, val_dev = [], []
    for i, (ytr_hat, vl_hat) in enumerate(
            zip(gbm.staged_predict(X[tr_idx]), gbm.staged_predict(X[vl_idx]))):
        if i % 40 == 0:
            train_dev.append(round(float(mean_absolute_error(y[tr_idx], ytr_hat)), 4))
            val_dev.append(round(float(mean_absolute_error(y[vl_idx], vl_hat)), 4))
    scores["train_mae_curve"] = train_dev
    scores["val_mae_curve"]   = val_dev
    scores["best_iter_by_val_mae"] = int(np.argmin(val_dev) * 40) if val_dev else 0

    # ── 5-fold cross-val R² and MAE ──
    cv_r2, cv_mae_cv = [], []
    for tr_i, vl_i in TimeSeriesSplit(n_splits=5).split(X):
        g = GradientBoostingRegressor(n_estimators=200, max_depth=4, learning_rate=0.05,
                                      subsample=0.8, loss="huber", random_state=42)
        g.fit(X[tr_i], y[tr_i])
        yh = np.maximum(0, g.predict(X[vl_i]))
        cv_r2.append(round(float(r2_score(y[vl_i], yh)), 4))
        cv_mae_cv.append(round(float(mean_absolute_error(y[vl_i], yh)), 4))
    scores["cv_r2_per_fold"]  = cv_r2
    scores["cv_r2_mean"]      = round(float(np.mean(cv_r2)), 4)
    scores["cv_r2_std"]       = round(float(np.std(cv_r2)), 4)
    scores["cv_mae_per_fold"] = cv_mae_cv
    scores["cv_mae_mean"]     = round(float(np.mean(cv_mae_cv)), 4)

    safe = mat_id.replace("-", "_")
    joblib.dump(gbm, MODELS_DIR / f"gbm_ss_{safe}.pkl")
    print(f"\n    ── {mat_id} Safety Stock GBM Scores ──")
    print(f"    MAE={scores['mae_units']} units  RMSE={scores['rmse_units']}  MAPE={scores['mape_pct']}%")
    print(f"    R²={scores['r2_score']}  ExplVar={scores['explained_variance']}  "
          f"MedianAE={scores['median_abs_error']}")
    print(f"    HuberLoss={scores['huber_loss']}  Bias={scores['bias_units']}  IQR=±{scores['prediction_iqr']}")
    print(f"    CV R²  (5-fold): {cv_r2}  → mean={scores['cv_r2_mean']} ±{scores['cv_r2_std']}")
    print(f"    CV MAE (5-fold): {cv_mae_cv}  → mean={scores['cv_mae_mean']}")
    print(f"    Best iter by val MAE: {scores['best_iter_by_val_mae']}")
    print(f"    Grade={scores['grade']}  Confidence={scores['confidence_pct']}%  {scores['note']}")
    print(f"    Top features: {', '.join(f'{k}({v})' for k,v in scores['top5_features'][:3])}")
    return {"model": gbm, "scores": scores}


def predict_optimal_ss(mat_id: str, mat_df: pd.DataFrame, gbm=None) -> dict:
    if gbm is None:
        safe = mat_id.replace("-", "_")
        gbm  = joblib.load(MODELS_DIR / f"gbm_ss_{safe}.pkl")
    latest    = mat_df.sort_values("date").iloc[-1]
    X         = np.array([[latest[f] for f in SS_FEATURES]])
    ml_ss     = max(0, float(gbm.predict(X)[0]))
    sap_ss    = MATERIALS.get(mat_id, {}).get("safety_stock", 0)
    delta     = ml_ss - sap_ss
    delta_pct = (delta / sap_ss * 100) if sap_ss else 0
    rec       = ("⬆️ INCREASE — under-protected"  if delta > sap_ss * 0.2 else
                 "⬇️ DECREASE — over-stocked"     if delta < -sap_ss * 0.2 else
                 "✅ OK — within acceptable range")
    return {
        "material_id":    mat_id,
        "current_ss_sap": sap_ss,
        "predicted_ss_ml": round(ml_ss, 1),
        "delta":          round(delta, 1),
        "delta_pct":      round(delta_pct, 1),
        "recommendation": rec,
    }


def run_safety_stock_optimizer(data_path="data/training_data.csv", retrain=True) -> dict:
    print("=" * 65)
    print("  MODULE 8: SAFETY STOCK OPTIMIZER (GBM)")
    print("=" * 65)
    df = pd.read_csv(data_path, parse_dates=["date"])
    results = {}
    for mat_id in sorted(df["material_id"].unique()):
        mat_df = df[df["material_id"] == mat_id].sort_values("date").reset_index(drop=True)
        safe   = mat_id.replace("-", "_")
        if retrain:
            r   = train_ss_optimizer(df, mat_id)
            gbm = r["model"]
            scores = r["scores"]
        else:
            gbm  = joblib.load(MODELS_DIR / f"gbm_ss_{safe}.pkl")
            scores = {"confidence_pct": 75.0, "grade": "N/A"}
        pred = predict_optimal_ss(mat_id, mat_df, gbm)
        pred["scores"] = scores
        results[mat_id] = pred
        print(f"  {mat_id}: SAP={pred['current_ss_sap']} → ML={pred['predicted_ss_ml']}  "
              f"Δ{pred['delta']:+.0f}  R²={scores.get('r2_score','N/A')}  "
              f"confidence={scores.get('confidence_pct','N/A')}%  {pred['recommendation']}")
    return results
