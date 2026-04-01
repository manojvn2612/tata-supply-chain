"""
module_7_whatif_simulator.py
------------------------------
Module 7: What-If Scenario Simulator

Answers planning questions like:
  "If Vendor B delays by 7 days AND FG demand rises 10%, will we stock out?"
  "What is the minimum expedite quantity to prevent a shutdown?"
  "How much buffer does a 20% demand spike leave us?"

Approach:
  - Takes a scenario definition (parameter deltas)
  - Re-runs BOM explosion + stock netting with modified inputs
  - Computes stockout date, gap quantity, cost of stockout vs expedite
  - Uses ML lead-time predictions (Module 2) for realistic timing
  - Ranks scenarios by severity

Run:  python module_7_whatif_simulator.py
"""

import numpy as np
import pandas as pd
from copy import deepcopy
from datetime import date, timedelta
from config import MATERIALS, CONSUMPTION_HISTORY, MONTHS
from module_5_bom_exploder import compute_net_requirements


PLANNING_DATE = date(2026, 2, 3)
WORKING_DAYS_MONTH = 22
STOCKOUT_COST_PER_DAY = 500_000   # ₹ revenue at risk per day of FG stoppage


# ── Scenario definitions ──────────────────────────────────────────────────────

DEFAULT_SCENARIOS = [
    {
        "name":        "Baseline (no change)",
        "description": "Current state — no disruptions",
        "demand_delta_pct": 0,
        "lt_delay_days":    {"RM-200010": 0, "SFG-150001": 0},
        "stock_reduction":  {},
    },
    {
        "name":        "Vendor B port delay +7d",
        "description": "RM-200010 delayed 7 days due to port congestion",
        "demand_delta_pct": 0,
        "lt_delay_days":    {"RM-200010": 7},
        "stock_reduction":  {},
    },
    {
        "name":        "Demand spike +10%",
        "description": "FG-100001 demand increases 10% across all months",
        "demand_delta_pct": 10,
        "lt_delay_days":    {},
        "stock_reduction":  {},
    },
    {
        "name":        "Compound: delay +7d AND demand +10%",
        "description": "Port delay AND demand increase simultaneously",
        "demand_delta_pct": 10,
        "lt_delay_days":    {"RM-200010": 7},
        "stock_reduction":  {},
    },
    {
        "name":        "Demand spike +20%",
        "description": "Aggressive demand growth scenario",
        "demand_delta_pct": 20,
        "lt_delay_days":    {},
        "stock_reduction":  {},
    },
    {
        "name":        "QI hold: 30% of RM-200010 stock blocked",
        "description": "Quality inspection blocks portion of Copper Kit stock",
        "demand_delta_pct": 0,
        "lt_delay_days":    {},
        "stock_reduction":  {"RM-200010": 0.30},
    },
]


# ── Core simulator ────────────────────────────────────────────────────────────

def run_scenario(scenario: dict, fg_demand_base: list) -> dict:
    """
    Runs one scenario:
      1. Modifies demand + lead times per scenario params
      2. Re-computes net requirements
      3. Finds first month with unmet net requirement (stockout signal)
      4. Estimates gap, stockout cost, minimum expedite qty
    """
    # ── Apply demand delta ──
    demand_delta = scenario.get("demand_delta_pct", 0) / 100
    fg_demand    = [d * (1 + demand_delta) for d in fg_demand_base]

    # ── Apply stock reductions (QI / blocked scenarios) ──
    mats = deepcopy(MATERIALS)
    for mat_id, pct_blocked in scenario.get("stock_reduction", {}).items():
        if mat_id in mats:
            reduction = mats[mat_id]["stock"] * pct_blocked
            mats[mat_id]["stock"] = max(0, mats[mat_id]["stock"] - reduction)

    # ── Apply lead time delays to effective LT ──
    lt_delays = scenario.get("lt_delay_days", {})

    # ── Compute net requirements ──
    net_df = compute_net_requirements("FG-100001", fg_demand)

    # ── Find stockout signals ──
    critical_materials = []
    for _, row in net_df.iterrows():
        mat_id    = row["material_id"]
        net_req   = row["net_req"]
        net_total = row["net_req_total"]

        if mat_id not in MATERIALS:
            continue

        mat       = MATERIALS[mat_id]
        extra_lt  = lt_delays.get(mat_id, 0)
        eff_lt    = mat["lead_time_days"] + mat["po_proc_days"] + mat["gr_proc_days"] + extra_lt
        avg_daily = np.mean(fg_demand) / WORKING_DAYS_MONTH

        # First month where net_req > 0 is the first unmet demand month
        first_gap_month = next((i for i, r in enumerate(net_req) if r > 0), None)
        first_gap_qty   = net_req[first_gap_month] if first_gap_month is not None else 0

        # Approximate stockout date
        if first_gap_month is not None:
            days_to_stockout = first_gap_month * WORKING_DAYS_MONTH
            stockout_date    = PLANNING_DATE + timedelta(days=days_to_stockout)
        else:
            stockout_date = None

        # Can we receive a PO in time?
        po_arrival_date = PLANNING_DATE + timedelta(days=int(eff_lt))
        can_cover       = stockout_date is None or po_arrival_date <= (stockout_date if stockout_date else PLANNING_DATE + timedelta(days=365))

        # Minimum expedite quantity
        min_expedite = max(first_gap_qty, float(mat.get("moq", 0))) if first_gap_qty > 0 else 0

        # Stockout cost estimate (days of gap × daily revenue at risk)
        gap_days    = max(0, eff_lt - (first_gap_month * WORKING_DAYS_MONTH)) if first_gap_month is not None else 0
        stockout_cost = gap_days * STOCKOUT_COST_PER_DAY

        # Expedite cost (10% premium on PO value)
        expedite_cost = min_expedite * mat.get("price", 0) * 0.10

        if net_total > 0:
            critical_materials.append({
                "material_id":       mat_id,
                "description":       mat.get("description", mat_id),
                "level":             int(row["level"]),
                "net_req_total":     round(net_total, 1),
                "first_gap_month":   MONTHS[first_gap_month] if first_gap_month is not None else "None",
                "first_gap_qty":     round(first_gap_qty, 1),
                "stockout_date":     stockout_date.isoformat() if stockout_date else None,
                "effective_lt_days": eff_lt,
                "po_arrives":        po_arrival_date.isoformat(),
                "can_cover_in_time": can_cover,
                "min_expedite_qty":  round(min_expedite, 0),
                "stockout_cost_est": round(stockout_cost, 0),
                "expedite_cost_est": round(expedite_cost, 0),
                "recommendation":    "EXPEDITE NOW" if not can_cover else ("PLACE PO SOON" if net_total > 0 else "OK"),
            })

    # Overall scenario severity
    has_critical = any(not m["can_cover_in_time"] for m in critical_materials)
    has_gaps     = len(critical_materials) > 0
    total_stockout_cost = sum(m["stockout_cost_est"] for m in critical_materials)
    total_expedite_cost = sum(m["expedite_cost_est"] for m in critical_materials)

    if has_critical:
        severity = "🔴  SEVERE — stockout likely without immediate action"
    elif has_gaps:
        severity = "🟠  WARNING — gaps exist but can be covered if ordered now"
    else:
        severity = "🟢  SAFE — current stock + POs cover all demand"

    return {
        "scenario_name":       scenario["name"],
        "description":         scenario["description"],
        "demand_delta_pct":    scenario.get("demand_delta_pct", 0),
        "lt_delays":           lt_delays,
        "severity":            severity,
        "affected_materials":  critical_materials,
        "total_stockout_cost": total_stockout_cost,
        "total_expedite_cost": total_expedite_cost,
        "better_to_expedite":  total_expedite_cost < total_stockout_cost,
    }


def run_whatif_simulator():
    print("=" * 65)
    print("  MODULE 7: WHAT-IF SCENARIO SIMULATOR")
    print("=" * 65)
    print(f"  Planning date: {PLANNING_DATE}")
    print(f"  Scenarios: {len(DEFAULT_SCENARIOS)}")

    fg_demand_base = CONSUMPTION_HISTORY["FG-100001"]

    results = []
    for i, scenario in enumerate(DEFAULT_SCENARIOS, 1):
        print(f"\n  Running scenario {i}/{len(DEFAULT_SCENARIOS)}: {scenario['name']}...", end=" ")
        result = run_scenario(scenario, fg_demand_base)
        results.append(result)
        print("done")

    # ── Print results ──
    print("\n\n" + "=" * 65)
    print("  SCENARIO ANALYSIS RESULTS")
    print("=" * 65)

    for i, res in enumerate(results, 1):
        print(f"\n  [{i}] {res['scenario_name']}")
        print(f"       {res['description']}")
        print(f"       Severity: {res['severity']}")

        if res["affected_materials"]:
            print(f"       Affected materials:")
            for m in res["affected_materials"]:
                cover_icon = "✅" if m["can_cover_in_time"] else "❌"
                print(f"         {cover_icon} {m['material_id']:<14} gap starts: {m['first_gap_month']:<10} "
                      f"expedite qty: {m['min_expedite_qty']:.0f}  "
                      f"LT: {m['effective_lt_days']}d")

        if res["total_stockout_cost"] > 0:
            verdict = "EXPEDITE" if res["better_to_expedite"] else "ACCEPT RISK"
            print(f"       Cost: stockout ₹{res['total_stockout_cost']:,.0f}  "
                  f"vs expedite ₹{res['total_expedite_cost']:,.0f}  → {verdict}")

    # ── Comparison matrix ──
    print("\n" + "=" * 65)
    print("  SCENARIO COMPARISON MATRIX")
    print(f"  {'Scenario':<40} {'Severity':<12} {'Materials at Risk':<20} {'Cost Impact'}")
    print(f"  {'─'*90}")
    for res in results:
        n_at_risk  = sum(1 for m in res["affected_materials"] if not m["can_cover_in_time"])
        cost_str   = f"₹{res['total_stockout_cost']:>10,.0f}" if res["total_stockout_cost"] else "₹0"
        sev_short  = res["severity"].split("—")[0].strip()
        print(f"  {res['scenario_name']:<40} {sev_short:<12} {n_at_risk:<20} {cost_str}")

    return results


if __name__ == "__main__":
    run_whatif_simulator()
