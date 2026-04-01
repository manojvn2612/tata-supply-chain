"""
module_5_bom_exploder.py
--------------------------
Module 5: Multi-Level BOM Exploder + Stock Netting

Takes FG demand forecast → explodes through multi-level BOM →
nets off available stock and open POs → computes net requirements
per component at every level.

This is the core MRP logic that SAP already does, reimplemented here
so the AI pipeline can:
  1. Run its own net requirements before SAP MRP runs
  2. Feed net requirements into the reorder proposal module
  3. Show planners the full demand cascade in one view

Explosion logic:
  Level 0 (FG):   demand as-is
  Level 1 (SFG):  FG demand × BOM qty
  Level 2 (RM):   FG demand × FG→RM qty  +  SFG demand × SFG→RM qty

Stock netting:
  Available = Unrestricted stock  (QI and Blocked excluded)
  Net requirement = Gross requirement - Available stock - Open PO qty (confirmed)
  Net requirement clipped at 0 (can't be negative)

Run:  python module_5_bom_exploder.py
"""

import numpy as np
import pandas as pd
from config import MATERIALS, BOM, CONSUMPTION_HISTORY, MONTHS


# ── BOM utilities ─────────────────────────────────────────────────────────────

def flatten_bom(fg_id: str, qty: float = 1.0, level: int = 0,
                _visited: set = None) -> list[dict]:
    """
    Recursively flatten BOM into a list of
    {component, total_qty_per_fg, level, parent}.
    Handles multi-level (n-deep) BOMs.
    Guards against circular references with _visited.
    """
    if _visited is None:
        _visited = set()
    if fg_id in _visited:
        return []
    _visited.add(fg_id)

    rows = []
    for component, comp_qty in BOM.get(fg_id, {}).items():
        total_qty = qty * comp_qty
        rows.append({
            "parent":             fg_id,
            "component":          component,
            "qty_per_parent":     comp_qty,
            "total_qty_per_fg":   total_qty,
            "level":              level + 1,
        })
        # Recurse into sub-assemblies
        rows.extend(flatten_bom(component, total_qty, level + 1, _visited))

    return rows


def get_flat_bom_df(fg_id: str) -> pd.DataFrame:
    rows = flatten_bom(fg_id)
    if not rows:
        return pd.DataFrame()
    df = pd.DataFrame(rows)
    # If a component appears via multiple paths, sum total qty
    df = (df.groupby(["component", "level"])
            .agg(total_qty_per_fg=("total_qty_per_fg", "sum"))
            .reset_index()
            .sort_values("level"))
    return df


# ── Open PO netting ───────────────────────────────────────────────────────────

def get_confirmed_po_qty(mat_id: str) -> float:
    """Sum of all open PO quantities for a material (confirmed receipts)."""
    mat = MATERIALS.get(mat_id, {})
    return sum(po["qty"] for po in mat.get("open_pos", []))


# ── Gross → Net requirements ──────────────────────────────────────────────────

def compute_net_requirements(fg_id: str, fg_demand_monthly: list) -> pd.DataFrame:
    """
    Given monthly FG demand, explode through BOM and compute gross + net
    requirements for every component over the planning horizon.

    Returns a DataFrame with columns:
      material_id, description, level, total_qty_per_fg,
      gross_req_monthly, available_stock, open_po_qty,
      net_req_monthly, net_req_total, stockout_risk
    """
    flat_bom = get_flat_bom_df(fg_id)

    # Always include the FG itself as level 0
    fg_row = pd.DataFrame([{
        "component": fg_id, "level": 0, "total_qty_per_fg": 1.0
    }])
    flat_bom = pd.concat([fg_row, flat_bom], ignore_index=True)

    results = []
    fg_demand = np.array(fg_demand_monthly, dtype=float)

    for _, row in flat_bom.iterrows():
        mat_id    = row["component"]
        qty_per_fg = row["total_qty_per_fg"]
        level      = row["level"]

        mat = MATERIALS.get(mat_id, {})

        # Gross requirement = FG demand × qty per FG unit
        gross_req = fg_demand * qty_per_fg

        # Stock netting
        avail_stock   = float(mat.get("stock", 0))
        open_po_qty   = get_confirmed_po_qty(mat_id)
        safety_stock  = float(mat.get("safety_stock", 0))

        # Net = gross - available above safety stock - open POs
        # Available to promise = stock - safety_stock + open_po_qty
        atp = max(0, avail_stock - safety_stock) + open_po_qty

        net_req_monthly = gross_req.copy()
        remaining_atp   = atp
        for i in range(len(net_req_monthly)):
            offset = min(remaining_atp, net_req_monthly[i])
            net_req_monthly[i] = max(0, net_req_monthly[i] - offset)
            remaining_atp = max(0, remaining_atp - offset)

        net_total = net_req_monthly.sum()

        # Stockout risk: does first-month net req exceed usable stock + PO?
        first_month_gap = gross_req[0] - atp
        if first_month_gap > 0 and avail_stock < safety_stock:
            stockout_risk = "🔴  CRITICAL"
        elif net_total > 0 and first_month_gap > 0:
            stockout_risk = "🟠  HIGH"
        elif net_total > 0:
            stockout_risk = "🟡  MEDIUM"
        else:
            stockout_risk = "🟢  COVERED"

        results.append({
            "material_id":        mat_id,
            "description":        mat.get("description", mat_id),
            "level":              int(level),
            "qty_per_fg":         round(qty_per_fg, 3),
            "gross_req":          [round(float(x), 1) for x in gross_req],
            "gross_req_total":    round(gross_req.sum(), 1),
            "available_stock":    avail_stock,
            "safety_stock":       safety_stock,
            "open_po_qty":        open_po_qty,
            "atp":                round(atp, 1),
            "net_req":            [round(float(x), 1) for x in net_req_monthly],
            "net_req_total":      round(net_total, 1),
            "stockout_risk":      stockout_risk,
        })

    return pd.DataFrame(results)


def run_bom_exploder(fg_demand_override: list = None):
    print("=" * 65)
    print("  MODULE 5: MULTI-LEVEL BOM EXPLODER + STOCK NETTING")
    print("=" * 65)

    fg_id = "FG-100001"
    fg_demand = fg_demand_override or CONSUMPTION_HISTORY[fg_id]

    print(f"\n  FG: {fg_id}  |  Months: {MONTHS}")
    print(f"  Demand: {fg_demand}")

    # ── Print flat BOM ──
    flat = get_flat_bom_df(fg_id)
    print(f"\n  Flat BOM ({fg_id}):")
    print(f"  {'Level':<8} {'Component':<15} {'Qty/FG':>8}")
    print(f"  {'─'*35}")
    for _, r in flat.iterrows():
        indent = "  " * int(r["level"])
        print(f"  L{int(r['level'])}      {indent}{r['component']:<15} {r['total_qty_per_fg']:>8.3f}")

    # ── Compute net requirements ──
    net_df = compute_net_requirements(fg_id, fg_demand)

    print(f"\n  NET REQUIREMENTS BY LEVEL")
    print(f"  {'─'*65}")

    for _, row in net_df.iterrows():
        indent = "  " * row["level"]
        print(f"\n  L{row['level']}  {indent}{row['material_id']}  —  {row['description']}")
        print(f"      Qty/FG:        {row['qty_per_fg']}")
        print(f"      Gross req:     {row['gross_req']}  (total: {row['gross_req_total']})")
        print(f"      Stock:         {row['available_stock']}  |  Safety stock: {row['safety_stock']}  "
              f"|  Open POs: {row['open_po_qty']}  |  ATP: {row['atp']}")
        print(f"      Net req:       {row['net_req']}  (total: {row['net_req_total']})")
        print(f"      Stockout risk: {row['stockout_risk']}")

    return net_df


if __name__ == "__main__":
    run_bom_exploder()
