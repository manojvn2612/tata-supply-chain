"""
module_6_reorder_proposals.py
-------------------------------
Module 6: Reorder Proposal Engine

Takes net requirements from Module 5 (BOM exploder) and generates
concrete purchase order proposals with:
  - SAP lot-size procedure logic (EX / FX / HB)
  - MOQ enforcement
  - ML-predicted lead time (from Module 2 XGBoost) for due-date calculation
  - Priority ranking by stockout risk + urgency
  - SAP BAPI write-back payload (for planner approval)

Lot-size procedures:
  EX (Lot-for-lot):  order exactly net requirement
  FX (Fixed):        order in fixed multiples of lot_size_value
  HB (Max stock):    order up to max stock level

Run:  python module_6_reorder_proposals.py
"""

import numpy as np
import pandas as pd
from datetime import date, timedelta
from config import MATERIALS, BOM, CONSUMPTION_HISTORY, MONTHS
from module_5_bom_exploder import run_bom_exploder, get_confirmed_po_qty

PLANNING_DATE = date(2026, 2, 3)   # "today" from dataset context
PRIORITY_RANK = {"🔴  CRITICAL": 0, "🟠  HIGH": 1, "🟡  MEDIUM": 2, "🟢  COVERED": 3}


# ── Lot-size logic ────────────────────────────────────────────────────────────

def apply_lot_size(net_req: float, mat_id: str) -> float:
    """
    Apply SAP lot-size procedure to compute the actual order quantity.
    Never orders less than MOQ.
    """
    if net_req <= 0:
        return 0

    mat       = MATERIALS[mat_id]
    procedure = mat["lot_size"]
    moq       = mat["moq"]
    ls_value  = mat["lot_size_value"]
    max_stock = mat["max_stock"]
    cur_stock = mat["stock"]

    if procedure == "EX":
        qty = net_req

    elif procedure == "FX":
        # Round up to nearest multiple of lot_size_value
        qty = np.ceil(net_req / ls_value) * ls_value

    elif procedure == "HB":
        # Order up to max stock level
        qty = max(net_req, max_stock - cur_stock)

    elif procedure == "MB":
        # Full monthly requirement
        monthly_avg = np.mean(CONSUMPTION_HISTORY.get(mat_id, [net_req]))
        qty = max(net_req, monthly_avg)

    else:
        qty = net_req

    # Always respect MOQ
    qty = max(qty, moq)
    return float(np.ceil(qty))


def compute_order_due_date(mat_id: str, use_p90: bool = True) -> date:
    """
    Compute the latest date a PO must be placed to receive material
    before stockout, using P90 lead time from Module 2 if available,
    otherwise SAP nominal.
    """
    mat = MATERIALS[mat_id]
    # Try to load XGBoost model, fall back to SAP nominal
    try:
        import joblib
        from pathlib import Path
        from module_2_xgboost_leadtime import LT_FEATURES
        safe_id   = mat_id.replace("-", "_")
        p90_model = joblib.load(Path("models") / f"xgb_lt_p90_{safe_id}.pkl")

        # Build feature row from latest known values
        latest = pd.read_csv("data/training_data.csv", parse_dates=["date"])
        latest = latest[latest["material_id"] == mat_id].sort_values("date").iloc[-1]
        row    = np.array([[latest[f] for f in LT_FEATURES]])
        lt     = float(p90_model.predict(row)[0])
    except Exception:
        lt = mat["lead_time_days"] + mat["po_proc_days"] + mat["gr_proc_days"]

    # Due date = today + effective lead time
    due_date = PLANNING_DATE + timedelta(days=int(np.ceil(lt)))
    return due_date, round(lt, 1)


# ── Proposal generator ────────────────────────────────────────────────────────

def generate_proposals(net_df: pd.DataFrame) -> list[dict]:
    """
    For every material with net_req_total > 0, generate a PO proposal.
    """
    proposals = []

    for _, row in net_df.iterrows():
        mat_id    = row["material_id"]
        net_total = row["net_req_total"]

        if mat_id not in MATERIALS:
            continue

        mat = MATERIALS[mat_id]

        # Only propose for materials with net requirements
        if net_total <= 0:
            proposals.append({
                "material_id":  mat_id,
                "description":  mat["description"],
                "level":        row["level"],
                "action":       "NO_ACTION",
                "reason":       "Net requirement = 0. Stock + open POs cover demand.",
                "order_qty":    0,
                "due_date":     None,
                "effective_lt": None,
                "price_total":  0,
                "risk":         row["stockout_risk"],
                "priority":     PRIORITY_RANK.get(row["stockout_risk"], 99),
            })
            continue

        order_qty = apply_lot_size(net_total, mat_id)
        due_date, lt = compute_order_due_date(mat_id)
        price_total  = order_qty * mat["price"]

        # Urgency: days of stock coverage at avg demand
        avg_daily = np.mean(CONSUMPTION_HISTORY.get(mat_id, [1])) / 22
        days_cover = (mat["stock"] - mat["safety_stock"]) / avg_daily if avg_daily > 0 else 999
        days_cover = max(0, days_cover)
        urgency    = max(0, days_cover - lt)

        if urgency <= 0:
            action = "PLACE_IMMEDIATELY"
        elif urgency <= 5:
            action = "PLACE_URGENT"
        elif urgency <= 14:
            action = "PLACE_THIS_WEEK"
        else:
            action = "PLAN_STANDARD"

        proposals.append({
            "material_id":     mat_id,
            "description":     mat["description"],
            "level":           row["level"],
            "action":          action,
            "reason":          f"Net req {net_total:.0f} units over {len(MONTHS)} months.",
            "order_qty":       int(order_qty),
            "lot_size_proc":   mat["lot_size"],
            "moq":             mat["moq"],
            "supplier":        mat["supplier"],
            "due_date":        due_date.isoformat(),
            "effective_lt":    lt,
            "days_until_must_order": round(urgency, 1),
            "price_per_unit":  mat["price"],
            "price_total":     round(price_total, 2),
            "risk":            row["stockout_risk"],
            "priority":        PRIORITY_RANK.get(row["stockout_risk"], 99),
        })

    proposals.sort(key=lambda x: (x["priority"], x.get("days_until_must_order", 99)))
    return proposals


def generate_sap_payload(proposals: list[dict]) -> list[dict]:
    """
    Formats proposals as SAP ME21N / BAPI_PO_CREATE1 payload.
    Always marked PENDING_PLANNER_APPROVAL — never auto-submitted.
    """
    payloads = []
    for p in proposals:
        if p["action"] == "NO_ACTION":
            continue
        payloads.append({
            "bapi":            "BAPI_PO_CREATE1",
            "plant":           "1000",
            "material":        p["material_id"],
            "vendor":          p["supplier"],
            "quantity":        p["order_qty"],
            "delivery_date":   p["due_date"],
            "net_price":       p["price_per_unit"],
            "total_value":     p["price_total"],
            "currency":        "INR",
            "action_code":     p["action"],
            "status":          "PENDING_PLANNER_APPROVAL",
        })
    return payloads


def run_reorder_proposals():
    print("=" * 65)
    print("  MODULE 6: REORDER PROPOSAL ENGINE")
    print("=" * 65)
    print(f"  Planning date: {PLANNING_DATE}")

    # Get net requirements from BOM exploder
    net_df    = run_bom_exploder()
    proposals = generate_proposals(net_df)

    print("\n" + "=" * 65)
    print("  PURCHASE ORDER PROPOSALS  (sorted by priority)")
    print("=" * 65)

    action_icons = {
        "PLACE_IMMEDIATELY": "🚨",
        "PLACE_URGENT":      "🔴",
        "PLACE_THIS_WEEK":   "🟠",
        "PLAN_STANDARD":     "🟡",
        "NO_ACTION":         "🟢",
    }

    total_value = 0
    for p in proposals:
        icon = action_icons.get(p["action"], "⚪")
        if p["action"] == "NO_ACTION":
            print(f"\n  {icon} {p['material_id']:<14} NO ACTION — {p['reason']}")
            continue

        total_value += p["price_total"]
        urgency_str = f"{p['days_until_must_order']:.0f}d window" if p["days_until_must_order"] else "overdue"

        print(f"\n  {icon} {p['action']:<22} {p['material_id']}  L{p['level']}")
        print(f"     {p['description']}")
        print(f"     Order qty:    {p['order_qty']:>6}  (lot-size: {p['lot_size_proc']}, MOQ: {p['moq']})")
        print(f"     Supplier:     {p['supplier']}")
        print(f"     Due date:     {p['due_date']}  (LT: {p['effective_lt']}d)")
        print(f"     Urgency:      {urgency_str}")
        print(f"     Value:        ₹{p['price_total']:,.0f}  (@₹{p['price_per_unit']:,.0f}/unit)")
        print(f"     Risk:         {p['risk']}")

    print(f"\n  {'─'*63}")
    print(f"  Total procurement value: ₹{total_value:,.0f}")

    # SAP payload
    payload = generate_sap_payload(proposals)
    print(f"\n  SAP WRITE-BACK PAYLOAD ({len(payload)} POs — awaiting planner approval)")
    for p in payload[:3]:   # show first 3
        print(f"    {p['material']:<14} qty={p['quantity']:>6}  due={p['delivery_date']}  "
              f"val=₹{p['total_value']:,.0f}  [{p['status']}]")
    if len(payload) > 3:
        print(f"    ... and {len(payload)-3} more")

    return proposals, payload


if __name__ == "__main__":
    run_reorder_proposals()
