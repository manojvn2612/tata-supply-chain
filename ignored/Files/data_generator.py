# # """
# # data_generator.py
# # ------------------
# # Generates 5 years of realistic daily MRP training data.
# # Saved to data/training_data.csv — all ML modules load from here.

# # Run:  python data_generator.py
# # """

# # import numpy as np
# # import pandas as pd
# # from pathlib import Path

# # SEED = 42
# # YEARS = 5
# # START_DATE = "2020-01-01"

# # MATERIALS = {
# #     "FG-100001": {
# #         "base_demand": 130, "demand_growth": 0.0015,
# #         "seasonality_amp": 0.18, "demand_noise": 0.10,
# #         "base_lt": 7,  "lt_std": 0.6,
# #         "p_delay": 0.05, "delay_mag": 2,
# #         "p_rejection": 0.002, "init_stock": 200,
# #         "safety_stock": 25, "moq": 10, "transit_risk": 0,
# #     },
# #     "SFG-150001": {
# #         "base_demand": 135, "demand_growth": 0.0015,
# #         "seasonality_amp": 0.18, "demand_noise": 0.12,
# #         "base_lt": 12, "lt_std": 1.2,
# #         "p_delay": 0.20, "delay_mag": 5,
# #         "p_rejection": 0.008, "init_stock": 300,
# #         "safety_stock": 20, "moq": 20, "transit_risk": 1,
# #     },
# #     "RM-200010": {
# #         "base_demand": 260, "demand_growth": 0.0015,
# #         "seasonality_amp": 0.20, "demand_noise": 0.15,
# #         "base_lt": 18, "lt_std": 3.5,
# #         "p_delay": 0.45, "delay_mag": 7,
# #         "p_rejection": 0.025, "init_stock": 800,
# #         "safety_stock": 250, "moq": 300, "transit_risk": 2,
# #     },
# # }

# # EXCEPTION_CLASSES = ["NO_EXCEPTION", "SHORTAGE", "EXPEDITE", "RESCHEDULE_IN", "DEMAND_SPIKE"]


# # def generate_material(mat_id, cfg, rng):
# #     dates = pd.date_range(START_DATE, periods=YEARS * 365, freq="D")
# #     n = len(dates)
# #     t = np.arange(n)

# #     # Demand = trend × seasonality × noise
# #     trend    = cfg["base_demand"] * (1 + cfg["demand_growth"]) ** t
# #     season   = 1 + cfg["seasonality_amp"] * (
# #         0.6 * np.sin(2 * np.pi * t / 365 + 1.0) +
# #         0.4 * np.sin(2 * np.pi * t / 30)
# #     )
# #     noise         = rng.normal(1.0, cfg["demand_noise"], n)
# #     daily_demand  = np.maximum(0.1, trend * season * noise / 22)

# #     # Lead time = base + gaussian noise + random port-delay events
# #     base_lt      = np.maximum(1, rng.normal(cfg["base_lt"], cfg["lt_std"], n))
# #     delay_event  = (rng.random(n) < cfg["p_delay"]).astype(int)
# #     delay_days   = rng.exponential(cfg["delay_mag"] * 0.8, n) * delay_event
# #     actual_lt    = np.round(base_lt + delay_days).astype(int)

# #     # PO confirmation slippage
# #     slippage = np.where(rng.random(n) < 0.15, rng.integers(1, 8, n), 0)

# #     # QI rejections
# #     qi_flag = (rng.random(n) < cfg["p_rejection"]).astype(int)

# #     # Rolling stock simulation
# #     stock = np.zeros(n)
# #     stock[0] = cfg["init_stock"]
# #     for i in range(1, n):
# #         receipt = daily_demand[i] * actual_lt[i] * rng.uniform(0.9, 1.1) if rng.random() < 0.08 else 0
# #         stock[i] = max(0, stock[i - 1] - daily_demand[i] + receipt)

# #     stock_cover = np.where(daily_demand > 0, stock / daily_demand, 999.0)

# #     # Rolling 30-day demand slope (manual, no apply needed)
# #     slope = np.zeros(n)
# #     for i in range(5, n):
# #         w = min(i, 30)
# #         x = np.arange(w)
# #         y = daily_demand[i - w:i]
# #         slope[i] = np.polyfit(x, y, 1)[0]

# #     # Fourier time features
# #     month_sin = np.sin(2 * np.pi * dates.month / 12)
# #     month_cos = np.cos(2 * np.pi * dates.month / 12)
# #     dow_sin   = np.sin(2 * np.pi * dates.dayofweek / 7)

# #     # Exception labels from domain rules (used as ground truth for RF)
# #     def label(i):
# #         if stock[i] < cfg["safety_stock"] * 0.5:
# #             return "SHORTAGE"
# #         if stock_cover[i] < actual_lt[i] * 1.2:
# #             return "EXPEDITE"
# #         if slope[i] > daily_demand[i] * 0.05:
# #             return "DEMAND_SPIKE"
# #         if stock[i] > cfg["safety_stock"] * 5:
# #             return "RESCHEDULE_IN"
# #         return "NO_EXCEPTION"

# #     labels = [label(i) for i in range(n)]

# #     return pd.DataFrame({
# #         "date":                     dates,
# #         "material_id":              mat_id,
# #         "daily_demand":             np.round(daily_demand, 3),
# #         "actual_lead_time_days":    actual_lt,
# #         "nominal_lead_time_days":   cfg["base_lt"],
# #         "lt_deviation_days":        actual_lt - cfg["base_lt"],
# #         "transit_delay_days":       np.round(delay_days, 2),
# #         "delay_event":              delay_event,
# #         "stock_level":              np.round(stock, 2),
# #         "stock_cover_days":         np.round(np.minimum(stock_cover, 999), 2),
# #         "po_slippage_days":         slippage,
# #         "qi_rejection":             qi_flag,
# #         "demand_rolling_slope":     np.round(slope, 5),
# #         "safety_stock":             cfg["safety_stock"],
# #         "moq":                      cfg["moq"],
# #         "transit_risk_level":       cfg["transit_risk"],
# #         "month_sin":                np.round(month_sin, 4),
# #         "month_cos":                np.round(month_cos, 4),
# #         "dow_sin":                  np.round(dow_sin, 4),
# #         "month":                    dates.month,
# #         "quarter":                  dates.quarter,
# #         "day_of_year":              dates.dayofyear,
# #         "exception_label":          labels,
# #     })


# # def generate_all(path="data/training_data.csv"):
# #     rng = np.random.default_rng(SEED)
# #     frames = []
# #     for mat_id, cfg in MATERIALS.items():
# #         print(f"  Generating {mat_id}...", end=" ", flush=True)
# #         df = generate_material(mat_id, cfg, rng)
# #         frames.append(df)
# #         print(f"{len(df):,} rows ✓")
# #     out = pd.concat(frames, ignore_index=True).sort_values(["material_id","date"]).reset_index(drop=True)
# #     Path(path).parent.mkdir(exist_ok=True)
# #     out.to_csv(path, index=False)
# #     print(f"\n  Saved → {path}  ({len(out):,} total rows, {len(out.columns)} columns)")
# #     print(f"  Date range: {out['date'].min()} → {out['date'].max()}")
# #     print(f"\n  Exception distribution:\n{out['exception_label'].value_counts().to_string()}")
# #     return out


# # if __name__ == "__main__":
# #     print("=" * 60)
# #     print("  DATA GENERATOR — Smart MRP ML Pipeline")
# #     print("=" * 60)
# #     generate_all()
# """
# data_generator.py
# ------------------
# Generates 5 years of realistic daily MRP training data.
# Saved to data/training_data.csv — all ML modules load from here.

# Run:  python data_generator.py
# """

# import numpy as np
# import pandas as pd
# from pathlib import Path

# SEED = 42
# YEARS = 5
# START_DATE = "2020-01-01"

# MATERIALS = {
#     "FG-100001": {
#         "base_demand": 130, "demand_growth": 0.0015,
#         "seasonality_amp": 0.18, "demand_noise": 0.10,
#         "base_lt": 7,  "lt_std": 0.6,
#         "p_delay": 0.05, "delay_mag": 2,
#         "p_rejection": 0.002, "init_stock": 200,
#         "safety_stock": 25, "moq": 10, "transit_risk": 0,
#     },
#     "SFG-150001": {
#         "base_demand": 135, "demand_growth": 0.0015,
#         "seasonality_amp": 0.18, "demand_noise": 0.12,
#         "base_lt": 12, "lt_std": 1.2,
#         "p_delay": 0.20, "delay_mag": 5,
#         "p_rejection": 0.008, "init_stock": 300,
#         "safety_stock": 20, "moq": 20, "transit_risk": 1,
#     },
#     "RM-200010": {
#         "base_demand": 260, "demand_growth": 0.0015,
#         "seasonality_amp": 0.20, "demand_noise": 0.15,
#         "base_lt": 18, "lt_std": 3.5,
#         "p_delay": 0.45, "delay_mag": 7,
#         "p_rejection": 0.025, "init_stock": 800,
#         "safety_stock": 250, "moq": 300, "transit_risk": 2,
#     },
# }

# EXCEPTION_CLASSES = ["NO_EXCEPTION", "SHORTAGE", "EXPEDITE", "RESCHEDULE_IN", "DEMAND_SPIKE"]


# def generate_material(mat_id, cfg, rng):
#     dates = pd.date_range(START_DATE, periods=YEARS * 365, freq="D")
#     n = len(dates)
#     t = np.arange(n)

#     # Demand = trend × seasonality × noise
#     trend    = cfg["base_demand"] * (1 + cfg["demand_growth"]) ** t
#     season   = 1 + cfg["seasonality_amp"] * (
#         0.6 * np.sin(2 * np.pi * t / 365 + 1.0) +
#         0.4 * np.sin(2 * np.pi * t / 30)
#     )
#     noise         = rng.normal(1.0, cfg["demand_noise"], n)
#     daily_demand  = np.maximum(0.1, trend * season * noise / 22)

#     # Lead time = base + gaussian noise + random port-delay events
#     base_lt      = np.maximum(1, rng.normal(cfg["base_lt"], cfg["lt_std"], n))
#     delay_event  = (rng.random(n) < cfg["p_delay"]).astype(int)
#     delay_days   = rng.exponential(cfg["delay_mag"] * 0.8, n) * delay_event
#     actual_lt    = np.round(base_lt + delay_days).astype(int)

#     # PO confirmation slippage
#     slippage = np.where(rng.random(n) < 0.15, rng.integers(1, 8, n), 0)

#     # QI rejections
#     qi_flag = (rng.random(n) < cfg["p_rejection"]).astype(int)

#     # Rolling stock simulation
#     stock = np.zeros(n)
#     stock[0] = cfg["init_stock"]
#     for i in range(1, n):
#         receipt = daily_demand[i] * actual_lt[i] * rng.uniform(0.9, 1.1) if rng.random() < 0.08 else 0
#         stock[i] = max(0, stock[i - 1] - daily_demand[i] + receipt)

#     stock_cover = np.where(daily_demand > 0, stock / daily_demand, 999.0)

#     # Rolling 30-day demand slope (manual, no apply needed)
#     slope = np.zeros(n)
#     for i in range(5, n):
#         w = min(i, 30)
#         x = np.arange(w)
#         y = daily_demand[i - w:i]
#         slope[i] = np.polyfit(x, y, 1)[0]

#     # Fourier time features
#     month_sin = np.sin(2 * np.pi * dates.month / 12)
#     month_cos = np.cos(2 * np.pi * dates.month / 12)
#     dow_sin   = np.sin(2 * np.pi * dates.dayofweek / 7)

#     # Exception labels from domain rules (used as ground truth for RF)
#     def label(i):
#         if stock[i] < cfg["safety_stock"] * 0.5:
#             return "SHORTAGE"
#         if stock_cover[i] < actual_lt[i] * 1.2:
#             return "EXPEDITE"
#         if slope[i] > daily_demand[i] * 0.05:
#             return "DEMAND_SPIKE"
#         if stock[i] > cfg["safety_stock"] * 5:
#             return "RESCHEDULE_IN"
#         return "NO_EXCEPTION"

#     labels = [label(i) for i in range(n)]

#     return pd.DataFrame({
#         "date":                     dates,
#         "material_id":              mat_id,
#         "daily_demand":             np.round(daily_demand, 3),
#         "actual_lead_time_days":    actual_lt,
#         "nominal_lead_time_days":   cfg["base_lt"],
#         "lt_deviation_days":        actual_lt - cfg["base_lt"],
#         "transit_delay_days":       np.round(delay_days, 2),
#         "delay_event":              delay_event,
#         "stock_level":              np.round(stock, 2),
#         "stock_cover_days":         np.round(np.minimum(stock_cover, 999), 2),
#         "po_slippage_days":         slippage,
#         "qi_rejection":             qi_flag,
#         "demand_rolling_slope":     np.round(slope, 5),
#         "safety_stock":             cfg["safety_stock"],
#         "moq":                      cfg["moq"],
#         "transit_risk_level":       cfg["transit_risk"],
#         "month_sin":                np.round(month_sin, 4),
#         "month_cos":                np.round(month_cos, 4),
#         "dow_sin":                  np.round(dow_sin, 4),
#         "month":                    dates.month,
#         "quarter":                  dates.quarter,
#         "day_of_year":              dates.dayofyear,
#         "exception_label":          labels,
#     })


# def generate_all(path="data/training_data1.csv"):
#     rng = np.random.default_rng(SEED)
#     frames = []
#     for mat_id, cfg in MATERIALS.items():
#         print(f"  Generating {mat_id}...", end=" ", flush=True)
#         df = generate_material(mat_id, cfg, rng)
#         frames.append(df)
#         print(f"{len(df):,} rows ✓")
#     out = pd.concat(frames, ignore_index=True).sort_values(["material_id","date"]).reset_index(drop=True)
#     Path(path).parent.mkdir(exist_ok=True)
#     out.to_csv(path, index=False)
#     print(f"\n  Saved → {path}  ({len(out):,} total rows, {len(out.columns)} columns)")
#     print(f"  Date range: {out['date'].min()} → {out['date'].max()}")
#     print(f"\n  Exception distribution:\n{out['exception_label'].value_counts().to_string()}")
#     return out


# if __name__ == "__main__":
#     print("=" * 60)
#     print("  DATA GENERATOR — Smart MRP ML Pipeline")
#     print("=" * 60)
#     generate_all()
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime, timedelta

SEED = 42
rng = np.random.default_rng(SEED)

START_MONTH = "2025-09-01"
MONTHS = 6

MONTH_LABELS = ["2025-09", "2025-10", "2025-11", "2025-12", "2026-01", "2026-02"]

# ----------------------------------------------------------------
# MASTER MATERIAL CONFIGURATION (Based on your current data)
# ----------------------------------------------------------------

MATERIAL_CONFIG = {
    "FG-100001": {
        "plant": "1000",
        "mat_type": "FERT",
        "desc": "E-Motor Assembly – High Efficiency Industrial Grade Motor used in Heavy Machinery Applications",
        "uom": "EA",
        "price": 18500,
        "std_price": "S (Std)",
        "bom_parent": "(self)",
        "bom_components": [
            ("RM-200010", 2, "EA"),
            ("RM-200020", 1, "EA"),
            ("SFG-150001", 1, "EA"),
        ],
        "base_demand": 135,
        "lead_time": 7,
        "supplier": "Vendor-managed",
        "transit_delay": "Low – Direct inland transport, no port dependency",
        "risk_level": "Low Risk"
    },
    "SFG-150001": {
        "plant": "1000",
        "mat_type": "HALB",
        "desc": "Stator Sub-Assembly – Precision Copper Wound Core for Electric Motor Applications",
        "uom": "EA",
        "price": 6200,
        "std_price": "S (Std)",
        "bom_parent": "FG-100001",
        "bom_components": [
            ("RM-200030", 1, "EA"),
            ("RM-200040", 0.5, "KG"),
        ],
        "base_demand": 145,
        "lead_time": 12,
        "supplier": "Vendor A",
        "transit_delay": "Medium – Occasional logistics congestion observed",
        "risk_level": "Medium Risk"
    },
    "RM-200010": {
        "plant": "1000",
        "mat_type": "ROH",
        "desc": "Copper Winding Kit – High Conductivity Copper Coil Assembly",
        "uom": "EA",
        "price": 950,
        "std_price": "MAP (Moving Avg)",
        "bom_parent": "FG-100001",
        "bom_components": [],
        "base_demand": 270,
        "lead_time": 18,
        "supplier": "Vendor B",
        "transit_delay": "High – Port delay +7 days possible due to import dependency",
        "risk_level": "High Risk"
    }
}

# ----------------------------------------------------------------
# HELPER FUNCTIONS
# ----------------------------------------------------------------

def generate_monthly_pattern(base):
    trend = np.linspace(0.95, 1.12, MONTHS)
    noise = rng.normal(1.0, 0.12, MONTHS)
    values = base * trend * noise
    return [int(max(0, v)) for v in values]

def format_bom(bom_list):
    if not bom_list:
        return "(n/a)"
    return ";\n".join([f"{comp}:{qty} {uom}" for comp, qty, uom in bom_list])

def generate_multiple_pos(material, uom):
    po_texts = []
    for _ in range(rng.integers(1, 3)):
        po_number = rng.integers(4500012000, 4500012999)
        qty = rng.integers(50, 500)
        due_date = datetime(2026, 2, 20) + timedelta(days=int(rng.integers(0, 30)))
        po_texts.append(f"PO#{po_number}: {qty} {uom} due {due_date.date()}")
    return "; ".join(po_texts)

def generate_inventory_comment(stock, safety, reorder):
    if stock < safety:
        return "CRITICAL: Stock below safety stock. Immediate replenishment recommended."
    elif stock < reorder:
        return "WARNING: Stock below reorder point. Monitor closely."
    else:
        return "HEALTHY: Inventory level sufficient for current demand forecast."

def generate_consumption_analysis(pattern):
    return (
        f"Consumption shows a progressive trend across {MONTH_LABELS[0]} to {MONTH_LABELS[-1]}. "
        f"Monthly demand values observed: {pattern}. "
        "Seasonal uplift visible towards year-end due to industrial production cycle."
    )

def generate_receipt_analysis(pattern):
    return (
        f"Receipt pattern aligned with planned procurement cycle. "
        f"Monthly receipts recorded as: {pattern}. "
        "Variation due to supplier confirmation timing and logistics variability."
    )

# ----------------------------------------------------------------
# MAIN GENERATOR
# ----------------------------------------------------------------

def generate_large_sap_table(path="data/extended_sap_planning_table.csv"):
    rows = []

    for mat_id, cfg in MATERIAL_CONFIG.items():

        consumption = generate_monthly_pattern(cfg["base_demand"])
        receipts = generate_monthly_pattern(cfg["base_demand"] * 0.92)

        stock_unrestricted = rng.integers(20, 120)
        stock_qi = rng.integers(0, 10)
        stock_blocked = rng.integers(0, 10)

        safety_stock = int(cfg["base_demand"] * 0.2)
        reorder_point = safety_stock + int(cfg["lead_time"] * (cfg["base_demand"] / 30))
        min_level = safety_stock
        max_level = safety_stock * 5
        moq = int(cfg["base_demand"] * 0.5)

        open_po = generate_multiple_pos(mat_id, cfg["uom"])
        inventory_comment = generate_inventory_comment(stock_unrestricted, safety_stock, reorder_point)

        row = {
            "Plant": cfg["plant"],
            "Material": mat_id,
            "Mat Type": cfg["mat_type"],
            "Description": cfg["desc"],
            "Base UoM": cfg["uom"],
            "Price/Unit": cfg["price"],
            "Std Price Ref": cfg["std_price"],
            "BOM Parent": cfg["bom_parent"],
            "BOM Components (Component:Qty/UoM)": format_bom(cfg["bom_components"]),
            "Consumption Pattern / Month (2025-09..2026-02)": ", ".join(map(str, consumption)),
            "Receipt Pattern / Month (2025-09..2026-02)": ", ".join(map(str, receipts)),
            "Consumption Analysis Commentary": generate_consumption_analysis(consumption),
            "Receipt Analysis Commentary": generate_receipt_analysis(receipts),
            "Stock (Unrestricted)": stock_unrestricted,
            "Stock (QI)": stock_qi,
            "Stock (Blocked)": stock_blocked,
            "Available PO (Open)": open_po,
            "Inventory Health Commentary": inventory_comment,
            "MIN": min_level,
            "MAX": max_level,
            "Reorder Point (Current)": reorder_point,
            "Safety Stock": safety_stock,
            "MOQ": moq,
            "Lot Size Proc": rng.choice(["EX", "FX", "HB"]),
            "Lot Size Value": moq * 2,
            "Rounding Profile": f"RP-{mat_id[-2:]}",
            "Rounding Value": int(moq / 2),
            "Supplier": cfg["supplier"],
            "Lead Time Supplier→Plant (Days)": cfg["lead_time"],
            "PO Processing (Days)": rng.integers(1, 4),
            "GR Processing (Days)": rng.integers(1, 3),
            "Transit Delay Scenario": cfg["transit_delay"],
            "Risk Classification": cfg["risk_level"],
            "Supplier Performance Notes": (
                f"Supplier {cfg['supplier']} categorized under {cfg['risk_level']}. "
                f"Lead time of {cfg['lead_time']} days with transit profile: {cfg['transit_delay']}."
            )
        }

        rows.append(row)

    df = pd.DataFrame(rows)
    Path(path).parent.mkdir(exist_ok=True)
    df.to_csv(path, index=False)

    print("Extended SAP Planning Table Generated Successfully.")
    print(df.head())

    return df


if __name__ == "__main__":
    generate_large_sap_table()