"""
data_generator.py
------------------
Generates 5 years of realistic daily MRP training data for all 6 materials.
Parameters sourced directly from Smart_MRP_Dummy_Data_Draft.xlsx.

Materials:
  FG-100001  E-Motor Assembly      (FERT)  monthly: 120,130,110,140,155,150
  SFG-150001 Stator Sub-Assembly   (HALB)  monthly: 125,135,115,145,160,155
  RM-200010  Copper Winding Kit    (ROH)   monthly: 240,260,220,280,310,300 (= FG x2)
  RM-200020  Insulation Wrap       (ROH)   BOM qty=1 from FG
  RM-200030  Stator Core Lam.      (ROH)   BOM qty=1 from SFG
  RM-200040  Resin Compound        (ROH)   BOM qty=0.5 from SFG

Run:  python data_generator.py
"""

import numpy as np
import pandas as pd
from pathlib import Path

SEED       = 42
YEARS      = 5
START_DATE = "2020-01-01"

# All numbers from Excel
MATERIALS = {
    "FG-100001": {
        "description": "E-Motor Assembly", "mat_type": "FERT",
        "base_demand": (120+130+110+140+155+150) / (6*22),  # 6.10/day
        "demand_growth": 0.0012, "seasonality_amp": 0.18, "demand_noise": 0.10,
        "init_stock": 42, "safety_stock": 25, "reorder_point": 30,
        "min_stock": 20, "max_stock": 120, "moq": 10,
        "lot_size": "EX", "lot_size_value": 50, "rounding_value": 5,
        "base_lt": 7, "lt_std": 0.8, "po_proc": 2, "gr_proc": 1,
        "transit_risk": 0, "p_delay": 0.05, "delay_mag": 2,
        "p_rejection": 0.002, "supplier": "Vendor-managed",
    },
    "SFG-150001": {
        "description": "Stator Sub-Assembly", "mat_type": "HALB",
        "base_demand": (125+135+115+145+160+155) / (6*22),  # 6.33/day
        "demand_growth": 0.0012, "seasonality_amp": 0.18, "demand_noise": 0.12,
        "init_stock": 18, "safety_stock": 20, "reorder_point": 25,
        "min_stock": 10, "max_stock": 90, "moq": 20,
        "lot_size": "FX", "lot_size_value": 100, "rounding_value": 10,
        "base_lt": 12, "lt_std": 1.5, "po_proc": 2, "gr_proc": 1,
        "transit_risk": 1, "p_delay": 0.20, "delay_mag": 5,
        "p_rejection": 0.008, "supplier": "Vendor A",
    },
    "RM-200010": {
        "description": "Copper Winding Kit", "mat_type": "ROH",
        "base_demand": (240+260+220+280+310+300) / (6*22),  # 12.19/day = FG x2
        "demand_growth": 0.0012, "seasonality_amp": 0.20, "demand_noise": 0.15,
        "init_stock": 75, "safety_stock": 250, "reorder_point": 350,
        "min_stock": 200, "max_stock": 1200, "moq": 300,
        "lot_size": "HB", "lot_size_value": 600, "rounding_value": 50,
        "base_lt": 18, "lt_std": 3.5, "po_proc": 3, "gr_proc": 2,
        "transit_risk": 2, "p_delay": 0.45, "delay_mag": 7,  # "+7d" from Excel
        "p_rejection": 0.025, "supplier": "Vendor B",
    },
    "RM-200020": {
        "description": "Insulation Wrap", "mat_type": "ROH",
        "base_demand": (120+130+110+140+155+150) / (6*22),  # BOM qty=1 from FG
        "demand_growth": 0.0012, "seasonality_amp": 0.18, "demand_noise": 0.10,
        "init_stock": 500, "safety_stock": 100, "reorder_point": 150,
        "min_stock": 100, "max_stock": 800, "moq": 200,
        "lot_size": "FX", "lot_size_value": 200, "rounding_value": 50,
        "base_lt": 10, "lt_std": 1.0, "po_proc": 2, "gr_proc": 1,
        "transit_risk": 0, "p_delay": 0.05, "delay_mag": 2,
        "p_rejection": 0.003, "supplier": "Vendor C",
    },
    "RM-200030": {
        "description": "Stator Core Lamination", "mat_type": "ROH",
        "base_demand": (125+135+115+145+160+155) / (6*22),  # BOM qty=1 from SFG
        "demand_growth": 0.0012, "seasonality_amp": 0.18, "demand_noise": 0.12,
        "init_stock": 90, "safety_stock": 50, "reorder_point": 80,
        "min_stock": 50, "max_stock": 400, "moq": 100,
        "lot_size": "FX", "lot_size_value": 100, "rounding_value": 25,
        "base_lt": 14, "lt_std": 1.5, "po_proc": 2, "gr_proc": 1,
        "transit_risk": 1, "p_delay": 0.20, "delay_mag": 5,
        "p_rejection": 0.010, "supplier": "Vendor A",
    },
    "RM-200040": {
        "description": "Resin Compound (KG)", "mat_type": "ROH",
        "base_demand": (125+135+115+145+160+155) / (6*22) * 0.5,  # BOM qty=0.5
        "demand_growth": 0.0012, "seasonality_amp": 0.18, "demand_noise": 0.10,
        "init_stock": 210, "safety_stock": 80, "reorder_point": 100,
        "min_stock": 80, "max_stock": 600, "moq": 150,
        "lot_size": "FX", "lot_size_value": 150, "rounding_value": 25,
        "base_lt": 8, "lt_std": 0.8, "po_proc": 2, "gr_proc": 1,
        "transit_risk": 0, "p_delay": 0.05, "delay_mag": 2,
        "p_rejection": 0.003, "supplier": "Vendor D",
    },
}


def _generate_material(mat_id: str, cfg: dict, rng: np.random.Generator) -> pd.DataFrame:
    dates = pd.date_range(START_DATE, periods=YEARS * 365, freq="D")
    n = len(dates)
    t = np.arange(n)

    # 1. Demand: trend x seasonality x noise
    trend  = cfg["base_demand"] * (1 + cfg["demand_growth"]) ** t
    season = 1 + cfg["seasonality_amp"] * (
        0.6 * np.sin(2 * np.pi * t / 365 + np.pi / 3) +
        0.4 * np.sin(2 * np.pi * t / 30)
    )
    daily_demand = np.maximum(0.1, trend * season * rng.normal(1.0, cfg["demand_noise"], n))

    # 2. Lead time: base + noise + transit delay events
    base_lt     = np.maximum(1.0, rng.normal(cfg["base_lt"], cfg["lt_std"], n))
    delay_event = (rng.random(n) < cfg["p_delay"]).astype(int)
    delay_days  = rng.exponential(cfg["delay_mag"] * 0.8, n) * delay_event
    actual_lt   = np.round(base_lt + delay_days).astype(int)

    # 3. PO slippage and QI rejections
    slippage = np.where(rng.random(n) < 0.15, rng.integers(1, 8, n), 0)
    qi_flag  = (rng.random(n) < cfg["p_rejection"]).astype(int)

    # 4. Stock simulation starting from Excel init_stock
    stock = np.zeros(n)
    stock[0] = cfg["init_stock"]
    for i in range(1, n):
        receipt = (daily_demand[i] * actual_lt[i] * rng.uniform(0.9, 1.1)
                   if rng.random() < 0.08 else 0)
        stock[i] = max(0.0, stock[i - 1] - daily_demand[i] + receipt)

    stock_cover = np.where(daily_demand > 0, stock / daily_demand, 999.0)

    # 5. Rolling 30-day demand slope
    slope = np.zeros(n)
    for i in range(5, n):
        w = min(i, 30)
        slope[i] = np.polyfit(np.arange(w), daily_demand[i - w:i], 1)[0]

    # 6. Fourier time features
    month_sin = np.sin(2 * np.pi * dates.month / 12)
    month_cos = np.cos(2 * np.pi * dates.month / 12)
    dow_sin   = np.sin(2 * np.pi * dates.dayofweek / 7)

    # 7. Exception labels (SAP MRP logic)
    def label(i):
        if stock[i] < cfg["safety_stock"] * 0.5:             return "SHORTAGE"
        if stock_cover[i] < actual_lt[i] * 1.2:              return "EXPEDITE"
        if slope[i] > daily_demand[i] * 0.05:                return "DEMAND_SPIKE"
        if stock[i] > cfg["safety_stock"] * 5:               return "RESCHEDULE_IN"
        return "NO_EXCEPTION"

    return pd.DataFrame({
        "date":                  dates,
        "material_id":           mat_id,
        "description":           cfg["description"],
        "mat_type":              cfg["mat_type"],
        "supplier":              cfg["supplier"],
        "daily_demand":          np.round(daily_demand, 3),
        "actual_lead_time_days": actual_lt,
        "nominal_lead_time_days": cfg["base_lt"],
        "lt_deviation_days":     actual_lt - cfg["base_lt"],
        "transit_delay_days":    np.round(delay_days, 2),
        "delay_event":           delay_event,
        "stock_level":           np.round(stock, 2),
        "stock_cover_days":      np.round(np.minimum(stock_cover, 999.0), 2),
        "po_slippage_days":      slippage,
        "qi_rejection":          qi_flag,
        "safety_stock":          cfg["safety_stock"],
        "reorder_point":         cfg["reorder_point"],
        "moq":                   cfg["moq"],
        "transit_risk_level":    cfg["transit_risk"],
        "demand_rolling_slope":  np.round(slope, 5),
        "month_sin":             np.round(month_sin, 4),
        "month_cos":             np.round(month_cos, 4),
        "dow_sin":               np.round(dow_sin, 4),
        "month":                 dates.month,
        "quarter":               dates.quarter,
        "day_of_year":           dates.dayofyear,
        "exception_label":       [label(i) for i in range(n)],
    })


def generate_all(path: str = "data/training_data.csv") -> pd.DataFrame:
    print("=" * 60)
    print("  DATA GENERATOR — Smart MRP ML Pipeline (Excel-faithful)")
    print("=" * 60)
    rng = np.random.default_rng(SEED)
    frames = []
    for mat_id, cfg in MATERIALS.items():
        print(f"  {mat_id} ({cfg['description']})...", end=" ", flush=True)
        df = _generate_material(mat_id, cfg, rng)
        frames.append(df)
        print(f"{len(df):,} rows  LT={cfg['base_lt']}d  transit_risk={cfg['transit_risk']} ✓")

    out = (pd.concat(frames, ignore_index=True)
             .sort_values(["material_id", "date"])
             .reset_index(drop=True))
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(path, index=False)
    print(f"\n  Saved → {path}  ({len(out):,} rows, {len(out.columns)} cols)")
    print(f"  Date range: {out['date'].min().date()} → {out['date'].max().date()}")
    print(f"\n  Exception distribution:")
    for lbl, cnt in out["exception_label"].value_counts().items():
        print(f"    {lbl:<22} {cnt:>5}  ({cnt/len(out):.1%})")
    return out


if __name__ == "__main__":
    generate_all()
