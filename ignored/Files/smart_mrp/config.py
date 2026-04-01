"""
config.py
----------
Central configuration — imported by all modules.
All values sourced from Smart_MRP_Dummy_Data_Draft.xlsx
"""

from pathlib import Path

BASE_DIR    = Path(__file__).parent
DATA_DIR    = BASE_DIR / "data"
MODELS_DIR  = BASE_DIR / "models"
OUTPUTS_DIR = BASE_DIR / "outputs"

for d in [DATA_DIR, MODELS_DIR, OUTPUTS_DIR]:
    d.mkdir(exist_ok=True)

TRAINING_DATA_PATH = DATA_DIR / "training_data.csv"
PLANT = "1000"

MATERIALS = {
    "FG-100001": {
        "description": "E-Motor Assembly", "mat_type": "FERT", "price": 18500,
        "stock": 42, "stock_qi": 3, "stock_blocked": 0,
        "safety_stock": 25, "reorder_point": 30, "moq": 10,
        "lot_size": "EX", "lot_size_value": 50, "min_stock": 20, "max_stock": 120,
        "lead_time_days": 7, "po_proc_days": 2, "gr_proc_days": 1,
        "supplier": "Vendor-managed", "transit_risk": "Low",
        "open_pos": [{"po": "PO#4500012345", "qty": 60, "due": "2026-02-25"}],
    },
    "SFG-150001": {
        "description": "Stator Sub-Assembly", "mat_type": "HALB", "price": 6200,
        "stock": 18, "stock_qi": 0, "stock_blocked": 0,
        "safety_stock": 20, "reorder_point": 25, "moq": 20,
        "lot_size": "FX", "lot_size_value": 100, "min_stock": 10, "max_stock": 90,
        "lead_time_days": 12, "po_proc_days": 2, "gr_proc_days": 1,
        "supplier": "Vendor A", "transit_risk": "Medium",
        "open_pos": [{"po": "PO#4500012388", "qty": 80, "due": "2026-03-02"}],
    },
    "RM-200010": {
        "description": "Copper Winding Kit", "mat_type": "ROH", "price": 950,
        "stock": 75, "stock_qi": 0, "stock_blocked": 5,
        "safety_stock": 250, "reorder_point": 350, "moq": 300,
        "lot_size": "HB", "lot_size_value": 600, "min_stock": 200, "max_stock": 1200,
        "lead_time_days": 18, "po_proc_days": 3, "gr_proc_days": 2,
        "supplier": "Vendor B", "transit_risk": "High",
        "open_pos": [
            {"po": "PO#4500012222", "qty": 500, "due": "2026-02-22"},
            {"po": "PO#4500012333", "qty": 300, "due": "2026-03-05"},
        ],
    },
    "RM-200020": {
        "description": "Insulation Wrap", "mat_type": "ROH", "price": 120,
        "stock": 500, "stock_qi": 0, "stock_blocked": 0,
        "safety_stock": 100, "reorder_point": 150, "moq": 200,
        "lot_size": "FX", "lot_size_value": 200, "min_stock": 100, "max_stock": 800,
        "lead_time_days": 10, "po_proc_days": 2, "gr_proc_days": 1,
        "supplier": "Vendor C", "transit_risk": "Low", "open_pos": [],
    },
    "RM-200030": {
        "description": "Stator Core Lamination", "mat_type": "ROH", "price": 430,
        "stock": 90, "stock_qi": 0, "stock_blocked": 0,
        "safety_stock": 50, "reorder_point": 80, "moq": 100,
        "lot_size": "FX", "lot_size_value": 100, "min_stock": 50, "max_stock": 400,
        "lead_time_days": 14, "po_proc_days": 2, "gr_proc_days": 1,
        "supplier": "Vendor A", "transit_risk": "Medium", "open_pos": [],
    },
    "RM-200040": {
        "description": "Resin Compound (KG)", "mat_type": "ROH", "price": 85,
        "stock": 210, "stock_qi": 0, "stock_blocked": 0,
        "safety_stock": 80, "reorder_point": 100, "moq": 150,
        "lot_size": "FX", "lot_size_value": 150, "min_stock": 80, "max_stock": 600,
        "lead_time_days": 8, "po_proc_days": 2, "gr_proc_days": 1,
        "supplier": "Vendor D", "transit_risk": "Low", "open_pos": [],
    },
}

BOM = {
    "FG-100001":  {"RM-200010": 2.0, "RM-200020": 1.0, "SFG-150001": 1.0},
    "SFG-150001": {"RM-200030": 1.0, "RM-200040": 0.5},
}

CONSUMPTION_HISTORY = {
    "FG-100001":  [120, 130, 110, 140, 155, 150],
    "SFG-150001": [125, 135, 115, 145, 160, 155],
    "RM-200010":  [240, 260, 220, 280, 310, 300],
    "RM-200020":  [120, 130, 110, 140, 155, 150],
    "RM-200030":  [125, 135, 115, 145, 160, 155],
    "RM-200040":  [ 62,  68,  58,  73,  80,  78],
}

MONTHS = ["Sep-25", "Oct-25", "Nov-25", "Dec-25", "Jan-26", "Feb-26"]

SERVICE_LEVEL      = 0.95
WORKING_DAYS_MONTH = 22

LOT_SIZE_PROCEDURES = {
    "EX": "Lot-for-lot",
    "FX": "Fixed lot size",
    "HB": "Replenish to max stock",
    "MB": "Monthly lot size",
}
