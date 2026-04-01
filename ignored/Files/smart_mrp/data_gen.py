import pandas as pd
import numpy as np
from pathlib import Path
import random

SEED = 42
TARGET_ROWS = 100
rng = np.random.default_rng(SEED)

INPUT_FILE = "dummy_main.csv"   # your 3-row CSV
OUTPUT_FILE = "dummy_main_100.csv"

df = pd.read_csv(INPUT_FILE)

expanded_rows = []

material_counter = 100000

for i in range(TARGET_ROWS):

    base_row = df.sample(1, random_state=rng.integers(0, 10000)).iloc[0].copy()

    # Generate new material ID
    prefix = base_row["Material"].split("-")[0]
    new_material = f"{prefix}-{material_counter+i}"

    base_row["Material"] = new_material
    base_row["Description"] = base_row["Description"] + f" Variant_{i}"

    # Slight variation in price
    base_row["Price/Unit"] = round(
        float(base_row["Price/Unit"]) * rng.uniform(0.85, 1.20),
        2
    )

    # Randomize stock values
    base_row["Stock (Unrestricted)"] = int(rng.integers(10, 500))
    base_row["Stock (QI)"] = int(rng.integers(0, 20))
    base_row["Stock (Blocked)"] = int(rng.integers(0, 10))

    # Randomize planning parameters
    base_row["MIN"] = int(rng.integers(10, 100))
    base_row["MAX"] = int(rng.integers(100, 1000))
    base_row["Reorder Point (Current)"] = int(rng.integers(20, 300))
    base_row["Safety Stock"] = int(rng.integers(10, 200))
    base_row["MOQ"] = int(rng.integers(5, 300))

    # Randomize lead time
    base_row["Lead Time Supplier→Plant (Days)"] = int(rng.integers(5, 25))
    base_row["PO Processing (Days)"] = int(rng.integers(1, 5))
    base_row["GR Processing (Days)"] = int(rng.integers(1, 3))

    # Randomize delay scenario
    base_row["Transit Delay Scenario"] = random.choice(
        ["Low", "Medium", "High", "High (port delay +7d)"]
    )

    expanded_rows.append(base_row)

expanded_df = pd.DataFrame(expanded_rows)

expanded_df.to_csv(OUTPUT_FILE, index=False)

print(f"✅ Generated {TARGET_ROWS} rows → {OUTPUT_FILE}")