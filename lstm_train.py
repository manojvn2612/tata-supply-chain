import pandas as pd
import numpy as np
import re

from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, Bidirectional

file_path = "Smart MRP_StatSampled.xlsx"
df = pd.read_excel(file_path)

print("Dataset Loaded:", df.shape)

#make series from text pattern and cleaning it slightly for better learning by model
def parse_series(text):

    if pd.isna(text):
        return []

    nums = []

    for i in str(text).split(","):
        try:
            nums.append(float(i.strip()))
        except:
            pass

    return nums


def extract_po_qty(text):

    if pd.isna(text):
        return 0

    match = re.search(r":\s*(\d+)", str(text))

    if match:
        return float(match.group(1))

    return 0


def extract_delay_days(text):

    if pd.isna(text):
        return 0

    match = re.search(r"\+(\d+)d", str(text))

    if match:
        return float(match.group(1))

    return 0


df["cons_series"] = df["Consumption Pattern / Month (2025-09..2026-02)"].apply(parse_series)
df["recv_series"] = df["Receipt Pattern / Month (2025-09..2026-02)"].apply(parse_series)

df["Available PO (Open)"] = df["Available PO (Open)"].apply(extract_po_qty)

if "Transit Delay Scenario" in df.columns:
    df["Transit Delay Scenario"] = df["Transit Delay Scenario"].apply(extract_delay_days)


numeric_cols = [
    "Stock (Unrestricted)",
    "Safety Stock",
    "MIN",
    "MAX",
    "Lead Time Supplier→Plant (Days)"
]

for col in numeric_cols:
    if col in df.columns:
        df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0)

#make sequences for LSTM training
X = []
y = []

materials = []
window = 4
#take into account lead time, safety stock, current stock, open PO and delay scenario as features along with consumption and receipt patterns in the window
for _, row in df.iterrows():
    cons = row["cons_series"]
    recv = row["recv_series"]
    if len(cons) <= window:
        continue
    lead = row["Lead Time Supplier→Plant (Days)"]
    safety = row["Safety Stock"]
    stock = row["Stock (Unrestricted)"]
    po = row["Available PO (Open)"]
    delay = 0
    if "Transit Delay Scenario" in row:
        delay = row["Transit Delay Scenario"]

    for i in range(len(cons) - window):

        features = []

        for j in range(window):

            features.append(cons[i+j])

            if j < len(recv):
                features.append(recv[i+j])
            else:
                features.append(0)

        features.extend([lead, safety, stock, po, delay])

        X.append(features)
        y.append(cons[i+window])
        materials.append(row["Material"])


X = np.array(X)
y = np.array(y)

print("Training samples:", X.shape)

# 5 NORMALIZE DATA

scalerX = MinMaxScaler()
scalery = MinMaxScaler()

X = scalerX.fit_transform(X)
y = scalery.fit_transform(y.reshape(-1,1))
X = X.reshape((X.shape[0], 1, X.shape[1]))

#model architecture
model = Sequential()

model.add(Bidirectional(LSTM(64, return_sequences=True), input_shape=(X.shape[1], X.shape[2])))

model.add(Dropout(0.2))

model.add(LSTM(32))

model.add(Dense(16, activation="relu"))

model.add(Dense(1))

model.compile(
    optimizer="adam",
    loss="mse",
    metrics=["mae"]
)

print(model.summary())

#training
history = model.fit(
    X,
    y,
    epochs=60,
    batch_size=16,
    validation_split=0.2
)

# 9 PREDICT DEMAND

pred = model.predict(X)

pred = scalery.inverse_transform(pred)

y_true = scalery.inverse_transform(y)

# 10 EVALUATION METRICS

mse = mean_squared_error(y_true, pred)
rmse = np.sqrt(mse)
mae = mean_absolute_error(y_true, pred)
r2 = r2_score(y_true, pred)

mape = np.mean(np.abs((y_true - pred) / y_true)) * 100

import joblib

joblib.dump(scalerX, "scalerX.pkl")
joblib.dump(scalery, "scalery.pkl")

print("\nModel Evaluation")
print("------------------")
print("MSE :", mse)
print("RMSE:", rmse)
print("MAE :", mae)
print("R2 :", r2)
print("MAPE:", mape)

model.save("mrp_lstm_model.h5")
# 11 GET LAST PREDICTION PER MATERIAL

pred_list = pred.flatten()

materials_unique = []
predictions = []

idx = 0

for _, row in df.iterrows():

    cons = row["cons_series"]

    if len(cons) <= window:
        continue

    samples = len(cons) - window

    last_pred = pred_list[idx + samples - 1]

    materials_unique.append(row["Material"])
    predictions.append(last_pred)

    idx += samples


results = pd.DataFrame({
    "Material": materials_unique,
    "Predicted Demand": predictions
})

#deterministic formula for reorder qty calculation based on predicted demand, safety stock, current stock and open PO. This can be further improved by incorporating delay scenario and lead time into the formula or by using a separate model to predict reorder qty directly.
results["Safety Stock"] = df["Safety Stock"].values[:len(results)]
results["Stock"] = df["Stock (Unrestricted)"].values[:len(results)]
results["Open PO"] = df["Available PO (Open)"].values[:len(results)]
#formula 
results["Reorder Qty"] = (
    results["Predicted Demand"]
    + results["Safety Stock"]
    - results["Stock"]
    - results["Open PO"]
)

# SAVE RESULTS

results.to_excel("MRP_AI_Predictions.xlsx", index=False)

scores = pd.DataFrame({
    "Metric":["MSE","RMSE","MAE","R2","MAPE"],
    "Value":[mse,rmse,mae,r2,mape]
})

scores.to_excel("MRP_Model_Scores.xlsx", index=False)

print("\nResults saved:")
print("MRP_AI_Predictions.xlsx")
print("MRP_Model_Scores.xlsx")