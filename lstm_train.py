import pandas as pd
import numpy as np
import os
import re
import joblib

from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import LSTM, Dense, Dropout, Bidirectional

MODEL_PATH = "model_weights/lstm_model.h5"
SCALER_X_PATH = "model_weights/scalerX.save"
SCALER_Y_PATH = "model_weights/scalerY.save"

# -----------------------------
# Helpers
# -----------------------------
def parse_series(text):
    if pd.isna(text):
        return []
    return [float(x.strip()) for x in str(text).split(",") if x.strip().replace('.', '', 1).isdigit()]

def extract_po(text):
    match = re.search(r":\s*(\d+)", str(text))
    return float(match.group(1)) if match else 0

# -----------------------------
# Train Model (only once)
# -----------------------------
def train_model(df):
    X, y = [], []
    window = 4

    for _, row in df.iterrows():
        cons = row["cons"]
        recv = row["recv"]

        if len(cons) <= window:
            continue

        for i in range(len(cons) - window):
            feat = []
            for j in range(window):
                feat.append(cons[i+j])
                feat.append(recv[i+j] if j < len(recv) else 0)
            X.append(feat)
            y.append(cons[i+window])

    X, y = np.array(X), np.array(y)

    scalerX = MinMaxScaler()
    scalery = MinMaxScaler()

    X = scalerX.fit_transform(X)
    y = scalery.fit_transform(y.reshape(-1,1))

    X = X.reshape((X.shape[0], 1, X.shape[1]))

    model = Sequential([
        Bidirectional(LSTM(64, return_sequences=True), input_shape=(1, X.shape[2])),
        Dropout(0.2),
        LSTM(32),
        Dense(16, activation="relu"),
        Dense(1)
    ])

    model.compile(optimizer="adam", loss="mse")
    model.fit(X, y, epochs=60, batch_size=16, verbose=0)

    model.save(MODEL_PATH)
    joblib.dump(scalerX, SCALER_X_PATH)
    joblib.dump(scalery, SCALER_Y_PATH)

    return model, scalerX, scalery

# -----------------------------
# Load or Train
# -----------------------------
def get_model(df):
    if os.path.exists(MODEL_PATH):
        # Load model without compiling to avoid deserialization errors from legacy metrics/losses
        model = load_model(MODEL_PATH, compile=False)
        scalerX = joblib.load(SCALER_X_PATH)
        scalery = joblib.load(SCALER_Y_PATH)
        # Re-compile with correct loss/metrics if needed
        model.compile(optimizer="adam", loss="mse")
    else:
        model, scalerX, scalery = train_model(df)
    return model, scalerX, scalery

# -----------------------------
# MAIN FUNCTION (used by agent)
# -----------------------------
def run_lstm_demand_forecast(df):

    df = df.copy()
    df["cons"] = df["Consumption Pattern / Month (2025-09..2026-02)"].apply(parse_series)
    df["recv"] = df["Receipt Pattern / Month (2025-09..2026-02)"].apply(parse_series)
    df["Available PO (Open)"] = df["Available PO (Open)"].apply(extract_po)

    model, scalerX, scalery = get_model(df)

    preds = []

    for _, row in df.iterrows():
        cons = row["cons"]
        recv = row["recv"]

        if len(cons) < 4:
            preds.append(0)
            continue

        feat = []
        for j in range(4):
            feat.append(cons[-4+j])
            feat.append(recv[-4+j] if j < len(recv) else 0)

        X = scalerX.transform([feat])
        X = X.reshape((1,1,X.shape[1]))

        pred = model.predict(X, verbose=0)
        pred = scalery.inverse_transform(pred)[0][0]

        preds.append(pred)

    results = pd.DataFrame({
        "Material": df["Material"],
        "Predicted Demand": preds
    })

    results["Safety Stock"] = df["Safety Stock"]
    results["Stock"] = df["Stock (Unrestricted)"]
    results["Open PO"] = df["Available PO (Open)"]
    results["Lead Time"] = df["Lead Time Supplier→Plant (Days)"]

    # 🔥 ROP
    results["ROP"] = (
        results["Predicted Demand"] * results["Lead Time"]
        + results["Safety Stock"]
    )

    # 🔥 Decision
    results["Decision"] = results.apply(
        lambda x: "REORDER" if (x["Stock"] + x["Open PO"]) < x["ROP"] else "NO ACTION",
        axis=1
    )

    # 🔥 Reorder Qty
    results["Reorder Qty"] = results.apply(
        lambda x: max(x["ROP"] - (x["Stock"] + x["Open PO"]), 0),
        axis=1
    )

    return results, {}