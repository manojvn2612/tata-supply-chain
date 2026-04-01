"""
module_1_lstm_demand_forecast.py
----------------------------------
ML Module 1: LSTM Demand Forecasting  (PyTorch)

Architecture:
  Input  : sequence of 30 days × 6 features
           [daily_demand, stock_cover_days, demand_rolling_slope,
            month_sin, month_cos, dow_sin]
  Model  : LSTM (2 layers, 128 hidden) → Dropout(0.2) → Linear → 14-day forecast
  Output : next 14 days of predicted daily demand (multi-step)

Why LSTM here vs statistics:
  - Learns non-linear seasonal patterns + demand momentum simultaneously
  - Cell state carries long-range memory (e.g. Q4 surge memory into Jan)
  - Outperforms ARIMA/ETS on datasets with multiple overlapping cycles

Run:  python module_1_lstm_demand_forecast.py
"""

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import MinMaxScaler
import joblib
from pathlib import Path

# ── Config ───────────────────────────────────────────────────────────────────
WINDOW       = 30
HORIZON      = 14
HIDDEN_SIZE  = 128
NUM_LAYERS   = 2
DROPOUT      = 0.2
EPOCHS       = 60
LR           = 1e-3
BATCH_SIZE   = 128
TRAIN_SPLIT  = 0.80
DEVICE       = "cuda" if torch.cuda.is_available() else "cpu"

FEATURES = [
    "daily_demand", "stock_cover_days", "demand_rolling_slope",
    "month_sin", "month_cos", "dow_sin"
]

MODELS_DIR = Path("models")
MODELS_DIR.mkdir(exist_ok=True)


# ── Dataset ───────────────────────────────────────────────────────────────────
class MRPSequenceDataset(Dataset):
    def __init__(self, X: np.ndarray, y: np.ndarray):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


# ── Model ─────────────────────────────────────────────────────────────────────
class DemandLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, horizon, dropout):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0.0,
            batch_first=True,
        )
        self.dropout = nn.Dropout(dropout)
        self.head    = nn.Linear(hidden_size, horizon)

    def forward(self, x):
        # x: (batch, seq_len, features)
        out, _ = self.lstm(x)
        out     = self.dropout(out[:, -1, :])   # last timestep
        return self.head(out)


# ── Helpers ───────────────────────────────────────────────────────────────────
def make_sequences(df: pd.DataFrame, scaler: MinMaxScaler, fit_scaler: bool):
    df = df.copy().sort_values("date").reset_index(drop=True)
    vals = df[FEATURES].values.astype(np.float32)

    if fit_scaler:
        vals = scaler.fit_transform(vals)
    else:
        vals = scaler.transform(vals)

    X, y = [], []
    for i in range(WINDOW, len(vals) - HORIZON):
        X.append(vals[i - WINDOW:i])
        # Target = raw (unscaled) demand for the next HORIZON days
        y.append(df["daily_demand"].values[i:i + HORIZON])

    return np.array(X), np.array(y)


def train_model(mat_id: str, df: pd.DataFrame) -> dict:
    print(f"\n  Training LSTM for {mat_id}...")
    df = df.sort_values("date").reset_index(drop=True)

    split_idx = int(len(df) * TRAIN_SPLIT)
    train_df  = df.iloc[:split_idx]
    val_df    = df.iloc[split_idx - WINDOW:]   # overlap window for continuity

    scaler = MinMaxScaler()
    X_train, y_train = make_sequences(train_df, scaler, fit_scaler=True)
    X_val,   y_val   = make_sequences(val_df,   scaler, fit_scaler=False)

    train_loader = DataLoader(MRPSequenceDataset(X_train, y_train), batch_size=BATCH_SIZE, shuffle=True)
    val_loader   = DataLoader(MRPSequenceDataset(X_val,   y_val),   batch_size=BATCH_SIZE)

    model = DemandLSTM(
        input_size=len(FEATURES),
        hidden_size=HIDDEN_SIZE,
        num_layers=NUM_LAYERS,
        horizon=HORIZON,
        dropout=DROPOUT,
    ).to(DEVICE)

    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=5, factor=0.5)
    criterion = nn.HuberLoss()   # robust to demand spikes vs plain MSE

    best_val_loss = float("inf")
    best_state    = None

    for epoch in range(1, EPOCHS + 1):
        # Train
        model.train()
        train_loss = 0
        for xb, yb in train_loader:
            xb, yb = xb.to(DEVICE), yb.to(DEVICE)
            optimizer.zero_grad()
            loss = criterion(model(xb), yb)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            train_loss += loss.item()

        # Validate
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for xb, yb in val_loader:
                xb, yb = xb.to(DEVICE), yb.to(DEVICE)
                val_loss += criterion(model(xb), yb).item()

        train_loss /= len(train_loader)
        val_loss   /= len(val_loader)
        scheduler.step(val_loss)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state    = {k: v.cpu().clone() for k, v in model.state_dict().items()}

        if epoch % 10 == 0:
            print(f"    Epoch {epoch:>3}/{EPOCHS}  train_loss={train_loss:.4f}  val_loss={val_loss:.4f}")

    # Save best
    model.load_state_dict(best_state)
    safe_id = mat_id.replace("-", "_")
    torch.save(model.state_dict(), MODELS_DIR / f"lstm_{safe_id}.pt")
    joblib.dump(scaler,            MODELS_DIR / f"lstm_scaler_{safe_id}.pkl")
    print(f"    Best val loss: {best_val_loss:.4f}  → saved to models/")

    return {"model": model, "scaler": scaler, "val_loss": best_val_loss}


def load_model(mat_id: str) -> tuple:
    safe_id = mat_id.replace("-", "_")
    scaler  = joblib.load(MODELS_DIR / f"lstm_scaler_{safe_id}.pkl")
    model   = DemandLSTM(len(FEATURES), HIDDEN_SIZE, NUM_LAYERS, HORIZON, DROPOUT)
    model.load_state_dict(torch.load(MODELS_DIR / f"lstm_{safe_id}.pt", map_location="cpu"))
    model.eval()
    return model, scaler


def forecast(mat_id: str, df: pd.DataFrame, model=None, scaler=None) -> dict:
    """Run inference on the last WINDOW days → 14-day forecast."""
    if model is None or scaler is None:
        model, scaler = load_model(mat_id)

    df = df.sort_values("date").reset_index(drop=True)
    last_window = df[FEATURES].values[-WINDOW:].astype(np.float32)
    last_window = scaler.transform(last_window)

    x = torch.tensor(last_window, dtype=torch.float32).unsqueeze(0)   # (1, 30, 6)
    with torch.no_grad():
        pred = model(x).numpy().flatten()

    pred = np.maximum(0, pred)   # demand can't be negative
    last_date = pd.to_datetime(df["date"].max())
    forecast_dates = pd.date_range(last_date + pd.Timedelta("1D"), periods=HORIZON)

    return {
        "material_id":     mat_id,
        "forecast_dates":  forecast_dates,
        "predicted_demand": pred.round(2),
        "14d_total":       round(pred.sum(), 1),
        "daily_avg":       round(pred.mean(), 2),
        "peak_day":        forecast_dates[pred.argmax()].strftime("%Y-%m-%d"),
        "peak_demand":     round(pred.max(), 2),
    }


# ── Main ──────────────────────────────────────────────────────────────────────
def run_lstm_forecast(data_path="data/training_data.csv", retrain=True):
    print("=" * 65)
    print("  MODULE 1: LSTM DEMAND FORECAST  (PyTorch)")
    print("=" * 65)
    print(f"  Device: {DEVICE}  |  Window: {WINDOW}d  |  Horizon: {HORIZON}d")
    print(f"  Architecture: LSTM({NUM_LAYERS}×{HIDDEN_SIZE}) → Dropout({DROPOUT}) → Linear({HORIZON})")

    df = pd.read_csv(data_path, parse_dates=["date"])
    all_forecasts = {}

    for mat_id in df["material_id"].unique():
        mat_df = df[df["material_id"] == mat_id].copy()

        if retrain:
            result = train_model(mat_id, mat_df)
            model, scaler = result["model"], result["scaler"]
        else:
            model, scaler = load_model(mat_id)

        fc = forecast(mat_id, mat_df, model, scaler)
        all_forecasts[mat_id] = fc

    # ── Print results ──
    print("\n" + "=" * 65)
    print("  14-DAY DEMAND FORECASTS")
    print("=" * 65)
    for mat_id, fc in all_forecasts.items():
        print(f"\n  {mat_id}")
        print(f"  Daily forecast: {fc['predicted_demand']}")
        print(f"  14-day total:   {fc['14d_total']} units")
        print(f"  Daily average:  {fc['daily_avg']} units/day")
        print(f"  Peak demand:    {fc['peak_demand']} on {fc['peak_day']}")

    return all_forecasts


if __name__ == "__main__":
    run_lstm_forecast(retrain=True)
