# """
# module_1_lstm_demand_forecast.py
# ----------------------------------
# ML Module 1: LSTM Demand Forecasting  (PyTorch)

# Architecture:
#   Input  : sequence of 30 days × 6 features
#            [daily_demand, stock_cover_days, demand_rolling_slope,
#             month_sin, month_cos, dow_sin]
#   Model  : LSTM (2 layers, 128 hidden) → Dropout(0.2) → Linear → 14-day forecast
#   Output : next 14 days of predicted daily demand (multi-step)

# Why LSTM here vs statistics:
#   - Learns non-linear seasonal patterns + demand momentum simultaneously
#   - Cell state carries long-range memory (e.g. Q4 surge memory into Jan)
#   - Outperforms ARIMA/ETS on datasets with multiple overlapping cycles

# Run:  python module_1_lstm_demand_forecast.py
# """

# import numpy as np
# import pandas as pd
# import torch
# import torch.nn as nn
# from torch.utils.data import Dataset, DataLoader
# from sklearn.preprocessing import MinMaxScaler
# import joblib
# from pathlib import Path

# # ── Config ───────────────────────────────────────────────────────────────────
# WINDOW       = 30
# HORIZON      = 14
# HIDDEN_SIZE  = 128
# NUM_LAYERS   = 2
# DROPOUT      = 0.2
# EPOCHS       = 60
# LR           = 1e-3
# BATCH_SIZE   = 128
# TRAIN_SPLIT  = 0.80
# DEVICE       = "cuda" if torch.cuda.is_available() else "cpu"

# FEATURES = [
#     "daily_demand", "stock_cover_days", "demand_rolling_slope",
#     "month_sin", "month_cos", "dow_sin"
# ]

# MODELS_DIR = Path("models")
# MODELS_DIR.mkdir(exist_ok=True)


# # ── Dataset ───────────────────────────────────────────────────────────────────
# class MRPSequenceDataset(Dataset):
#     def __init__(self, X: np.ndarray, y: np.ndarray):
#         self.X = torch.tensor(X, dtype=torch.float32)
#         self.y = torch.tensor(y, dtype=torch.float32)

#     def __len__(self):
#         return len(self.X)

#     def __getitem__(self, idx):
#         return self.X[idx], self.y[idx]


# # ── Model ─────────────────────────────────────────────────────────────────────
# class DemandLSTM(nn.Module):
#     def __init__(self, input_size, hidden_size, num_layers, horizon, dropout):
#         super().__init__()
#         self.lstm = nn.LSTM(
#             input_size=input_size,
#             hidden_size=hidden_size,
#             num_layers=num_layers,
#             dropout=dropout if num_layers > 1 else 0.0,
#             batch_first=True,
#         )
#         self.dropout = nn.Dropout(dropout)
#         self.head    = nn.Linear(hidden_size, horizon)

#     def forward(self, x):
#         # x: (batch, seq_len, features)
#         out, _ = self.lstm(x)
#         out     = self.dropout(out[:, -1, :])   # last timestep
#         return self.head(out)


# # ── Helpers ───────────────────────────────────────────────────────────────────
# def make_sequences(df: pd.DataFrame, scaler: MinMaxScaler, fit_scaler: bool):
#     df = df.copy().sort_values("date").reset_index(drop=True)
#     vals = df[FEATURES].values.astype(np.float32)

#     if fit_scaler:
#         vals = scaler.fit_transform(vals)
#     else:
#         vals = scaler.transform(vals)

#     X, y = [], []
#     for i in range(WINDOW, len(vals) - HORIZON):
#         X.append(vals[i - WINDOW:i])
#         # Target = raw (unscaled) demand for the next HORIZON days
#         y.append(df["daily_demand"].values[i:i + HORIZON])

#     return np.array(X), np.array(y)


# def train_model(mat_id: str, df: pd.DataFrame) -> dict:
#     print(f"\n  Training LSTM for {mat_id}...")
#     df = df.sort_values("date").reset_index(drop=True)

#     split_idx = int(len(df) * TRAIN_SPLIT)
#     train_df  = df.iloc[:split_idx]
#     val_df    = df.iloc[split_idx - WINDOW:]   # overlap window for continuity

#     scaler = MinMaxScaler()
#     X_train, y_train = make_sequences(train_df, scaler, fit_scaler=True)
#     X_val,   y_val   = make_sequences(val_df,   scaler, fit_scaler=False)

#     train_loader = DataLoader(MRPSequenceDataset(X_train, y_train), batch_size=BATCH_SIZE, shuffle=True)
#     val_loader   = DataLoader(MRPSequenceDataset(X_val,   y_val),   batch_size=BATCH_SIZE)

#     model = DemandLSTM(
#         input_size=len(FEATURES),
#         hidden_size=HIDDEN_SIZE,
#         num_layers=NUM_LAYERS,
#         horizon=HORIZON,
#         dropout=DROPOUT,
#     ).to(DEVICE)

#     optimizer = torch.optim.Adam(model.parameters(), lr=LR)
#     scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=5, factor=0.5)
#     criterion = nn.HuberLoss()   # robust to demand spikes vs plain MSE

#     best_val_loss = float("inf")
#     best_state    = None

#     for epoch in range(1, EPOCHS + 1):
#         # Train
#         model.train()
#         train_loss = 0
#         for xb, yb in train_loader:
#             xb, yb = xb.to(DEVICE), yb.to(DEVICE)
#             optimizer.zero_grad()
#             loss = criterion(model(xb), yb)
#             loss.backward()
#             nn.utils.clip_grad_norm_(model.parameters(), 1.0)
#             optimizer.step()
#             train_loss += loss.item()

#         # Validate
#         model.eval()
#         val_loss = 0
#         with torch.no_grad():
#             for xb, yb in val_loader:
#                 xb, yb = xb.to(DEVICE), yb.to(DEVICE)
#                 val_loss += criterion(model(xb), yb).item()

#         train_loss /= len(train_loader)
#         val_loss   /= len(val_loader)
#         scheduler.step(val_loss)

#         if val_loss < best_val_loss:
#             best_val_loss = val_loss
#             best_state    = {k: v.cpu().clone() for k, v in model.state_dict().items()}

#         if epoch % 10 == 0:
#             print(f"    Epoch {epoch:>3}/{EPOCHS}  train_loss={train_loss:.4f}  val_loss={val_loss:.4f}")

#     # Save best
#     model.load_state_dict(best_state)
#     safe_id = mat_id.replace("-", "_")
#     torch.save(model.state_dict(), MODELS_DIR / f"lstm_{safe_id}.pt")
#     joblib.dump(scaler,            MODELS_DIR / f"lstm_scaler_{safe_id}.pkl")
#     print(f"    Best val loss: {best_val_loss:.4f}  → saved to models/")

#     return {"model": model, "scaler": scaler, "val_loss": best_val_loss}


# def load_model(mat_id: str) -> tuple:
#     safe_id = mat_id.replace("-", "_")
#     scaler  = joblib.load(MODELS_DIR / f"lstm_scaler_{safe_id}.pkl")
#     model   = DemandLSTM(len(FEATURES), HIDDEN_SIZE, NUM_LAYERS, HORIZON, DROPOUT)
#     model.load_state_dict(torch.load(MODELS_DIR / f"lstm_{safe_id}.pt", map_location="cpu"))
#     model.eval()
#     return model, scaler


# def forecast(mat_id: str, df: pd.DataFrame, model=None, scaler=None) -> dict:
#     """Run inference on the last WINDOW days → 14-day forecast."""
#     if model is None or scaler is None:
#         model, scaler = load_model(mat_id)

#     df = df.sort_values("date").reset_index(drop=True)
#     last_window = df[FEATURES].values[-WINDOW:].astype(np.float32)
#     last_window = scaler.transform(last_window)

#     x = torch.tensor(last_window, dtype=torch.float32).unsqueeze(0)   # (1, 30, 6)
#     with torch.no_grad():
#         pred = model(x).numpy().flatten()

#     pred = np.maximum(0, pred)   # demand can't be negative
#     last_date = pd.to_datetime(df["date"].max())
#     forecast_dates = pd.date_range(last_date + pd.Timedelta("1D"), periods=HORIZON)

#     return {
#         "material_id":     mat_id,
#         "forecast_dates":  forecast_dates,
#         "predicted_demand": pred.round(2),
#         "14d_total":       round(pred.sum(), 1),
#         "daily_avg":       round(pred.mean(), 2),
#         "peak_day":        forecast_dates[pred.argmax()].strftime("%Y-%m-%d"),
#         "peak_demand":     round(pred.max(), 2),
#     }


# # ── Main ──────────────────────────────────────────────────────────────────────
# def run_lstm_forecast(data_path="data/training_data.csv", retrain=True):
#     print("=" * 65)
#     print("  MODULE 1: LSTM DEMAND FORECAST  (PyTorch)")
#     print("=" * 65)
#     print(f"  Device: {DEVICE}  |  Window: {WINDOW}d  |  Horizon: {HORIZON}d")
#     print(f"  Architecture: LSTM({NUM_LAYERS}×{HIDDEN_SIZE}) → Dropout({DROPOUT}) → Linear({HORIZON})")

#     df = pd.read_csv(data_path, parse_dates=["date"])
#     all_forecasts = {}

#     for mat_id in df["material_id"].unique():
#         mat_df = df[df["material_id"] == mat_id].copy()

#         if retrain:
#             result = train_model(mat_id, mat_df)
#             model, scaler = result["model"], result["scaler"]
#         else:
#             model, scaler = load_model(mat_id)

#         fc = forecast(mat_id, mat_df, model, scaler)
#         all_forecasts[mat_id] = fc

#     # ── Print results ──
#     print("\n" + "=" * 65)
#     print("  14-DAY DEMAND FORECASTS")
#     print("=" * 65)
#     for mat_id, fc in all_forecasts.items():
#         print(f"\n  {mat_id}")
#         print(f"  Daily forecast: {fc['predicted_demand']}")
#         print(f"  14-day total:   {fc['14d_total']} units")
#         print(f"  Daily average:  {fc['daily_avg']} units/day")
#         print(f"  Peak demand:    {fc['peak_demand']} on {fc['peak_day']}")

#     return all_forecasts


# if __name__ == "__main__":
#     run_lstm_forecast(retrain=True)
"""
module_1_global_lstm_forecast.py
----------------------------------
ML Module 1 (v2): GLOBAL LSTM Demand Forecast  (PyTorch)

Replaces 3 separate per-material LSTMs with ONE global model that:
  - Trains on all materials simultaneously (3× more data)
  - Uses a learned material embedding to distinguish FG / SFG / RM behaviour
  - Learns cross-material patterns (FG demand surge → expect RM demand surge)
  - Scales to 100s of materials without retraining separate models
  - Cold-starts on brand-new materials by inheriting shared patterns

Architecture:
  ┌──────────────────────────────────────────────────────┐
  │  material_id  →  Embedding(num_materials, embed_dim) │  ← learned per-material vector
  │  time_features → LSTM(2 layers, 128 hidden)          │  ← shared sequence encoder
  │  [lstm_out ; embedding] → Linear(128+embed → 64)     │  ← fusion layer
  │  → Dropout(0.2) → Linear(64 → horizon)               │  ← multi-step output
  └──────────────────────────────────────────────────────┘

Input per timestep:  [daily_demand, stock_cover_days, demand_rolling_slope,
                      month_sin, month_cos, dow_sin]  (6 features)
Sequence length:     30 days
Forecast horizon:    14 days (multi-step)

Why global > per-material:
  - 5,475 rows vs 1,825 rows per model → better generalisation
  - Embedding learns "RM-200010 is high-volume, volatile" automatically
  - Shared LSTM weights capture universal seasonality patterns
  - One .pt file to deploy, one API to call

Run:
  python module_1_global_lstm_forecast.py              # train + forecast
  python module_1_global_lstm_forecast.py --no-retrain # load saved, forecast only
"""

import sys
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import MinMaxScaler
import joblib
from pathlib import Path

# ── Config ────────────────────────────────────────────────────────────────────
WINDOW       = 30       # days of history fed to LSTM
HORIZON      = 14       # days ahead to forecast
EMBED_DIM    = 16       # material embedding size
HIDDEN_SIZE  = 128
NUM_LAYERS   = 2
DROPOUT      = 0.2
EPOCHS       = 80
LR           = 1e-3
BATCH_SIZE   = 128
TRAIN_SPLIT  = 0.80
DEVICE       = "cuda" if torch.cuda.is_available() else "cpu"

TIME_FEATURES = [
    "daily_demand",
    "stock_cover_days",
    "demand_rolling_slope",
    "month_sin",
    "month_cos",
    "dow_sin",
]

MODELS_DIR = Path("models")
MODELS_DIR.mkdir(exist_ok=True)

MODEL_PATH   = MODELS_DIR / "global_lstm.pt"
SCALER_PATH  = MODELS_DIR / "global_lstm_scaler.pkl"
ENCODER_PATH = MODELS_DIR / "global_lstm_material_encoder.pkl"


# ── Dataset ───────────────────────────────────────────────────────────────────

class GlobalMRPDataset(Dataset):
    """
    Builds (material_idx, sequence, target) triplets from ALL materials.
    material_idx is an integer ID fed into the embedding layer.
    """
    def __init__(self, df: pd.DataFrame, scaler: MinMaxScaler,
                 mat_encoder: dict, fit_scaler: bool = False):
        self.samples = []

        # Scale time-features globally across all materials
        feat_vals = df[TIME_FEATURES].values.astype(np.float32)
        if fit_scaler:
            feat_vals = scaler.fit_transform(feat_vals)
        else:
            feat_vals = scaler.transform(feat_vals)

        df = df.copy()
        df[TIME_FEATURES] = feat_vals

        for mat_id, mat_df in df.groupby("material_id"):
            mat_df  = mat_df.sort_values("date").reset_index(drop=True)
            mat_idx = mat_encoder[mat_id]
            vals    = mat_df[TIME_FEATURES].values.astype(np.float32)
            demand  = mat_df["daily_demand"].values.astype(np.float32)   # raw target

            for i in range(WINDOW, len(vals) - HORIZON):
                seq    = vals[i - WINDOW : i]           # (30, 6)
                target = demand[i : i + HORIZON]        # (14,)  — raw, unscaled
                self.samples.append((mat_idx, seq, target))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        mat_idx, seq, target = self.samples[idx]
        return (
            torch.tensor(mat_idx, dtype=torch.long),
            torch.tensor(seq,     dtype=torch.float32),
            torch.tensor(target,  dtype=torch.float32),
        )


# ── Model ─────────────────────────────────────────────────────────────────────

class GlobalDemandLSTM(nn.Module):
    """
    Shared LSTM encoder + per-material embedding → multi-step demand forecast.

    The embedding vector is concatenated with the LSTM's last hidden state
    before the output head. This lets the model produce different forecasts
    for different materials while sharing all sequence-learning weights.
    """
    def __init__(self, num_materials: int, embed_dim: int,
                 input_size: int, hidden_size: int,
                 num_layers: int, horizon: int, dropout: float):
        super().__init__()

        # Material-specific learned representation
        self.embedding = nn.Embedding(num_materials, embed_dim)

        # Shared sequence encoder
        self.lstm = nn.LSTM(
            input_size  = input_size,
            hidden_size = hidden_size,
            num_layers  = num_layers,
            dropout     = dropout if num_layers > 1 else 0.0,
            batch_first = True,
        )

        # Fusion + output head
        fusion_dim = hidden_size + embed_dim
        self.head = nn.Sequential(
            nn.Linear(fusion_dim, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, horizon),
        )

    def forward(self, mat_idx: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        """
        mat_idx : (batch,)          — integer material IDs
        x       : (batch, seq, feat)— time-series features
        returns : (batch, horizon)  — demand forecast
        """
        emb          = self.embedding(mat_idx)               # (batch, embed_dim)
        lstm_out, _  = self.lstm(x)                          # (batch, seq, hidden)
        last_hidden  = lstm_out[:, -1, :]                    # (batch, hidden)
        fused        = torch.cat([last_hidden, emb], dim=1)  # (batch, hidden+embed)
        return self.head(fused)                              # (batch, horizon)


# ── Training ──────────────────────────────────────────────────────────────────

def train_global_model(data_path: str = "data/training_data.csv") -> dict:
    print(f"\n  Training GLOBAL LSTM  (device: {DEVICE})")
    print(f"  Architecture: Embedding({'{'}num_mats{'}'},{EMBED_DIM}) + "
          f"LSTM({NUM_LAYERS}×{HIDDEN_SIZE}) → Linear({HORIZON})")

    df = pd.read_csv(data_path, parse_dates=["date"])

    # Build material → integer encoder
    materials   = sorted(df["material_id"].unique())
    mat_encoder = {m: i for i, m in enumerate(materials)}
    num_mats    = len(materials)
    print(f"  Materials ({num_mats}): {materials}")

    # Train / val split — last 20% of dates (time-ordered, no leakage)
    cutoff  = df["date"].quantile(TRAIN_SPLIT)
    train_df = df[df["date"] <= cutoff].copy()
    val_df   = df[df["date"] >  cutoff].copy()

    # For validation continuity, include the last WINDOW rows of training
    # per material so sequences can be formed across the split boundary
    overlap_rows = []
    for mat_id in materials:
        tail = df[df["material_id"] == mat_id].sort_values("date").tail(WINDOW)
        overlap_rows.append(tail)
    overlap_df = pd.concat(overlap_rows)
    val_df_ext = pd.concat([overlap_df, val_df]).sort_values(["material_id","date"])

    scaler   = MinMaxScaler()
    train_ds = GlobalMRPDataset(train_df,   scaler, mat_encoder, fit_scaler=True)
    val_ds   = GlobalMRPDataset(val_df_ext, scaler, mat_encoder, fit_scaler=False)

    print(f"  Train samples: {len(train_ds):,}  |  Val samples: {len(val_ds):,}")

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,  num_workers=0)
    val_loader   = DataLoader(val_ds,   batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

    model = GlobalDemandLSTM(
        num_materials = num_mats,
        embed_dim     = EMBED_DIM,
        input_size    = len(TIME_FEATURES),
        hidden_size   = HIDDEN_SIZE,
        num_layers    = NUM_LAYERS,
        horizon       = HORIZON,
        dropout       = DROPOUT,
    ).to(DEVICE)

    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  Trainable parameters: {total_params:,}")

    optimizer = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)
    criterion = nn.HuberLoss(delta=1.0)   # robust to demand spikes

    best_val   = float("inf")
    best_state = None
    history    = {"train": [], "val": []}

    for epoch in range(1, EPOCHS + 1):
        # ── Train ──
        model.train()
        train_loss = 0.0
        for mat_idx, x, y in train_loader:
            mat_idx = mat_idx.to(DEVICE)
            x       = x.to(DEVICE)
            y       = y.to(DEVICE)
            optimizer.zero_grad()
            loss = criterion(model(mat_idx, x), y)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            train_loss += loss.item()

        # ── Validate ──
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for mat_idx, x, y in val_loader:
                mat_idx = mat_idx.to(DEVICE)
                x, y    = x.to(DEVICE), y.to(DEVICE)
                val_loss += criterion(model(mat_idx, x), y).item()

        train_loss /= max(len(train_loader), 1)
        val_loss   /= max(len(val_loader), 1)
        scheduler.step()

        history["train"].append(train_loss)
        history["val"].append(val_loss)

        if val_loss < best_val:
            best_val   = val_loss
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}

        if epoch % 10 == 0:
            lr_now = scheduler.get_last_lr()[0]
            print(f"    Epoch {epoch:>3}/{EPOCHS}  "
                  f"train={train_loss:.4f}  val={val_loss:.4f}  "
                  f"lr={lr_now:.2e}")

    # Save best checkpoint
    model.load_state_dict(best_state)
    torch.save({
        "model_state":   best_state,
        "mat_encoder":   mat_encoder,
        "num_materials": num_mats,
        "config": {
            "embed_dim":   EMBED_DIM,
            "hidden_size": HIDDEN_SIZE,
            "num_layers":  NUM_LAYERS,
            "horizon":     HORIZON,
            "dropout":     DROPOUT,
            "features":    TIME_FEATURES,
        },
    }, MODEL_PATH)
    joblib.dump(scaler, SCALER_PATH)

    print(f"\n  Best val loss: {best_val:.4f}")
    print(f"  Saved → {MODEL_PATH}  |  {SCALER_PATH}")

    return {"model": model, "scaler": scaler,
            "mat_encoder": mat_encoder, "best_val": best_val}


# ── Inference ─────────────────────────────────────────────────────────────────

def load_global_model():
    """Load saved global model and scaler from disk."""
    checkpoint   = torch.load(MODEL_PATH, map_location="cpu")
    cfg          = checkpoint["config"]
    mat_encoder  = checkpoint["mat_encoder"]
    num_mats     = checkpoint["num_materials"]

    model = GlobalDemandLSTM(
        num_materials = num_mats,
        embed_dim     = cfg["embed_dim"],
        input_size    = len(cfg["features"]),
        hidden_size   = cfg["hidden_size"],
        num_layers    = cfg["num_layers"],
        horizon       = cfg["horizon"],
        dropout       = cfg["dropout"],
    )
    model.load_state_dict(checkpoint["model_state"])
    model.eval()

    scaler = joblib.load(SCALER_PATH)
    return model, scaler, mat_encoder


def forecast_all(df: pd.DataFrame = None,
                 data_path: str = "data/training_data.csv",
                 model=None, scaler=None, mat_encoder=None) -> dict:
    """
    Run inference for every material using the last WINDOW days of each.
    Returns a dict of {material_id: forecast_result}.
    """
    if model is None:
        model, scaler, mat_encoder = load_global_model()

    if df is None:
        df = pd.read_csv(data_path, parse_dates=["date"])

    model.eval()
    results = {}

    for mat_id in sorted(df["material_id"].unique()):
        if mat_id not in mat_encoder:
            print(f"  ⚠️  {mat_id} not in training vocab — skipping")
            continue

        mat_df = df[df["material_id"] == mat_id].sort_values("date")
        window = mat_df[TIME_FEATURES].values[-WINDOW:].astype(np.float32)
        window = scaler.transform(window)

        mat_idx_t = torch.tensor([mat_encoder[mat_id]], dtype=torch.long)
        x_t       = torch.tensor(window, dtype=torch.float32).unsqueeze(0)  # (1,30,6)

        with torch.no_grad():
            pred = model(mat_idx_t, x_t).numpy().flatten()

        pred = np.maximum(0, pred)
        last_date      = pd.to_datetime(mat_df["date"].max())
        forecast_dates = pd.date_range(last_date + pd.Timedelta("1D"), periods=HORIZON)

        results[mat_id] = {
            "material_id":      mat_id,
            "forecast_dates":   forecast_dates,
            "predicted_demand": pred.round(3),
            "14d_total":        round(float(pred.sum()), 1),
            "daily_avg":        round(float(pred.mean()), 3),
            "peak_day":         forecast_dates[int(pred.argmax())].strftime("%Y-%m-%d"),
            "peak_demand":      round(float(pred.max()), 3),
        }

    return results


def forecast_single(mat_id: str, last_30_days_df: pd.DataFrame,
                    model=None, scaler=None, mat_encoder=None) -> dict:
    """
    Forecast for ONE material given a DataFrame of its last 30+ days.
    Useful when called from other modules or a live system.
    """
    if model is None:
        model, scaler, mat_encoder = load_global_model()

    mat_df = last_30_days_df.sort_values("date").tail(WINDOW)
    window = mat_df[TIME_FEATURES].values.astype(np.float32)
    window = scaler.transform(window)

    mat_idx_t = torch.tensor([mat_encoder[mat_id]], dtype=torch.long)
    x_t       = torch.tensor(window, dtype=torch.float32).unsqueeze(0)

    with torch.no_grad():
        pred = model(mat_idx_t, x_t).numpy().flatten()

    pred = np.maximum(0, pred)
    last_date      = pd.to_datetime(mat_df["date"].max())
    forecast_dates = pd.date_range(last_date + pd.Timedelta("1D"), periods=HORIZON)

    return {
        "material_id":      mat_id,
        "forecast_dates":   forecast_dates,
        "predicted_demand": pred.round(3),
        "14d_total":        round(float(pred.sum()), 1),
        "daily_avg":        round(float(pred.mean()), 3),
        "peak_day":         forecast_dates[int(pred.argmax())].strftime("%Y-%m-%d"),
        "peak_demand":      round(float(pred.max()), 3),
    }


# ── Embedding inspector ───────────────────────────────────────────────────────

def inspect_embeddings(model, mat_encoder: dict):
    """
    Print learned material embeddings — shows what the model
    has learned to distinguish between materials.
    High values in a dimension = strong signal for that material.
    """
    print("\n  LEARNED MATERIAL EMBEDDINGS")
    print(f"  (each material is a {EMBED_DIM}-dim vector the model uses to specialise)")
    print()

    weights = model.embedding.weight.detach().numpy()
    for mat_id, idx in sorted(mat_encoder.items(), key=lambda x: x[1]):
        vec      = weights[idx]
        top_dims = np.argsort(np.abs(vec))[-3:][::-1]
        print(f"  {mat_id}  →  norm={np.linalg.norm(vec):.3f}  "
              f"top dims: {[(int(d), round(float(vec[d]),3)) for d in top_dims]}")


# ── Main ──────────────────────────────────────────────────────────────────────

def run_global_lstm_forecast(data_path: str = "data/training_data.csv",
                              retrain: bool = True) -> dict:
    print("=" * 65)
    print("  MODULE 1 (v2): GLOBAL LSTM DEMAND FORECAST")
    print("=" * 65)
    print(f"  Mode: {'TRAIN + FORECAST' if retrain else 'LOAD + FORECAST'}")
    print(f"  Device: {DEVICE}")

    if retrain or not MODEL_PATH.exists():
        train_result = train_global_model(data_path)
        model        = train_result["model"]
        scaler       = train_result["scaler"]
        mat_encoder  = train_result["mat_encoder"]
    else:
        print("  Loading saved global model...")
        model, scaler, mat_encoder = load_global_model()

    # Inspect what embeddings learned
    inspect_embeddings(model, mat_encoder)

    # Run forecasts for all materials
    df       = pd.read_csv(data_path, parse_dates=["date"])
    forecasts = forecast_all(df, model=model, scaler=scaler, mat_encoder=mat_encoder)

    # ── Print results ──
    print("\n" + "=" * 65)
    print(f"  14-DAY FORECASTS  (ONE GLOBAL MODEL, {len(forecasts)} materials)")
    print("=" * 65)
    for mat_id, fc in forecasts.items():
        dates_short = [d.strftime("%d%b") for d in fc["forecast_dates"]]
        print(f"\n  {mat_id}")
        print(f"  Dates:   {dates_short}")
        print(f"  Demand:  {fc['predicted_demand'].tolist()}")
        print(f"  14d total: {fc['14d_total']}  |  daily avg: {fc['daily_avg']}  "
              f"|  peak: {fc['peak_demand']} on {fc['peak_day']}")

    return forecasts


if __name__ == "__main__":
    retrain = "--no-retrain" not in sys.argv
    run_global_lstm_forecast(retrain=retrain)