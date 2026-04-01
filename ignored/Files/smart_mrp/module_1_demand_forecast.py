"""
module_1_demand_forecast.py
----------------------------
ML Module 1: LSTM Demand Forecasting (PyTorch)

Architecture: LSTM(2x128) -> Dropout(0.2) -> Linear(14)
Input: 30-day window of [demand, stock_cover, slope, month_sin, month_cos, dow_sin]
Output: 14-day daily demand forecast

Confidence score: rolling backtest MAE converted to ± uncertainty band
"""

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error
import joblib
from pathlib import Path

WINDOW      = 30
HORIZON     = 14
HIDDEN_SIZE = 128
NUM_LAYERS  = 2
DROPOUT     = 0.2
EPOCHS      = 60
LR          = 1e-3
BATCH_SIZE  = 128
TRAIN_SPLIT = 0.80
DEVICE      = "cuda" if torch.cuda.is_available() else "cpu"

FEATURES = [
    "daily_demand", "stock_cover_days", "demand_rolling_slope",
    "month_sin", "month_cos", "dow_sin"
]

MODELS_DIR = Path("models")
MODELS_DIR.mkdir(exist_ok=True)


class MRPSequenceDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32)
    def __len__(self):  return len(self.X)
    def __getitem__(self, i): return self.X[i], self.y[i]


class DemandLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, horizon, dropout):
        super().__init__()
        self.lstm    = nn.LSTM(input_size, hidden_size, num_layers,
                               dropout=dropout if num_layers > 1 else 0.0,
                               batch_first=True)
        self.dropout = nn.Dropout(dropout)
        self.head    = nn.Linear(hidden_size, horizon)

    def forward(self, x):
        out, _ = self.lstm(x)
        return self.head(self.dropout(out[:, -1, :]))


def _make_sequences(df, scaler, fit_scaler):
    df   = df.sort_values("date").reset_index(drop=True)
    vals = df[FEATURES].values.astype(np.float32)
    vals = scaler.fit_transform(vals) if fit_scaler else scaler.transform(vals)
    X, y = [], []
    for i in range(WINDOW, len(vals) - HORIZON):
        X.append(vals[i - WINDOW:i])
        y.append(df["daily_demand"].values[i:i + HORIZON])
    return np.array(X), np.array(y)


def _compute_scores(model, scaler, df):
    """
    Full scoring suite on last 10% held-out data.
    Metrics: MAE, RMSE, MAPE, R², Huber loss, bias, directional accuracy.
    """
    from sklearn.metrics import r2_score as sk_r2
    df    = df.sort_values("date").reset_index(drop=True)
    split = int(len(df) * 0.90)
    test_df = df.iloc[split - WINDOW:]
    try:
        full_scaler = MinMaxScaler().fit(df[FEATURES].values)
        X, y = _make_sequences(test_df, full_scaler, False)
        if len(X) == 0:
            return _empty_scores()
        preds = []
        model.eval()
        with torch.no_grad():
            for i in range(len(X)):
                xb = torch.tensor(X[i:i+1], dtype=torch.float32)
                preds.append(model(xb).numpy().flatten())

        y_true = y[:len(preds)].flatten()
        y_pred = np.maximum(0, np.array(preds).flatten())

        mae    = float(mean_absolute_error(y_true, y_pred))
        rmse   = float(np.sqrt(mean_squared_error(y_true, y_pred)))
        r2     = float(sk_r2(y_true, y_pred))

        # MAPE — skip near-zero actuals
        mask   = y_true > 0.01
        mape   = float(np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100) if mask.sum() else 0.0

        # Huber loss (delta=1.0, same as training criterion)
        delta    = 1.0
        abs_err  = np.abs(y_true - y_pred)
        huber    = float(np.where(abs_err <= delta,
                                  0.5 * abs_err**2,
                                  delta * (abs_err - 0.5 * delta)).mean())

        # Bias: signed mean error (positive = model over-predicts)
        bias     = float(np.mean(y_pred - y_true))

        # Directional accuracy: does the model predict the correct direction of change?
        if len(y_true) > 1:
            dir_true = np.sign(np.diff(y_true))
            dir_pred = np.sign(np.diff(y_pred))
            dir_acc  = float(np.mean(dir_true == dir_pred))
        else:
            dir_acc = float("nan")

        mean_d = float(y_true.mean()) or 1.0
        conf   = round(max(50, min(99, 100 - (mae / mean_d * 100))), 1)

        return {
            "mae":                round(mae, 4),
            "rmse":               round(rmse, 4),
            "mape_pct":           round(mape, 2),
            "r2_score":           round(r2, 4),
            "huber_loss":         round(huber, 4),
            "bias":               round(bias, 4),
            "directional_acc":    round(dir_acc, 4) if not np.isnan(dir_acc) else None,
            "uncertainty_band":   f"±{mae:.3f} units/day",
            "confidence_pct":     conf,
            "n_test_samples":     int(len(y_true)),
            "grade": ("A" if mae / mean_d < 0.05 else
                      "B" if mae / mean_d < 0.10 else
                      "C" if mae / mean_d < 0.20 else "D"),
        }
    except Exception as e:
        return _empty_scores()


def _empty_scores():
    return {"mae": None, "rmse": None, "mape_pct": None, "r2_score": None,
            "huber_loss": None, "bias": None, "directional_acc": None,
            "uncertainty_band": "±unknown", "confidence_pct": 70.0,
            "n_test_samples": 0, "grade": "N/A"}


# keep old name as alias so mrp_agent.py still works
_compute_confidence = _compute_scores


def train_model(mat_id: str, df: pd.DataFrame) -> dict:
    print(f"    Training LSTM for {mat_id}...", end=" ", flush=True)
    df = df.sort_values("date").reset_index(drop=True)
    split_idx = int(len(df) * TRAIN_SPLIT)
    scaler    = MinMaxScaler()
    X_tr, y_tr = _make_sequences(df.iloc[:split_idx], scaler, True)
    X_vl, y_vl = _make_sequences(df.iloc[split_idx - WINDOW:], scaler, False)

    tr_loader = DataLoader(MRPSequenceDataset(X_tr, y_tr), BATCH_SIZE, shuffle=True)
    vl_loader = DataLoader(MRPSequenceDataset(X_vl, y_vl), BATCH_SIZE)

    model     = DemandLSTM(len(FEATURES), HIDDEN_SIZE, NUM_LAYERS, HORIZON, DROPOUT).to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=5, factor=0.5)
    criterion = nn.HuberLoss()

    best_loss, best_state  = float("inf"), None
    best_epoch             = 1
    patience_counter       = 0
    EARLY_STOP_PATIENCE    = 15          # stop if val_loss doesn't improve for 15 epochs
    train_loss_curve       = []
    val_loss_curve         = []
    lr_curve               = []
    grad_norm_curve        = []

    for epoch in range(1, EPOCHS + 1):
        # ── Train ──
        model.train()
        epoch_train_loss = 0.0
        epoch_grad_norm  = 0.0
        n_batches        = 0
        for xb, yb in tr_loader:
            xb, yb = xb.to(DEVICE), yb.to(DEVICE)
            optimizer.zero_grad()
            loss = criterion(model(xb), yb)
            loss.backward()
            # Gradient norm before clipping — measures learning signal strength
            grad_norm = float(
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            )
            optimizer.step()
            epoch_train_loss += loss.item()
            epoch_grad_norm  += grad_norm
            n_batches        += 1

        epoch_train_loss /= max(n_batches, 1)
        epoch_grad_norm  /= max(n_batches, 1)

        # ── Validate ──
        model.eval()
        epoch_val_loss = 0.0
        with torch.no_grad():
            for xb, yb in vl_loader:
                xb, yb = xb.to(DEVICE), yb.to(DEVICE)
                epoch_val_loss += criterion(model(xb), yb).item()
        epoch_val_loss /= max(len(vl_loader), 1)

        # Current LR from optimizer
        current_lr = optimizer.param_groups[0]["lr"]

        # Record curves
        train_loss_curve.append(round(epoch_train_loss, 6))
        val_loss_curve.append(round(epoch_val_loss, 6))
        lr_curve.append(round(current_lr, 8))
        grad_norm_curve.append(round(epoch_grad_norm, 4))

        scheduler.step(epoch_val_loss)

        # Best model tracking
        if epoch_val_loss < best_loss:
            best_loss    = epoch_val_loss
            best_epoch   = epoch
            best_state   = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            patience_counter = 0
        else:
            patience_counter += 1

        if epoch % 10 == 0:
            print(f"      Epoch {epoch:>3}/{EPOCHS}  "
                  f"train_loss={epoch_train_loss:.4f}  "
                  f"val_loss={epoch_val_loss:.4f}  "
                  f"lr={current_lr:.2e}  "
                  f"grad_norm={epoch_grad_norm:.3f}")

        # Early stopping
        if patience_counter >= EARLY_STOP_PATIENCE:
            print(f"      Early stop at epoch {epoch} (no improvement for {EARLY_STOP_PATIENCE} epochs)")
            break

    model.load_state_dict(best_state)
    safe_id = mat_id.replace("-", "_")
    torch.save(model.state_dict(), MODELS_DIR / f"lstm_{safe_id}.pt")
    joblib.dump(scaler, MODELS_DIR / f"lstm_scaler_{safe_id}.pkl")

    # ── Training summary ──
    final_train  = train_loss_curve[-1]
    final_val    = val_loss_curve[-1]
    overfit_gap  = round(final_val - final_train, 6)
    converged_at = next((i+1 for i, (t, v) in enumerate(zip(train_loss_curve, val_loss_curve))
                         if v <= best_loss * 1.01), len(train_loss_curve))

    training_scores = {
        "best_val_loss":      round(best_loss, 6),
        "best_epoch":         best_epoch,
        "final_train_loss":   round(final_train, 6),
        "final_val_loss":     round(final_val, 6),
        "overfit_gap":        overfit_gap,           # positive = overfitting
        "converged_at_epoch": converged_at,
        "early_stopped":      patience_counter >= EARLY_STOP_PATIENCE,
        "epochs_run":         len(train_loss_curve),
        "final_lr":           lr_curve[-1],
        "avg_grad_norm":      round(float(np.mean(grad_norm_curve)), 4),
        "min_grad_norm":      round(float(np.min(grad_norm_curve)), 4),
        "max_grad_norm":      round(float(np.max(grad_norm_curve)), 4),
        # Store sampled curves (every 5 epochs) to keep output manageable
        "train_loss_curve":   train_loss_curve[::5],
        "val_loss_curve":     val_loss_curve[::5],
        "lr_curve":           lr_curve[::5],
    }

    print(f"    best_val_loss={best_loss:.4f} at epoch {best_epoch}  "
          f"overfit_gap={overfit_gap:+.4f}  "
          f"early_stopped={training_scores['early_stopped']} ✓")

    return {
        "model":          model,
        "scaler":         scaler,
        "training_scores": training_scores,
    }


def load_model(mat_id: str):
    safe_id = mat_id.replace("-", "_")
    scaler  = joblib.load(MODELS_DIR / f"lstm_scaler_{safe_id}.pkl")
    model   = DemandLSTM(len(FEATURES), HIDDEN_SIZE, NUM_LAYERS, HORIZON, DROPOUT)
    model.load_state_dict(torch.load(MODELS_DIR / f"lstm_{safe_id}.pt", map_location="cpu"))
    model.eval()
    return model, scaler


def forecast(mat_id: str, df: pd.DataFrame, model=None, scaler=None,
             training_scores: dict = None) -> dict:
    if model is None:
        model, scaler = load_model(mat_id)
    df = df.sort_values("date").reset_index(drop=True)
    last = scaler.transform(df[FEATURES].values[-WINDOW:].astype(np.float32))
    x    = torch.tensor(last, dtype=torch.float32).unsqueeze(0)
    with torch.no_grad():
        pred = np.maximum(0, model(x).numpy().flatten())
    eval_scores = _compute_scores(model, scaler, df)
    last_date      = pd.to_datetime(df["date"].max())
    forecast_dates = pd.date_range(last_date + pd.Timedelta("1D"), periods=HORIZON)
    return {
        "material_id":       mat_id,
        "forecast_dates":    [d.strftime("%Y-%m-%d") for d in forecast_dates],
        "predicted_demand":  pred.round(2).tolist(),
        "14d_total":         round(float(pred.sum()), 1),
        "daily_avg":         round(float(pred.mean()), 2),
        "peak_day":          forecast_dates[pred.argmax()].strftime("%Y-%m-%d"),
        "peak_demand":       round(float(pred.max()), 2),
        "scores":            eval_scores,
        "training_scores":   training_scores or {},
    }


def run_demand_forecast(data_path="data/training_data.csv", retrain=True) -> dict:
    print("=" * 65)
    print("  MODULE 1: LSTM DEMAND FORECAST")
    print("=" * 65)
    df = pd.read_csv(data_path, parse_dates=["date"])
    results = {}
    for mat_id in sorted(df["material_id"].unique()):
        mat_df = df[df["material_id"] == mat_id].copy()
        if retrain:
            r = train_model(mat_id, mat_df)
            model, scaler     = r["model"], r["scaler"]
            training_scores   = r["training_scores"]
        else:
            model, scaler   = load_model(mat_id)
            training_scores = {}
        fc = forecast(mat_id, mat_df, model, scaler, training_scores)
        results[mat_id] = fc
        s  = fc["scores"]
        ts = fc["training_scores"]
        print(f"  {mat_id}:")
        print(f"    Forecast       : 14d={fc['14d_total']}  avg={fc['daily_avg']}/day  "
              f"peak={fc['peak_demand']} on {fc['peak_day']}")
        print(f"    Eval MAE       : {s.get('mae','N/A')}  RMSE: {s.get('rmse','N/A')}  "
              f"MAPE: {s.get('mape_pct','N/A')}%  R²: {s.get('r2_score','N/A')}")
        print(f"    Huber Loss     : {s.get('huber_loss','N/A')}  Bias: {s.get('bias','N/A')}  "
              f"Dir.Acc: {s.get('directional_acc','N/A')}  Grade: {s.get('grade','N/A')}")
        print(f"    Confidence     : {s.get('confidence_pct','N/A')}%  {s.get('uncertainty_band','')}")
        if ts:
            print(f"    Training       : best_val_loss={ts.get('best_val_loss','N/A')} "
                  f"@ epoch {ts.get('best_epoch','N/A')}/{ts.get('epochs_run','N/A')}  "
                  f"early_stopped={ts.get('early_stopped','N/A')}")
            print(f"    Loss Gap       : overfit_gap={ts.get('overfit_gap','N/A'):+}  "
                  f"(positive=overfitting, negative=underfitting)")
            print(f"    Gradient Norm  : avg={ts.get('avg_grad_norm','N/A')}  "
                  f"min={ts.get('min_grad_norm','N/A')}  max={ts.get('max_grad_norm','N/A')}  "
                  f"final_lr={ts.get('final_lr','N/A')}")
    return results


if __name__ == "__main__":
    run_demand_forecast(retrain=True)
