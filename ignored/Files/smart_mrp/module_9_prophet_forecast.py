"""
module_9_prophet_forecast.py
------------------------------
Module 9: Prophet Demand Forecasting  (Facebook Prophet)

Serves as a strong statistical baseline to compare against
the LSTM (Module 1). Prophet is excellent at:
  - Automatic seasonality detection (yearly + weekly)
  - Holiday effects and special events
  - Trend changepoints (detects when demand pattern shifts)
  - Explainable components (trend + seasonality + residual)

Comparison output: Prophet vs LSTM on same validation window.
Rule: Use Prophet when LSTM overfits (<1000 training rows).
      Use LSTM when dataset is large and non-linear patterns dominate.

Run:  python module_9_prophet_forecast.py
"""

import numpy as np
import pandas as pd
from pathlib import Path

MODELS_DIR  = Path("models")
OUTPUTS_DIR = Path("outputs")
MODELS_DIR.mkdir(exist_ok=True)
OUTPUTS_DIR.mkdir(exist_ok=True)

FORECAST_HORIZON = 14   # days ahead


def run_prophet_forecast(data_path="data/training_data.csv"):
    print("=" * 65)
    print("  MODULE 9: PROPHET DEMAND FORECAST  (Baseline vs LSTM)")
    print("=" * 65)

    try:
        from prophet import Prophet
    except ImportError:
        print("\n  ⚠️  prophet not installed.")
        print("  Run:  pip install prophet")
        print("  Falling back to linear trend extrapolation for comparison.\n")
        _run_linear_baseline(data_path)
        return {}

    df = pd.read_csv(data_path, parse_dates=["date"])
    results = {}

    for mat_id in df["material_id"].unique():
        print(f"\n  Training Prophet for {mat_id}...")
        mat_df = (df[df["material_id"] == mat_id]
                    .sort_values("date")
                    .rename(columns={"date": "ds", "daily_demand": "y"})
                    [["ds", "y"]]
                    .reset_index(drop=True))

        # Train/val split — last 30 days as validation
        train = mat_df.iloc[:-30]
        val   = mat_df.iloc[-30:]

        model = Prophet(
            yearly_seasonality=True,
            weekly_seasonality=True,
            daily_seasonality=False,
            seasonality_mode="multiplicative",   # demand × season factor
            changepoint_prior_scale=0.05,        # regularise trend changes
        )
        model.fit(train)

        # Forecast
        future = model.make_future_dataframe(periods=FORECAST_HORIZON + 30)
        fc     = model.predict(future)

        # Validation metrics on held-out 30 days
        val_pred = fc[fc["ds"].isin(val["ds"])]["yhat"].values
        val_true = val["y"].values[:len(val_pred)]
        mae  = np.mean(np.abs(val_true - val_pred))
        mape = np.mean(np.abs((val_true - val_pred) / (val_true + 1e-9))) * 100

        print(f"    Validation MAE={mae:.3f}  MAPE={mape:.1f}%")

        # Next FORECAST_HORIZON days
        future_rows = fc[fc["ds"] > mat_df["ds"].max()].head(FORECAST_HORIZON)
        pred        = np.maximum(0, future_rows["yhat"].values)

        # Save forecast
        fc_path = OUTPUTS_DIR / f"prophet_forecast_{mat_id.replace('-','_')}.csv"
        future_rows[["ds","yhat","yhat_lower","yhat_upper"]].to_csv(fc_path, index=False)

        results[mat_id] = {
            "material_id":      mat_id,
            "validation_mae":   round(mae, 3),
            "validation_mape":  round(mape, 1),
            "forecast_14d":     pred.round(3).tolist(),
            "14d_total":        round(pred.sum(), 1),
            "daily_avg":        round(pred.mean(), 3),
        }

        print(f"    14-day forecast: total={pred.sum():.1f}  avg={pred.mean():.2f}/day")
        print(f"    Saved → {fc_path}")

    print("\n" + "=" * 65)
    print("  PROPHET vs LSTM COMPARISON  (validation MAE)")
    print("=" * 65)
    print("  Lower MAE = better on this dataset")
    print()

    for mat_id, res in results.items():
        print(f"  {mat_id}")
        print(f"    Prophet  MAE={res['validation_mae']:.3f}  MAPE={res['validation_mape']:.1f}%")
        print(f"    (Run Module 1 LSTM to compare)")
        print()

    print("  Note: For <1000 rows → Prophet usually wins.")
    print("        For 5yr daily data → LSTM captures non-linear patterns better.")

    return results


def _run_linear_baseline(data_path: str):
    """Simple linear trend fallback when Prophet is not installed."""
    print("  Running linear trend baseline instead...\n")
    df = pd.read_csv(data_path, parse_dates=["date"])

    for mat_id in df["material_id"].unique():
        mat_df = df[df["material_id"] == mat_id].sort_values("date")
        y = mat_df["daily_demand"].values
        x = np.arange(len(y))

        slope, intercept = np.polyfit(x, y, 1)

        # Forecast next 14 days by extrapolating
        future_x    = np.arange(len(y), len(y) + FORECAST_HORIZON)
        future_pred = np.maximum(0, slope * future_x + intercept)

        print(f"  {mat_id}")
        print(f"    Trend slope: {slope:+.4f} units/day")
        print(f"    14-day forecast: {[round(v,2) for v in future_pred]}")
        print(f"    14-day total:    {future_pred.sum():.1f}")
        print()


if __name__ == "__main__":
    run_prophet_forecast()
