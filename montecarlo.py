import pandas as pd
import numpy as np

def run_montecarlo_risk(pred_df, rmse=96, num_simulations=10000):
    """
    Takes a DataFrame with columns: Material, Predicted Demand, Stock, Safety Stock, Open PO, Lead Time
    Returns a DataFrame with risk statistics (Material, Predicted Demand, Mean Demand, Demand Std, 5%/95% Demand, Stockout Probability, etc.)
    """
    simulated_demands = []
    for demand in pred_df["Predicted Demand"]:
        sims = np.random.normal(
            loc=demand,
            scale=rmse,
            size=num_simulations
        )
        simulated_demands.append(sims)
    sim_df = pd.DataFrame(simulated_demands).T

    risk_df = pd.DataFrame()
    risk_df["Material"] = pred_df["Material"]
    risk_df["Predicted Demand"] = pred_df["Predicted Demand"]
    risk_df["Mean Demand"] = sim_df.mean().values
    risk_df["Demand Std"] = sim_df.std().values
    risk_df["5% Demand"] = sim_df.quantile(0.05).values
    risk_df["95% Demand"] = sim_df.quantile(0.95).values

    # Stockout Probability
    stockout_probs = []
    for i in range(len(pred_df)):
        stock_level = pred_df["Stock"].iloc[i]
        demand_sim = sim_df.iloc[:, i]
        prob = np.mean(demand_sim > stock_level)
        stockout_probs.append(prob)
    risk_df["Stockout Probability"] = stockout_probs

    # Optionally add ROP calculation if Lead Time is present
    if "Lead Time" in pred_df.columns:
        risk_df["ROP"] = (
            pred_df["Predicted Demand"] * pred_df["Lead Time"]
            + pred_df["Safety Stock"]
        )

    return risk_df
