# """
# Monte Carlo Inventory Simulation — Prediction-Driven
# =====================================================
# Inputs  : MRP_AI_Predictions.xlsx  (AI-predicted demand, safety stock,
#           current stock, open PO, reorder qty)
# Method  : 5,000-run × 90-day stochastic simulation per material
#           Demand  ~ Normal(μ = predicted_monthly/30,  σ = μ × CV)
#           LT      ~ Normal(μ = lead_time_mean,        σ = lt_std)
# Outputs : stockout probability, avg inventory, service level,
#           P50/P90 stockout day, recommended safety stock,
#           cost savings vs naive policy, day-by-day percentile bands
# """

# import numpy as np
# import pandas as pd
# from scipy import stats
# from datetime import date

# # ─────────────────────────────────────────────────────────────
# # 1. LOAD AI PREDICTIONS
# # ─────────────────────────────────────────────────────────────

# PRED_PATH = 'MRP_AI_Predictions.xlsx'

# df_pred = pd.read_excel(PRED_PATH)
# df_pred.columns = df_pred.columns.str.strip()

# # Normalise column names robustly
# col_map = {
#     'Material':         'material',
#     'Predicted Demand': 'pred_demand',
#     'Safety Stock':     'safety_stock',
#     'Stock':            'stock',
#     'Open PO':          'open_po',
#     'Reorder Qty':      'reorder_qty',
# }
# df_pred = df_pred.rename(columns=col_map)
# df_pred['material'] = df_pred['material'].astype(str).str.strip()

# # ─────────────────────────────────────────────────────────────
# # 2. SUPPLIER / LEAD-TIME PROFILE LOOKUP
# #    (derived from original ERP data; extended to all new SKUs
# #     by material-type prefix)
# # ─────────────────────────────────────────────────────────────

# def get_profile(mat_id: str) -> dict:
#     """Return lead-time and cost profile by material prefix."""
#     if mat_id.startswith('FG-'):
#         return dict(price=18_500, lt_mean=10, lt_std=1.5,
#                     moq=10,  rounding=5,  max_stock=120,
#                     lot_size='EX', holding_rate_pa=0.25,
#                     stockout_penalty_mult=2.0)
#     if mat_id.startswith('SFG-'):
#         return dict(price=6_200,  lt_mean=15, lt_std=2.5,
#                     moq=20,  rounding=10, max_stock=90,
#                     lot_size='FX', holding_rate_pa=0.25,
#                     stockout_penalty_mult=2.0)
#     # RM- series
#     return dict(price=950,    lt_mean=23, lt_std=3.5,
#                 moq=300, rounding=50, max_stock=1_200,
#                 lot_size='HB', holding_rate_pa=0.25,
#                 stockout_penalty_mult=2.0)

# # ─────────────────────────────────────────────────────────────
# # 3. CORE SIMULATION
# # ─────────────────────────────────────────────────────────────

# HORIZON  = 90       # days
# N_SIMS   = 5_000
# CV       = 0.15     # demand coefficient of variation
# Z_SL     = 1.645    # 95% service level

# def simulate_one(mat_row: pd.Series, seed: int = 42) -> dict:
#     rng = np.random.default_rng(seed)

#     mat_id      = mat_row['material']
#     pred_demand = float(mat_row['pred_demand'])     # monthly
#     safety_stk  = float(mat_row['safety_stock'])
#     init_stock  = float(mat_row['stock'])
#     open_po_qty = max(0.0, float(mat_row['open_po']))
#     reorder_qty_pred = float(mat_row['reorder_qty'])

#     prof        = get_profile(mat_id)
#     price       = prof['price']
#     lt_mean     = prof['lt_mean']
#     lt_std      = prof['lt_std']
#     moq         = prof['moq']
#     rounding    = prof['rounding']
#     max_stock   = prof['max_stock']
#     lot_size    = prof['lot_size']
#     hold_rate   = price * prof['holding_rate_pa'] / 365   # ₹/unit/day
#     so_penalty  = price * prof['stockout_penalty_mult']

#     # Daily demand parameters derived from AI prediction
#     d_mu  = max(0.5, pred_demand / 30.0)
#     d_sig = d_mu * CV

#     # Schedule open PO: assume arrives at day = lt_mean (best estimate)
#     po_arrival_day = int(round(lt_mean))
#     po_schedule = {po_arrival_day: open_po_qty} if open_po_qty > 0 else {}

#     # Hadley-Whitin recommended safety stock
#     rec_ss = int(np.ceil(Z_SL * np.sqrt(
#         lt_mean * d_sig**2 + d_mu**2 * lt_std**2
#     )))
#     rec_ss = max(rec_ss, moq)

#     # ── 5,000-run simulation ──────────────────────────────────
#     so_flags      = np.zeros(N_SIMS, dtype=bool)
#     first_so_days = np.full(N_SIMS, np.nan)
#     avg_inv_arr   = np.zeros(N_SIMS)
#     so_units_arr  = np.zeros(N_SIMS)
#     n_orders_arr  = np.zeros(N_SIMS, dtype=int)
#     daily_traces  = np.zeros((N_SIMS, HORIZON))

#     for sim in range(N_SIMS):
#         stock   = float(init_stock)
#         pending = {}
#         inv_sum = so_sum = 0.0
#         first_so = None
#         n_ord = 0

#         for day in range(HORIZON):
#             # Receive scheduled POs
#             if day in po_schedule:
#                 stock += po_schedule[day]
#             if day in pending:
#                 stock += pending.pop(day)

#             # Stochastic demand
#             demand = max(0.0, rng.normal(d_mu, d_sig))
#             if demand > stock:
#                 so_sum += demand - stock
#                 if first_so is None:
#                     first_so = day
#                 stock = 0.0
#             else:
#                 stock -= demand

#             inv_sum += stock
#             daily_traces[sim, day] = stock

#             # Reorder trigger (use predicted reorder qty as base)
#             reorder_pt = max(safety_stk, d_mu * lt_mean)
#             if stock <= reorder_pt and not pending:
#                 if lot_size == 'HB':
#                     oq = max_stock - stock
#                 elif lot_size == 'FX':
#                     oq = moq * 3
#                 else:   # EX
#                     oq = max(moq, abs(reorder_qty_pred) + safety_stk - stock)
#                 oq  = max(moq, round(oq / rounding) * rounding)
#                 arr = day + max(1, int(round(rng.normal(lt_mean, lt_std))))
#                 pending[arr] = pending.get(arr, 0) + oq
#                 n_ord += 1

#         if first_so is not None:
#             so_flags[sim]      = True
#             first_so_days[sim] = first_so
#         avg_inv_arr[sim]  = inv_sum / HORIZON
#         so_units_arr[sim] = so_sum
#         n_orders_arr[sim] = n_ord

#     # ── Aggregate ─────────────────────────────────────────────
#     so_prob    = float(so_flags.mean())
#     avg_inv    = float(avg_inv_arr.mean())
#     valid_days = first_so_days[~np.isnan(first_so_days)]
#     p50_day    = int(np.percentile(valid_days, 50)) if len(valid_days) else None
#     p90_day    = int(np.percentile(valid_days, 90)) if len(valid_days) else None
#     svc_lvl    = 1.0 - float(so_units_arr.sum()) / (d_mu * HORIZON * N_SIMS)

#     # Cost comparison
#     naive_so_prob   = min(0.95, so_prob * 1.8)
#     units_at_risk   = d_mu * HORIZON
#     smart_hold      = avg_inv * hold_rate * HORIZON
#     naive_hold      = avg_inv * 0.55 * hold_rate * HORIZON
#     smart_so_cost   = so_prob       * units_at_risk * so_penalty * 0.02
#     naive_so_cost   = naive_so_prob * units_at_risk * so_penalty * 0.02
#     cost_saving     = (naive_hold + naive_so_cost) - (smart_hold + smart_so_cost)

#     # Urgency
#     days_of_cover = (init_stock + open_po_qty) / d_mu if d_mu > 0 else 999
#     days_to_action = max(0, days_of_cover - lt_mean - 3)
#     if days_to_action == 0:      urgency = 'CRITICAL'
#     elif days_to_action <= 7:    urgency = 'HIGH'
#     elif days_to_action <= 14:   urgency = 'MEDIUM'
#     else:                        urgency = 'LOW'

#     return {
#         # Identifiers
#         'material':              mat_id,
#         'mat_type':              'FG' if mat_id.startswith('FG') else ('SFG' if mat_id.startswith('SFG') else 'RM'),
#         # Inputs from predictions
#         'predicted_monthly_demand': round(pred_demand, 2),
#         'daily_demand_mean':     round(d_mu, 3),
#         'daily_demand_std':      round(d_sig, 3),
#         'current_stock':         init_stock,
#         'open_po_qty':           open_po_qty,
#         'predicted_reorder_qty': round(reorder_qty_pred, 1),
#         'current_safety_stock':  safety_stk,
#         # Lead time
#         'lt_mean_days':          lt_mean,
#         'lt_std_days':           lt_std,
#         # Outputs
#         'stockout_probability':  round(so_prob, 4),
#         'stockout_pct':          round(so_prob * 100, 1),
#         'avg_inventory_units':   round(avg_inv, 1),
#         'avg_so_units_per_run':  round(float(so_units_arr.mean()), 2),
#         'service_level_pct':     round(svc_lvl * 100, 2),
#         'p50_first_so_day':      p50_day,
#         'p90_first_so_day':      p90_day,
#         'recommended_safety_stock': rec_ss,
#         'ss_delta':              rec_ss - int(safety_stk),
#         'days_of_cover':         round(days_of_cover, 1),
#         'days_to_action':        round(days_to_action, 1),
#         'urgency':               urgency,
#         'cost_saving_inr':       round(cost_saving),
#         'unit_price':            price,
#         # Internal for Excel charts
#         '_daily_traces':         daily_traces,
#     }


# # ─────────────────────────────────────────────────────────────
# # 4. RUN ALL 33 MATERIALS
# # ─────────────────────────────────────────────────────────────

# def run_all() -> list:
#     results = []
#     print(f'{"Material":<15} {"SO Prob":>9} {"Svc Lvl":>9} {"Avg Inv":>9} {"Urgency":>10} {"Rec SS":>8}')
#     print('-' * 65)
#     for i, row in df_pred.iterrows():
#         r = simulate_one(row, seed=42 + i)
#         results.append(r)
#         print(f'{r["material"]:<15} {r["stockout_pct"]:>8.1f}% {r["service_level_pct"]:>8.2f}%'
#               f' {r["avg_inventory_units"]:>9.1f} {r["urgency"]:>10} {r["recommended_safety_stock"]:>8}')
#     return results


# if __name__ == '__main__':
#     print(f'Monte Carlo Simulation — Prediction-Driven')
#     print(f'Materials: {len(df_pred)}  |  Runs: {N_SIMS:,}  |  Horizon: {HORIZON}d  |  CV: {CV*100:.0f}%\n')
#     all_results = run_all()
#     print(f'\nTotal materials simulated: {len(all_results)}')
#     critical = [r for r in all_results if r['urgency'] == 'CRITICAL']
#     high     = [r for r in all_results if r['urgency'] == 'HIGH']
#     print(f'CRITICAL: {len(critical)}  |  HIGH: {len(high)}')


# Refactored as a function for agent use
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
