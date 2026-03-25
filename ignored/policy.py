import numpy as np
import pandas as pd

def simulate_policy(days, demand_mean, demand_std, lead_time,
                    reorder_point, order_qty, init_inventory):

    inventory = init_inventory
    pipeline_orders = []

    stockouts = 0
    holding_cost = 0
    stockout_cost = 0

    inventory_history = []

    for day in range(days):
        demand = max(0, int(np.random.normal(demand_mean, demand_std)))
        inventory -= demand

        if inventory < 0:
            stockouts += 1
            stockout_cost += abs(inventory) * 10
            inventory = 0

        for order in pipeline_orders:
            if order["arrival_day"] == day:
                inventory += order["qty"]

        pipeline_orders = [o for o in pipeline_orders if o["arrival_day"] > day]

        if inventory < reorder_point:
            pipeline_orders.append({
                "arrival_day": day + int(lead_time),
                "qty": order_qty
            })

        holding_cost += inventory * 0.5
        inventory_history.append(inventory)

    return {
        "stockouts": stockouts,
        "avg_inventory": np.mean(inventory_history),
        "total_cost": holding_cost + stockout_cost
    }


def run_policy_simulator(df, run_lstm_demand_forecast):
    results, _ = run_lstm_demand_forecast(df)

    if results is None:
        return None

    output = []

    for i, row in results.iterrows():
        material = row["Material"]

        demand_mean = row["Predicted Demand"]
        demand_std = demand_mean * 0.2

        lead_time = df.iloc[i]["Lead Time Supplier→Plant (Days)"]
        init_inventory = row["Stock"]
        safety_stock = row["Safety Stock"]

        # 🔴 Naive Policy
        naive = simulate_policy(
            days=60,
            demand_mean=demand_mean,
            demand_std=demand_std,
            lead_time=lead_time,
            reorder_point=100,
            order_qty=200,
            init_inventory=init_inventory
        )

        # 🟢 Smart Policy (YOUR LOGIC)
        reorder_point = demand_mean * lead_time + safety_stock
        order_qty = max(row["Reorder Qty"], demand_mean * lead_time)

        smart = simulate_policy(
            days=60,
            demand_mean=demand_mean,
            demand_std=demand_std,
            lead_time=lead_time,
            reorder_point=reorder_point,
            order_qty=order_qty,
            init_inventory=init_inventory
        )

        output.append({
            "Material": material,

            "Naive_Stockouts": naive["stockouts"],
            "Smart_Stockouts": smart["stockouts"],

            "Naive_Cost": naive["total_cost"],
            "Smart_Cost": smart["total_cost"],

            "Improvement_Stockouts": naive["stockouts"] - smart["stockouts"],
            "Cost_Savings": naive["total_cost"] - smart["total_cost"]
        })

    return pd.DataFrame(output)