"""
Policy Optimizer for Tata Supply Chain AI
------------------------------------------
Implements Naive vs Smart inventory policy simulation and comparison.
Called by app.py via:
    from policy_optimizer import optimize_policy, format_policy_output
"""

import numpy as np
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple


# ─── Data classes ─────────────────────────────────────────────────────────────

@dataclass
class PolicyMetrics:
    total_stockouts: int
    stockout_rate: float        # % of periods with stockout
    average_inventory: float
    max_inventory: float
    min_inventory: float
    total_ordering_cost: float
    total_holding_cost: float
    total_stockout_cost: float
    total_cost: float
    number_of_orders: int
    avg_order_quantity: float
    inventory_history: List[float]
    cost_savings_vs_naive: Optional[float] = None   # filled in later
    stockout_reduction_vs_naive: Optional[int] = None


@dataclass
class OptimizationResult:
    naive_metrics: PolicyMetrics
    smart_metrics: PolicyMetrics
    recommended_policy: str
    recommended_reorder_point: float
    recommended_order_quantity: float
    safety_stock: float
    lead_time: int
    service_level: float
    supplier_risk: str
    stockout_risk: float
    cost_savings: float
    stockout_reduction: int
    simulation_days: int
    initial_stock: float
    avg_demand: float
    demand_std: float


# ─── Simulation core ──────────────────────────────────────────────────────────

def _simulate(
    demand_series: np.ndarray,
    lead_time: int,
    reorder_point: float,
    order_quantity: float,
    initial_inventory: float,
    ordering_cost: float,
    holding_cost_per_unit: float,
    stockout_cost_per_unit: float,
    dynamic: bool = False,
    forecast_series: Optional[np.ndarray] = None,
    service_level: float = 0.95,
) -> PolicyMetrics:
    """
    Generic inventory simulation.
    If dynamic=True, recalculates ROP/EOQ each period from forecast (Smart policy).
    """
    inv = initial_inventory
    inventory_history: List[float] = [inv]
    pending_orders: List[Tuple[int, float]] = []  # (arrival_period, qty)

    total_ordering_cost = 0.0
    total_holding_cost = 0.0
    total_stockout_cost = 0.0
    stockout_count = 0
    order_count = 0
    order_quantities: List[float] = []

    n = len(demand_series)

    # Z-scores for common service levels
    z_table = {0.80: 0.84, 0.85: 1.04, 0.90: 1.28, 0.95: 1.645, 0.99: 2.33}
    z = z_table.get(service_level, 1.645)

    for period in range(n):
        # Receive pending orders
        still_pending = []
        for arrival, qty in pending_orders:
            if arrival <= period:
                inv += qty
            else:
                still_pending.append((arrival, qty))
        pending_orders = still_pending

        demand = float(demand_series[period])

        # Stockout check
        if inv < demand:
            shortage = demand - inv
            total_stockout_cost += shortage * stockout_cost_per_unit
            stockout_count += 1
            inv = 0.0
        else:
            inv -= demand

        total_holding_cost += inv * holding_cost_per_unit
        inventory_history.append(inv)

        # Ordering decision
        if dynamic and forecast_series is not None:
            fcast = float(forecast_series[period])
            demand_cv = 0.20
            demand_std_local = fcast * demand_cv
            safety_stock = z * demand_std_local * np.sqrt(lead_time)
            rop = fcast * lead_time + safety_stock
            target = rop + fcast * 14
            oq = max(target - inv, fcast * 7)
        else:
            rop = reorder_point
            oq = order_quantity

        if inv <= rop:
            arrival_period = period + lead_time
            pending_orders.append((arrival_period, oq))
            order_count += 1
            order_quantities.append(oq)
            total_ordering_cost += ordering_cost

    inv_array = np.array(inventory_history[:-1]) if len(inventory_history) > 1 else np.array(inventory_history)
    total_cost = total_ordering_cost + total_holding_cost + total_stockout_cost

    return PolicyMetrics(
        total_stockouts=stockout_count,
        stockout_rate=(stockout_count / n) * 100,
        average_inventory=float(inv_array.mean()) if len(inv_array) else 0.0,
        max_inventory=float(inv_array.max()) if len(inv_array) else 0.0,
        min_inventory=float(inv_array.min()) if len(inv_array) else 0.0,
        total_ordering_cost=total_ordering_cost,
        total_holding_cost=total_holding_cost,
        total_stockout_cost=total_stockout_cost,
        total_cost=total_cost,
        number_of_orders=order_count,
        avg_order_quantity=float(np.mean(order_quantities)) if order_quantities else 0.0,
        inventory_history=inventory_history,
    )


# ─── Public API ───────────────────────────────────────────────────────────────

def optimize_policy(
    data: dict,
    forecast: List[float],
    stockout_risk: float,
    supplier_risk: str,
    simulation_days: int = 60,
    ordering_cost: float = 500.0,
    holding_cost: float = 2.0,
    stockout_cost: float = 150.0,
) -> OptimizationResult:
    """
    Core optimization entry point called by app.py.

    Parameters
    ----------
    data : dict with keys:
        'initial_stock'  – starting inventory level
        'forecast'       – list of predicted demand values (≥ simulation_days)
    forecast : full forecast list (same as data['forecast'])
    stockout_risk : float in [0, 1], average stockout probability from Monte Carlo
    supplier_risk : str, e.g. "LOW" | "MEDIUM" | "HIGH"
    """
    initial_stock = float(data.get("initial_stock", 200))
    forecast_vals = np.array(forecast, dtype=float)

    # If fewer forecast values than simulation days, tile/repeat
    if len(forecast_vals) < simulation_days:
        repeats = int(np.ceil(simulation_days / len(forecast_vals)))
        forecast_vals = np.tile(forecast_vals, repeats)
    forecast_vals = forecast_vals[:simulation_days]
    forecast_vals = np.maximum(forecast_vals, 0)

    avg_demand = float(forecast_vals.mean())
    demand_std = float(forecast_vals.std())

    # Adjust service level based on risk inputs
    if supplier_risk == "HIGH" or stockout_risk > 0.35:
        service_level = 0.99
    elif supplier_risk == "LOW" and stockout_risk < 0.10:
        service_level = 0.90
    else:
        service_level = 0.95

    # Lead time: adjust by supplier risk
    lead_time_map = {"LOW": 7, "MEDIUM": 10, "HIGH": 14}
    lead_time = lead_time_map.get(str(supplier_risk).upper(), 10)

    # Z-score
    z_table = {0.80: 0.84, 0.85: 1.04, 0.90: 1.28, 0.95: 1.645, 0.99: 2.33}
    z = z_table.get(service_level, 1.645)

    # Smart policy parameters
    safety_stock = z * demand_std * np.sqrt(lead_time)
    smart_rop = avg_demand * lead_time + safety_stock
    smart_eoq = max(smart_rop + avg_demand * 14 - initial_stock, avg_demand * 7)

    # Naive policy parameters (simple fixed values)
    naive_rop = avg_demand * lead_time * 1.5   # 50% buffer, no safety stock
    naive_eoq = avg_demand * 5                  # 5 days of stock

    # ── Run simulations ──────────────────────────────────────────────────────
    naive_metrics = _simulate(
        demand_series=forecast_vals,
        lead_time=lead_time,
        reorder_point=naive_rop,
        order_quantity=max(naive_eoq, 1),
        initial_inventory=initial_stock,
        ordering_cost=ordering_cost,
        holding_cost_per_unit=holding_cost,
        stockout_cost_per_unit=stockout_cost,
        dynamic=False,
    )

    smart_metrics = _simulate(
        demand_series=forecast_vals,
        lead_time=lead_time,
        reorder_point=smart_rop,
        order_quantity=max(smart_eoq, 1),
        initial_inventory=initial_stock,
        ordering_cost=ordering_cost,
        holding_cost_per_unit=holding_cost,
        stockout_cost_per_unit=stockout_cost,
        dynamic=True,
        forecast_series=forecast_vals,
        service_level=service_level,
    )

    cost_savings = naive_metrics.total_cost - smart_metrics.total_cost
    stockout_reduction = naive_metrics.total_stockouts - smart_metrics.total_stockouts

    smart_metrics.cost_savings_vs_naive = cost_savings
    smart_metrics.stockout_reduction_vs_naive = stockout_reduction

    recommended = "Smart Policy" if cost_savings >= 0 or stockout_reduction >= 0 else "Naive Policy"

    return OptimizationResult(
        naive_metrics=naive_metrics,
        smart_metrics=smart_metrics,
        recommended_policy=recommended,
        recommended_reorder_point=round(smart_rop, 2),
        recommended_order_quantity=round(smart_eoq, 2),
        safety_stock=round(safety_stock, 2),
        lead_time=lead_time,
        service_level=service_level,
        supplier_risk=str(supplier_risk).upper(),
        stockout_risk=round(stockout_risk, 4),
        cost_savings=round(cost_savings, 2),
        stockout_reduction=stockout_reduction,
        simulation_days=simulation_days,
        initial_stock=initial_stock,
        avg_demand=round(avg_demand, 2),
        demand_std=round(demand_std, 2),
    )


def format_policy_output(result: OptimizationResult) -> str:
    """
    Produce the structured text returned to the LLM as 'raw_output'.
    The LLM wraps this with an explanation; keep it information-dense.
    """
    n = result.naive_metrics
    s = result.smart_metrics

    cost_arrow = "↓" if result.cost_savings >= 0 else "↑"
    stockout_arrow = "↓" if result.stockout_reduction >= 0 else "↑"

    lines = [
        "=== POLICY OPTIMIZATION RESULT ===",
        f"Simulation Period : {result.simulation_days} days",
        f"Initial Stock     : {result.initial_stock:.0f} units",
        f"Avg Forecast Demand: {result.avg_demand:.2f} units/day",
        f"Demand Std Dev    : {result.demand_std:.2f}",
        f"Lead Time         : {result.lead_time} days",
        f"Stockout Risk     : {result.stockout_risk:.1%}",
        f"Supplier Risk     : {result.supplier_risk}",
        f"Service Level     : {result.service_level:.0%}",
        "",
        "--- SMART POLICY PARAMETERS ---",
        f"Safety Stock      : {result.safety_stock:.2f} units",
        f"Reorder Point     : {result.recommended_reorder_point:.2f} units",
        f"Order Quantity    : {result.recommended_order_quantity:.2f} units",
        "",
        "--- NAIVE POLICY PERFORMANCE ---",
        f"Total Stockouts   : {n.total_stockouts}",
        f"Stockout Rate     : {n.stockout_rate:.1f}%",
        f"Avg Inventory     : {n.average_inventory:.1f} units",
        f"Number of Orders  : {n.number_of_orders}",
        f"Avg Order Qty     : {n.avg_order_quantity:.1f}",
        f"Ordering Cost     : ${n.total_ordering_cost:,.2f}",
        f"Holding Cost      : ${n.total_holding_cost:,.2f}",
        f"Stockout Cost     : ${n.total_stockout_cost:,.2f}",
        f"Total Cost        : ${n.total_cost:,.2f}",
        "",
        "--- SMART POLICY PERFORMANCE ---",
        f"Total Stockouts   : {s.total_stockouts}",
        f"Stockout Rate     : {s.stockout_rate:.1f}%",
        f"Avg Inventory     : {s.average_inventory:.1f} units",
        f"Number of Orders  : {s.number_of_orders}",
        f"Avg Order Qty     : {s.avg_order_quantity:.1f}",
        f"Ordering Cost     : ${s.total_ordering_cost:,.2f}",
        f"Holding Cost      : ${s.total_holding_cost:,.2f}",
        f"Stockout Cost     : ${s.total_stockout_cost:,.2f}",
        f"Total Cost        : ${s.total_cost:,.2f}",
        "",
        "--- COMPARISON ---",
        f"Cost Savings      : ${abs(result.cost_savings):,.2f} {cost_arrow} ({('saving' if result.cost_savings >= 0 else 'extra cost')})",
        f"Stockout Reduction: {abs(result.stockout_reduction)} events {stockout_arrow}",
        "",
        f"RECOMMENDED       : {result.recommended_policy}",
        "===================================",
    ]

    return "\n".join(lines)
