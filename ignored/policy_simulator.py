"""
Policy Simulator for Inventory Management
Compares Naive vs Smart inventory policies based on forecast, lead time, and risk
"""

import pandas as pd
import numpy as np
from typing import Dict, Tuple, List
from dataclasses import dataclass
from enum import Enum
import warnings

warnings.filterwarnings('ignore')


@dataclass
class PolicyMetrics:
    """Stores performance metrics for a policy"""
    total_stockouts: int
    stockout_rate: float  # percentage of periods with stockout
    average_inventory: float
    max_inventory: float
    min_inventory: float
    total_ordering_cost: float
    total_stockout_cost: float
    total_cost: float
    number_of_orders: int
    avg_order_quantity: float
    

class InventoryPolicy:
    """Base class for inventory policies"""
    
    def __init__(self, name: str):
        self.name = name
        self.orders = []
        self.inventory_history = []
        self.stockouts = []
        
    def calculate_order_qty(self, current_inventory: float, demand: float, 
                           lead_time: int, forecasted_demand: float = None) -> Tuple[float, bool]:
        """
        Calculate order quantity if needed.
        Returns: (order_qty, should_order)
        """
        raise NotImplementedError
    
    def simulate(self, demand_series: np.ndarray, lead_time: int, 
                 forecasted_demand: np.ndarray = None, 
                 ordering_cost: float = 100, holding_cost: float = 1,
                 stockout_cost: float = 50, initial_inventory: float = 100) -> PolicyMetrics:
        """
        Simulate inventory policy over a period
        """
        current_inv = initial_inventory
        inventory_levels = [current_inv]
        pending_orders = []  # List of (arrival_period, quantity)
        total_ordering_cost = 0
        total_stockout_cost = 0
        total_holding_cost = 0
        stockout_count = 0
        order_count = 0
        order_quantities = []
        
        n_periods = len(demand_series)
        
        for period in range(n_periods):
            # Process pending orders
            newly_arrived = []
            still_pending = []
            for arrival_period, qty in pending_orders:
                if arrival_period <= period:
                    current_inv += qty
                    newly_arrived.append(qty)
                else:
                    still_pending.append((arrival_period, qty))
            pending_orders = still_pending
            
            # Current demand
            demand = demand_series[period]
            
            # Check for stockout
            if current_inv < demand:
                shortage = demand - current_inv
                stockout_cost_this_period = shortage * stockout_cost
                total_stockout_cost += stockout_cost_this_period
                stockout_count += 1
                self.stockouts.append({
                    'period': period,
                    'shortage': shortage,
                    'cost': stockout_cost_this_period
                })
                current_inv = 0
            else:
                current_inv -= demand
            
            # Calculate holding cost
            total_holding_cost += current_inv * holding_cost
            
            # Decide to order or not
            forecast = forecasted_demand[period] if forecasted_demand is not None else demand
            order_qty, should_order = self.calculate_order_qty(
                current_inv, demand, lead_time, forecast
            )
            
            if should_order and order_qty > 0:
                arrival_period = period + lead_time
                pending_orders.append((arrival_period, order_qty))
                order_count += 1
                order_quantities.append(order_qty)
                total_ordering_cost += ordering_cost
                self.orders.append({
                    'period': period,
                    'quantity': order_qty,
                    'arrival_period': arrival_period
                })
            
            inventory_levels.append(current_inv)
        
        self.inventory_history = inventory_levels
        
        # Calculate metrics
        inventory_array = np.array(inventory_levels)
        avg_inventory = inventory_array[:-1].mean()  # Exclude final state
        max_inventory = inventory_array.max()
        min_inventory = inventory_array.min()
        
        total_cost = total_ordering_cost + total_holding_cost + total_stockout_cost
        stockout_rate = (stockout_count / n_periods) * 100
        avg_order_qty = np.mean(order_quantities) if order_quantities else 0
        
        return PolicyMetrics(
            total_stockouts=stockout_count,
            stockout_rate=stockout_rate,
            average_inventory=avg_inventory,
            max_inventory=max_inventory,
            min_inventory=min_inventory,
            total_ordering_cost=total_ordering_cost,
            total_stockout_cost=total_stockout_cost,
            total_cost=total_cost,
            number_of_orders=order_count,
            avg_order_quantity=avg_order_qty
        )


class NaivePolicy(InventoryPolicy):
    """
    Naive Policy: Fixed reorder point and fixed order quantity
    - Reorder point: a fixed value based on average demand and lead time
    - Order quantity: a fixed value (EOQ approximation)
    """
    
    def __init__(self, reorder_point: float, order_quantity: float):
        super().__init__("Naive (Fixed ROP & Fixed EOQ)")
        self.reorder_point = reorder_point
        self.order_quantity = order_quantity
    
    def calculate_order_qty(self, current_inventory: float, demand: float,
                           lead_time: int, forecasted_demand: float = None) -> Tuple[float, bool]:
        """
        Order when inventory drops to or below reorder point
        """
        should_order = current_inventory <= self.reorder_point
        if should_order:
            return self.order_quantity, True
        return 0, False


class SmartPolicy(InventoryPolicy):
    """
    Smart Policy: Based on forecast + lead time + risk
    - Reorder point: Dynamic, based on forecasted demand, lead time, and demand variability
    - Order quantity: Dynamic, based on forecasted demand and service level
    - Uses safety stock to account for demand variability and lead time uncertainty
    """
    
    def __init__(self, service_level: float = 0.95):
        super().__init__(f"Smart (Forecast + Lead Time + Risk, SL={service_level})")
        self.service_level = service_level
        self.z_score = self._get_z_score(service_level)
    
    @staticmethod
    def _get_z_score(service_level: float) -> float:
        """Get Z-score for service level (simplified)"""
        # Approximate Z-scores for common service levels
        z_scores = {
            0.80: 0.84,
            0.85: 1.04,
            0.90: 1.28,
            0.95: 1.645,
            0.99: 2.33
        }
        return z_scores.get(service_level, 1.645)
    
    def calculate_order_qty(self, current_inventory: float, demand: float,
                           lead_time: int, forecasted_demand: float = None) -> Tuple[float, bool]:
        """
        Dynamic reorder point and quantity based on forecast and variability
        """
        if forecasted_demand is None:
            forecasted_demand = demand
        
        # Estimate demand variability (using coefficient of variation concept)
        # In practice, this would be calculated from historical data
        demand_cv = 0.15  # 15% coefficient of variation (moderate variability)
        
        # Calculate safety stock
        # Safety Stock = Z * σ_d * √(LT)
        # where σ_d = mean_demand * CV
        demand_std = forecasted_demand * demand_cv
        safety_stock = self.z_score * demand_std * np.sqrt(lead_time)
        
        # Reorder point = (Average demand during lead time) + Safety stock
        reorder_point = (forecasted_demand * lead_time) + safety_stock
        
        # Order quantity based on economic order quantity principles
        # For simplicity: EOQ = √(2*D*S/H) where we estimate based on forecast
        # Using a simplified formula: Order when below ROP, order enough to reach target level
        target_inventory = reorder_point + (forecasted_demand * 14)  # 2 weeks of buffer
        
        should_order = current_inventory <= reorder_point
        
        if should_order:
            order_qty = max(target_inventory - current_inventory, forecasted_demand * 7)
            return order_qty, True
        
        return 0, False


class PolicySimulator:
    """Orchestrates the simulation of multiple policies"""
    
    def __init__(self, demand_data: np.ndarray, lead_time: int,
                 forecasted_demand: np.ndarray = None,
                 initial_inventory: float = 100,
                 ordering_cost: float = 100,
                 holding_cost: float = 1,
                 stockout_cost: float = 50):
        """
        Initialize simulator
        
        Args:
            demand_data: Array of historical/actual demand
            lead_time: Supplier lead time in periods
            forecasted_demand: Array of forecasted demand (optional)
            initial_inventory: Starting inventory level
            ordering_cost: Cost to place an order
            holding_cost: Cost to hold one unit per period
            stockout_cost: Cost per unit of shortage
        """
        self.demand_data = demand_data
        self.lead_time = lead_time
        self.forecasted_demand = forecasted_demand if forecasted_demand is not None else demand_data
        self.initial_inventory = initial_inventory
        self.ordering_cost = ordering_cost
        self.holding_cost = holding_cost
        self.stockout_cost = stockout_cost
        
        self.results = {}
        self.policies = {}
    
    def add_policy(self, policy: InventoryPolicy):
        """Add a policy to simulate"""
        self.policies[policy.name] = policy
    
    def run_simulation(self):
        """Run simulation for all policies"""
        for policy_name, policy in self.policies.items():
            metrics = policy.simulate(
                self.demand_data,
                self.lead_time,
                self.forecasted_demand,
                self.ordering_cost,
                self.holding_cost,
                self.stockout_cost,
                self.initial_inventory
            )
            self.results[policy_name] = metrics
    
    def get_comparison_dataframe(self) -> pd.DataFrame:
        """Get a dataframe comparing all policies"""
        data = []
        for policy_name, metrics in self.results.items():
            data.append({
                'Policy': policy_name,
                'Total Stockouts': metrics.total_stockouts,
                'Stockout Rate (%)': f"{metrics.stockout_rate:.2f}%",
                'Avg Inventory': f"{metrics.average_inventory:.2f}",
                'Max Inventory': f"{metrics.max_inventory:.2f}",
                'Min Inventory': f"{metrics.min_inventory:.2f}",
                'Ordering Cost': f"${metrics.total_ordering_cost:.2f}",
                'Stockout Cost': f"${metrics.total_stockout_cost:.2f}",
                'Total Cost': f"${metrics.total_cost:.2f}",
                'Number of Orders': metrics.number_of_orders,
                'Avg Order Qty': f"{metrics.avg_order_quantity:.2f}"
            })
        return pd.DataFrame(data)
    
    def get_raw_metrics(self) -> Dict[str, PolicyMetrics]:
        """Get raw metrics for custom analysis"""
        return self.results
    
    def print_summary(self):
        """Print a summary of results"""
        print("=" * 120)
        print("POLICY COMPARISON SUMMARY")
        print("=" * 120)
        print(self.get_comparison_dataframe().to_string(index=False))
        print("=" * 120)
        
        # Key insights
        print("\nKEY INSIGHTS:")
        if self.results:
            metrics_list = list(self.results.items())
            
            # Stockouts comparison
            stockouts_comparison = sorted(metrics_list, key=lambda x: x[1].total_stockouts)
            print(f"\nStockouts (Lower is Better):")
            for name, metrics in stockouts_comparison:
                print(f"  {name}: {metrics.total_stockouts} events ({metrics.stockout_rate:.2f}%)")
            
            # Cost comparison
            cost_comparison = sorted(metrics_list, key=lambda x: x[1].total_cost)
            print(f"\nTotal Cost (Lower is Better):")
            for name, metrics in cost_comparison:
                print(f"  {name}: ${metrics.total_cost:.2f}")
            
            # Inventory comparison
            inv_comparison = sorted(metrics_list, key=lambda x: x[1].average_inventory)
            print(f"\nAverage Inventory Level (Lower is Better):")
            for name, metrics in inv_comparison:
                print(f"  {name}: {metrics.average_inventory:.2f} units")
        
        print("=" * 120)


def example_simulation():
    """Run an example simulation with sample data"""
    # Create sample demand data (e.g., 365 days)
    np.random.seed(42)
    
    # Generate realistic demand with trend and seasonality
    t = np.arange(365)
    base_demand = 100
    trend = 0.02 * t
    seasonality = 20 * np.sin(2 * np.pi * t / 365)
    noise = np.random.normal(0, 10, 365)
    demand = base_demand + trend + seasonality + noise
    demand = np.maximum(demand, 5)  # Ensure non-negative demand
    
    # Generate forecast (slightly noisy version of true demand)
    forecast = demand + np.random.normal(0, 5, 365)
    forecast = np.maximum(forecast, 5)
    
    # Simulation parameters
    lead_time = 7  # 7-day lead time
    initial_inventory = 800
    
    # Create simulator
    simulator = PolicySimulator(
        demand_data=demand,
        lead_time=lead_time,
        forecasted_demand=forecast,
        initial_inventory=initial_inventory,
        ordering_cost=100,
        holding_cost=1,
        stockout_cost=50
    )
    
    # Add policies
    # Naive: Fixed ROP = average demand * lead time, Fixed EOQ = average demand
    avg_demand = demand.mean()
    naive_rop = avg_demand * lead_time
    naive_eoq = avg_demand * 5  # 5 days of stock
    
    simulator.add_policy(NaivePolicy(
        reorder_point=naive_rop,
        order_quantity=naive_eoq
    ))
    
    # Smart: Adaptive based on forecast and risk
    simulator.add_policy(SmartPolicy(service_level=0.95))
    
    # Run simulation
    simulator.run_simulation()
    
    # Display results
    simulator.print_summary()
    
    return simulator


if __name__ == "__main__":
    simulator = example_simulation()
