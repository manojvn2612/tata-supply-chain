"""
Policy Simulator Integration with Real Supply Chain Data
Demonstrates naive vs smart policies on actual data
"""

import pandas as pd
import numpy as np
import sys
import os
from pathlib import Path

# Import the policy simulator
from policy_simulator import PolicySimulator, NaivePolicy, SmartPolicy


def load_supply_chain_data(csv_path: str) -> pd.DataFrame:
    """Load supply chain data from CSV"""
    try:
        df = pd.read_csv(csv_path)
        print(f"✓ Loaded data with {len(df)} products")
        return df
    except FileNotFoundError:
        print(f"✗ File not found: {csv_path}")
        return None


def generate_demand_series(df: pd.DataFrame, product_sku: str, num_periods: int = 90) -> np.ndarray:
    """
    Generate demand series for a product
    Uses historical sales data and adds realistic variation
    """
    if product_sku not in df['SKU'].values:
        print(f"✗ SKU {product_sku} not found")
        return None
    
    product = df[df['SKU'] == product_sku].iloc[0]
    avg_sold = product['Number of products sold']
    
    # Create realistic demand with seasonality and noise
    np.random.seed(hash(product_sku) % 2**32)
    t = np.arange(num_periods)
    
    # Base demand with small trend
    base = avg_sold + 0.1 * t
    
    # Seasonal pattern (weekly)
    seasonality = 0.2 * avg_sold * np.sin(2 * np.pi * t / 7)
    
    # Random noise
    noise = np.random.normal(0, 0.1 * avg_sold, num_periods)
    
    demand = base + seasonality + noise
    demand = np.maximum(demand, 1)  # Ensure positive
    
    return demand


def generate_forecast(demand: np.ndarray, accuracy: float = 0.85) -> np.ndarray:
    """
    Generate forecast with specified accuracy
    accuracy: 0-1, where 1 is perfect forecast
    """
    error_std = (1 - accuracy) * demand.mean()
    forecast = demand + np.random.normal(0, error_std, len(demand))
    return np.maximum(forecast, 1)


def run_policy_simulation_for_sku(df: pd.DataFrame, sku: str, num_periods: int = 90) -> dict:
    """
    Run full policy simulation for a specific SKU
    """
    print(f"\n{'='*80}")
    print(f"SIMULATING POLICIES FOR SKU: {sku}")
    print(f"{'='*80}")
    
    product = df[df['SKU'] == sku].iloc[0]
    
    # Get product parameters
    lead_time = int(product['Lead time'])
    initial_stock = int(product['Stock levels'])
    
    # Generate demand and forecast
    demand = generate_demand_series(df, sku, num_periods)
    forecast = generate_forecast(demand, accuracy=0.85)
    
    print(f"\nProduct Details:")
    print(f"  Name:              SKU {sku}")
    print(f"  Lead Time:         {lead_time} days")
    print(f"  Current Stock:     {initial_stock} units")
    print(f"  Price:             ${product['Price']:.2f}")
    print(f"  Avg Monthly Sales: {product['Number of products sold']:.0f} units")
    
    print(f"\nDemand Statistics (90-day period):")
    print(f"  Mean Demand:       {demand.mean():.2f} units/day")
    print(f"  Std Deviation:     {demand.std():.2f} units/day")
    print(f"  Min Demand:        {demand.min():.2f} units/day")
    print(f"  Max Demand:        {demand.max():.2f} units/day")
    
    # Create simulator with realistic costs
    unit_price = product['Price']
    ordering_cost = 50  # Fixed cost per order
    holding_cost = unit_price * 0.2  # 20% of unit price per unit per period
    stockout_cost = unit_price * 2  # 2x unit price per stockout
    
    simulator = PolicySimulator(
        demand_data=demand,
        lead_time=lead_time,
        forecasted_demand=forecast,
        initial_inventory=initial_stock,
        ordering_cost=ordering_cost,
        holding_cost=holding_cost,
        stockout_cost=stockout_cost
    )
    
    print(f"\nCost Parameters:")
    print(f"  Ordering Cost:     ${ordering_cost:.2f} per order")
    print(f"  Holding Cost:      ${holding_cost:.2f} per unit per day")
    print(f"  Stockout Cost:     ${stockout_cost:.2f} per unit short")
    
    # Add naive policy
    avg_demand = demand.mean()
    naive_rop = avg_demand * lead_time * 1.5  # 1.5x safety factor
    naive_eoq = avg_demand * 7
    
    print(f"\nNaive Policy Configuration:")
    print(f"  Reorder Point:     {naive_rop:.2f} units")
    print(f"  Order Quantity:    {naive_eoq:.2f} units")
    
    simulator.add_policy(NaivePolicy(
        reorder_point=naive_rop,
        order_quantity=naive_eoq
    ))
    
    # Add smart policy
    print(f"\nSmart Policy Configuration:")
    print(f"  Service Level:     95%")
    print(f"  Dynamic ROP:       Based on forecast & lead time")
    print(f"  Order Quantity:    Adaptive")
    
    simulator.add_policy(SmartPolicy(service_level=0.95))
    
    # Run simulation
    print(f"\n{'─'*80}")
    print(f"Running simulation...")
    simulator.run_simulation()
    
    # Display results
    simulator.print_summary()
    
    # Analysis
    metrics = simulator.get_raw_metrics()
    policies = list(metrics.keys())
    
    if len(policies) == 2:
        metric_names = ['Naive (Fixed ROP & Fixed EOQ)', 'Smart (Forecast + Lead Time + Risk, SL=0.95)']
        naive_metrics = metrics[metric_names[0]]
        smart_metrics = metrics[metric_names[1]]
        
        print("\nPERFORMANCE COMPARISON:")
        print(f"  Stockout Reduction:     {((naive_metrics.total_stockouts - smart_metrics.total_stockouts) / max(naive_metrics.total_stockouts, 1) * 100):.1f}%")
        print(f"  Cost Savings:           ${(naive_metrics.total_cost - smart_metrics.total_cost):.2f} ({((naive_metrics.total_cost - smart_metrics.total_cost) / naive_metrics.total_cost * 100):.1f}%)")
        print(f"  Inventory Reduction:    {((naive_metrics.average_inventory - smart_metrics.average_inventory) / naive_metrics.average_inventory * 100):.1f}%")
    
    return {
        'sku': sku,
        'simulator': simulator,
        'demand': demand,
        'forecast': forecast
    }


def run_batch_simulation(csv_path: str, num_skus: int = 3) -> list:
    """
    Run simulation for multiple SKUs
    """
    df = load_supply_chain_data(csv_path)
    if df is None:
        return []
    
    results = []
    
    # Select SKUs
    skus = df['SKU'].unique()[:num_skus]
    
    for sku in skus:
        try:
            result = run_policy_simulation_for_sku(df, sku, num_periods=90)
            results.append(result)
        except Exception as e:
            print(f"✗ Error simulating {sku}: {e}")
    
    return results


def main():
    """Main entry point"""
    # Try to find supply chain data
    current_dir = Path(__file__).parent.absolute()
    
    possible_paths = [
        current_dir / "supply_chain_data.csv",
        Path("d:/Manoj/Engineering/Btech/4th year/tata-tech/supply_chain_data.csv"),
        Path("d:/Manoj/Engineering/Btech/4th year/tata-tech/tata-supply-chain/supply_chain_data.csv"),
    ]
    
    csv_path = None
    for path in possible_paths:
        if path.exists():
            csv_path = str(path)
            print(f"✓ Found data file: {csv_path}")
            break
    
    if csv_path is None:
        print("✗ supply_chain_data.csv not found")
        print("Please provide the path to your supply chain data CSV file")
        return
    
    # Run batch simulation
    print("\nStarting Policy Simulator...")
    results = run_batch_simulation(csv_path, num_skus=3)
    
    if results:
        print(f"\n{'='*80}")
        print(f"SUMMARY: Simulated {len(results)} products")
        print(f"{'='*80}")


if __name__ == "__main__":
    main()
