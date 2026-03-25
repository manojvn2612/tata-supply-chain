## Policy Simulator - Inventory Management

A comprehensive framework for comparing **Naive** and **Smart** inventory policies using simulation and analysis.

---

## 📋 Overview

This package implements two inventory management policies and compares them based on:

### Metrics Compared
- **Stockouts**: Number of periods with insufficient inventory
- **Average Inventory**: Mean inventory level (indicator of capital tied up)
- **Total Cost**: Sum of ordering, holding, and stockout costs

### Policy Implementations

#### 1. **Naive Policy** (Fixed ROP & Fixed EOQ)
- **Reorder Point (ROP)**: Fixed value based on average demand × lead time
- **Order Quantity (EOQ)**: Fixed quantity (typically 5-7 days of demand)
- **Characteristics**: Simple, easy to implement, doesn't adapt to demand changes

**Formula:**
```
ROP = Average Demand × Lead Time × Safety Factor (1.5)
EOQ = Average Demand × Order Days (5)
```

#### 2. **Smart Policy** (Forecast + Lead Time + Risk)
- **Dynamic ROP**: Based on forecasted demand and demand variability
- **Dynamic Order Qty**: Based on service level and safety stock calculations
- **Safety Stock**: Z × σ × √(Lead Time) - accounts for uncertainty

**Formula:**
```
Safety Stock = Z-score × Demand Std Dev × √(Lead Time)
ROP = (Forecast Demand × Lead Time) + Safety Stock
Order Qty = Target Level - Current Inventory
Target Level = ROP + (Forecast × 14 days)
```

---

## 📁 Files

### Core Implementation
1. **`policy_simulator.py`** - Main framework
   - Base classes: `InventoryPolicy`, `NaivePolicy`, `SmartPolicy`
   - `PolicySimulator` orchestrator class
   - `PolicyMetrics` dataclass for results
   - Example simulation function

2. **`policy_simulator_integration.py`** - Real data integration
   - Load supply chain data from CSV
   - Generate realistic demand patterns
   - Run batch simulations for multiple SKUs
   - Print detailed analysis

### Analysis & Visualization
3. **`Policy_Simulator.ipynb`** - Jupyter notebook
   - Complete simulation walkthrough
   - Interactive visualizations
   - Cost breakdowns
   - Performance comparisons

---

## 🚀 Quick Start

### Option 1: Run the Example (No Data Required)
```bash
cd tata-supply-chain
python policy_simulator.py
```

This runs a built-in simulation with synthetic demand data.

### Option 2: Run with Real Data
```bash
python policy_simulator_integration.py
```

Make sure your supply chain data CSV has these columns:
- `SKU`: Product SKU
- `Lead time`: Supplier lead time in days
- `Stock levels`: Current inventory
- `Number of products sold`: Historical demand
- `Price`: Unit price

### Option 3: Interactive Jupyter Notebook
```bash
jupyter notebook Policy_Simulator.ipynb
```

Run cells sequentially to see step-by-step analysis and visualizations.

---

## 📊 Example Output

### Metrics Comparison Table
```
Policy                          | Stockouts | Avg Inventory | Total Cost
─────────────────────────────────────────────────────────────────────────
Naive (Fixed ROP & Fixed EOQ)  |    12     |    234.50     | $24,650.00
Smart (Forecast + Risk, SL=95%)|     3     |    189.30     | $22,140.00
─────────────────────────────────────────────────────────────────────────

KEY INSIGHTS:
  Stockout Reduction:     75.0%
  Cost Savings:           $2,510.00 (10.2%)
  Inventory Reduction:    19.2%
```

### Generated Visualizations
1. **Demand Pattern** - Actual vs Forecasted demand
2. **Inventory Levels** - Comparison over time (full year + zoomed)
3. **Cost Breakdown** - Pie charts showing cost composition
4. **Performance Metrics** - Bar charts for stockouts, inventory, costs
5. **Cumulative Costs** - Trending and gains analysis

---

## 🔧 Customization

### Adjust Simulation Parameters

In `policy_simulator.py` or notebook:
```python
# Simulation period
SIMULATION_DAYS = 365

# Cost parameters
ORDERING_COST = 100      # $ per order
HOLDING_COST = 1         # $ per unit per day
STOCKOUT_COST = 50       # $ per unit short

# Lead time
LEAD_TIME = 7            # days

# Service level (Smart Policy)
SERVICE_LEVEL = 0.95     # 95% fill rate target
```

### Create Custom Policies

Extend `InventoryPolicy` class:
```python
from policy_simulator import InventoryPolicy

class MyPolicy(InventoryPolicy):
    def __init__(self):
        super().__init__("My Custom Policy")
    
    def calculate_order_qty(self, current_inv, demand, lead_time, forecast=None):
        # Your logic here
        should_order = current_inv < threshold
        order_qty = 100
        return order_qty, should_order

# Use in simulation
simulator.add_policy(MyPolicy())
```

---

## 📈 Key Insights & Interpretation

### When Naive Policy is Better
- **Stable demand** patterns
- **Low service level requirements**
- **High administrative costs** for frequent changes
- **Simplicity is valued** over optimization

### When Smart Policy is Better
- **Variable demand** (seasonal, trending)
- **High stockout costs** (critical items)
- **Forecast data available**
- **Cost optimization is priority**

### Cost Breakdown Analysis
- **Ordering Cost**: ↑ with frequency of orders
- **Holding Cost**: ↑ with inventory levels (biggest component typically)
- **Stockout Cost**: ↑ with demand variability and low safety stock

---

## 🧮 Mathematical Background

### Economic Order Quantity (EOQ)
Traditional formula (implemented in naive baseline):
$$EOQ = \sqrt{\frac{2DS}{H}}$$

Where:
- D = Annual demand
- S = Ordering cost per order
- H = Annual holding cost per unit

### Safety Stock & Service Level
For Normal distribution:
$$SS = Z_{\alpha} \times \sigma_{D} \times \sqrt{LT}$$

Where:
- Z = Service level Z-score
- σ_D = Demand standard deviation  
- LT = Lead time
- Common Z-scores:
  - 80% SL: Z = 0.84
  - 90% SL: Z = 1.28
  - 95% SL: Z = 1.645
  - 99% SL: Z = 2.33

---

## 📝 Data Requirements

Supply chain data CSV should contain:
```
SKU|Lead time|Stock levels|Number of products sold|Price|...
SKU0|7|58|802|69.81|...
SKU1|30|53|736|14.84|...
```

The simulator automatically:
- Generates demand patterns with seasonality
- Creates forecast estimates
- Scales costs to match unit prices
- Handles missing or variable data

---

## 🎯 Use Cases

1. **Product Classification**: Use Smart for high-value items, Naive for low-margin
2. **Supplier Evaluation**: Test different lead times
3. **Service Level Setting**: Adjust target fill rates
4. **Seasonal Planning**: Input seasonal demand forecasts
5. **Cost Sensitivity Analysis**: Test different cost parameters

---

## ⚠️ Assumptions

- **Demand**: Normally distributed with historical mean and variance
- **Lead Time**: Constant (no variability)
- **Ordering**: Instantaneous (no batch constraints)
- **Planning Horizon**: Adequate history for demand estimation
- **Costs**: Linear, constant across units

---

## 🔮 Future Enhancements

- [ ] Multi-echelon inventory systems
- [ ] Stochastic lead times
- [ ] Demand forecasting integration
- [ ] Multi-product optimization
- [ ] Constraints (storage, budget, min/max orders)
- [ ] Real-time dashboard

---

## 📚 References

- Wilson, R. H. (1934). "A Scientific Routine for Stock Control"
- Silver, E. A., & Peterson, R. (1985). "Decision Systems for Inventory Management"
- Nahmias, S. (2015). "Production and Operations Analysis" (7th ed.)

---

## 📧 Questions?

Refer to the Jupyter notebook for detailed walkthroughs and visual explanations.
