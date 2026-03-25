#!/usr/bin/env python3
"""
QUICK START GUIDE - Policy Simulator

Run this file to quickly see the policy simulator in action!
"""

import sys
import os
from pathlib import Path

# Add the current directory to path
sys.path.insert(0, str(Path(__file__).parent))

def main():
    """Main entry point for quick start"""
    
    print("\n" + "="*80)
    print(" "*20 + "POLICY SIMULATOR - QUICK START")
    print("="*80 + "\n")
    
    print("📦 Importing modules...")
    try:
        from policy_simulator import example_simulation
        print("✓ Successfully imported policy_simulator\n")
    except ImportError as e:
        print(f"✗ Error importing: {e}")
        print("  Make sure you're in the correct directory")
        return
    
    print("🎯 OPTION 1: Run Built-in Simulation (No data file needed)")
    print("-" * 80)
    print("This will run with synthetic demand data (365 days)\n")
    
    try:
        response = input("Run example simulation? (y/n): ").strip().lower()
        if response == 'y':
            print("\n🚀 Running BUILT-IN EXAMPLE SIMULATION...\n")
            simulator = example_simulation()
            print("\n✓ Simulation completed successfully!")
    except KeyboardInterrupt:
        print("\n⚠️  Cancelled by user")
        return
    
    print("\n" + "="*80)
    print("📊 NEXT STEPS")
    print("="*80)
    print("""
1. READ THE DOCUMENTATION:
   Open: POLICY_SIMULATOR_README.md
   
2. RUN INTERACTIVE NOTEBOOK:
   Command: jupyter notebook Policy_Simulator.ipynb
   
3. USE YOUR OWN DATA:
   Command: python policy_simulator_integration.py
   Make sure you have supply_chain_data.csv in this directory
   
4. CUSTOMIZE & EXTEND:
   Edit policy_simulator.py to create custom policies
   Add new simulation parameters as needed

KEY FILES:
  ✓ policy_simulator.py                 - Core implementation
  ✓ policy_simulator_integration.py     - Real data integration  
  ✓ Policy_Simulator.ipynb             - Interactive notebook
  ✓ POLICY_SIMULATOR_README.md         - Full documentation
""")
    
    print("="*80)
    print("💡 TIP: The Jupyter notebook provides the best visualization experience!")
    print("="*80 + "\n")


if __name__ == "__main__":
    main()
