import numpy as np
import matplotlib.pyplot as plt
from utils import *


def runit():
    """
    Main function to set up parameters, compute equilibrium, and simulate the market.
    """
    # Model Parameters
    c = {
        "MAX_FIRMS": 3,  # Max number of active firms
        "KMAX": 19,  # Max efficiency level
        "START_FIRMS": 1,  # Start equilibrium computation with 1 firm
        "ENTRY_LOW": 0.15,  # Lower bound for entry cost
        "ENTRY_HIGH": 0.25,  # Upper bound for entry cost
        "SCRAP_VAL": 0.1,  # Scrap value
        "ENTRY_AT": 4,  # Efficiency level at which new firms enter
        "BETA": 0.925,  # Discount factor
        "DELTA": 0.7,  # Probability of industry decline
        "INV_MULT": 3,  # Investment cost parameter
        "INTERCEPT": 3,  # Cournot demand intercept
        "FIXED_COST": 0.2,  # Fixed cost for firms
        "GAMMA": 1,  # Marginal cost coefficient
        "TOL": 0.1,  # Convergence tolerance
        "PROFIT_DONE": 0,  # Indicator for profit computation
        "EQL_DONE": 0,  # Indicator for equilibrium computation
        "PREFIX": "cc",  # Prefix for saved results
        "DS_WSTART": np.array([6,0,0]),  # Initial state for simulation (ENTRY_AT+2)
        "DS_NSIMX": 10000  # Number of simulation periods
    }

    ## **Compute Static Profits**
    print("Computing static profits...")
    static_profit(c)

    ## **Solve Dynamic Equilibrium**
    print("Solving for dynamic equilibrium...")
    eql_ma(c)

    ## **Simulate Entry & Exit**
    print("Simulating industry evolution (Baseline case)...")
    ds_ma(c, "Data/Out/baseline")

    ## **Low Entry Cost Case**
    c["ENTRY_LOW"] = 0.01
    c["ENTRY_HIGH"] = 0.11
    c["PREFIX"] = "cc_low"

    print("Computing static profits for low entry cost case...")
    static_profit(c)

    print("Solving for equilibrium under low entry cost...")
    eql_ma(c)

    print("Simulating industry evolution (Low entry cost case)...")
    ds_ma(c, "Data/Out/low_entry_cost")

if __name__ == "__main__":
    # runit()

    # baseline
    firms_count_avg = np.load("Data/Out/baseline_avg.npz")["firms_count_avg"]
    print("baseline firms count: ", firms_count_avg)
    investment_period_avg = np.load("Data/Out/baseline_avg.npz")["investment_period_avg"]
    print("baseline investment period: ", investment_period_avg)
    # low entry cost
    firms_count_avg = np.load("Data/Out/low_entry_cost_avg.npz")["firms_count_avg"]
    print("low entry cost firms count: ", firms_count_avg)
    investment_period_avg = np.load("Data/Out/low_entry_cost_avg.npz")["investment_period_avg"]
    print("low entry cost investment period: ", investment_period_avg)

    # Plot firms count
    firms_count_b=np.load("Data/Out/baseline.npz")["firms_count"]
    firms_count_l=np.load("Data/Out/low_entry_cost.npz")["firms_count"]
    plt.figure(figsize=(12, 6))
    plt.plot(firms_count_b[:100], label='Baseline', color='red')
    plt.plot(firms_count_l[:100], label='Low Entry Cost', color='blue')
    plt.legend()
    plt.xlabel('Time Period')
    plt.ylabel('Firms Count')
    plt.title('Firms Count over Time')
    plt.legend()
    plt.grid(True)
    plt.savefig(f'Figures/Firms_count_start.pdf')

    plt.plot(firms_count_b[-100:], label='Baseline', color='red')
    plt.plot(firms_count_l[-100:0], label='Low Entry Cost', color='blue')
    plt.legend()
    plt.xlabel('Time Period')
    plt.ylabel('Firms Count')
    plt.title('Firms Count over Time')
    plt.legend()
    plt.grid(True)
    plt.savefig(f'Figures/Firms_count_end.pdf')

    # Plot average investment per period
    
    invesment_history_b=np.load("Data/Out/baseline.npz")["investment_history"]
    investment_history_b = np.mean(invesment_history_b,axis=1)
    invesment_history_l=np.load("Data/Out/low_entry_cost.npz")["investment_history"]
    investment_history_l = np.mean(invesment_history_l,axis=1)
    plt.figure(figsize=(12, 6))
    plt.plot(investment_history_b[:100], label='Baseline', color='red')
    plt.plot(investment_history_l[:100], label='Low Entry Cost', color='blue')
    plt.legend()
    plt.xlabel('Time Period')
    plt.ylabel('Average Investment')
    plt.title('Average Investment per Period (Baseline)')
    plt.grid(True)
    plt.savefig(f'Figures/Average_investment_trajectory_start.pdf')

    plt.figure(figsize=(12, 6))
    plt.plot(investment_history_b[-100:0], label='Baseline', color='red')
    plt.plot(investment_history_l[-100:0], label='Low Entry Cost', color='blue')
    plt.legend()
    plt.xlabel('Time Period')
    plt.ylabel('Average Investment')
    plt.title('Average Investment per Period (Baseline)')
    plt.grid(True)
    plt.savefig(f'Figures/Average_investment_trajectory_end.pdf')




    
 

