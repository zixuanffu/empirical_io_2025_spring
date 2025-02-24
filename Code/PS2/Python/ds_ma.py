import numpy as np

def ds_ma(c, out_file):
    """
    Simulates the dynamic oligopoly game for a specified number of periods 
    and computes summary statistics.

    Args:
        c (dict): Model parameters containing:
            - DS_WSTART (list or np.ndarray): Initial state for simulation.
            - DS_NSIMX (int): Number of simulation periods.
        out_file (str): Output file name for saving results.
    """
    wstart = np.array(c["DS_WSTART"])  # Initial state for simulation
    numtimes = c["DS_NSIMX"]  # Number of simulation periods

    # Initialize state tracking
    state_history = np.zeros((numtimes, len(wstart)), dtype=int)
    firms_count = np.zeros(numtimes, dtype=int)  # Track active firms count
    total_investment = np.zeros(numtimes, dtype=float)  # Track total investment

    # Set initial state
    state = wstart.copy()

    for t in range(numtimes):
        state_history[t] = state  # Record current state
        firms_count[t] = np.sum(state > 0)  # Count active firms

        # Compute investment decision using optimize function
        encoded_state = qencode(state.tolist(), etable, multfac)
        investment, _ = optimize(encoded_state, oldvalue, oldx, isentry, profit, dtable, etable, multfac, two_n, kmax, nfirms, mask, phi, entry_k, beta, delta, a)

        total_investment[t] = np.sum(investment)  # Track total investment

        # Update state based on investment and exogenous shocks
        state = update_state(state, investment, c)

    # Compute statistics
    avg_firms = np.mean(firms_count)
    avg_investment = np.mean(total_investment)

    # Save results
    np.savez(out_file, avg_firms=avg_firms, avg_investment=avg_investment, state_history=state_history)

    print(f"Simulation completed. Results saved in {out_file}")
