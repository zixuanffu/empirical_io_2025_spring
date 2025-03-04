def optimize(w, oldvalue, oldx, isentry, profit, dtable, etable, multfac, two_n, kmax, nfirms, mask, phi, entry_k, beta, delta, a):
    """
    Computes optimal investment and value function for a given state w.

    Args:
        w (int): State code.
        oldvalue (np.ndarray): Previous value function.
        oldx (np.ndarray): Previous investment policy.
        isentry (np.ndarray): Entry probability for a new firm.
        profit (np.ndarray): Profit function for each state.
        dtable (np.ndarray): Decoding table for states.
        etable (np.ndarray): Encoding table for quick lookup.
        multfac (np.ndarray): Multiplication factors for encoding.
        two_n (int): Number of rival action combinations (2^(nfirms-1)).
        kmax (int): Maximum efficiency level.
        nfirms (int): Number of firms.
        mask (np.ndarray): Binary outcomes of rivals.
        phi (float): Scrap value.
        entry_k (int): Entry efficiency level.
        beta (float): Discount factor.
        delta (float): Probability of industry aggregate decline.
        a (float): Investment cost multiplier.

    Returns:
        tuple: (nx_t, nval_t) - Optimal investment strategy and updated value function.
    """
    # Decode the state
    locw = qdecode(w, dtable)  # Efficiency levels of firms
    locwx = locw.copy()  # State after exit decisions
    oval = oldvalue[w, :].copy()  # Old value function for the state
    ox = oldx[w, :].copy()  # Old investment levels
    nval = np.zeros(nfirms)  # New value function
    nx = np.zeros(nfirms)  # New investment policy

    ## **Exit Decision: Identify Firms That Exit**
    for j in range(nfirms):
        if locwx[j] == 0:
            nval[j] = phi  # Firm exits, gets scrap value
        else:
            for k in range(j + 1, nfirms):  # If lower efficiency firms exist, they also exit
                if locwx[k] <= locwx[j]:
                    locwx[k] = 0  # Mark as exited

    ## **Entry Decision: Compute Entry Probability**
    locwe = locwx.copy()  # Copy state for the case of entry
    if locwe[-1] == 0:  # If the last position is empty, entry is possible
        locwe[-1] = entry_k  # Assign entry efficiency level


    ## **Compute Optimal Investment Strategy**
    for j in range(nfirms):
        if locwx[j] == 0:  # Firm exits
            nval[j] = phi
            continue

        # Compute continuation values for investing and not investing
        val_up, val_stay = calcval(j + 1, locwx, ox, locwx[j], oldvalue, etable, multfac, two_n, kmax, nfirms, mask, delta, a)

        # Compute expected value
        expected_val = (1 - isentry[w]) * val_stay + isentry[w] * val_up

        # Optimal investment decision (bounded to [0,1])
        nx[j] = max(0, min(1, (expected_val - profit[w, j]) / a))

        # Update value function
        nval[j] = profit[w, j] + beta * expected_val

        # Optional refinement: recheck exit decision
        if nval[j] < phi:
            nval[j] = phi
            nx[j] = 0
            locwx[j] = 0
            for k in range(j+1, nfirms):
                if locwx[k] <= locwx[j]:
                    locwx[k] = 0

        # Update investment policy for remaining firms
        ox[j] = nx[j]

    return nx.tolist(), nval.tolist()

