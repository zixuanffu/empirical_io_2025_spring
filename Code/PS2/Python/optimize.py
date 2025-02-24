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
    locwe = locwx.copy()  # Copy state for entry decision
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

        # Compute optimal investment using a simple closed-form formula
        nx[j] = max(0, min(1, (expected_val - profit[w, j]) / a))

        # Compute new value function with investment
        nval[j] = profit[w, j] + beta * expected_val

    return nx.tolist(), nval.tolist()

def ccprofit(nfirms, descn, binom, D, f, ggamma):
    """
    Computes the profit and market share for the static Cournot competition 
    for each firm in all states.

    Args:
        nfirms (int): Number of active firms.
        descn (int): Number of possible industry structures.
        binom (numpy.ndarray): Binomial coefficient matrix.
        D (float): Cournot demand intercept.
        f (float): Fixed cost per firm.
        ggamma (float): Marginal cost coefficient.

    Returns:
        numpy.ndarray: Profit matrix of shape (descn, nfirms).
    """
    profit = np.zeros((descn, nfirms))  # Initialize profit matrix

    for i in range(descn):
        if i % 50 == 0:  # Print progress every 50 iterations
            print(f"  Computed: {i}")

        w = decode(i, nfirms, binom)  # Decode state i into efficiency levels
        theta = ggamma * np.exp(-(np.array(w) - 4))  # Compute marginal cost

        # Solve for Cournot equilibrium: Reduce number of firms until all produce positive quantity
        n = nfirms
        p = (D + np.sum(theta[:n])) / (n + 1)  # Equilibrium price

        while not ((p - theta[n - 1] >= 0) or (n == 1)):  # Reduce n if price makes last firm unprofitable
            n -= 1
            p = (D + np.sum(theta[:n])) / (n + 1)

        q = np.zeros(nfirms)  # Initialize output quantities
        if p - theta[n - 1] > 0:  # Ensure positive quantity production
            q[:n] = p - theta[:n]

        quan = q  # Equilibrium quantity

        pstar = D - np.sum(quan)  # Equilibrium price
        profstar = (pstar > theta) * (pstar - theta) * quan - f  # Compute profits

        profit[i, :] = profstar  # Store computed profits

    return profit

