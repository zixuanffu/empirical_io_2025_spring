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
