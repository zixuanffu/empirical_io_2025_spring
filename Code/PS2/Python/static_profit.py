def static_profit(c):
    """
    Computes static Cournot competition profits for different industry structures.

    Args:
        c (dict): Model parameters containing:
            - MAX_FIRMS (int): Max number of active firms.
            - KMAX (int): Max efficiency level.
            - INTERCEPT (float): Cournot demand intercept.
            - FIXED_COST (float): Fixed cost per firm.
            - GAMMA (float): Cournot marginal cost coefficient.
            - PREFIX (str): Filename prefix for saving results.
    """

    nfmax = c["MAX_FIRMS"]  # Max number of active firms
    kkmax = c["KMAX"]  # Max efficiency level

    # Set up binomial coefficients for encoding/decoding n-tuples
    kmax = kkmax
    binom = np.eye(nfmax + kmax + 2, dtype=int)
    binom = np.hstack((np.zeros((nfmax + kmax + 2, 1), dtype=int), binom))  # Add leading zero column

    for i in range(1, nfmax + kmax + 2):
        binom[i, 1:i] = binom[i - 1, 1:i] + binom[i - 1, :i - 1]

    for nfirms in range(1,nfmax+1):
        print(f"\nFirms: {nfirms}")

        # Compute the number of descending n-tuples
        descn = binom[nfirms + kmax, kmax + 1]
        print(f"Industry structures to compute: {descn}")

        # Extract Cournot parameters
        D = c["INTERCEPT"]
        f = c["FIXED_COST"]
        ggamma = c["GAMMA"]

        # Compute static Cournot profits
        profit = ccprofit(nfirms, descn, binom, D, f, ggamma)

        # Save output
        filename = f"a.{c['PREFIX']}_pr{nfirms}.npz"
        np.savez(filename, profit=profit)

    c["PROFIT_DONE"] = 1  # Mark profit computation as done
