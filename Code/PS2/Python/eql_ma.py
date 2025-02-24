def eql_ma(c):
    """
    Computes the dynamic equilibrium using backward induction.

    Args:
        c (dict): Model parameters containing:
            - MAX_FIRMS (int): Max number of active firms.
            - KMAX (int): Max efficiency level.
            - START_FIRMS (int): Starting number of firms for equilibrium computation.
            - ENTRY_LOW (float): Lower bound for entry cost.
            - ENTRY_HIGH (float): Upper bound for entry cost.
            - SCRAP_VAL (float): Scrap value.
            - ENTRY_AT (int): Efficiency level at which new firms enter.
            - BETA (float): Discount factor.
            - DELTA (float): Probability of industry aggregate decline.
            - INV_MULT (float): Investment cost multiplier.
            - TOL (float): Tolerance for convergence.
            - PREFIX (str): Prefix for saving results.
    """
    rlnfirms = c["MAX_FIRMS"]
    kmax = c["KMAX"]
    stfirm = c["START_FIRMS"]

    x_entryl = c["ENTRY_LOW"]
    x_entryh = c["ENTRY_HIGH"]
    phi = c["SCRAP_VAL"]
    entry_k = c["ENTRY_AT"]
    beta = c["BETA"]
    delta = c["DELTA"]
    a = c["INV_MULT"]
    tol = c["TOL"]

    # Set up binomial coefficients for encoding/decoding n-tuples
    binom = np.eye(rlnfirms + kmax + 1, dtype=int)
    binom = np.hstack((np.zeros((rlnfirms + kmax + 1, 1), dtype=int), binom))
    
    for i in range(2, rlnfirms + kmax + 1):
        binom[i, 1:i] = binom[i - 1, 1:i] + binom[i - 1, :i - 1]

    # Solve for the game starting with small firms and expanding
    for nfirms in range(stfirm, rlnfirms + 1):
        wmax = binom[nfirms + kmax + 1, kmax + 2]  # Compute number of industry states
        
        print(f"\nFirms: {nfirms}   States: {wmax}\nInitialization ...")

        # Load profit data
        filename = f"a.{c['PREFIX']}_pr{nfirms}.npz"
        profit = np.load(filename)["profit"]

        two_n = 2 ** (nfirms - 1)  # Number of rival actions

        # Generate binary matrix of all rival investment outcomes
        mask = np.array([[int(x) for x in format(i, f"0{nfirms-1}b")] for i in range(2 ** (nfirms - 1))])

        ## Create decoding table: maps state index to industry structure
        dtable = np.zeros((nfirms, wmax), dtype=int)
        for i in range(wmax):
            dtable[:, i] = decode(i, nfirms, binom)

        ## Create encoding table
        multfac = (kmax + 1) ** np.arange(nfirms)  # Allows mapping without sorting

        # Generate all possible states
        wgrid = np.meshgrid(*[np.arange(kmax + 1)] * nfirms, indexing="ij")
        wtable = np.column_stack([wg.flatten() for wg in wgrid])
        wtable = np.sort(wtable, axis=1)[:, ::-1]  # Ensure weakly descending order

        # Encode each state into a unique index
        etable = np.array([encode(w.tolist(), nfirms, binom) for w in wtable])

        ## Initialize value and policy functions
        oldvalue, oldx = initialize(dtable, nfirms, wmax, binom, None, None)

        ## Perform iterations using contraction mapping
        print("Contraction ...")
        ix = 1
        norm = tol + 1
        avgnorm = norm
        while norm > tol and avgnorm > 0.0001 * tol:
            newvalue, newx, isentry = contract(
                oldvalue, oldx, profit, dtable, etable, multfac, wmax, two_n, kmax, nfirms, mask,
                x_entryl, x_entryh, phi, entry_k, beta, delta, a
            )

            norm = np.max(np.abs(oldvalue - newvalue))
            avgnorm = np.mean(np.abs(oldvalue - newvalue))

            print(f"  {ix:2d}    Sup norm: {norm:8.4f}      Mean norm: {avgnorm:8.4f}")
            ix += 1

            oldx = newx
            oldvalue = newvalue

        ## Check if there is investment at highest efficiency level
        w = np.zeros(nfirms, dtype=int)
        w[0] = kmax  # Set highest efficiency level
        if np.max(newx[qencode(w.tolist(), etable, multfac): wmax, 0]) > 0:
            print("Warning: Positive investment recorded at highest efficiency level.")
            print("Consider increasing the maximum efficiency level (kmax).")

        ## Save results
        prising = a * newx / (1 + a * newx)
        np.savez(f"a.{c['PREFIX']}_markov{nfirms}.npz", newvalue=newvalue, newx=newx, prising=prising, isentry=isentry)

    c["EQL_DONE"] = 1  # Mark equilibrium computation as complete
