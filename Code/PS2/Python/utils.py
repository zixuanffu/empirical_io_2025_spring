import numpy as np

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

def decode(code, nfirms, binom):
    """
    Decodes an integer state code into a weakly descending n-tuple.

    Args:
        code (int): Encoded integer state index.
        nfirms (int): Number of firms (size of the tuple).
        binom (numpy.ndarray): Binomial coefficient matrix.

    Returns:
        list: Weakly descending N-tuple.
    """
    ntuple = np.zeros(nfirms, dtype=int)  # Initialize output n-tuple
    w = len(ntuple) - 1  # Maximum possible value for tuple elements

    # Iterate over each firm
    for i in range(nfirms):
        # Find the largest possible value for the current tuple element
        while binom[nfirms - i + w, w + 1] > code:
            w -= 1
        
        # Assign the decoded value ensuring a weakly descending order
        ntuple[i] = w + 1
        # Adjust the code value
        code -= binom[nfirms - i + w, w + 1]

    return ntuple.tolist()


def encode(ntuple, nfirms, binom):
    """
    Encodes a weakly descending n-tuple into an integer state code.

    Args:
        ntuple (list): Weakly descending n-tuple.
        nfirms (int): Number of firms.
        binom (numpy.ndarray): Binomial coefficient matrix.

    Returns:
        int: Encoded integer state code.
    """
    code = 0  # Initialize encoded state code
    w = 0  # Tracks the largest element processed

    # Iterate over each firm
    for i in range(nfirms):
        w = ntuple[i] - 1  # Convert to zero-based index
        code += binom[nfirms - i + w, w + 1]  # Add the skipped states

    return code

def qdecode(code, dtable):
    """
    Quickly decodes a previously encoded number into a weakly descending n-tuple.

    Args:
        code (int): Encoded integer state code.
        dtable (numpy.ndarray): Decoding lookup table.

    Returns:
        numpy.ndarray: Decoded weakly descending n-tuple.
    """
    return dtable[:, code]  # Retrieve the n-tuple from the decoding table

def qencode(ntuple, etable, multfac):
    """
    Quickly encodes a weakly descending n-tuple using a precomputed lookup table.

    Args:
        ntuple (list): Weakly descending n-tuple.
        etable (numpy.ndarray): Encoding lookup table.
        multfac (numpy.ndarray): Multiplication factor for encoding.

    Returns:
        int: Encoded integer state code.
    """
    index = np.sum(np.array(ntuple) * np.array(multfac)) + 1  # Compute index
    return etable[index]  # Lookup encoded value

def calcval(place, w, x, k, oldvalue, etable, multfac, two_n, kmax, nfirms, mask, delta, a):
    """
    Computes the continuation value for increasing the efficiency level (moving up) and staying 
    at the same efficiency level for a given state w.

    Args:
        place (int): Position of the firm in the state tuple.
        w (list or np.ndarray): State tuple (nfirms x 1).
        x (list or np.ndarray): Investment strategy in state w (nfirms x 1).
        k (int): Own efficiency level (integer).
        oldvalue (np.ndarray): Value function from previous iteration.
        etable (np.ndarray): Encoding lookup table.
        multfac (np.ndarray): Multiplication factors for encoding.
        two_n (int): Number of rival action combinations (2^(nfirms-1)).
        kmax (int): Maximum efficiency level.
        nfirms (int): Number of firms.
        mask (np.ndarray): Binary outcomes of rivals.
        delta (float): Probability of industry aggregate decline.
        a (float): Investment cost multiplier.

    Returns:
        tuple: (val_up, val_stay) - The value of moving up and staying at the same efficiency level.
    """
    # Lower and upper bounds for efficiency levels
    z1 = np.zeros(nfirms, dtype=int)  # Lower bound (0)
    z2 = np.full(nfirms, kmax, dtype=int)  # Upper bound (kmax)

    # Adjust "mask" based on firm's position
    if nfirms > 1:
        if place == 1:
            locmask = np.vstack([np.zeros((1, two_n)), mask])
        elif place == nfirms:
            locmask = np.vstack([mask, np.zeros((1, two_n))])
        else:
            locmask = np.vstack([mask[:place - 1], np.zeros((1, two_n)), mask[place - 1:]])
    else:
        locmask = np.zeros((1, 1), dtype=int)

    # Modify investment and state
    x[place - 1] = 0  # Own investment is set to zero
    w[place - 1] = k  # Own efficiency level is updated
    justone = np.zeros(nfirms, dtype=int)  # Dummy vector
    justone[place - 1] = 1  # Mark this firm's position

    # Probability of moving up
    p_up = (a * np.array(x)) / (1 + a * np.array(x))

    # Initialize output values
    val_up = 0
    val_stay = 0

    for i in range(two_n):
        # Compute transition probability
        probmask = np.prod((locmask[:, i] * p_up) + ((1 - locmask[:, i]) * (1 - p_up)))

        # Value when firm does NOT move up
        d = np.array(w) + locmask[:, i]  # Private shock
        temp = np.column_stack([d, justone])
        temp = temp[temp[:, 0].argsort()[::-1]]  # Sort in descending order
        d = temp[:, 0]
        e = d - 1  # Aggregate shock

        # Check boundaries
        e = np.maximum(e, z1)
        d = np.minimum(d, z2)
        pl1 = np.argmax(temp[:, 1])  # Find "place" in the new state

        # Update expected value for staying at efficiency level
        val_stay += ((1 - delta) * oldvalue[qencode(d.tolist(), etable, multfac), pl1] +
                     delta * oldvalue[qencode(e.tolist(), etable, multfac), pl1]) * probmask

        # **Task Completed: Compute value for k_v + 1 (moving up in efficiency)**
        new_d = d.copy()
        new_d[pl1] = min(new_d[pl1] + 1, kmax)  # Increase efficiency level
        
        # Compute expected value when firm moves up
        val_up += ((1 - delta) * oldvalue[qencode(new_d.tolist(), etable, multfac), pl1] +
                   delta * oldvalue[qencode(e.tolist(), etable, multfac), pl1]) * probmask

    return val_up, val_stay

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


import numpy as np
from qdecode import qdecode

def contract(oldvalue, oldx, profit, dtable, etable, multfac, wmax, two_n, kmax, nfirms, mask, x_entryl, x_entryh, phi, entry_k, beta, delta, a):
    """
    Performs one iteration of policy and value function updates.

    Args:
        oldvalue (np.ndarray): Previous value function.
        oldx (np.ndarray): Previous investment strategy.
        profit (np.ndarray): Profit function for each state.
        dtable (np.ndarray): Decoding table for states.
        etable (np.ndarray): Encoding table for quick lookup.
        multfac (np.ndarray): Multiplication factors for encoding.
        wmax (int): Maximum number of states.
        two_n (int): Number of rival action combinations (2^(nfirms-1)).
        kmax (int): Maximum efficiency level.
        nfirms (int): Number of firms.
        mask (np.ndarray): Binary outcomes of rivals.
        x_entryl (float): Lower bound for entry cost.
        x_entryh (float): Upper bound for entry cost.
        phi (float): Scrap value.
        entry_k (int): Efficiency level at which new firms enter.
        beta (float): Discount factor.
        delta (float): Probability of industry aggregate decline.
        a (float): Investment cost multiplier.

    Returns:
        tuple: (newvalue, newx, isentry) - Updated value function, policy, and entry probabilities.
    """
    ## **Step 1: Entry Decision Calculation**
    isentry = np.zeros(wmax)  # Initialize entry probability

    for w in range(wmax):
        locw = qdecode(w, dtable)  # Decode state

        # Check if entry is possible (if last firm slot is empty)
        if locw[nfirms - 1] == 0:
            _, v1 = calcval(nfirms, locw, oldx[w, :], entry_k, oldvalue, etable, multfac, two_n, kmax, nfirms, mask, delta, a)
            val = beta * v1  # Compute expected value of entry
            isentry[w] = (val - x_entryl) / (x_entryh - x_entryl)

    # Ensure probabilities stay within [0,1]
    isentry = np.clip(isentry, 0, 1)

    ## **Step 2: Investment Decision and Value Function Update**
    newx = np.zeros((wmax, nfirms))  # Initialize investment policy
    newvalue = np.zeros((wmax, nfirms))  # Initialize value function

    for w in range(wmax):
        newx[w, :], newvalue[w, :] = optimize(w, oldvalue, oldx, isentry, profit, dtable, etable, multfac, two_n, kmax, nfirms, mask, phi, entry_k, beta, delta, a)

    return newvalue, newx, isentry

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

def initialize(dtable, nfirms, wmax, binom, newvalue, newx):
    """
    Initializes the value and policy functions for equilibrium computation.

    Args:
        dtable (np.ndarray): Decoding table for states.
        nfirms (int): Number of firms.
        wmax (int): Maximum number of states.
        binom (np.ndarray): Binomial coefficient matrix.
        newvalue (np.ndarray): Solved newvalue matrix for (nfirms - 1).
        newx (np.ndarray): Solved newx matrix for (nfirms - 1).

    Returns:
        tuple: (oldvalue, oldx) - Initial value function and policy function.
    """
    oldx = np.zeros((wmax, nfirms))  # Initialize policy function
    oldvalue = np.zeros((wmax, nfirms))  # Initialize value function

    if nfirms == 1:
        oldvalue = 1 + 0.1 * np.arange(1, wmax + 1)  # Base case for a single firm
    else:
        for w in range(wmax):
            tuple_w = qdecode(w, dtable)  # Decode the state

            # Map current state to corresponding (nfirms-1) equilibrium
            n = encode(tuple_w[:nfirms - 1], nfirms - 1, binom)
            oldvalue[w, :nfirms - 1] = newvalue[n, :nfirms - 1]
            oldx[w, :nfirms - 1] = newx[n, :nfirms - 1]

            # Initialize the last firm by swapping the last two firms and setting last firm to state 0
            tuple_w[nfirms - 2] = tuple_w[nfirms - 1]  # Swap last two firms
            tuple_w[nfirms - 1] = 0  # Set last firm to zero state
            n = encode(tuple_w, nfirms, binom)

            # Copy the results from (nfirms-1) game to initialize last firm
            oldvalue[w, nfirms - 1] = oldvalue[n, nfirms - 2]
            oldx[w, nfirms - 1] = oldx[n, nfirms - 2]

    return oldvalue, oldx

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
