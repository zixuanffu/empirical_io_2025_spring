import numpy as np
import matplotlib.pyplot as plt

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
        binom[i, 1:i+1] = binom[i - 1, 1:i+1] + binom[i - 1, :i]
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
        filename = f"Data/Out/a.{c['PREFIX']}_pr{nfirms}.npz"
        np.savez(filename, profit=profit)

    c["PROFIT_DONE"] = 1  # Mark profit computation as done

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
        p = (D + np.sum(theta)) / (n + 1)  # Equilibrium price

        while not ((p - theta[n - 1] >= 0) or (n == 1)):  # Reduce n if price makes last firm unprofitable
            n -= 1
            p = (D + np.sum(theta[:n])) / (n + 1)

        q = np.zeros(nfirms)  # Initialize output quantities
        if p - theta[n - 1] > 0:  # Ensure positive quantity production
            q[:n] = p - theta[:n]

        quan = q  # Equilibrium quantity

        pstar = D - np.sum(quan)  # Equilibrium price
        profstar = (pstar > theta) * (pstar - theta) * quan - f # Compute profits
        profit[i, :] = profstar  # Store computed profits
        
    return profit

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
    
    # Iterate over each firm in the tuple
    for i in range(nfirms):
        row = nfirms - i - 1
        col = 1
        while (code >= binom[row, col]):
            code -= binom[row, col]
            row += 1
            col += 1
        ntuple[i] = col-1

    return ntuple


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
    code = 0  # Initialize state code
    for i in range(nfirms):
        for j in range(ntuple[i]):
            code += binom[nfirms - i -1 +j ,1+j]

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
    return dtable[:, code].copy()   # Retrieve the n-tuple from the decoding table

def qencode(ntuple, etable, multfac):
    """
    Quickly encodes a weakly descending n-tuple using a precomputed lookup table.

    Args:
        ntuple (list): Weakly descending n-tuple.
        etable (numpy.ndarray): Encoding lookup table.
        multfac (numpy.ndarray): Multiplication factor for encoding.

    Returns:s
        int: Encoded integer state code.
    """
    index = np.sum(np.array(ntuple) * np.array(multfac)).astype(int)+ 1  # Compute index
    return etable[index-1].copy()  # Lookup encoded value

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

    # Adjust "mask" based on firm's position
    if nfirms > 1:
        zeros_row = np.zeros((1, two_n))
        if place == 0:
            locmask = np.vstack([zeros_row, mask])
        elif place == nfirms-1:
            locmask = np.vstack([mask, zeros_row])
        else:
            locmask = np.vstack([mask[:place], zeros_row, mask[place:]])
    else:
        locmask = np.zeros((1, 1), dtype=int)
    
    # Modify investment and state
    x[place] = 0  # Own investment is set to zero
    w[place] = k  # Own efficiency level is updated

    # Probability of moving up
    p_up = a * x / (1 + a * x)

    # Initialize output values
    val_up = 0
    val_stay = 0

    for i in range(two_n):
        # Compute transition probability
        probmask = np.prod((locmask[:, i] * p_up) + ((1 - locmask[:, i]) * (1 - p_up)))

        # Value when firm does NOT move up
        d = w + locmask[:, i]  # Private shock
        sorted_idx1 = np.argsort(d)[::-1]  # Sort in descending order
        pl1 = np.where(sorted_idx1 == place)[0][0]  # Find "place" in the new state
        d = d[sorted_idx1]
        e = d - 1  # Aggregate shock
    
        # Check boundaries
        e = np.maximum(e, 0)
        d = np.minimum(d, kmax)
     
        # issue: turn every one dimensional vector into a matrix (solved)
        # Update expected value for staying at efficiency level
        val_stay += ((1 - delta) * oldvalue[qencode(d, etable, multfac), pl1] +
                     delta * oldvalue[qencode(e, etable, multfac), pl1]) * probmask

        # **Task Completed: Compute value for k_v + 1 (moving up in efficiency)**
        new_d = w + locmask[:, i]  # Private shock
        # issue: should not copy d which is already sorted (solved)
        new_d[place] = k+1  # Increase efficiency level
        sorted_idx2 = np.argsort(new_d)[::-1]  # Sort in descending order
        pl2 = np.where(sorted_idx2 == place)[0][0]  # Find "place" in the new state
        new_d = new_d[sorted_idx2]
        new_e = new_d - 1  # Aggregate shock

        new_e = np.maximum(new_e, 0)  # Check lower bound
        new_d = np.minimum(new_d, kmax)  # Check upper bound
        
        # Compute expected value when firm moves up
        val_up += ((1 - delta) * oldvalue[qencode(new_d, etable, multfac), pl2] +
                   delta * oldvalue[qencode(new_e, etable, multfac), pl2]) * probmask

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
        (nx_t, nval_t): Optimal investment strategy and updated value function.
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
        if oval[j]<phi:
            locwx[j] = 0
            locwx[locwx<=locwx[j]] = 0
    
    ## **Entry Decision: Compute Entry Probability** 
    locwe = locwx.copy()  # Copy state for entry decision
    if locwe[-1] == 0:  # If the last position is empty, entry is possible
        pentry = isentry[w]  # Entry probability
        locwe[-1] = entry_k  # Assign entry efficiency level
    else: 
        pentry = 0

    ## **Compute Optimal Investment Strategy**
    for j in range(nfirms):
        if locwx[j] == 0:  # Firm exits
            nval[j] = phi
            nx[j] = 0
        else:

            # Compute (sub)continuation values when the potential entrant does not enter $\tilde{u}$
            val_up, val_stay = calcval(j, locwx, ox, locwx[j], oldvalue, etable, multfac, two_n, kmax, nfirms, mask, delta, a)

            # Compute (sub)continuation values when the potential entrant enters $\tilde{u}$
            # issue: sort the state after entry decision and get the new index
            sorted_idx = np.argsort(locwe)[::-1]
            j_e = np.where(sorted_idx == j)[0][0]
            locwe = locwe[sorted_idx]

            val_up_e, val_stay_e = calcval(j_e, locwe, ox, locwe[j_e], oldvalue, etable, multfac, two_n, kmax, nfirms, mask, delta, a)
        

            # Compute expected value \tilde{v}
            # if the firm realizes the investment,
            val_up_both = (1-pentry)*val_up+pentry*val_up_e
            val_stay_both = (1-pentry)*val_stay+pentry*val_stay_e
            
            # Compute optimal investment level using closed form formula
            # \frac{1}{a}(\sqrt{a(v_up_both-v_stay_both}-1)
            if locwx[j] == kmax or val_up_both<val_stay_both: # if the firm is at the highest efficiency level
                nx[j] = 0
            else:
                nx[j] = max(0,1/a*(np.sqrt(a*(val_up_both-val_stay_both)*beta)-1))
        
            p_up = (a * nx[j]) / (1 + a * nx[j])  # Probability of moving up
            
            expected_val = p_up*val_up_both+(1-p_up)*val_stay_both
            # Update value function with investment 
            # issue: investment cost (solved)
            pr = profit[qencode(locwx, etable, multfac), j]  # Profit for firm j
            nval[j] = pr - nx[j] + beta * expected_val

            # Optional refinement: recheck exit decision
            if nval[j] < phi:
                nval[j] = phi
                nx[j] = 0
                locwx[j] = 0
                for k in range(j+1, nfirms):
                    if locwx[k] <= locwx[j]:
                        locwx[k] = 0
                        nval[k] = phi
                        nx[k] = 0

                # issue: update everything including locwe, pentry (solved)
                locwe = locwx.copy()
                if locwe[-1] == 0:
                    pentry = isentry[qencode(locwx, etable, multfac)]
                    locwe[-1] = entry_k
                else:
                    pentry = 0
        # issue: should we update the state code? because the state changes while updating.  we should not
        
        # Update investment policy for remaining firms
        ox[j] = nx[j]

    return nx, nval


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
            _, v1 = calcval(nfirms-1, locw, oldx[w, :], entry_k, oldvalue, etable, multfac, two_n, kmax, nfirms, mask, delta, a)
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
    
    for i in range(1, rlnfirms + kmax + 1):
        binom[i, 1:i+1] = binom[i - 1, 1:i+1] + binom[i - 1, :i]
        

    # Solve for the game starting with small firms and expanding
    for nfirms in range(stfirm, rlnfirms + 1):
        wmax = binom[nfirms + kmax, kmax + 1]  # Compute number of industry states
        
        print(f"\nFirms: {nfirms}   States: {wmax}\nInitialization ...")

        # Load profit data
        filename = f"Data/Out/a.{c['PREFIX']}_pr{nfirms}.npz"
        profit = np.load(filename)["profit"]

        two_n = 2 ** (nfirms - 1)  # Number of rival actions

        # Generate binary matrix of all rival investment outcomes
        binary_strings = [format(i, f'0{nfirms-1}b') for i in range(two_n)]
        mask = np.array([[int(bit) for bit in binary_string] for binary_string in binary_strings]).T
       
        ## Create decoding table: maps state index to industry structure
        dtable = np.zeros((nfirms, wmax), dtype=int)
        for i in range(wmax):
            dtable[:, i] = decode(i, nfirms, binom)
        
        ## Create encoding table 
        # issue: not sure how it is done # note: from  0 to nfirm-1
        multfac = (kmax + 1) ** np.arange(nfirms)  # Allows mapping without sorting 
        
        
        # Generate all possible states
        wgrid = np.meshgrid(*[np.arange(kmax + 1)] * nfirms, indexing="ij")
        wtable = np.column_stack([wg.flatten() for wg in wgrid])
        wtable = np.sort(wtable, axis=1)[:, ::-1]  # Ensure weakly descending order

        # Encode each state into a unique index
        etable = np.array([encode(w, nfirms, binom) for w in wtable])
        np.savez(f"Data/Out/a.{c['PREFIX']}_table{nfirms}.npz", dtable = dtable, multfac=multfac,etable=etable, wtable=wtable,wgrid=wgrid)
        ## Initialize value and policy functions
        # issue: ensure even when nfirms = 1, the initialization is done correctly (solved)
        if nfirms == 1:
            oldvalue, oldx = initialize(dtable, nfirms, wmax, binom, None, None)
        else:
            oldvalue, oldx = initialize(dtable, nfirms, wmax, binom, newvalue, newx)

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
        if np.max(newx[qencode(w, etable, multfac): wmax, 0]) > 0:
            print("Warning: Positive investment recorded at highest efficiency level.")
            print("Consider increasing the maximum efficiency level (kmax).")

        ## Save results
        prising = a * newx / (1 + a * newx)
        np.savez(f"Data/Out/a.{c['PREFIX']}_markov{nfirms}.npz", newvalue=newvalue, newx=newx, prising=prising, isentry=isentry)

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
        # reshape to 2d array
        oldvalue = oldvalue.reshape(-1, 1)
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

def ds_ma(c, out_file):
    """
    Simulates the dynamic oligopoly game for a specified number of periods 
    and computes summary statistics.

    Args:
        c (dict): Model parameters containing:
            - DS_WSTART (list): Initial state for simulation.
            - DS_NSIMX (int): Number of simulation periods.
            - INV_MULT (float): Investment cost multiplier.
            - DELTA (float): Probability of industry aggregate decline.
            - KMAX (int): Maximum efficiency level.
            - SCRAP_VAL (float): Scrap value.
        out_file (str): Output file name for saving results.
    """
    wstart = np.array(c["DS_WSTART"])  # Initial state for simulation
    numtimes = c["DS_NSIMX"]  # Number of simulation periods
    a = c["INV_MULT"]  # Investment cost multiplier
    delta = c["DELTA"]  # Probability of industry aggregate decline
    kmax = c["KMAX"] # Maximum efficiency level
    nfirms = len(wstart) # Number of firms
    phi = c["SCRAP_VAL"]  # Scrap value
    entry_k = c["ENTRY_AT"]  # Entry efficiency level

    # Initialize state tracking
    state_history = np.zeros((numtimes, len(wstart)), dtype=int)
    firms_count_history = np.zeros(numtimes, dtype=int)  # Track active firms count
    investment_history= np.zeros((numtimes, len(wstart)), dtype=float)  # Track total investment

    # Simulate all the aggregate shocks with Bernoulli distribution with probability delta
    nu = np.random.binomial(1, delta, numtimes)

    # Load equilibrium objects
    newx = np.load(f"Data/Out/a.{c['PREFIX']}_markov{nfirms}.npz")["newx"]
    newvalue= np.load(f"Data/Out/a.{c['PREFIX']}_markov{nfirms}.npz")["newvalue"]
    isentry = np.load(f"Data/Out/a.{c['PREFIX']}_markov{nfirms}.npz")["isentry"]

    # Load encoding table
    multfac = np.load(f"Data/Out/a.{c['PREFIX']}_table{nfirms}.npz")["multfac"]
    etable = np.load(f"Data/Out/a.{c['PREFIX']}_table{nfirms}.npz")["etable"]

    # Initialize the state
    current_state = wstart.copy()

    for t in range(numtimes):
        state_code = qencode(current_state, etable, multfac)
        # Record the state
        state_history[t, :] = current_state # state at the start of the period
        firms_count_history[t] = np.sum(current_state > 0) # count of firm at the start of the period

        # issue: consider entry decision
        for i in range(nfirms):
            if current_state[i] == 0:
                entry_prob = isentry[state_code]
                entry = np.random.binomial(1, entry_prob)
                if entry:
                    e_idx = i
                break
        # issue: consider exist decision by comparing value function with scrap value (solved)
        sorted_idx = np.argsort(current_state)[::-1]
        for j in range(nfirms):
            if newvalue[state_code, :][sorted_idx][j]< phi:
                current_state[j] = 0
                current_state[current_state <= current_state[j]] = 0

        # Solve for the optimal investment
        # newx is sorted by current_state may not be
        investment_policy = (current_state>0)*(newx[state_code, :][sorted_idx])
        investment_history[t, :] = investment_policy # record investment policy
        # note: those who just enter and those who exit do not invest

        if entry:
            current_state[e_idx] = entry_k
    
        # Simulate the individual shocks
        individual_shocks_prob = (a * investment_policy) / (1 + a * investment_policy)
        individual_shocks = np.random.binomial(1, individual_shocks_prob)
        # note: entrant has no individual shock

        # Update the state of the industry
        current_state = np.maximum(np.minimum(current_state + individual_shocks - nu[t], kmax),0)

    # Take average 
    firms_count_avg = np.mean(firms_count_history)
    investment_period_avg = np.mean(investment_history)
    # Save results
    np.savez(out_file, state_history=state_history, firms_count=firms_count_history, investment_history=investment_history)
    np.savez(out_file+"_avg", firms_count_avg=firms_count_avg, investment_period_avg=investment_period_avg)