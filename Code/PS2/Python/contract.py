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
