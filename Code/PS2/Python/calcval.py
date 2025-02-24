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
