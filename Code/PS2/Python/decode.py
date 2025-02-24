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


