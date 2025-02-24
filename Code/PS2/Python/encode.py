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
