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
