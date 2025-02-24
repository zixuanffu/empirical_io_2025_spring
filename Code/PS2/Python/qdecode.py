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