def cross_one_out(kfold):  # Deprecated

    """Through Stratified splits, retrieve indices of one vs. others
    For example, if kfold was setup with n_splits=10,
    this will return
        indices of 10% data
        and indices of rest of the data

    Returns:
        ([list], [list])
    """

    idx = next(kfold)[1]
    leftover_idx = []
    while True:
        try:
            leftover_idx.extend(next(kfold)[1])
        except:
            break
    return idx.tolist(), leftover_idx
