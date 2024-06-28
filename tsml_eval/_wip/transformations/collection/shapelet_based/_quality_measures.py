"""Quality measures to use with STC."""

from numba import njit


@njit(fastmath=True, cache=True)
def binary_information_gain(orderline, c1, c2):
    """Find binary information gain for an order line.

    Parameters
    ----------

    orderline: 2D np.array (?) of what?
        Orderline for a given shapelet
    c1: int
        Number of cases of the class of interest
    c2: int
        Number of cases of all other classes
    """

    def _binary_entropy(c1, c2):
        ent = 0
        if c1 != 0:
            ent -= c1 / (c1 + c2) * np.log2(c1 / (c1 + c2))
        if c2 != 0:
            ent -= c2 / (c1 + c2) * np.log2(c2 / (c1 + c2))
        return ent

    initial_ent = _binary_entropy(c1, c2)

    total_all = c1 + c2

    bsf_ig = 0
    c1_count = 0
    c2_count = 0

    # evaluate each split point
    for split in range(len(orderline)):
        next_class = orderline[split][1]  # +1 if this class, -1 if other
        if next_class > 0:
            c1_count += 1
        else:
            c2_count += 1

        left_prop = (split + 1) / total_all
        ent_left = _binary_entropy(c1_count, c2_count)

        right_prop = 1 - left_prop
        ent_right = _binary_entropy(
            c1 - c1_count,
            c2 - c2_count,
        )

        ig = initial_ent - left_prop * ent_left - right_prop * ent_right
        bsf_ig = max(ig, bsf_ig)

    return bsf_ig

def f_stat(c1, c2):
    """Calculate the F-statistic for two classes.

    Parameters
    ----------
    c1 : np.ndarray
        Distance values for class 1
    c2 : np.ndarray
        Distance values for class 2

    Returns
    -------
    float
        F-statistic value
    """
    return 0.0

