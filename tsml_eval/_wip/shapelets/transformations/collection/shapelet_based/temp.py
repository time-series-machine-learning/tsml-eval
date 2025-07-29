"""

ST structure

fit
-_fit_shapelets
    creates empty list for shapelets
    extract in batches
        for each batch, call
            _extract_random_shapelet: This is not in numba
                this does loads of randomly samples shapelet, then calls
                _find_shapelet_quality. THIS IS IN NUMBA. Hard coded to IG.








    List not array?
            # shapelet list content: quality, length, position, channel, inst_idx, cls_idx
        shapelets = List(
            [
                List([List([-1.0, -1, -1, -1, -1, -1, -1])])
                for _ in range(self.n_classes_)
            ]
        )
shapelets = List(
    [
        List([               # Outer list contains one item per class
            List([           # Middle list contains one item (a list)
                -1.0, -1, -1, -1, -1, -1, -1  # Inner-most list of values
            ])
        ])
        for _ in range(self.n_classes_)
    ]
)


"""



@njit(fastmath=True, cache=True)
def _f_stat_shapelet_quality(
    X,
    y,
    shapelet,
    sorted_indicies,
    position,
    length,
    dim,
    inst_idx,
    this_cls_count,
    other_cls_count,
    worst_quality,
):
    distances1 = np.zeros(this_cls_count - 1)
    distances2 = np.zeros(other_cls_count)
    c1 = 0
    c2 = 0
    for i, series in enumerate(X):
        if i != inst_idx:
            distance = _online_shapelet_distance(
                series[dim], shapelet, sorted_indicies, position, length
            )
            if y[i] == y[inst_idx]:
                distances1[c1] = distance
                c1 = c1 + 1
            else:
                distances2[c2] = distance
                c2 = c2 + 1

    quality = qm.f_stat(distances1, distances2)

    return round(quality, 12)


@njit(fastmath=True, cache=True)
def _moods_median_shapelet_quality(
    X,
    y,
    shapelet,
    sorted_indicies,
    position,
    length,
    dim,
    inst_idx,
    this_cls_count,
    other_cls_count,
    worst_quality,
):
    distances1 = np.zeros(this_cls_count - 1)
    distances2 = np.zeros(other_cls_count)
    c1 = 0
    c2 = 0
    for i, series in enumerate(X):
        if i != inst_idx:
            distance = _online_shapelet_distance(
                series[dim], shapelet, sorted_indicies, position, length
            )
            if y[i] == y[inst_idx]:
                distances1[c1] = distance
                c1 += 1
            else:
                distances2[c2] = distance
                c2 += 1

    quality = qm._moods_median(distances1, distances2)

    return round(quality, 12)


@njit(fastmath=True, cache=True)
def _kruskal_wallis_shapelet_quality(
    X,
    y,
    shapelet,
    sorted_indicies,
    position,
    length,
    dim,
    inst_idx,
    this_cls_count,
    other_cls_count,
    worst_quality,
    ranks,
    tie_correction,
    n1,
    n2,
    n,
):
    quality = qm.kruskal_wallis_test(ranks, n1, n2, n, tie_correction)

    return round(quality, 12)


@njit(fastmath=True, cache=True)
def _wasserstein_shapelet_quality(
    X,
    y,
    shapelet,
    sorted_indicies,
    position,
    length,
    dim,
    inst_idx,
    this_cls_count,
    other_cls_count,
    worst_quality,
    mu1,
    Sigma1,
    mu2,
    Sigma2,
):
    mu1 = np.array([mu1])  # Make sure it's an array if it's not
    mu2 = np.array([mu2])
    quality = qm.wasserstein_distance_gaussian(mu1, Sigma1, mu2, Sigma2)

    return round(quality, 12)


def _wasser_emp_shapelet_quality(
    self,
    X,
    y,
    shapelet,
    sorted_indicies,
    position,
    length,
    dim,
    inst_idx,
    this_cls_count,
    other_cls_count,
    worst_quality,
):
    distances1 = np.zeros(this_cls_count - 1)
    distances2 = np.zeros(other_cls_count)
    c1 = 0
    c2 = 0
    for i, series in enumerate(X):
        if i != inst_idx:
            distance = _online_shapelet_distance(
                series[dim], shapelet, sorted_indicies, position, length
            )
            if y[i] == y[inst_idx]:
                distances1[c1] = distance
                c1 = c1 + 1
            else:
                distances2[c2] = distance
                c2 = c2 + 1

    quality = qm.wasserstein_distance_empirical(distances1, distances2)

    return round(quality, 12)


def _kolmogorov_shapelet_quality(
    self,
    X,
    y,
    shapelet,
    sorted_indicies,
    position,
    length,
    dim,
    inst_idx,
    this_cls_count,
    other_cls_count,
    worst_quality,
):
    distances1 = np.zeros(this_cls_count - 1)
    distances2 = np.zeros(other_cls_count)
    c1 = 0
    c2 = 0
    for i, series in enumerate(X):
        if i != inst_idx:
            distance = _online_shapelet_distance(
                series[dim], shapelet, sorted_indicies, position, length
            )
            if y[i] == y[inst_idx]:
                distances1[c1] = distance
                c1 = c1 + 1
            else:
                distances2[c2] = distance
                c2 = c2 + 1

    quality = qm.kolmogorov_test(distances1, distances2)

    return round(quality, 12)

@njit(fastmath=True, cache=True)
def _online_shapelet_distance(series, shapelet, sorted_indicies, position, length):
    subseq = series[position : position + length]

    sum = 0.0
    sum2 = 0.0
    for i in subseq:
        sum += i
        sum2 += i * i

    mean = sum / length
    std = math.sqrt((sum2 - mean * mean * length) / length)
    if std > AEON_NUMBA_STD_THRESHOLD:
        subseq = (subseq - mean) / std
    else:
        subseq = np.zeros(length)

    best_dist = 0
    for i, n in zip(shapelet, subseq):
        temp = i - n
        best_dist += temp * temp

    i = 1
    traverse = [True, True]
    sums = [sum, sum]
    sums2 = [sum2, sum2]

    while traverse[0] or traverse[1]:
        for n in range(2):
            mod = -1 if n == 0 else 1
            pos = position + mod * i
            traverse[n] = pos >= 0 if n == 0 else pos <= len(series) - length

            if not traverse[n]:
                continue

            start = series[pos - n]
            end = series[pos - n + length]

            sums[n] += mod * end - mod * start
            sums2[n] += mod * end * end - mod * start * start

            mean = sums[n] / length
            std = math.sqrt((sums2[n] - mean * mean * length) / length)

            dist = 0
            use_std = std > AEON_NUMBA_STD_THRESHOLD
            for j in range(length):
                val = (series[pos + sorted_indicies[j]] - mean) / std if use_std else 0
                temp = shapelet[sorted_indicies[j]] - val
                dist += temp * temp

                if dist > best_dist:
                    break

            if dist < best_dist:
                best_dist = dist

        i += 1

    return best_dist if best_dist == 0 else 1 / length * best_dist


@njit(fastmath=True, cache=True)
def _calc_early_binary_ig(
    orderline,
    c1_traversed,
    c2_traversed,
    c1_to_add,
    c2_to_add,
    worst_quality,
):
    initial_ent = _binary_entropy(
        c1_traversed + c1_to_add,
        c2_traversed + c2_to_add,
    )

    total_all = c1_traversed + c2_traversed + c1_to_add + c2_to_add

    bsf_ig = 0
    # actual observations in orderline
    c1_count = 0
    c2_count = 0

    # evaluate each split point
    for split in range(len(orderline)):
        next_class = orderline[split][1]  # +1 if this class, -1 if other
        if next_class > 0:
            c1_count += 1
        else:
            c2_count += 1

        # optimistically add this class to left side first and other to right
        left_prop = (split + 1 + c1_to_add) / total_all
        ent_left = _binary_entropy(c1_count + c1_to_add, c2_count)

        # because right side must optimistically contain everything else
        right_prop = 1 - left_prop

        ent_right = _binary_entropy(
            c1_traversed - c1_count,
            c2_traversed - c2_count + c2_to_add,
        )

        ig = initial_ent - left_prop * ent_left - right_prop * ent_right
        bsf_ig = max(ig, bsf_ig)

        # now optimistically add this class to right, other to left
        left_prop = (split + 1 + c2_to_add) / total_all
        ent_left = _binary_entropy(c1_count, c2_count + c2_to_add)

        # because right side must optimistically contain everything else
        right_prop = 1 - left_prop

        ent_right = _binary_entropy(
            c1_traversed - c1_count + c1_to_add,
            c2_traversed - c2_count,
        )

        ig = initial_ent - left_prop * ent_left - right_prop * ent_right
        bsf_ig = max(ig, bsf_ig)

        if bsf_ig > worst_quality:
            return bsf_ig

    return bsf_ig


@njit(fastmath=True, cache=True)
def _calc_binary_ig(orderline, c1, c2):
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


@njit(fastmath=True, cache=True)
def _binary_entropy(c1, c2):
    ent = 0
    if c1 != 0:
        ent -= c1 / (c1 + c2) * np.log2(c1 / (c1 + c2))
    if c2 != 0:
        ent -= c2 / (c1 + c2) * np.log2(c2 / (c1 + c2))
    return ent


@njit(fastmath=True, cache=True)
def _is_self_similar(s1, s2):
    # not self similar if from different series or dimension
    if s1[4] == s2[4] and s1[3] == s2[3]:
        if s2[2] <= s1[2] <= s2[2] + s2[1]:
            return True
        if s1[2] <= s2[2] <= s1[2] + s1[1]:
            return True

    return False


@njit(fastmath=True, cache=True)
def f_stat(class0, class1):
    """
    Calculate the F-statistic for shapelet quality based on two numpy arrays of distances for two classes.
    Parameters:
    - class0 (np.array): Array of distances for the first class.
    - class1 (np.array): Array of distances for the second class.
    Returns:
    - float: The computed F-statistic.
    """

    if len(class0) == 0 or len(class1) == 0:
        return np.inf  # Use NumPy's inf representation

    # Calculate means
    mean_class0 = np.mean(class0)
    mean_class1 = np.mean(class1)
    all_distances = np.concatenate((class0, class1))
    overall_mean = np.mean(all_distances)

    n0 = len(class0)
    n1 = len(class1)
    total_n = n0 + n1

    # Between-class sum of squares
    ssb = (
        n0 * (mean_class0 - overall_mean) ** 2 + n1 * (mean_class1 - overall_mean) ** 2
    )

    # Within-class sum of squares
    ssw = np.sum((class0 - mean_class0) ** 2) + np.sum((class1 - mean_class1) ** 2)

    # Degrees of freedom
    df_between = 1
    df_within = total_n - 2

    # Avoid division by zero
    if df_within <= 0:

        return np.inf

    F_stat = (ssb / df_between) / (ssw / df_within)
    return F_stat
