"""Inter-annotator agreement metrics.

Implements Cohen's kappa (pairwise) and Fleiss' kappa (multi-rater) with
lightweight numpy-free implementations suitable for unit tests.
"""
from typing import List


def cohens_kappa(rater1: List[int], rater2: List[int]) -> float:
    """Compute Cohen's kappa for two raters with categorical labels.

    Labels should be integers starting at 0. Returns kappa in [-1,1].
    """
    if len(rater1) != len(rater2):
        raise ValueError("Rater lists must be the same length")
    n = len(rater1)
    if n == 0:
        return 0.0

    # Build contingency and marginals
    labels = sorted(set(rater1) | set(rater2))
    label_to_idx = {l: i for i, l in enumerate(labels)}
    k = len(labels)

    obs = [[0] * k for _ in range(k)]
    for a, b in zip(rater1, rater2):
        obs[label_to_idx[a]][label_to_idx[b]] += 1

    p0 = sum(obs[i][i] for i in range(k)) / n

    # Marginal probabilities
    p_a = [sum(obs[i][j] for j in range(k)) / n for i in range(k)]
    p_b = [sum(obs[i][j] for i in range(k)) / n for j in range(k)]

    pe = sum(p_a[i] * p_b[i] for i in range(k))
    if pe == 1.0:
        return 1.0
    return (p0 - pe) / (1 - pe)


def fleiss_kappa(ratings: List[List[int]]) -> float:
    """Compute Fleiss' kappa for multiple raters and items.

    `ratings` is a list of items where each item is a list of category indices
    assigned by raters (categories must be non-negative ints). All items must
    have the same number of ratings.
    """
    if not ratings:
        return 0.0
    n = len(ratings)
    m = len(ratings[0])
    # Gather categories
    categories = sorted({c for item in ratings for c in item})
    k = len(categories)
    if k == 0:
        return 0.0

    # Build category counts per item
    p = [0] * k
    P = []
    for item in ratings:
        counts = [0] * k
        for c in item:
            counts[categories.index(c)] += 1
        P_i = sum((cnt * (cnt - 1)) for cnt in counts) / (m * (m - 1))
        P.append(P_i)
        for idx, cnt in enumerate(counts):
            p[idx] += cnt

    p = [x / (n * m) for x in p]
    P_bar = sum(P) / n
    P_e = sum(pi * pi for pi in p)
    if P_e == 1.0:
        return 1.0
    return (P_bar - P_e) / (1 - P_e)
