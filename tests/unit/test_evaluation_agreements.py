"""Tests for agreement metrics."""

from code_explainer.evaluation.agreements import cohens_kappa, fleiss_kappa


def test_cohens_kappa_perfect_agreement():
    a = [0, 1, 0, 1]
    b = [0, 1, 0, 1]
    assert cohens_kappa(a, b) == 1.0


def test_fleiss_kappa_simple():
    # 3 items, 3 raters each
    ratings = [
        [0, 0, 0],
        [1, 1, 1],
        [0, 1, 0]
    ]
    k = fleiss_kappa(ratings)
    assert isinstance(k, float)
*** End Patch