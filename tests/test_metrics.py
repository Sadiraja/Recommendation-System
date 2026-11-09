"""
Unit tests for ranking metrics.
"""

import numpy as np
import pytest
from src.metrics import (
    average_precision_at_k,
    hit_rate_at_k,
    mrr_at_k,
    ndcg_at_k,
    precision_at_k,
    recall_at_k,
)


def test_precision_at_k():
    actual = [1, 2, 3]
    pred = [1, 4, 2, 5, 6]

    assert precision_at_k(actual, pred, k=1) == 1.0  # [1] -> 1/1
    assert precision_at_k(actual, pred, k=2) == 0.5  # [1, 4] -> 1/2
    assert precision_at_k(actual, pred, k=3) == pytest.approx(2 / 3)  # [1, 4, 2] -> 2/3
    assert precision_at_k(actual, pred, k=5) == 2 / 5

    # Edge cases
    assert precision_at_k([], pred, k=5) == 0.0
    assert precision_at_k(actual, [], k=5) == 0.0
    assert precision_at_k(actual, pred, k=0) == 0.0


def test_recall_at_k():
    actual = [1, 2, 3, 4]
    pred = [1, 5, 2, 6, 7]

    assert recall_at_k(actual, pred, k=1) == 1 / 4
    assert recall_at_k(actual, pred, k=3) == 2 / 4
    assert recall_at_k(actual, pred, k=5) == 2 / 4

    # Edge cases
    assert recall_at_k([], pred, k=5) == 0.0
    assert recall_at_k(actual, [], k=5) == 0.0


def test_ndcg_at_k():
    actual = [1, 2]
    # Perfect order: item 1 at rank 1, item 2 at rank 2
    perfect_pred = [1, 2, 3, 4]
    assert ndcg_at_k(actual, perfect_pred, k=2) == pytest.approx(1.0)

    # Inverted order: item 2 at rank 1, item 1 at rank 2 -> DCG equals IDCG for binary
    inverted_pred = [2, 1, 3, 4]
    assert ndcg_at_k(actual, inverted_pred, k=2) == pytest.approx(1.0)

    # Displaced order: item at rank 1 is irrelevant, relevant items at rank 2 and 3
    delayed_pred = [99, 1, 2]
    score = ndcg_at_k(actual, delayed_pred, k=3)
    assert 0.0 < score < 1.0

    # No matches
    disjoint_pred = [98, 99, 100]
    assert ndcg_at_k(actual, disjoint_pred, k=3) == 0.0


def test_hit_rate_and_mrr():
    actual = [10, 20]
    pred = [5, 10, 30]

    assert hit_rate_at_k(actual, pred, k=1) == 0.0
    assert hit_rate_at_k(actual, pred, k=2) == 1.0
    assert mrr_at_k(actual, pred, k=1) == 0.0
    assert mrr_at_k(actual, pred, k=2) == 0.5  # First hit at rank 2 -> 1/2


def test_average_precision_at_k():
    actual = [1, 3]
    pred = [1, 2, 3, 4]
    # Rank 1: item 1 (hit, P@1 = 1.0)
    # Rank 2: item 2 (miss)
    # Rank 3: item 3 (hit, P@3 = 2/3)
    # AP@3 = (1.0 + 2/3) / 2 = 5/6
    assert average_precision_at_k(actual, pred, k=3) == pytest.approx((1.0 + 2 / 3) / 2)
