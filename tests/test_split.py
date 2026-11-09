"""
Unit tests for train/test splitting logic.
"""

import pandas as pd
import pytest
from src.split import leave_k_out_split, temporal_per_user_split


@pytest.fixture
def sample_ratings():
    # 2 users, each with 10 interactions sorted by timestamp
    rows = []
    for u in [1, 2]:
        for i in range(1, 11):
            rows.append(
                {
                    "userId": u,
                    "movieId": 100 + i,
                    "rating": 4.0 if i % 2 == 0 else 2.0,
                    "timestamp": 1000 + (u * 100) + i,
                }
            )
    return pd.DataFrame(rows)


def test_temporal_per_user_split(sample_ratings):
    split = temporal_per_user_split(sample_ratings, test_ratio=0.2, min_positive_rating=3.5)

    # 10 items * 0.2 = 2 test items per user -> 4 test items total
    assert len(split.test) == 4
    assert len(split.train) == 16

    # Verify no temporal data leakage: train max timestamp < test min timestamp per user
    for u in [1, 2]:
        train_u = split.train[split.train["userId"] == u]
        test_u = split.test[split.test["userId"] == u]
        assert train_u["timestamp"].max() < test_u["timestamp"].min()

    # Relevant items are those with rating >= 3.5
    assert (split.positive_test["rating"] >= 3.5).all()


def test_leave_k_out_split(sample_ratings):
    split = leave_k_out_split(sample_ratings, k=3, min_positive_rating=3.5)

    # 3 items held out per user
    assert len(split.test) == 6
    assert len(split.train) == 14

    for u in [1, 2]:
        train_u = split.train[split.train["userId"] == u]
        test_u = split.test[split.test["userId"] == u]
        assert len(test_u) == 3
        assert train_u["timestamp"].max() < test_u["timestamp"].min()
