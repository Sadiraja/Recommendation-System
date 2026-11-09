"""
Train/Test splitting strategies for recommender systems.
Implements temporal per-user splitting and leave-k-out splitting to prevent
data leakage and reflect realistic recommendation evaluation.
"""

from dataclasses import dataclass
from typing import Tuple

import numpy as np
import pandas as pd


@dataclass
class TrainTestSplit:
    train: pd.DataFrame
    test: pd.DataFrame
    positive_test: pd.DataFrame
    min_positive_rating: float


def temporal_per_user_split(
    ratings: pd.DataFrame,
    test_ratio: float = 0.2,
    min_positive_rating: float = 3.5,
    min_train_items: int = 3,
) -> TrainTestSplit:
    """
    Performs chronological (temporal) train/test split per user.
    For each user, sorts interactions by timestamp, reserving the last
    `test_ratio` portion for testing and earlier interactions for training.

    This avoids temporal data leakage (evaluating whether a model can
    predict future interactions given past history).

    Args:
        ratings: DataFrame containing [userId, movieId, rating, timestamp].
        test_ratio: Fraction of most recent user interactions to hold out (default: 0.2).
        min_positive_rating: Threshold for an interaction to be considered relevant/liked.
        min_train_items: Minimum number of train items required per user.

    Returns:
        TrainTestSplit container.
    """
    train_chunks = []
    test_chunks = []

    # Ensure sorted chronologically per user
    sorted_ratings = ratings.sort_values(by=["userId", "timestamp"]).reset_index(drop=True)

    for user_id, user_group in sorted_ratings.groupby("userId"):
        n_items = len(user_group)
        if n_items < (min_train_items + 1):
            train_chunks.append(user_group)
            continue

        n_test = int(np.ceil(n_items * test_ratio))
        n_test = max(1, min(n_test, n_items - min_train_items))

        train_part = user_group.iloc[:-n_test]
        test_part = user_group.iloc[-n_test:]

        train_chunks.append(train_part)
        test_chunks.append(test_part)

    train_df = pd.concat(train_chunks, ignore_index=True)
    test_df = pd.concat(test_chunks, ignore_index=True) if test_chunks else pd.DataFrame(columns=ratings.columns)

    # Positive test items (ground-truth relevant items for ranking metrics)
    positive_test = test_df[test_df["rating"] >= min_positive_rating].copy()

    return TrainTestSplit(
        train=train_df,
        test=test_df,
        positive_test=positive_test,
        min_positive_rating=min_positive_rating,
    )


def leave_k_out_split(
    ratings: pd.DataFrame,
    k: int = 5,
    min_positive_rating: float = 3.5,
) -> TrainTestSplit:
    """
    Holds out the last `k` chronological interactions per user for testing.

    Args:
        ratings: DataFrame containing [userId, movieId, rating, timestamp].
        k: Number of most recent interactions held out per user.
        min_positive_rating: Threshold for an interaction to be considered relevant.

    Returns:
        TrainTestSplit container.
    """
    train_chunks = []
    test_chunks = []

    sorted_ratings = ratings.sort_values(by=["userId", "timestamp"]).reset_index(drop=True)

    for user_id, user_group in sorted_ratings.groupby("userId"):
        n_items = len(user_group)
        if n_items <= k:
            # Not enough items to hold out k, keep at least 1 in train
            n_holdout = max(1, n_items // 2)
        else:
            n_holdout = k

        train_chunks.append(user_group.iloc[:-n_holdout])
        test_chunks.append(user_group.iloc[-n_holdout:])

    train_df = pd.concat(train_chunks, ignore_index=True)
    test_df = pd.concat(test_chunks, ignore_index=True)
    positive_test = test_df[test_df["rating"] >= min_positive_rating].copy()

    return TrainTestSplit(
        train=train_df,
        test=test_df,
        positive_test=positive_test,
        min_positive_rating=min_positive_rating,
    )
