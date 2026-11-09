"""
Abstract base class for recommendation models.
"""

from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Sequence, Set
import pandas as pd


class BaseRecommender(ABC):
    """
    Abstract Base Class for Recommenders.
    Standardizes interface for training, prediction, and top-N ranking generation.
    """

    def __init__(self, name: str = "BaseRecommender"):
        self.name = name
        self.is_fitted = False
        self.user_seen_items: Dict[int, Set[int]] = {}
        self.all_items: List[int] = []

    @abstractmethod
    def fit(self, train_ratings: pd.DataFrame, items_metadata: Optional[pd.DataFrame] = None) -> "BaseRecommender":
        """
        Fits the recommender model on training ratings and optional item metadata.
        """
        pass

    @abstractmethod
    def recommend(
        self,
        user_id: int,
        n: int = 10,
        exclude_seen: bool = True,
        candidate_items: Optional[Sequence[int]] = None,
    ) -> List[int]:
        """
        Recommends top-n items for a specific user.

        Args:
            user_id: Target user ID.
            n: Number of recommendations to produce.
            exclude_seen: Whether to filter out items the user interacted with in train set.
            candidate_items: Optional list of item IDs to rank from. If None, considers all known items.

        Returns:
            List of recommended item IDs ranked from highest to lowest score.
        """
        pass

    def predict(self, user_id: int, item_id: int) -> float:
        """
        Predicts score/rating for a specific user-item pair.
        """
        return 0.0

    def recommend_for_all(
        self,
        user_ids: Sequence[int],
        n: int = 10,
        exclude_seen: bool = True,
    ) -> Dict[int, List[int]]:
        """
        Generates top-n recommendations for a collection of users.
        """
        recommendations = {}
        for uid in user_ids:
            recommendations[uid] = self.recommend(user_id=uid, n=n, exclude_seen=exclude_seen)
        return recommendations

    def _store_seen_items(self, train_ratings: pd.DataFrame):
        """
        Caches user seen items from training set.
        """
        self.user_seen_items = (
            train_ratings.groupby("userId")["movieId"]
            .apply(set)
            .to_dict()
        )
        self.all_items = list(train_ratings["movieId"].unique())
