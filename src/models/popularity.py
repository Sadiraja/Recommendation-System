"""
Popularity-based recommendation baseline.
Uses Bayesian average (IMDb weighted rating formula) to balance
interaction count and rating quality, preventing niche items with 1 rating
from distorting rankings.
"""

from typing import Dict, List, Optional, Sequence
import numpy as np
import pandas as pd
from .base import BaseRecommender


class PopularityRecommender(BaseRecommender):
    """
    Popularity Recommender.
    Scores items using Bayesian Weighted Rating:
        WR = (v / (v + m)) * R + (m / (v + m)) * C
    where:
        v = number of ratings for the item
        m = minimum ratings threshold / damping factor (e.g. 75th percentile of counts)
        R = average rating of the item
        C = mean rating across the entire dataset
    """

    def __init__(self, damping_quantile: float = 0.70, name: str = "PopularityBaseline"):
        super().__init__(name=name)
        self.damping_quantile = damping_quantile
        self.global_mean_rating: float = 3.5
        self.min_ratings_threshold: float = 10.0
        self.item_scores: Dict[int, float] = {}
        self.ranked_items: List[int] = []
        self.genre_ranked_items: Dict[str, List[int]] = {}

    def fit(self, train_ratings: pd.DataFrame, items_metadata: Optional[pd.DataFrame] = None) -> "PopularityRecommender":
        """
        Fits the popularity model by computing weighted Bayesian scores for all items.
        """
        self._store_seen_items(train_ratings)
        self.global_mean_rating = float(train_ratings["rating"].mean())

        # Aggregate counts and means per movie
        item_stats = (
            train_ratings.groupby("movieId")
            .agg(vote_count=("rating", "count"), vote_average=("rating", "mean"))
            .reset_index()
        )

        # Compute damping threshold m
        m = float(item_stats["vote_count"].quantile(self.damping_quantile))
        self.min_ratings_threshold = max(5.0, m)
        C = self.global_mean_rating

        # Weighted rating formula
        v = item_stats["vote_count"]
        R = item_stats["vote_average"]
        item_stats["score"] = (v / (v + self.min_ratings_threshold)) * R + (
            self.min_ratings_threshold / (v + self.min_ratings_threshold)
        ) * C

        # Sort descending
        item_stats = item_stats.sort_values(by=["score", "vote_count"], ascending=[False, False])
        self.item_scores = dict(zip(item_stats["movieId"], item_stats["score"]))
        self.ranked_items = item_stats["movieId"].tolist()

        # Build genre-specific popularity rankings if metadata is available
        if items_metadata is not None and "genres_list" in items_metadata.columns:
            meta_with_scores = items_metadata.merge(item_stats[["movieId", "score"]], on="movieId", how="inner")
            self.genre_ranked_items = {}
            # Flatten genres
            for _, row in meta_with_scores.iterrows():
                movie_id = row["movieId"]
                genres = row.get("genres_list", [])
                for g in genres:
                    g_clean = g.strip().lower()
                    if g_clean not in self.genre_ranked_items:
                        self.genre_ranked_items[g_clean] = []
                    self.genre_ranked_items[g_clean].append(movie_id)

        self.is_fitted = True
        return self

    def predict(self, user_id: int, item_id: int) -> float:
        """
        Predicts score for item (popularity score is user-independent).
        """
        return self.item_scores.get(item_id, self.global_mean_rating)

    def recommend(
        self,
        user_id: int,
        n: int = 10,
        exclude_seen: bool = True,
        candidate_items: Optional[Sequence[int]] = None,
    ) -> List[int]:
        """
        Recommends top-n globally popular items not yet seen by the user.
        """
        if not self.is_fitted:
            raise RuntimeError("Model must be fitted before calling recommend().")

        seen = self.user_seen_items.get(user_id, set()) if exclude_seen else set()
        pool = candidate_items if candidate_items is not None else self.ranked_items

        recommendations = []
        for item in pool:
            if item not in seen:
                recommendations.append(item)
                if len(recommendations) >= n:
                    break

        return recommendations

    def recommend_for_genre(self, genre: str, n: int = 10, exclude_items: Optional[Sequence[int]] = None) -> List[int]:
        """
        Recommends top popular items within a given genre (useful for new user onboarding).
        """
        genre_key = genre.strip().lower()
        pool = self.genre_ranked_items.get(genre_key, self.ranked_items)
        excluded = set(exclude_items) if exclude_items else set()

        recommendations = []
        for item in pool:
            if item not in excluded:
                recommendations.append(item)
                if len(recommendations) >= n:
                    break
        return recommendations
