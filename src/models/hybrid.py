"""
Hybrid Recommender combining Collaborative Filtering, Content-Based, and Popularity.
Features dynamic confidence weighting based on user interaction count,
smoothly transitioning from Popularity/Content for cold-start users
to Collaborative Filtering for mature users.
"""

from typing import Dict, List, Optional, Sequence
import numpy as np
import pandas as pd
from .base import BaseRecommender
from .collaborative import CollaborativeFilteringRecommender
from .content_based import ContentBasedRecommender
from .popularity import PopularityRecommender


class HybridRecommender(BaseRecommender):
    """
    Adaptive Hybrid Recommender.
    Blends Collaborative Filtering (SVD) and Content-Based (TF-IDF) predictions
    using dynamic weighting alpha based on user history length:
        alpha(n_u) = n_u / (n_u + gamma)
    where:
        n_u = number of user ratings in train set
        gamma = half-saturation constant (default: 15)
    If user has 0 ratings (cold-start), falls back to Popularity (or Content if genres supplied).
    """

    def __init__(
        self,
        cf_weight: float = 0.6,
        gamma: float = 15.0,
        adaptive_weighting: bool = True,
        name: str = "Hybrid_CF_Content_Popularity",
    ):
        super().__init__(name=name)
        self.base_cf_weight = cf_weight
        self.gamma = gamma
        self.adaptive_weighting = adaptive_weighting

        self.popularity_model = PopularityRecommender()
        self.cf_model = CollaborativeFilteringRecommender()
        self.cb_model = ContentBasedRecommender()

        self.user_interaction_counts: Dict[int, int] = {}

    def fit(self, train_ratings: pd.DataFrame, items_metadata: Optional[pd.DataFrame] = None) -> "HybridRecommender":
        """
        Fits all underlying models (Popularity, CF, and Content-Based).
        """
        self._store_seen_items(train_ratings)
        self.user_interaction_counts = train_ratings["userId"].value_counts().to_dict()

        self.popularity_model.fit(train_ratings, items_metadata=items_metadata)
        self.cf_model.fit(train_ratings, items_metadata=items_metadata)
        self.cb_model.fit(train_ratings, items_metadata=items_metadata)

        self.is_fitted = True
        return self

    def _compute_alpha(self, user_id: int) -> float:
        """
        Computes CF weight alpha for user_id.
        If adaptive_weighting is disabled, returns base_cf_weight.
        """
        if not self.adaptive_weighting:
            return self.base_cf_weight

        count = self.user_interaction_counts.get(user_id, 0)
        if count == 0:
            return 0.0
        # Saturation curve: starts near 0 for few ratings, reaches ~0.7-0.9 for active users
        return float(count / (count + self.gamma))

    @staticmethod
    def _normalize_scores(scores: np.ndarray) -> np.ndarray:
        """
        Min-max normalizes scores to [0.0, 1.0].
        """
        s_min = np.min(scores)
        s_max = np.max(scores)
        if s_max - s_min < 1e-9:
            return np.ones_like(scores) * 0.5
        return (scores - s_min) / (s_max - s_min)

    def predict(self, user_id: int, item_id: int) -> float:
        """
        Predicts blended rating for user_id and item_id.
        """
        if not self.is_fitted:
            raise RuntimeError("Model must be fitted before predict().")

        alpha = self._compute_alpha(user_id)
        if alpha == 0.0:
            # Cold-start user
            return self.popularity_model.predict(user_id, item_id)

        cf_pred = self.cf_model.predict(user_id, item_id)
        cb_pred = self.cb_model.predict(user_id, item_id)
        return float(alpha * cf_pred + (1.0 - alpha) * cb_pred)

    def recommend(
        self,
        user_id: int,
        n: int = 10,
        exclude_seen: bool = True,
        candidate_items: Optional[Sequence[int]] = None,
    ) -> List[int]:
        """
        Recommends top-n items using weighted score combination.
        Falls back to popularity for complete cold-start users.
        """
        if not self.is_fitted:
            raise RuntimeError("Model must be fitted before recommend().")

        alpha = self._compute_alpha(user_id)
        seen = self.user_seen_items.get(user_id, set()) if exclude_seen else set()

        # Pure cold-start: user has never been seen in train set
        if alpha == 0.0 or user_id not in self.user_interaction_counts:
            return self.popularity_model.recommend(
                user_id=user_id,
                n=n,
                exclude_seen=exclude_seen,
                candidate_items=candidate_items,
            )

        # Get candidates / all items
        cand_list = list(candidate_items) if candidate_items is not None else self.all_items
        if not cand_list:
            return []

        # Fast vector scoring
        cf_all = self.cf_model.score_all_items(user_id)
        cb_all = self.cb_model.score_all_items(user_id)

        # Map to candidate list
        cand_items_arr = []
        cf_vals = []
        cb_vals = []

        for item in cand_list:
            if item in seen:
                continue
            cand_items_arr.append(item)
            cf_idx = self.cf_model.item_to_idx.get(item)
            cb_idx = self.cb_model.item_to_idx.get(item)
            cf_vals.append(cf_all[cf_idx] if cf_idx is not None else self.cf_model.global_mean)
            cb_vals.append(cb_all[cb_idx] if cb_idx is not None else 0.0)

        if not cand_items_arr:
            return []

        cf_arr = self._normalize_scores(np.array(cf_vals))
        cb_arr = self._normalize_scores(np.array(cb_vals))

        hybrid_scores = alpha * cf_arr + (1.0 - alpha) * cb_arr
        top_k = min(n, len(hybrid_scores))

        top_indices = np.argpartition(-hybrid_scores, top_k - 1)[:top_k]
        top_indices = top_indices[np.argsort(-hybrid_scores[top_indices])]
        return [cand_items_arr[idx] for idx in top_indices]

    def recommend_for_new_user(
        self,
        preferred_genres: Optional[Sequence[str]] = None,
        initial_ratings: Optional[Dict[int, float]] = None,
        n: int = 10,
    ) -> List[int]:
        """
        Explicit cold-start handler for new users with no historical train data.
        - If initial_ratings provided: builds on-the-fly user profile.
        - If preferred_genres provided: uses content-based genre profile.
        - Otherwise: returns Bayesian popularity baseline.
        """
        if initial_ratings:
            m_ids = list(initial_ratings.keys())
            rats = list(initial_ratings.values())
            profile = self.cb_model.build_user_profile(m_ids, rats)
            if profile is not None:
                # Rank all items by similarity to this profile
                all_ids = list(self.cb_model.idx_to_item.values())
                cands = [i for i in all_ids if i not in initial_ratings]
                cand_indices = [self.cb_model.item_to_idx[i] for i in cands]
                features = self.cb_model.item_features[cand_indices]
                scores = np.dot(features, profile)
                top_part = np.argsort(-scores)[:n]
                return [cands[idx] for idx in top_part]

        if preferred_genres:
            return self.cb_model.recommend_for_cold_start_genres(preferred_genres, n=n)

        # Complete fallback
        return self.popularity_model.recommend(user_id=-1, n=n, exclude_seen=False)
