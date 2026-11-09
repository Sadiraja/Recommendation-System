"""
Content-Based Recommender using TF-IDF on item genres, tags, and titles.
Builds user preference profiles from historical positive ratings,
computing cosine similarity between user profile vectors and candidate item vectors.
Also supports zero-history onboarding via explicit genre preferences.
"""

from typing import Dict, List, Optional, Sequence
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import normalize
from .base import BaseRecommender


class ContentBasedRecommender(BaseRecommender):
    """
    Content-Based Recommender.
    Represents items using TF-IDF vectors of their genres, crowdsourced tags, and titles.
    Constructs user interest vectors as weighted linear combinations of liked items.
    """

    def __init__(
        self,
        min_df: int = 2,
        rating_threshold: float = 3.0,
        name: str = "ContentBased_TFIDF",
    ):
        super().__init__(name=name)
        self.min_df = min_df
        self.rating_threshold = rating_threshold

        self.vectorizer = TfidfVectorizer(
            stop_words="english",
            min_df=self.min_df,
            ngram_range=(1, 2),
            sublinear_tf=True,
        )

        self.item_to_idx: Dict[int, int] = {}
        self.idx_to_item: Dict[int, int] = {}
        self.item_features: np.ndarray = np.array([])  # (n_items, n_features)
        self.user_profiles: Dict[int, np.ndarray] = {}  # userId -> (n_features,)

    def fit(self, train_ratings: pd.DataFrame, items_metadata: Optional[pd.DataFrame] = None) -> "ContentBasedRecommender":
        """
        Fits TF-IDF vectorizer on item metadata and constructs user preference profiles.
        """
        self._store_seen_items(train_ratings)

        if items_metadata is None or "combined_content" not in items_metadata.columns:
            raise ValueError("items_metadata with 'combined_content' column is required for ContentBasedRecommender.")

        # Ensure consistent item ordering
        items_meta = items_metadata.drop_duplicates(subset=["movieId"]).reset_index(drop=True)
        movie_ids = items_meta["movieId"].tolist()

        self.item_to_idx = {m_id: idx for idx, m_id in enumerate(movie_ids)}
        self.idx_to_item = {idx: m_id for idx, m_id in enumerate(movie_ids)}

        # Fit TF-IDF on item textual metadata
        tfidf_sparse = self.vectorizer.fit_transform(items_meta["combined_content"].fillna(""))
        # Normalize to unit length for fast cosine similarity via dot product
        self.item_features = normalize(tfidf_sparse, norm="l2", axis=1).toarray()

        # Build user profile vectors
        self.user_profiles = {}
        for user_id, group in train_ratings.groupby("userId"):
            profile = self.build_user_profile(
                movie_ids=group["movieId"].tolist(),
                ratings=group["rating"].tolist(),
            )
            if profile is not None:
                self.user_profiles[user_id] = profile

        self.is_fitted = True
        return self

    def build_user_profile(self, movie_ids: Sequence[int], ratings: Sequence[float]) -> Optional[np.ndarray]:
        """
        Constructs a normalized user preference vector from rated items.
        Weight = rating - threshold (e.g. 3.0), so liked items have positive weights.
        """
        weights = []
        vectors = []

        for m_id, r in zip(movie_ids, ratings):
            if m_id in self.item_to_idx:
                idx = self.item_to_idx[m_id]
                weight = max(0.1, r - self.rating_threshold)
                weights.append(weight)
                vectors.append(self.item_features[idx])

        if not vectors:
            return None

        # Weighted sum of item vectors
        profile = np.average(vectors, axis=0, weights=weights)
        norm = np.linalg.norm(profile)
        if norm > 1e-9:
            profile = profile / norm
        return profile

    def build_profile_from_preferences(self, preferred_genres_or_keywords: Sequence[str]) -> np.ndarray:
        """
        Creates a user profile vector directly from preferred genres or text keywords.
        Used for new user cold-start onboarding.
        """
        text = " ".join(preferred_genres_or_keywords).lower().strip()
        vec = self.vectorizer.transform([text]).toarray()[0]
        norm = np.linalg.norm(vec)
        if norm > 1e-9:
            vec = vec / norm
        return vec

    def predict(self, user_id: int, item_id: int) -> float:
        """
        Predicts content match score (cosine similarity scaled to [0.5, 5.0]).
        """
        if not self.is_fitted:
            raise RuntimeError("Model must be fitted before predict().")

        if user_id not in self.user_profiles or item_id not in self.item_to_idx:
            return 3.0

        user_vec = self.user_profiles[user_id]
        item_idx = self.item_to_idx[item_id]
        item_vec = self.item_features[item_idx]

        sim = float(np.dot(user_vec, item_vec))
        # Scale cosine similarity [0, 1] into rating range [1.0, 5.0]
        return 1.0 + 4.0 * max(0.0, sim)

    def score_all_items(self, user_id: int) -> np.ndarray:
        """
        Computes cosine similarity scores for all items known to the model.
        Returns np.ndarray of shape (n_items,) aligned with idx_to_item.
        """
        user_vec = self.user_profiles.get(user_id)
        if user_vec is None:
            return np.zeros(len(self.idx_to_item), dtype=float)
        return np.dot(self.item_features, user_vec)

    def recommend(
        self,
        user_id: int,
        n: int = 10,
        exclude_seen: bool = True,
        candidate_items: Optional[Sequence[int]] = None,
    ) -> List[int]:
        """
        Generates top-n content recommendations by computing cosine similarity
        between user profile vector and candidate item vectors.
        """
        if not self.is_fitted:
            raise RuntimeError("Model must be fitted before recommend().")

        scores = self.score_all_items(user_id).copy()
        seen = self.user_seen_items.get(user_id, set()) if exclude_seen else set()

        if candidate_items is not None:
            cand_set = set(candidate_items)
            for idx, item in self.idx_to_item.items():
                if item not in cand_set or item in seen:
                    scores[idx] = -np.inf
        else:
            for item in seen:
                idx = self.item_to_idx.get(item)
                if idx is not None:
                    scores[idx] = -np.inf

        valid_count = np.sum(np.isfinite(scores))
        if valid_count == 0:
            return []

        top_k = min(n, int(valid_count))
        top_indices = np.argpartition(-scores, top_k - 1)[:top_k]
        top_indices = top_indices[np.argsort(-scores[top_indices])]
        return [self.idx_to_item[idx] for idx in top_indices]


    def recommend_for_cold_start_genres(
        self,
        preferred_genres: Sequence[str],
        n: int = 10,
        candidate_items: Optional[Sequence[int]] = None,
    ) -> List[int]:
        """
        Recommends items directly from onboarding genre preferences.
        """
        profile = self.build_profile_from_preferences(preferred_genres)
        if candidate_items is not None:
            valid_cand_indices = [
                self.item_to_idx[item] for item in candidate_items if item in self.item_to_idx
            ]
        else:
            valid_cand_indices = list(self.idx_to_item.keys())

        if not valid_cand_indices:
            return []

        cand_indices_arr = np.array(valid_cand_indices)
        cand_features = self.item_features[cand_indices_arr]
        scores = np.dot(cand_features, profile)

        top_part = np.argpartition(-scores, min(n, len(scores) - 1))[:n]
        top_order = top_part[np.argsort(-scores[top_part])]
        return [self.idx_to_item[cand_indices_arr[idx]] for idx in top_order]
