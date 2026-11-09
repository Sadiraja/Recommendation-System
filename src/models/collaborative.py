"""
Collaborative Filtering Recommender using Matrix Factorization (SVD with user & item biases).
Decomposes user-item interaction matrix into latent factor representations:
    r_hat(u, i) = mu + b_u + b_i + p_u^T * q_i
"""

from typing import Dict, List, Optional, Sequence
import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import svds
from .base import BaseRecommender


class CollaborativeFilteringRecommender(BaseRecommender):
    """
    Biased Matrix Factorization Recommender via SVD.
    Learns global mean, user bias b_u, item bias b_i, and k latent factors for users and items.
    """

    def __init__(
        self,
        n_factors: int = 40,
        reg_user: float = 10.0,
        reg_item: float = 10.0,
        name: str = "CollaborativeFiltering_SVD",
    ):
        super().__init__(name=name)
        self.n_factors = n_factors
        self.reg_user = reg_user
        self.reg_item = reg_item

        self.global_mean: float = 0.0
        self.user_to_idx: Dict[int, int] = {}
        self.idx_to_user: Dict[int, int] = {}
        self.item_to_idx: Dict[int, int] = {}
        self.idx_to_item: Dict[int, int] = {}

        self.user_biases: np.ndarray = np.array([])
        self.item_biases: np.ndarray = np.array([])
        self.user_factors: np.ndarray = np.array([])  # (n_users, n_factors)
        self.item_factors: np.ndarray = np.array([])  # (n_items, n_factors)

    def fit(self, train_ratings: pd.DataFrame, items_metadata: Optional[pd.DataFrame] = None) -> "CollaborativeFilteringRecommender":
        """
        Fits Matrix Factorization on the user-item interaction matrix.
        """
        self._store_seen_items(train_ratings)

        users = sorted(train_ratings["userId"].unique())
        items = sorted(train_ratings["movieId"].unique())

        self.user_to_idx = {u: idx for idx, u in enumerate(users)}
        self.idx_to_user = {idx: u for idx, u in enumerate(users)}
        self.item_to_idx = {i: idx for idx, i in enumerate(items)}
        self.idx_to_item = {idx: i for idx, i in enumerate(items)}

        n_users = len(users)
        n_items = len(items)

        self.global_mean = float(train_ratings["rating"].mean())

        # Compute regularized user biases: b_u = sum(r_ui - mu) / (|I_u| + reg_user)
        user_groups = train_ratings.groupby("userId")["rating"]
        user_sums = user_groups.apply(lambda r: np.sum(r - self.global_mean))
        user_counts = user_groups.count()
        user_bias_series = user_sums / (user_counts + self.reg_user)

        self.user_biases = np.zeros(n_users)
        for u, b in user_bias_series.items():
            if u in self.user_to_idx:
                self.user_biases[self.user_to_idx[u]] = b

        # Compute regularized item biases: b_i = sum(r_ui - mu - b_u) / (|U_i| + reg_item)
        temp_df = train_ratings.copy()
        temp_df["u_idx"] = temp_df["userId"].map(self.user_to_idx)
        temp_df["u_bias"] = self.user_biases[temp_df["u_idx"].values]
        temp_df["residual_user"] = temp_df["rating"] - self.global_mean - temp_df["u_bias"]

        item_groups = temp_df.groupby("movieId")["residual_user"]
        item_sums = item_groups.sum()
        item_counts = item_groups.count()
        item_bias_series = item_sums / (item_counts + self.reg_item)

        self.item_biases = np.zeros(n_items)
        for i, b in item_bias_series.items():
            if i in self.item_to_idx:
                self.item_biases[self.item_to_idx[i]] = b

        # Compute residuals for matrix factorization: R_tilde = r_ui - (mu + b_u + b_i)
        temp_df["i_idx"] = temp_df["movieId"].map(self.item_to_idx)
        temp_df["i_bias"] = self.item_biases[temp_df["i_idx"].values]
        residuals = (
            temp_df["rating"].values
            - self.global_mean
            - temp_df["u_bias"].values
            - temp_df["i_bias"].values
        )

        rows = temp_df["u_idx"].values
        cols = temp_df["i_idx"].values
        sparse_res = csr_matrix((residuals, (rows, cols)), shape=(n_users, n_items))

        # Perform Truncated SVD
        actual_k = min(self.n_factors, min(n_users, n_items) - 1)
        actual_k = max(2, actual_k)

        U, sigma, Vt = svds(sparse_res, k=actual_k)
        sigma_sqrt = np.diag(np.sqrt(sigma))

        self.user_factors = np.dot(U, sigma_sqrt)  # (n_users, actual_k)
        self.item_factors = np.dot(sigma_sqrt, Vt).T  # (n_items, actual_k)

        self.is_fitted = True
        return self

    def predict(self, user_id: int, item_id: int) -> float:
        """
        Predicts rating for user_id and item_id.
        Handles unseen users and unseen items gracefully.
        """
        if not self.is_fitted:
            raise RuntimeError("Model must be fitted before predict().")

        u_idx = self.user_to_idx.get(user_id)
        i_idx = self.item_to_idx.get(item_id)

        # Base expectation: global mean
        score = self.global_mean

        if i_idx is not None:
            score += self.item_biases[i_idx]

        if u_idx is not None:
            score += self.user_biases[u_idx]

        if u_idx is not None and i_idx is not None:
            score += float(np.dot(self.user_factors[u_idx], self.item_factors[i_idx]))

        return float(np.clip(score, 0.5, 5.0))

    def score_all_items(self, user_id: int) -> np.ndarray:
        """
        Computes predicted scores for all items known to the model.
        Returns np.ndarray of shape (n_items,) aligned with idx_to_item.
        """
        u_idx = self.user_to_idx.get(user_id)
        if u_idx is not None:
            u_bias = self.user_biases[u_idx]
            u_factor = self.user_factors[u_idx]
            return self.global_mean + u_bias + self.item_biases + np.dot(self.item_factors, u_factor)
        else:
            return self.global_mean + self.item_biases

    def recommend(
        self,
        user_id: int,
        n: int = 10,
        exclude_seen: bool = True,
        candidate_items: Optional[Sequence[int]] = None,
    ) -> List[int]:
        """
        Generates top-n collaborative recommendations for user_id.
        Vectorized across all candidate items for maximum speed.
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

