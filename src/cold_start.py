"""
Cold-Start Handling Module.
Implements a multi-tier strategy for users and items with zero or sparse history:
1. Zero-History Fallback: Bayesian Weighted Popularity.
2. Preference Elicitation: Genre-based profile generation.
3. Few-Shot Rating Elicitation: Instant on-the-fly user profiling.
4. Item Cold-Start: Content-based nearest neighbor retrieval for unrated items.
"""

from typing import Dict, List, Optional, Sequence
import numpy as np
import pandas as pd
from .models.content_based import ContentBasedRecommender
from .models.popularity import PopularityRecommender


class ColdStartHandler:
    """
    Manages cold-start scenarios for both new users and new items.
    """

    def __init__(
        self,
        popularity_model: PopularityRecommender,
        content_model: ContentBasedRecommender,
        metadata: pd.DataFrame,
    ):
        self.popularity_model = popularity_model
        self.content_model = content_model
        self.metadata = metadata.drop_duplicates(subset=["movieId"]).set_index("movieId")

    def handle_anonymous_user(self, n: int = 10) -> List[Dict]:
        """
        Strategy 1: Anonymous / Zero-History User.
        Recommends top globally popular and critically acclaimed movies.
        """
        movie_ids = self.popularity_model.recommend(user_id=-1, n=n, exclude_seen=False)
        return self._format_recommendations(movie_ids, reason="Global Bayesian Popularity Baseline")

    def handle_genre_preferences(self, preferred_genres: Sequence[str], n: int = 10) -> List[Dict]:
        """
        Strategy 2: New user provides genre selections during onboarding.
        Builds on-the-fly preference vector and matches with item content representations.
        """
        movie_ids = self.content_model.recommend_for_cold_start_genres(preferred_genres=preferred_genres, n=n)
        return self._format_recommendations(
            movie_ids,
            reason=f"Content match with preferred genres: {', '.join(preferred_genres)}",
        )

    def handle_interactive_elicitation(
        self,
        seed_ratings: Dict[int, float],
        n: int = 10,
    ) -> List[Dict]:
        """
        Strategy 3: User rates 3-5 seed items during onboarding.
        Constructs an instant weighted profile vector.
        """
        movie_ids = list(seed_ratings.keys())
        ratings = list(seed_ratings.values())

        profile = self.content_model.build_user_profile(movie_ids=movie_ids, ratings=ratings)
        if profile is None:
            return self.handle_anonymous_user(n=n)

        all_ids = [m for m in self.content_model.idx_to_item.values() if m not in seed_ratings]
        cand_indices = [self.content_model.item_to_idx[m] for m in all_ids]
        features = self.content_model.item_features[cand_indices]

        scores = np.dot(features, profile)
        top_k_idx = np.argsort(-scores)[:n]
        recs = [all_ids[idx] for idx in top_k_idx]

        return self._format_recommendations(recs, reason="Synthesized from onboarding ratings")

    def handle_new_item_cold_start(self, target_movie_id: int, n: int = 5) -> List[Dict]:
        """
        Strategy 4: New Movie Cold-Start.
        When a new movie is added to the catalog with zero ratings,
        finds structurally similar movies using content similarity.
        """
        if target_movie_id not in self.content_model.item_to_idx:
            return []

        target_idx = self.content_model.item_to_idx[target_movie_id]
        target_vec = self.content_model.item_features[target_idx]

        all_features = self.content_model.item_features
        sims = np.dot(all_features, target_vec)

        # Exclude self
        sims[target_idx] = -1.0
        top_indices = np.argsort(-sims)[:n]
        similar_ids = [self.content_model.idx_to_item[idx] for idx in top_indices]

        return self._format_recommendations(similar_ids, reason=f"Content neighbors to movie {target_movie_id}")

    def _format_recommendations(self, movie_ids: Sequence[int], reason: str) -> List[Dict]:
        """
        Enriches movie IDs with metadata for presentation.
        """
        results = []
        for rank, m_id in enumerate(movie_ids, 1):
            if m_id in self.metadata.index:
                row = self.metadata.loc[m_id]
                title = row.get("title", f"Movie #{m_id}")
                genres = row.get("genres", "N/A")
                year = row.get("year", "N/A")
            else:
                title = f"Movie #{m_id}"
                genres = "N/A"
                year = "N/A"

            results.append(
                {
                    "rank": rank,
                    "movieId": int(m_id),
                    "title": title,
                    "genres": genres,
                    "year": year,
                    "explanation": reason,
                }
            )
        return results
