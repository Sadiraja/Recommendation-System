"""
Unit tests for cold-start handler module.
"""

import pandas as pd
import pytest
from src.cold_start import ColdStartHandler
from src.models.content_based import ContentBasedRecommender
from src.models.popularity import PopularityRecommender


@pytest.fixture
def cold_start_setup():
    ratings = pd.DataFrame(
        [
            {"userId": 1, "movieId": 1, "rating": 5.0, "timestamp": 1},
            {"userId": 1, "movieId": 2, "rating": 4.5, "timestamp": 2},
            {"userId": 2, "movieId": 1, "rating": 5.0, "timestamp": 3},
            {"userId": 2, "movieId": 3, "rating": 2.0, "timestamp": 4},
        ]
    )
    metadata = pd.DataFrame(
        [
            {"movieId": 1, "title": "The Matrix (1999)", "genres": "Action|Sci-Fi", "genres_list": ["Action", "Sci-Fi"], "year": 1999, "combined_content": "the matrix action sci-fi cyberpunk"},
            {"movieId": 2, "title": "Interstellar (2014)", "genres": "Adventure|Drama|Sci-Fi", "genres_list": ["Adventure", "Drama", "Sci-Fi"], "year": 2014, "combined_content": "interstellar adventure drama sci-fi space"},
            {"movieId": 3, "title": "The Notebook (2004)", "genres": "Drama|Romance", "genres_list": ["Drama", "Romance"], "year": 2004, "combined_content": "the notebook drama romance"},
        ]
    )

    pop = PopularityRecommender().fit(ratings, metadata)
    cb = ContentBasedRecommender(min_df=1).fit(ratings, metadata)
    handler = ColdStartHandler(popularity_model=pop, content_model=cb, metadata=metadata)
    return handler


def test_anonymous_user(cold_start_setup):
    handler = cold_start_setup
    recs = handler.handle_anonymous_user(n=2)
    assert len(recs) == 2
    assert recs[0]["rank"] == 1
    assert "explanation" in recs[0]


def test_genre_preferences(cold_start_setup):
    handler = cold_start_setup
    recs = handler.handle_genre_preferences(["Sci-Fi", "Cyberpunk"], n=1)
    assert len(recs) == 1
    assert recs[0]["movieId"] == 1  # The Matrix


def test_seed_ratings(cold_start_setup):
    handler = cold_start_setup
    recs = handler.handle_interactive_elicitation(seed_ratings={1: 5.0}, n=1)
    assert len(recs) == 1
    # Interstellar (2) also has Sci-Fi
    assert recs[0]["movieId"] == 2


def test_new_item_cold_start(cold_start_setup):
    handler = cold_start_setup
    # Finding neighbors for The Matrix (1)
    recs = handler.handle_new_item_cold_start(target_movie_id=1, n=1)
    assert len(recs) == 1
    assert recs[0]["movieId"] == 2  # Interstellar shares sci-fi
