"""
Unit tests for recommender models (Popularity, CF SVD, Content-Based, Hybrid).
"""

import pandas as pd
import pytest
from src.models.collaborative import CollaborativeFilteringRecommender
from src.models.content_based import ContentBasedRecommender
from src.models.hybrid import HybridRecommender
from src.models.popularity import PopularityRecommender


@pytest.fixture
def mock_data():
    ratings = pd.DataFrame(
        [
            {"userId": 1, "movieId": 10, "rating": 5.0, "timestamp": 1},
            {"userId": 1, "movieId": 20, "rating": 4.0, "timestamp": 2},
            {"userId": 1, "movieId": 30, "rating": 1.0, "timestamp": 3},
            {"userId": 2, "movieId": 10, "rating": 5.0, "timestamp": 4},
            {"userId": 2, "movieId": 20, "rating": 5.0, "timestamp": 5},
            {"userId": 2, "movieId": 40, "rating": 4.0, "timestamp": 6},
            {"userId": 3, "movieId": 30, "rating": 5.0, "timestamp": 7},
            {"userId": 3, "movieId": 40, "rating": 2.0, "timestamp": 8},
            {"userId": 3, "movieId": 50, "rating": 5.0, "timestamp": 9},
        ]
    )

    metadata = pd.DataFrame(
        [
            {"movieId": 10, "title": "Star Wars (1977)", "genres_list": ["Action", "Sci-Fi"], "combined_content": "star wars sci-fi action space"},
            {"movieId": 20, "title": "Star Trek (1979)", "genres_list": ["Action", "Sci-Fi"], "combined_content": "star trek sci-fi action space enterprise"},
            {"movieId": 30, "title": "The Notebook (2004)", "genres_list": ["Drama", "Romance"], "combined_content": "the notebook drama romance love"},
            {"movieId": 40, "title": "Titanic (1997)", "genres_list": ["Drama", "Romance"], "combined_content": "titanic drama romance love ship"},
            {"movieId": 50, "title": "Pride & Prejudice (2005)", "genres_list": ["Drama", "Romance"], "combined_content": "pride prejudice drama romance"},
        ]
    )
    return ratings, metadata


def test_popularity_recommender(mock_data):
    ratings, metadata = mock_data
    pop = PopularityRecommender()
    pop.fit(ratings, metadata)

    # Movies 10 and 20 received high ratings consistently
    recs = pop.recommend(user_id=1, n=2, exclude_seen=True)
    assert len(recs) == 2
    # User 1 has seen 10, 20, 30. Remaining are 40, 50
    assert 10 not in recs
    assert 20 not in recs

    # Cold-start recommendations for brand new user
    cold_recs = pop.recommend(user_id=999, n=3, exclude_seen=False)
    assert len(cold_recs) == 3


def test_collaborative_filtering(mock_data):
    ratings, metadata = mock_data
    cf = CollaborativeFilteringRecommender(n_factors=2)
    cf.fit(ratings, metadata)

    # Predict rating for user 1 on movie 40
    pred = cf.predict(user_id=1, item_id=40)
    assert 0.5 <= pred <= 5.0

    recs = cf.recommend(user_id=1, n=2, exclude_seen=True)
    assert len(recs) <= 2
    assert 10 not in recs  # Seen item filtered out


def test_content_based_recommender(mock_data):
    ratings, metadata = mock_data
    cb = ContentBasedRecommender(min_df=1)
    cb.fit(ratings, metadata)

    # User 1 loved Sci-Fi (10, 20) and hated Romance (30)
    # The unseen movies are 40 and 50 (Romance).
    recs = cb.recommend(user_id=1, n=2, exclude_seen=True)
    assert len(recs) == 2

    # Test onboarding genre recommendations
    genre_recs = cb.recommend_for_cold_start_genres(preferred_genres=["space", "sci-fi"], n=2)
    assert len(genre_recs) == 2
    # Top should be Star Wars or Star Trek
    assert set(genre_recs).intersection({10, 20})


def test_hybrid_recommender(mock_data):
    ratings, metadata = mock_data
    hybrid = HybridRecommender(cf_weight=0.6, gamma=5.0)
    hybrid.fit(ratings, metadata)

    # Known user
    recs = hybrid.recommend(user_id=2, n=2, exclude_seen=True)
    assert len(recs) == 2

    # Cold start user with no history
    cold_recs = hybrid.recommend(user_id=999, n=2, exclude_seen=True)
    assert len(cold_recs) == 2

    # New user with genre preferences
    new_user_genre_recs = hybrid.recommend_for_new_user(preferred_genres=["drama", "romance"], n=2)
    assert len(new_user_genre_recs) == 2
