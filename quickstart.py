"""
Quickstart demo script for MovieLens Recommender System.
Demonstrates:
1. Training models on MovieLens dataset.
2. Generating top-5 recommendations for an existing user.
3. Cold-start recommendation for a new user with no history.
4. Cold-start recommendation with genre onboarding.
"""

from src.cold_start import ColdStartHandler
from src.data_loader import load_movielens_data
from src.models.collaborative import CollaborativeFilteringRecommender
from src.models.content_based import ContentBasedRecommender
from src.models.hybrid import HybridRecommender
from src.models.popularity import PopularityRecommender
from src.split import temporal_per_user_split


def main():
    print("=" * 70)
    print("  MOVIELENS RECOMMENDER SYSTEM - QUICKSTART DEMO")
    print("=" * 70)

    # 1. Load Data and Split
    print("\n[1/4] Loading MovieLens dataset...")
    dataset = load_movielens_data(data_dir="data")
    split = temporal_per_user_split(dataset.ratings, test_ratio=0.2)
    print(f"Loaded {len(dataset.ratings):,} ratings across {dataset.ratings['movieId'].nunique():,} movies.")

    # 2. Train Hybrid Model
    print("\n[2/4] Training Adaptive Hybrid Recommender (CF + Content + Popularity)...")
    model = HybridRecommender(cf_weight=0.6, gamma=15.0)
    model.fit(split.train, dataset.metadata)
    print("Model fitted successfully!")

    # 3. Recommendations for an existing user
    user_id = 1
    print(f"\n[3/4] Generating Top-5 Recommendations for existing User #{user_id}:")
    rec_ids = model.recommend(user_id=user_id, n=5, exclude_seen=True)

    meta_indexed = dataset.metadata.set_index("movieId")
    for rank, m_id in enumerate(rec_ids, 1):
        row = meta_indexed.loc[m_id]
        print(f"  #{rank}: {row['title']} | Genres: {row['genres']}")

    # 4. Cold-Start Scenarios
    print("\n[4/4] Cold-Start Handling for Brand New Users:")
    # Scenario A: Anonymous User (0 history)
    print("\n  Scenario A: Brand new user with 0 history (Bayesian Popularity Fallback):")
    anon_recs = model.recommend(user_id=-999, n=3, exclude_seen=False)
    for rank, m_id in enumerate(anon_recs, 1):
        row = meta_indexed.loc[m_id]
        print(f"    #{rank}: {row['title']} | Genres: {row['genres']}")

    # Scenario B: Onboarding Genre Preference
    genres = ["Sci-Fi", "Action"]
    print(f"\n  Scenario B: Brand new user picked genres {genres}:")
    new_user_recs = model.recommend_for_new_user(preferred_genres=genres, n=3)
    for rank, m_id in enumerate(new_user_recs, 1):
        row = meta_indexed.loc[m_id]
        print(f"    #{rank}: {row['title']} | Genres: {row['genres']}")

    print("\n" + "=" * 70)
    print("Demo completed successfully! Run 'python run_evaluation.py' for full benchmarks,")
    print("or run 'streamlit run app.py' to launch the interactive UI.")
    print("=" * 70)


if __name__ == "__main__":
    main()
