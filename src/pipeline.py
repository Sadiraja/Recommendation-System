"""
Evaluation and benchmark pipeline.
Fits models, evaluates ranking metrics across test users,
and computes cold-start simulations.
"""

from typing import Dict, List, Optional, Tuple
import numpy as np
import pandas as pd
from tqdm import tqdm

from .cold_start import ColdStartHandler
from .data_loader import MovieLensDataset, load_movielens_data
from .metrics import evaluate_recommendations_batch
from .models.base import BaseRecommender
from .models.collaborative import CollaborativeFilteringRecommender
from .models.content_based import ContentBasedRecommender
from .models.hybrid import HybridRecommender
from .models.popularity import PopularityRecommender
from .split import TrainTestSplit, temporal_per_user_split


def run_pipeline(
    data_dir: str = "data",
    test_ratio: float = 0.2,
    min_positive_rating: float = 3.5,
    k_list: Tuple[int, ...] = (5, 10, 20),
    max_eval_users: Optional[int] = None,
) -> Tuple[Dict[str, Dict[str, float]], TrainTestSplit, MovieLensDataset, Dict[str, BaseRecommender]]:
    """
    Runs the complete training and evaluation pipeline on MovieLens.

    Args:
        data_dir: Path to dataset folder.
        test_ratio: Temporal holdout ratio.
        min_positive_rating: Threshold for relevant items in evaluation.
        k_list: Cutoffs for Precision, Recall, NDCG.
        max_eval_users: Optional limit on evaluation users for quick checks.

    Returns:
        results: Dict mapping model name to metrics dictionary.
        split_data: TrainTestSplit object.
        dataset: MovieLensDataset object.
        models: Dict of trained model instances.
    """
    print("=" * 65)
    print(" 1. LOADING MOVIELENS DATASET")
    print("=" * 65)
    dataset = load_movielens_data(data_dir=data_dir)
    print(f"Total ratings: {len(dataset.ratings):,}")
    print(f"Unique users:  {dataset.ratings['userId'].nunique():,}")
    print(f"Unique movies: {dataset.ratings['movieId'].nunique():,}")

    print("\n" + "=" * 65)
    print(" 2. TEMPORAL TRAIN/TEST SPLIT (PER-USER CHRONOLOGICAL)")
    print("=" * 65)
    split_data = temporal_per_user_split(
        dataset.ratings,
        test_ratio=test_ratio,
        min_positive_rating=min_positive_rating,
    )
    print(f"Train interactions: {len(split_data.train):,}")
    print(f"Test interactions:  {len(split_data.test):,}")
    print(f"Test positive items (rating >= {min_positive_rating}): {len(split_data.positive_test):,}")

    # Prepare ground truth dictionary for test users
    user_actuals: Dict[int, List[int]] = {}
    for user_id, group in split_data.positive_test.groupby("userId"):
        user_actuals[user_id] = group["movieId"].tolist()

    eval_user_ids = sorted(user_actuals.keys())
    if max_eval_users and max_eval_users < len(eval_user_ids):
        eval_user_ids = eval_user_ids[:max_eval_users]
        user_actuals = {u: user_actuals[u] for u in eval_user_ids}

    print(f"Evaluating on {len(eval_user_ids)} test users who have ground-truth positive test items.")

    print("\n" + "=" * 65)
    print(" 3. MODEL INITIALIZATION AND TRAINING")
    print("=" * 65)
    models: Dict[str, BaseRecommender] = {
        "Popularity Baseline": PopularityRecommender(),
        "Collaborative Filtering (SVD)": CollaborativeFilteringRecommender(n_factors=30),
        "Content-Based (TF-IDF)": ContentBasedRecommender(),
        "Hybrid Recommender": HybridRecommender(cf_weight=0.6, gamma=15.0),
    }

    results: Dict[str, Dict[str, float]] = {}
    max_k = max(k_list)

    for name, model in models.items():
        print(f"\n--> Training {name}...")
        model.fit(train_ratings=split_data.train, items_metadata=dataset.metadata)

        print(f"--> Generating top-{max_k} recommendations for test users...")
        user_predictions: Dict[int, List[int]] = {}
        for uid in tqdm(eval_user_ids, desc=f"Evaluating {name}"):
            user_predictions[uid] = model.recommend(user_id=uid, n=max_k, exclude_seen=True)

        print(f"--> Calculating ranking metrics at K = {k_list}...")
        model_metrics = evaluate_recommendations_batch(
            user_actuals=user_actuals,
            user_predictions=user_predictions,
            k_list=k_list,
        )
        results[name] = model_metrics

    return results, split_data, dataset, models
