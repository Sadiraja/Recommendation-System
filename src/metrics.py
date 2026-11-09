"""
Ranking evaluation metrics for recommender systems.
Implements:
- Precision@K
- Recall@K
- NDCG@K (Normalized Discounted Cumulative Gain)
- MAP@K (Mean Average Precision)
- HitRate@K (Hit Rate)
- MRR@K (Mean Reciprocal Rank)
"""

from typing import Dict, Iterable, List, Sequence, Set, Union
import numpy as np


def precision_at_k(actual: Sequence[Union[int, str]], predicted: Sequence[Union[int, str]], k: int) -> float:
    """
    Computes Precision@K.
    Proportion of recommended items in the top-K that are relevant.

    Args:
        actual: Collection of ground-truth relevant item IDs.
        predicted: Ordered list of recommended item IDs.
        k: Cutoff rank.

    Returns:
        Precision@K score in [0.0, 1.0].
    """
    if k <= 0:
        return 0.0
    actual_set = set(actual)
    if not actual_set:
        return 0.0

    pred_k = predicted[:k]
    if not pred_k:
        return 0.0

    hits = sum(1 for item in pred_k if item in actual_set)
    return hits / float(k)


def recall_at_k(actual: Sequence[Union[int, str]], predicted: Sequence[Union[int, str]], k: int) -> float:
    """
    Computes Recall@K.
    Proportion of all relevant items retrieved in the top-K recommendations.

    Args:
        actual: Collection of ground-truth relevant item IDs.
        predicted: Ordered list of recommended item IDs.
        k: Cutoff rank.

    Returns:
        Recall@K score in [0.0, 1.0].
    """
    if k <= 0:
        return 0.0
    actual_set = set(actual)
    if not actual_set:
        return 0.0

    pred_k = predicted[:k]
    hits = sum(1 for item in pred_k if item in actual_set)
    return hits / float(len(actual_set))


def ndcg_at_k(
    actual: Sequence[Union[int, str]],
    predicted: Sequence[Union[int, str]],
    k: int,
    relevance_scores: Dict[Union[int, str], float] = None,
) -> float:
    """
    Computes Normalized Discounted Cumulative Gain (NDCG@K).
    Measures the ranking quality, rewarding relevant items placed at higher ranks.

    Args:
        actual: Ground-truth relevant items.
        predicted: Ordered list of recommended item IDs.
        k: Cutoff rank.
        relevance_scores: Optional mapping from item ID to graded relevance value.
                          If None, binary relevance is assumed (1.0 for actual items, 0.0 otherwise).

    Returns:
        NDCG@K score in [0.0, 1.0].
    """
    if k <= 0:
        return 0.0
    actual_set = set(actual)
    if not actual_set:
        return 0.0

    pred_k = predicted[:k]
    if not pred_k:
        return 0.0

    # Calculate DCG@K
    dcg = 0.0
    for idx, item in enumerate(pred_k):
        if item in actual_set:
            rel = relevance_scores.get(item, 1.0) if relevance_scores else 1.0
            dcg += rel / np.log2(idx + 2)  # idx + 2 because 1-based rank (idx=0 -> rank=1 -> log2(2)=1)

    # Calculate Ideal DCG@K (IDCG@K)
    if relevance_scores:
        ideal_rels = sorted([relevance_scores.get(item, 1.0) for item in actual_set], reverse=True)[:k]
    else:
        ideal_rels = [1.0] * min(k, len(actual_set))

    idcg = sum(rel / np.log2(idx + 2) for idx, rel in enumerate(ideal_rels))

    if idcg <= 0.0:
        return 0.0

    return dcg / idcg


def average_precision_at_k(actual: Sequence[Union[int, str]], predicted: Sequence[Union[int, str]], k: int) -> float:
    """
    Computes Average Precision at K (AP@K) for a single query/user.
    """
    if k <= 0:
        return 0.0
    actual_set = set(actual)
    if not actual_set:
        return 0.0

    pred_k = predicted[:k]
    hits = 0
    sum_precisions = 0.0

    for idx, item in enumerate(pred_k):
        if item in actual_set:
            hits += 1
            sum_precisions += hits / float(idx + 1)

    max_possible_hits = min(k, len(actual_set))
    if max_possible_hits == 0:
        return 0.0

    return sum_precisions / float(max_possible_hits)


def hit_rate_at_k(actual: Sequence[Union[int, str]], predicted: Sequence[Union[int, str]], k: int) -> float:
    """
    Computes Hit Rate at K (HR@K).
    1.0 if at least one relevant item appears in top-K, 0.0 otherwise.
    """
    if k <= 0:
        return 0.0
    actual_set = set(actual)
    if not actual_set:
        return 0.0

    pred_k = predicted[:k]
    return 1.0 if any(item in actual_set for item in pred_k) else 0.0


def mrr_at_k(actual: Sequence[Union[int, str]], predicted: Sequence[Union[int, str]], k: int) -> float:
    """
    Computes Mean Reciprocal Rank at K (MRR@K).
    1 / rank of the first relevant item in top-K, or 0.0 if not found.
    """
    if k <= 0:
        return 0.0
    actual_set = set(actual)
    if not actual_set:
        return 0.0

    pred_k = predicted[:k]
    for idx, item in enumerate(pred_k):
        if item in actual_set:
            return 1.0 / float(idx + 1)
    return 0.0


def evaluate_recommendations_batch(
    user_actuals: Dict[Union[int, str], Sequence[Union[int, str]]],
    user_predictions: Dict[Union[int, str], Sequence[Union[int, str]]],
    k_list: Sequence[int] = (5, 10, 20),
) -> Dict[str, float]:
    """
    Evaluates top-K recommendations across all users for multiple cutoffs K.

    Args:
        user_actuals: Dict mapping userId -> list/set of relevant ground-truth itemIds.
        user_predictions: Dict mapping userId -> ranked list of recommended itemIds.
        k_list: Tuple or list of K values to evaluate at.

    Returns:
        Dictionary of mean metric scores (e.g. {'Precision@5': 0.12, 'Recall@5': 0.08, ...}).
    """
    metrics = {f"{metric}@{k}": [] for k in k_list for metric in ("Precision", "Recall", "NDCG", "MAP", "HitRate", "MRR")}

    eval_users = [u for u in user_actuals.keys() if len(user_actuals[u]) > 0 and u in user_predictions]
    if not eval_users:
        return {metric_name: 0.0 for metric_name in metrics}

    for user in eval_users:
        actual = user_actuals[user]
        pred = user_predictions[user]

        for k in k_list:
            metrics[f"Precision@{k}"].append(precision_at_k(actual, pred, k))
            metrics[f"Recall@{k}"].append(recall_at_k(actual, pred, k))
            metrics[f"NDCG@{k}"].append(ndcg_at_k(actual, pred, k))
            metrics[f"MAP@{k}"].append(average_precision_at_k(actual, pred, k))
            metrics[f"HitRate@{k}"].append(hit_rate_at_k(actual, pred, k))
            metrics[f"MRR@{k}"].append(mrr_at_k(actual, pred, k))

    # Compute mean across evaluated users
    return {metric_name: float(np.mean(values)) for metric_name, values in metrics.items()}
