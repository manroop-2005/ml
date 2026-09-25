"""
Evaluation metrics implementing the competition's macro-averaged F_0.5 score.
"""

from typing import Dict, List, Set, Tuple
import numpy as np


def compute_entity_f_beta(
    true_matches: Set[str],
    pred_matches: Set[str],
    beta: float = 0.5
) -> float:
    """
    Compute F_beta for a single Source 1 entity.
    Singletons:
      - If ground truth is empty and prediction is empty: 1.0
      - If ground truth is empty and prediction is non-empty: 0.0
      - If ground truth is non-empty and prediction is empty: 0.0
    """
    if not true_matches:
        return 1.0 if not pred_matches else 0.0

    if not pred_matches:
        return 0.0

    tp = len(true_matches & pred_matches)
    if tp == 0:
        return 0.0

    precision = tp / len(pred_matches)
    recall = tp / len(true_matches)

    beta_sq = beta ** 2
    denom = (beta_sq * precision) + recall
    if denom == 0.0:
        return 0.0

    return (1 + beta_sq) * precision * recall / denom


def compute_macro_f_beta(
    ground_truth: Dict[str, Set[str]],
    predictions: Dict[str, Set[str]],
    beta: float = 0.5
) -> Tuple[float, float, float]:
    """
    Compute Macro-average F_0.5, Precision, and Recall across all Source 1 entities.
    """
    all_s1_ids = sorted(ground_truth.keys())
    if not all_s1_ids:
        return 0.0, 0.0, 0.0

    f_scores = []
    precisions = []
    recalls = []

    for s1_id in all_s1_ids:
        true_set = ground_truth.get(s1_id, set())
        pred_set = predictions.get(s1_id, set())

        f_val = compute_entity_f_beta(true_set, pred_set, beta=beta)
        f_scores.append(f_val)

        if true_set and pred_set:
            tp = len(true_set & pred_set)
            precisions.append(tp / len(pred_set))
            recalls.append(tp / len(true_set))
        elif not true_set and not pred_set:
            precisions.append(1.0)
            recalls.append(1.0)
        elif not true_set and pred_set:
            precisions.append(0.0)
            recalls.append(1.0)
        else:
            precisions.append(1.0)
            recalls.append(0.0)

    macro_f = float(np.mean(f_scores))
    macro_p = float(np.mean(precisions))
    macro_r = float(np.mean(recalls))
    return macro_f, macro_p, macro_r


def find_optimal_threshold(
    ground_truth: Dict[str, Set[str]],
    candidate_scores: Dict[str, List[Tuple[str, float]]],
    threshold_grid: List[float] = None,
    beta: float = 0.5
) -> Tuple[float, float]:
    """
    Search for the probability threshold that maximizes Macro F_0.5 on validation data.
    """
    if threshold_grid is None:
        threshold_grid = [0.25, 0.30, 0.35, 0.40, 0.45, 0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80]

    best_thresh = 0.50
    best_score = -1.0

    for thresh in threshold_grid:
        preds: Dict[str, Set[str]] = {}
        for s1_id, cands in candidate_scores.items():
            preds[s1_id] = {cid for cid, score in cands if score >= thresh}

        score, _, _ = compute_macro_f_beta(ground_truth, preds, beta=beta)
        if score > best_score:
            best_score = score
            best_thresh = thresh

    return best_thresh, best_score
