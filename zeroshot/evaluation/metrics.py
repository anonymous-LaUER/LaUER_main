"""Detailed multi-label metrics with MELP-compatible macro AUROC."""

from __future__ import annotations

from typing import Dict, Sequence

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    coverage_error,
    label_ranking_average_precision_score,
    label_ranking_loss,
    precision_recall_curve,
    roc_auc_score,
)

def _finite_scores(scores: np.ndarray) -> np.ndarray:
    scores = np.asarray(scores, dtype=np.float64)
    if not np.isfinite(scores).all():
        raise ValueError("Scores contain NaN or Inf")
    return scores

def _best_f1_threshold(y_true: np.ndarray, score: np.ndarray):
    precision, recall, thresholds = precision_recall_curve(y_true, score)
    f1 = 2.0 * precision * recall / np.maximum(precision + recall, 1e-12)
    index = int(np.nanargmax(f1))
    if index >= len(thresholds):
        threshold = float(np.nextafter(score.min(), -np.inf))
    else:
        threshold = float(thresholds[index])
    return threshold, float(f1[index])

def _binary_confusion(y_true: np.ndarray, predicted: np.ndarray) -> Dict:
    y_true = y_true.astype(bool)
    predicted = predicted.astype(bool)
    tp = int(np.logical_and(y_true, predicted).sum())
    tn = int(np.logical_and(~y_true, ~predicted).sum())
    fp = int(np.logical_and(~y_true, predicted).sum())
    fn = int(np.logical_and(y_true, ~predicted).sum())
    return {
        "tp": tp,
        "tn": tn,
        "fp": fp,
        "fn": fn,
        "sensitivity": tp / (tp + fn) if tp + fn else None,
        "specificity": tn / (tn + fp) if tn + fp else None,
        "precision": tp / (tp + fp) if tp + fp else None,
        "accuracy": (tp + tn) / (tp + tn + fp + fn),
    }

def _ranking_metrics(y_true: np.ndarray, scores: np.ndarray) -> Dict:
    positive_per_row = y_true.sum(axis=1)
    valid = positive_per_row > 0
    result: Dict = {
        "rows_with_positive": int(valid.sum()),
        "rows_without_positive": int((~valid).sum()),
    }
    if not valid.any():
        return result
    truth = y_true[valid]
    values = scores[valid]
    result.update(
        {
            "label_ranking_average_precision": float(
                label_ranking_average_precision_score(truth, values)
            ),
            "coverage_error": float(coverage_error(truth, values)),
            "label_ranking_loss": float(label_ranking_loss(truth, values)),
        }
    )
    order = np.argsort(-values, axis=1)
    ranked_truth = np.take_along_axis(truth, order, axis=1)
    first_positions = np.argmax(ranked_truth > 0, axis=1) + 1
    result["mean_reciprocal_rank"] = float(
        np.mean(1.0 / first_positions)
    )
    for k in sorted({1, min(3, truth.shape[1]), min(5, truth.shape[1])}):
        top_truth = ranked_truth[:, :k]
        result[f"hit_at_{k}"] = float((top_truth.sum(axis=1) > 0).mean())
        result[f"precision_at_{k}"] = float(top_truth.mean())
        result[f"recall_at_{k}"] = float(
            np.mean(
                top_truth.sum(axis=1)
                / np.maximum(truth.sum(axis=1), 1)
            )
        )
    return result

def evaluate_multilabel_scores(
    y_true: np.ndarray,
    scores: np.ndarray,
    labels: Sequence[str],
) -> Dict:
    """Evaluate one score family.

    ``macro_auroc`` exactly follows MELP: compute binary AUROC per class, skip
    classes that are constant in the evaluated split, then take the mean.
    Threshold-dependent values are explicitly named ``oracle_max_f1`` because
    their thresholds are selected on the same test set and are diagnostic,
    not deployment estimates.
    """
    truth = np.asarray(y_true)
    values = _finite_scores(scores)
    if truth.shape != values.shape:
        raise ValueError(
            f"Shape mismatch: y_true={truth.shape}, scores={values.shape}"
        )
    if truth.ndim != 2 or truth.shape[1] != len(labels):
        raise ValueError("Expected [N,C] arrays aligned to labels")
    if not np.isin(truth, (0, 1)).all():
        raise ValueError("y_true must be binary")
    truth = truth.astype(np.int8)

    per_class: Dict[str, Dict] = {}
    aurocs = []
    average_precisions = []
    for index, label in enumerate(labels):
        target = truth[:, index]
        score = values[:, index]
        positives = int(target.sum())
        negatives = int(len(target) - positives)
        class_result: Dict = {
            "positives": positives,
            "negatives": negatives,
            "prevalence": positives / len(target) if len(target) else None,
        }
        if positives == 0 or negatives == 0:
            class_result.update(
                {
                    "valid_for_auc": False,
                    "auroc": None,
                    "average_precision": None,
                    "oracle_max_f1": None,
                    "oracle_threshold": None,
                    "oracle_confusion": None,
                }
            )
        else:
            auroc = float(roc_auc_score(target, score))
            average_precision = float(
                average_precision_score(target, score)
            )
            threshold, best_f1 = _best_f1_threshold(target, score)
            predicted = score >= threshold
            class_result.update(
                {
                    "valid_for_auc": True,
                    "auroc": auroc,
                    "average_precision": average_precision,
                    "oracle_max_f1": best_f1,
                    "oracle_threshold": threshold,
                    "oracle_confusion": _binary_confusion(target, predicted),
                }
            )
            aurocs.append(auroc)
            average_precisions.append(average_precision)
        per_class[label] = class_result

    flattened = truth.ravel()
    flattened_scores = values.ravel()
    micro_auroc = (
        float(roc_auc_score(flattened, flattened_scores))
        if np.unique(flattened).size == 2
        else None
    )
    micro_average_precision = (
        float(average_precision_score(flattened, flattened_scores))
        if np.unique(flattened).size == 2
        else None
    )
    return {
        "sample_count": int(truth.shape[0]),
        "class_count": int(truth.shape[1]),
        "valid_auc_class_count": len(aurocs),
        "constant_class_count": int(truth.shape[1] - len(aurocs)),
"macro_auroc": float(np.mean(aurocs)) if aurocs else None,
"macro_average_precision": (
            float(np.mean(average_precisions)) if average_precisions else None
        ),
        "micro_auroc": micro_auroc,
        "micro_average_precision": micro_average_precision,
        "ranking": _ranking_metrics(truth, values),
        "per_class": per_class,
    }
