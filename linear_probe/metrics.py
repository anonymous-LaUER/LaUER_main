"""Classwise AUROC, average precision, and thresholded multilabel metrics."""

from __future__ import annotations

from typing import Dict, List, Optional, Sequence

import numpy as np

def roc_auc_binary(y_true: np.ndarray, y_score: np.ndarray) -> float:
    y_true = np.asarray(y_true).astype(bool, copy=False)
    y_score = np.asarray(y_score, dtype=np.float64)
    n_pos = int(y_true.sum())
    n_neg = int(y_true.size - n_pos)
    if n_pos == 0 or n_neg == 0:
        return float("nan")

    order = np.argsort(y_score, kind="mergesort")
    sorted_scores = y_score[order]
    ranks = np.empty(y_score.size, dtype=np.float64)

    index = 0
    total = y_score.size
    while index < total:
        end = index
        value = sorted_scores[index]
        while end + 1 < total and sorted_scores[end + 1] == value:
            end += 1
        ranks[order[index:end + 1]] = 0.5 * (index + end) + 1.0
        index = end + 1

    sum_ranks_pos = float(ranks[y_true].sum())
    return (sum_ranks_pos - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg)

def macro_f1(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    y_true = np.asarray(y_true).astype(bool, copy=False)
    y_pred = np.asarray(y_pred).astype(bool, copy=False)
    tp = (y_true & y_pred).sum(axis=0).astype(np.float64)
    fp = (~y_true & y_pred).sum(axis=0).astype(np.float64)
    fn = (y_true & ~y_pred).sum(axis=0).astype(np.float64)
    denominator = 2.0 * tp + fp + fn
    per_class = np.divide(2.0 * tp, denominator,
                          out=np.zeros_like(tp), where=denominator > 0)
    return float(per_class.mean()) if per_class.size else 0.0

def subset_accuracy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    y_true = np.asarray(y_true).astype(bool, copy=False)
    y_pred = np.asarray(y_pred).astype(bool, copy=False)
    if y_true.shape[0] == 0:
        return 0.0
    return float((y_true == y_pred).all(axis=1).mean())

def evaluate_scores(y_true: np.ndarray,
                    y_prob: np.ndarray,
                    label_names: Optional[Sequence[str]] = None,
                    threshold: float = 0.5,
                    score_space: str = "probability") -> Dict[str, object]:
    y_true = np.asarray(y_true)
    y_prob = np.asarray(y_prob, dtype=np.float64)
    if y_true.ndim != 2 or y_prob.shape != y_true.shape:
        raise ValueError(
            f"shape mismatch: y_true {y_true.shape} vs y_prob {y_prob.shape}"
        )
    num_classes = y_true.shape[1]
    names = list(label_names) if label_names is not None else [
        f"c{i}" for i in range(num_classes)
    ]
    if len(names) != num_classes:
        raise ValueError(
            f"label_names has {len(names)} entries, expected {num_classes}"
        )

    per_class: Dict[str, float] = {}
    undefined: List[str] = []
    positives: Dict[str, int] = {}
    for index, name in enumerate(names):
        column = y_true[:, index]
        positives[name] = int(column.sum())
        auc = roc_auc_binary(column, y_prob[:, index])
        if np.isnan(auc):
            undefined.append(name)
        else:
            per_class[name] = float(auc)

    scored = len(per_class)
    macro = float(np.mean(list(per_class.values()))) if scored else float("nan")

    if score_space == "probability":
        y_pred = y_prob >= threshold
    elif score_space == "logit":
        y_pred = y_prob >= 0.0
    else:
        raise ValueError(f"score_space must be probability|logit, got {score_space!r}")

    return {
        "macro_auroc": macro,
        "macro_f1": macro_f1(y_true, y_pred),
        "subset_accuracy": subset_accuracy(y_true, y_pred),
        "n_classes_total": num_classes,
        "n_classes_scored": scored,
        "undefined_classes": undefined,
        "per_class_auroc": per_class,
        "per_class_positives": positives,
        "threshold": threshold,
        "score_space": score_space,
    }

def format_metrics(metrics: Dict[str, object], prefix: str = "") -> str:
    macro = metrics.get("macro_auroc")
    macro_text = "nan" if macro is None or np.isnan(float(macro)) else f"{float(macro):.4f}"
    return (
        f"{prefix}AUROC={macro_text} "
        f"F1={float(metrics['macro_f1']):.4f} "
        f"Acc={float(metrics['subset_accuracy']):.4f} "
        f"[{metrics['n_classes_scored']}/{metrics['n_classes_total']} classes scored]"
    )
