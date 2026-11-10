import numpy as np
import pytest
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score

from linear_probe.metrics import (
    evaluate_scores,
    format_metrics,
    macro_f1,
    roc_auc_binary,
    subset_accuracy,
)

def test_auc_matches_sklearn_without_ties():
    rng = np.random.default_rng(0)
    for _ in range(30):
        y = (rng.random(200) < 0.4).astype(int)
        if y.sum() in (0, 200):
            continue
        score = rng.random(200)
        assert roc_auc_binary(y, score) == pytest.approx(
            roc_auc_score(y, score), abs=1e-12
        )

def test_auc_matches_sklearn_with_heavy_ties():
    rng = np.random.default_rng(1)
    worst = 0.0
    for _ in range(50):
        y = (rng.random(150) < 0.35).astype(int)
        if y.sum() in (0, 150):
            continue
        score = np.round(rng.random(150) * 4) / 4
        worst = max(worst, abs(roc_auc_binary(y, score) - roc_auc_score(y, score)))
    assert worst < 1e-12

def test_auc_undefined_for_single_class():
    assert np.isnan(roc_auc_binary(np.zeros(10), np.arange(10, dtype=float)))
    assert np.isnan(roc_auc_binary(np.ones(10), np.arange(10, dtype=float)))

def test_auc_perfect_separation():
    y = np.array([0, 0, 1, 1])
    assert roc_auc_binary(y, np.array([0.1, 0.2, 0.8, 0.9])) == pytest.approx(1.0)
    assert roc_auc_binary(y, np.array([0.9, 0.8, 0.2, 0.1])) == pytest.approx(0.0)

def test_macro_f1_and_accuracy_match_sklearn():
    rng = np.random.default_rng(2)
    y = (rng.random((300, 12)) < 0.25)
    pred = rng.random((300, 12)) >= 0.5
    assert macro_f1(y, pred) == pytest.approx(
        f1_score(y, pred, average="macro", zero_division=0), abs=1e-12
    )
    assert subset_accuracy(y, pred) == pytest.approx(accuracy_score(y, pred))

def test_evaluate_scores_accounts_for_undefined_classes():
    y = np.zeros((10, 3), dtype=np.float32)
    y[:, 0] = [1, 1, 1, 0, 0, 0, 1, 0, 1, 0]
    y[:, 1] = 1.0
    y[:, 2] = 0.0
    scores = np.random.default_rng(3).random((10, 3))
    metrics = evaluate_scores(y, scores, ["A", "B", "C"])
    assert metrics["n_classes_total"] == 3
    assert metrics["n_classes_scored"] == 1
    assert set(metrics["undefined_classes"]) == {"B", "C"}
    assert metrics["per_class_positives"]["B"] == 10
    assert "1/3 classes scored" in format_metrics(metrics)

def test_auc_is_invariant_to_monotone_transform():
    rng = np.random.default_rng(4)
    y = (rng.random((200, 4)) < 0.3).astype(np.float32)
    logits = rng.normal(size=(200, 4)) * 3
    probability = 1.0 / (1.0 + np.exp(-logits))
    a = evaluate_scores(y, logits, score_space="logit")
    b = evaluate_scores(y, probability, score_space="probability")
    assert a["macro_auroc"] == pytest.approx(b["macro_auroc"], abs=1e-12)

def test_threshold_only_affects_f1_and_accuracy():
    rng = np.random.default_rng(5)
    y = (rng.random((200, 5)) < 0.3).astype(np.float32)
    probability = rng.random((200, 5))
    low = evaluate_scores(y, probability, threshold=0.2)
    high = evaluate_scores(y, probability, threshold=0.8)
    assert low["macro_auroc"] == high["macro_auroc"]
    assert low["macro_f1"] >= high["macro_f1"]

def test_shape_mismatch_raises():
    with pytest.raises(ValueError):
        evaluate_scores(np.zeros((4, 2)), np.zeros((4, 3)))
    with pytest.raises(ValueError):
        evaluate_scores(np.zeros((4, 2)), np.zeros((4, 2)), ["only-one"])
