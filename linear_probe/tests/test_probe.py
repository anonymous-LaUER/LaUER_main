import numpy as np
import pytest
import torch

from linear_probe.config import BudgetSpec
from linear_probe.probe import (
    ProbeError,
    evaluate_features,
    train_linear_probe,
)

def _synthetic(n=600, dim=24, classes=4, seed=0):
    rng = np.random.default_rng(seed)
    weights = rng.normal(size=(dim, classes))
    feats = rng.normal(size=(n, dim)).astype(np.float32)
    logits = feats @ weights
    labels = (logits > np.median(logits, axis=0)).astype(np.float32)
    return (torch.from_numpy(feats), torch.from_numpy(labels))

def _budget(**kwargs):
    base = dict(
        name="test", optimizer="adamw", lr=1e-2, weight_decay=1e-4,
        batch_size=32, epochs=8, scheduler="cosine", eta_min=1e-5,
        early_stop_patience=None, select="last",
    )
    base.update(kwargs)
    return BudgetSpec(**base)

def _split(feats, labels):
    cut = int(feats.shape[0] * 0.7)
    return feats[:cut], labels[:cut], feats[cut:], labels[cut:]

def test_learns_separable_synthetic_data():
    feats, labels = _synthetic()
    tr_f, tr_y, va_f, va_y = _split(feats, labels)
    names = [f"c{i}" for i in range(labels.shape[1])]
    outcome = train_linear_probe(tr_f, tr_y, va_f, va_y, names,
                                 _budget(epochs=30), device="cpu", verbose=False)
    assert outcome.val_metrics["macro_auroc"] > 0.9
    assert outcome.in_dim == feats.shape[1]
    assert outcome.num_classes == labels.shape[1]

def test_last_epoch_selection_reports_final_weights():
    feats, labels = _synthetic(seed=1)
    tr_f, tr_y, va_f, va_y = _split(feats, labels)
    names = [f"c{i}" for i in range(labels.shape[1])]
    outcome = train_linear_probe(tr_f, tr_y, va_f, va_y, names,
                                 _budget(epochs=6, select="last"),
                                 device="cpu", verbose=False)
    assert outcome.epochs_run == 6
    assert outcome.best_epoch == 5
    assert not outcome.stopped_early
    final_metrics, _ = evaluate_features(outcome.classifier, va_f, va_y, names, "cpu")
    assert final_metrics["macro_auroc"] == pytest.approx(
        outcome.val_metrics["macro_auroc"], abs=1e-9
    )

def test_best_val_selection_keeps_best_checkpoint():
    feats, labels = _synthetic(seed=2)
    tr_f, tr_y, va_f, va_y = _split(feats, labels)
    names = [f"c{i}" for i in range(labels.shape[1])]
    outcome = train_linear_probe(tr_f, tr_y, va_f, va_y, names,
                                 _budget(epochs=40, select="best_val"),
                                 device="cpu", verbose=False)
    history = outcome.history
    best_epoch = max(history, key=lambda row: (
        -1e9 if not np.isfinite(row["val_macro_auroc"]) else row["val_macro_auroc"]
    ))["epoch"]
    assert outcome.best_epoch == best_epoch
    assert outcome.val_metrics["macro_auroc"] == pytest.approx(
        history[best_epoch]["val_macro_auroc"], abs=1e-9
    )

    assert outcome.val_metrics["macro_auroc"] >= history[-1]["val_macro_auroc"] - 1e-9

def test_early_stopping_triggers_with_patience():
    feats, labels = _synthetic(seed=3)
    tr_f, tr_y, va_f, va_y = _split(feats, labels)
    names = [f"c{i}" for i in range(labels.shape[1])]
    outcome = train_linear_probe(
        tr_f, tr_y, va_f, va_y, names,
        _budget(epochs=200, select="best_val", early_stop_patience=3),
        device="cpu", verbose=False,
    )
    assert outcome.stopped_early
    assert outcome.epochs_run < 200

def test_no_early_stop_runs_all_epochs():
    feats, labels = _synthetic(seed=4)
    tr_f, tr_y, va_f, va_y = _split(feats, labels)
    names = [f"c{i}" for i in range(labels.shape[1])]
    outcome = train_linear_probe(tr_f, tr_y, va_f, va_y, names,
                                 _budget(epochs=9, select="last",
                                         early_stop_patience=None),
                                 device="cpu", verbose=False)
    assert outcome.epochs_run == 9
    assert not outcome.stopped_early

def test_training_is_deterministic_for_fixed_seed():
    feats, labels = _synthetic(seed=5)
    tr_f, tr_y, va_f, va_y = _split(feats, labels)
    names = [f"c{i}" for i in range(labels.shape[1])]
    budget = _budget(epochs=5)
    first = train_linear_probe(tr_f, tr_y, va_f, va_y, names, budget,
                               device="cpu", seed=7, verbose=False)
    second = train_linear_probe(tr_f, tr_y, va_f, va_y, names, budget,
                                device="cpu", seed=7, verbose=False)
    for key in first.state_dict:
        assert torch.equal(first.state_dict[key], second.state_dict[key])
    assert first.val_metrics["macro_auroc"] == second.val_metrics["macro_auroc"]

def test_different_seeds_differ():
    feats, labels = _synthetic(seed=6)
    tr_f, tr_y, va_f, va_y = _split(feats, labels)
    names = [f"c{i}" for i in range(labels.shape[1])]
    budget = _budget(epochs=4)
    first = train_linear_probe(tr_f, tr_y, va_f, va_y, names, budget,
                               device="cpu", seed=1, verbose=False)
    second = train_linear_probe(tr_f, tr_y, va_f, va_y, names, budget,
                                device="cpu", seed=2, verbose=False)
    assert not torch.equal(first.state_dict["weight"], second.state_dict["weight"])

def test_all_undefined_classes_is_not_fatal():
    feats, labels = _synthetic(seed=7)
    tr_f, tr_y, va_f, va_y = _split(feats, labels)
    single = torch.ones_like(va_y)
    names = [f"c{i}" for i in range(labels.shape[1])]
    outcome = train_linear_probe(tr_f, tr_y, va_f, single, names,
                                 _budget(epochs=4, select="best_val",
                                         early_stop_patience=2),
                                 device="cpu", verbose=False)

    assert outcome.degraded
    assert outcome.val_metrics["n_classes_scored"] == 0
    assert outcome.best_epoch == outcome.epochs_run - 1

    assert outcome.epochs_run <= 4

def test_empty_training_set_raises():
    feats, labels = _synthetic(seed=8)
    names = [f"c{i}" for i in range(labels.shape[1])]
    with pytest.raises(ProbeError):
        train_linear_probe(feats[:0], labels[:0], feats, labels, names,
                           _budget(epochs=1), device="cpu", verbose=False)

def test_class_count_mismatch_raises():
    feats, labels = _synthetic(seed=9)
    with pytest.raises(ProbeError):
        train_linear_probe(feats, labels, feats, labels[:, :2],
                           ["a", "b"], _budget(epochs=1), device="cpu",
                           verbose=False)

def test_batch_size_larger_than_train_set_is_clamped():
    feats, labels = _synthetic(n=20, seed=10)
    names = [f"c{i}" for i in range(labels.shape[1])]
    outcome = train_linear_probe(feats, labels, feats, labels, names,
                                 _budget(epochs=2, batch_size=512),
                                 device="cpu", verbose=False)
    assert outcome.epochs_run == 2

def test_multistep_scheduler_per_epoch_lowers_lr_once_past_milestone():
    feats, labels = _synthetic(seed=11)
    tr_f, tr_y, va_f, va_y = _split(feats, labels)
    names = [f"c{i}" for i in range(labels.shape[1])]
    outcome = train_linear_probe(
        tr_f, tr_y, va_f, va_y, names,
        _budget(epochs=6, scheduler="multistep", milestones=(2,), gamma=0.1,
                lr=1e-2, select="last", schedule_step="epoch"),
        device="cpu", verbose=False,
    )
    learning_rates = [row["lr"] for row in outcome.history]
    assert learning_rates[0] == pytest.approx(1e-2)
    assert learning_rates[2] == pytest.approx(1e-3)
    assert learning_rates[-1] == pytest.approx(1e-3)
