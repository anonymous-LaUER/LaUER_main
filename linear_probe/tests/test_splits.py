from dataclasses import replace

import numpy as np
import pandas as pd
import pytest
from sklearn.model_selection import train_test_split

from linear_probe.config import TASK_ORDER, get_task
from linear_probe.splits import (
    CohortError,
    load_public_splits,
    subsample_train,
    subsample_train_melp,
    subsample_train_merl,
)

@pytest.mark.parametrize("task_key", TASK_ORDER)
def test_public_cohort_counts_match_published(task_key):
    task = get_task(task_key)
    cohort = load_public_splits(task)
    counts = tuple(cohort.counts()[s] for s in ("train", "val", "test"))
    assert counts == task.expected_counts
    assert cohort.num_classes == len(cohort.label_names)

@pytest.mark.parametrize("task_key", TASK_ORDER)
def test_no_split_leakage(task_key):
    cohort = load_public_splits(get_task(task_key))
    assert not (set(cohort.ids["train"]) & set(cohort.ids["val"]))
    assert not (set(cohort.ids["train"]) & set(cohort.ids["test"]))
    assert not (set(cohort.ids["val"]) & set(cohort.ids["test"]))

@pytest.mark.parametrize("task_key", TASK_ORDER)
def test_no_dead_class_in_train(task_key):
    cohort = load_public_splits(get_task(task_key))
    assert (cohort.labels["train"].sum(axis=0) > 0).all()

@pytest.mark.parametrize("task_key", TASK_ORDER)
def test_labels_are_binary(task_key):
    cohort = load_public_splits(get_task(task_key))
    for split in ("train", "val", "test"):
        matrix = cohort.labels[split]
        assert set(np.unique(matrix)).issubset({0.0, 1.0})

def test_counts_guard_raises_on_wrong_expectation():
    broken = replace(get_task("icbeb"), expected_counts=(1, 2, 3))
    with pytest.raises(CohortError, match="expected 1"):
        load_public_splits(broken)

def test_name_order_realigns_ptbxl_sub_class():
    cohort = load_public_splits(get_task("ptbxl_sub_class"))
    permutation = cohort.label_permutation["val"]
    # Corrected native splits share one canonical named order.
    assert permutation == list(range(len(permutation)))
    assert sorted(permutation) == list(range(len(permutation)))
    positional = load_public_splits(get_task("ptbxl_sub_class"),
                                    label_order="positional")
    assert np.array_equal(cohort.labels["val"], positional.labels["val"])

def test_other_tasks_have_canonical_column_order():
    for task_key in TASK_ORDER:
        if task_key == "ptbxl_sub_class":
            continue
        cohort = load_public_splits(get_task(task_key))
        for split in ("train", "val", "test"):
            permutation = cohort.label_permutation[split]
            assert permutation == list(range(len(permutation)))

def test_fingerprint_is_deterministic_and_content_based():
    first = load_public_splits(get_task("icbeb"))
    second = load_public_splits(get_task("icbeb"))
    assert first.fingerprint == second.fingerprint
    assert first.fingerprint != load_public_splits(get_task("chapman")).fingerprint

def test_subsample_full_ratio_keeps_csv_order():
    ids = [str(i) for i in range(100)]
    for scheme in ("melp", "merl"):
        assert subsample_train(ids, 1.0, scheme=scheme) == ids

def test_subsample_sizes_and_uniqueness():
    ids = [str(i) for i in range(1000)]
    for scheme in ("melp", "merl"):
        assert len(subsample_train(ids, 0.01, scheme=scheme)) == 10
        assert len(subsample_train(ids, 0.1, scheme=scheme)) == 100
        assert len(set(subsample_train(ids, 0.1, scheme=scheme))) == 100

def test_subsample_is_deterministic():
    ids = [f"id{i}" for i in range(500)]
    assert subsample_train_melp(ids, 0.1) == subsample_train_melp(ids, 0.1)
    assert subsample_train_merl(ids, 0.1) == subsample_train_merl(ids, 0.1)

def test_subsample_is_a_subset_of_train():
    ids = [f"id{i}" for i in range(500)]
    for scheme in ("melp", "merl"):
        assert set(subsample_train(ids, 0.1, scheme=scheme)) <= set(ids)

def test_melp_scheme_matches_pandas_sample():
    frame = pd.DataFrame({"ecg_id": [f"id{i}" for i in range(400)]})
    expected = set(frame.sample(frac=0.1, random_state=42)["ecg_id"].tolist())
    got = set(subsample_train_melp(frame["ecg_id"].tolist(), 0.1, seed=42))
    assert got == expected

def test_merl_scheme_matches_train_test_split_of_dataframe():
    frame = pd.DataFrame({"ecg_id": [f"id{i}" for i in range(400)]})
    expected, _ = train_test_split(frame, train_size=0.1, random_state=42)
    got = set(subsample_train_merl(frame["ecg_id"].tolist(), 0.1, seed=42))
    assert got == set(expected["ecg_id"].tolist())

def test_two_schemes_pick_different_rows():
    ids = [f"id{i}" for i in range(1000)]
    assert set(subsample_train_melp(ids, 0.1)) != set(subsample_train_merl(ids, 0.1))
