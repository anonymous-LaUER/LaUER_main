
import numpy as np
import pytest

from linear_probe.label_audit import compare, _normalise_subtask, _strip_code

def test_strip_code_removes_both_sides():
    assert _strip_code("ISC_") == "ISC"
    assert _strip_code("_AVB") == "AVB"
    assert _strip_code("STD_") == "STD"
    assert _strip_code("NORM") == "NORM"
    assert _strip_code(" NST_ ") == "NST"

def test_normalise_subtask_accepts_both_forms():
    assert _normalise_subtask("super") == ("super", "ptbxl_super_class")
    assert _normalise_subtask("super_class") == ("super", "ptbxl_super_class")
    assert _normalise_subtask("ptbxl_sub_class") == ("sub", "ptbxl_sub_class")
    assert _normalise_subtask("ptbxl_form") == ("form", "ptbxl_form")
    with pytest.raises(Exception):
        _normalise_subtask("ptbxl_unknown")

def _vector(task_key: str, positives=()) -> np.ndarray:
    from linear_probe.label_sets import expected_label_names

    vector = np.zeros(len(expected_label_names(task_key)), dtype=np.float32)
    for index in positives:
        vector[index] = 1.0
    return vector

def test_compare_reports_perfect_agreement():
    labels = {f"id{i}": _vector("icbeb", (0, 2)) for i in range(10)}
    result = compare("icbeb", labels, dict(labels), "fake")
    assert result.exact_match_rate == 1.0
    assert result.dead_classes_in_raw == []
    assert result.n_common == 10

def test_compare_detects_dead_class_in_raw():
    csv_labels = {f"id{i}": _vector("ptbxl_super_class", (0,)) for i in range(6)}
    raw_labels = {f"id{i}": _vector("ptbxl_super_class") for i in range(6)}
    result = compare("ptbxl_super_class", csv_labels, raw_labels, "fake")
    assert result.exact_match_rate == 0.0
    assert result.dead_classes_in_raw == ["HYP"]
    assert result.per_class["HYP"]["csv_positives"] == 6
    assert result.per_class["HYP"]["raw_positives"] == 0
    assert result.csv_only_examples

def test_compare_reports_raw_only_positives():
    csv_labels = {"a": _vector("ptbxl_super_class")}
    raw_labels = {"a": _vector("ptbxl_super_class", (0,))}
    result = compare("ptbxl_super_class", csv_labels, raw_labels, "fake")
    assert result.per_class["HYP"]["raw_only"] == 1
    assert result.dead_classes_in_raw == []

def test_compare_handles_disjoint_id_sets():
    result = compare("icbeb", {"a": _vector("icbeb")},
                     {"b": _vector("icbeb")}, "fake")
    assert result.n_common == 0
    assert result.exact_match_rate == 0.0