import numpy as np

from linear_probe.cohort import (
    coverage_report,
    merge_family_cohorts,
    resolve_sample_paths,
)
from linear_probe.config import TASK_ORDER, PathsConfig, PreprocessConfig, get_task
from linear_probe.splits import load_public_splits
from linear_probe.verify import check_code_invariant, extract_band_traces

def test_lead_order_code_invariant():
    result = check_code_invariant()
    assert result["signal_invariant_ok"] is True
    assert result["image_invariant_ok"] is True
    assert result["ok"] is True

def test_extract_band_traces_reads_synthetic_bands():
    image = np.full((720, 200, 3), 255, dtype=np.uint8)
    for lead in range(12):
        row = lead * 60 + 30
        image[row - 2:row + 3, :, :] = 0
    traces = extract_band_traces(image, x0=0)
    assert traces.shape == (12, 200)
    for lead in range(12):
        assert np.nanmean(traces[lead]) == 30.0

def test_extract_band_traces_rejects_non_divisible_height():
    assert extract_band_traces(np.zeros((700, 100, 3), dtype=np.uint8)) is None

def test_family_union_covers_every_task():
    task_keys = [t for t in TASK_ORDER if get_task(t).family == "ptbxl"]
    cohorts = [load_public_splits(get_task(t)) for t in task_keys]
    union = merge_family_cohorts(cohorts)
    union_ids = set(union.all_ids())
    for cohort in cohorts:
        assert set(cohort.all_ids()) <= union_ids

    super_ids = set(cohorts[0].all_ids())
    form_ids = set(cohorts[2].all_ids())
    assert len(union_ids) > len(super_ids)
    assert form_ids - super_ids

def test_merge_rejects_mixed_families():
    import pytest

    with pytest.raises(ValueError):
        merge_family_cohorts([
            load_public_splits(get_task("icbeb")),
            load_public_splits(get_task("chapman")),
        ])

def test_coverage_report_counts_missing_files(tmp_path):
    task = get_task("icbeb")
    cohort = load_public_splits(task)
    empty_root = tmp_path / "empty"
    empty_root.mkdir()
    paths = PathsConfig(icbeb_root=str(empty_root))
    report = coverage_report(cohort, paths)
    assert report["train"]["n"] == task.expected_counts[0]
    assert report["train"]["missing_signal"] == task.expected_counts[0]
    assert report["train"]["missing_image"] == task.expected_counts[0]

def test_resolve_sample_paths_raises_when_files_absent(tmp_path):
    from linear_probe.cohort import CoverageError

    cohort = load_public_splits(get_task("icbeb"))
    empty_root = tmp_path / "empty"
    empty_root.mkdir()
    paths = PathsConfig(icbeb_root=str(empty_root))
    try:
        resolve_sample_paths(cohort, paths, check_exists=True)
    except CoverageError as error:
        assert "signal file missing" in str(error)
    else:
        raise AssertionError("expected CoverageError")

def test_preprocess_tag_changes_with_settings():
    base = PreprocessConfig()
    assert base.tag() != PreprocessConfig(signal_length=2500).tag()
    assert base.tag() != PreprocessConfig(image_swap_bands=False).tag()
    assert base.tag() != PreprocessConfig(signal_swap_leads=False).tag()
    assert base.tag() != PreprocessConfig(csn_apply_gain=False).tag()
    assert base.tag() == PreprocessConfig().tag()
    assert "meeti_aligned" in base.tag()
