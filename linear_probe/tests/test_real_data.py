
import os
from pathlib import Path

import pytest

from linear_probe.config import TASK_ORDER, PathsConfig, PreprocessConfig, get_task
from linear_probe.cohort import coverage_report, resolve_sample_paths
from linear_probe.dataset import read_wfdb_gain
from linear_probe.splits import load_public_splits
from linear_probe.verify import check_transformed_lead_order

PATHS = PathsConfig.from_env()

def _family_available(family: str) -> bool:
    return Path(PATHS.root_for(family)).is_dir()

BASELINE_MIN_POSITIVES = {
    "ptbxl_super_class": (2119, 268, 262),
    "ptbxl_sub_class": (13, 2, 2),
    "ptbxl_form": (8, 1, 1),
    "ptbxl_rhythm": (16, 2, 2),
    "icbeb": (170, 12, 38),
    "chapman": (3, 0, 0),
}

@pytest.mark.parametrize("task_key", TASK_ORDER)
def test_min_positive_per_class_matches_audit(task_key):
    task = get_task(task_key)
    if not _family_available(task.family):
        pytest.skip(f"{task.family} dataset not present")
    cohort = load_public_splits(task)
    for index, split in enumerate(("train", "val", "test")):
        smallest = int(cohort.labels[split].sum(axis=0).min())
        assert smallest >= 0
        if split == "train":
            assert smallest > 0, f"{task_key}/{split} has an unlearnable class"
        assert smallest == BASELINE_MIN_POSITIVES[task_key][index], (
            f"{task_key}/{split}: min positives {smallest} != audit baseline "
            f"{BASELINE_MIN_POSITIVES[task_key][index]}"
        )

@pytest.mark.parametrize("task_key", ["ptbxl_form", "icbeb", "chapman"])
def test_coverage_is_complete(task_key):
    task = get_task(task_key)
    if not _family_available(task.family):
        pytest.skip(f"{task.family} dataset not present")
    cohort = load_public_splits(task)
    report = coverage_report(cohort, PATHS)
    assert report["all"]["missing_signal"] == 0
    assert report["all"]["missing_image"] == 0

@pytest.mark.parametrize("task_key", ["ptbxl_form", "icbeb", "chapman"])
def test_image_size_mode_and_opacity(task_key):
    from linear_probe.cohort import probe_image_sizes

    task = get_task(task_key)
    if not _family_available(task.family):
        pytest.skip(f"{task.family} dataset not present")
    cohort = load_public_splits(task)
    resolved = resolve_sample_paths(cohort, PATHS, check_exists=True)
    sample_ids = list(cohort.ids["test"][:6])
    probe = probe_image_sizes(resolved, sample_ids)
    assert probe["matches_expected"], probe
    assert probe["band_divides_evenly"] and probe["band_ok"]
    assert probe["fully_opaque"], probe
    assert all("A" in mode for mode in probe["modes"])

@pytest.mark.parametrize("task_key", ["ptbxl_form", "icbeb", "chapman"])
def test_post_transform_lead_alignment_and_negative_control(task_key):
    task = get_task(task_key)
    if not _family_available(task.family):
        pytest.skip(f"{task.family} dataset not present")
    cohort = load_public_splits(task)
    resolved = resolve_sample_paths(cohort, PATHS, check_exists=True)
    sample_ids = []
    for split in ("train", "val", "test"):
        sample_ids.extend(cohort.ids[split][:3])

    good = check_transformed_lead_order(cohort, resolved, PreprocessConfig(),
                                        sample_ids)
    assert good["ok"], good
    assert good["band4_is_row4"] and good["band5_is_row5"]

    assert good["band4_vs_row4_avg"] - good["band4_vs_row5_avg"] > 0.2

    image_unswapped = check_transformed_lead_order(
        cohort, resolved, PreprocessConfig(image_swap_bands=False), sample_ids
    )
    assert not image_unswapped["ok"], "one-sided image swap was not detected"

    signal_unswapped = check_transformed_lead_order(
        cohort, resolved, PreprocessConfig(signal_swap_leads=False), sample_ids
    )
    assert not signal_unswapped["ok"], "one-sided signal swap was not detected"

def test_csn_header_gain_is_1000():
    if not _family_available("chapman"):
        pytest.skip("chapman dataset not present")
    import glob

    headers = sorted(glob.glob(str(Path(PATHS.chapman_root) / "WFDBRecords" / "**" / "*.hea"),
                              recursive=True))
    assert headers, "no CSN .hea files found"
    for header in headers[::997]:
        assert read_wfdb_gain(header) == 1000.0

def test_csn_signal_is_millivolts_after_gain():
    if not _family_available("chapman"):
        pytest.skip("chapman dataset not present")
    from linear_probe.dataset import SIGNAL_READERS, _read_chapman_signal

    cohort = load_public_splits(get_task("chapman"))
    resolved = resolve_sample_paths(cohort, PATHS, check_exists=True)
    for sample_id in cohort.ids["test"][:5]:
        path = resolved[sample_id].signal
        raw = _read_chapman_signal(path, apply_gain=False)
        milli = _read_chapman_signal(path, apply_gain=True)
        assert abs(float(milli.std()) * 1000.0 - float(raw.std())) < 1.0

        assert 0.02 < float(milli.std()) < 0.8
