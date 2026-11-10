
from dataclasses import asdict

import pytest

from linear_probe.config import PathsConfig, PreprocessConfig, ProbeConfig, get_task
from linear_probe.run import (
    RunRecord,
    aggregate_serialized,
    main,
    save_records,
)
from linear_probe.splits import load_public_splits

def _record(run_id: str, seed: int, auroc: float, ratio: float = 1.0) -> RunRecord:
    return RunRecord(
        run_id=run_id, task="icbeb", display_name="CPSC2018", family="icbeb",
        mode="signal_only", ratio=ratio, budget="melp", subset_scheme="melp",
        seed=seed, in_dim=192, num_classes=9, n_train=4950, n_val=551,
        n_test=1376,
        val_metrics={"macro_auroc": auroc, "macro_f1": auroc,
                     "subset_accuracy": auroc, "n_classes_scored": 9,
                     "n_classes_total": 9},
        test_metrics={"macro_auroc": auroc, "macro_f1": auroc,
                      "subset_accuracy": auroc, "n_classes_scored": 9,
                      "n_classes_total": 9},
        epochs_run=10, best_epoch=9, stopped_early=True, degraded=False,
    )

def test_preprocessing_does_not_depend_on_mode():
    base = PreprocessConfig()
    assert base.signal_length == 5000
    assert not hasattr(base, "icbeb_first_n")
    assert not hasattr(base, "signal_minmax")
    assert base.csn_apply_gain is True

def test_only_intended_switches_remain():
    from dataclasses import fields

    names = {f.name for f in fields(PreprocessConfig)}
    assert names == {"csn_apply_gain", "signal_length", "n_leads",
                     "signal_swap_leads", "image_swap_bands"}

def test_aggregate_serialized_means_over_seeds():
    runs = [asdict(_record("a", 1, 0.90)), asdict(_record("b", 2, 0.94))]
    agg = aggregate_serialized(runs)
    key = "icbeb|signal_only|1|melp"
    assert agg[key]["n_seeds"] == 2
    assert agg[key]["auroc_mean"] == pytest.approx(92.0, abs=1e-6)
    assert agg[key]["auroc_std"] == pytest.approx(2.8284271, abs=1e-4)
    assert agg[key]["seeds"] == [1, 2]

def test_save_records_merges_instead_of_overwriting(tmp_path):
    paths = PathsConfig(output_dir=str(tmp_path / "runs"))
    probe = ProbeConfig()
    preprocess = PreprocessConfig()
    cohorts = {"icbeb": load_public_splits(get_task("icbeb"))}

    first = save_records([_record("run-a", 1, 0.90)], paths, preprocess, probe,
                         cohorts, "tag", "enc")
    save_records([_record("run-b", 2, 0.93)], paths, preprocess, probe,
                 cohorts, "tag", "enc")

    import json

    payload = json.loads(first.read_text(encoding="utf-8"))
    run_ids = sorted(item["run_id"] for item in payload["runs"])
    assert run_ids == ["run-a", "run-b"]
    assert payload["aggregates"]["icbeb|signal_only|1|melp"]["n_seeds"] == 2
    assert payload["label_source"] == "melp_public_split_csv"
    assert payload["preprocess"]["tag"] == preprocess.tag()

def test_save_records_replaces_same_run_id(tmp_path):
    paths = PathsConfig(output_dir=str(tmp_path / "runs"))
    probe = ProbeConfig()
    preprocess = PreprocessConfig()
    cohorts = {"icbeb": load_public_splits(get_task("icbeb"))}

    save_records([_record("run-a", 1, 0.90)], paths, preprocess, probe,
                 cohorts, "tag", "enc")
    path = save_records([_record("run-a", 1, 0.99)], paths, preprocess, probe,
                        cohorts, "tag", "enc")

    import json

    payload = json.loads(path.read_text(encoding="utf-8"))
    assert len(payload["runs"]) == 1
    assert payload["runs"][0]["test_metrics"]["macro_auroc"] == pytest.approx(0.99)

def test_save_records_refuses_to_merge_different_encoder_tags(tmp_path):
    import json

    paths = PathsConfig(output_dir=str(tmp_path / "runs"))
    probe = ProbeConfig()
    preprocess = PreprocessConfig()
    cohorts = {"icbeb": load_public_splits(get_task("icbeb"))}

    save_records([_record("run-a", 1, 0.90)], paths, preprocess, probe,
                 cohorts, "tag", "enc-A")
    with pytest.raises(RuntimeError, match="different encoders"):
        save_records([_record("run-b", 2, 0.93)], paths, preprocess, probe,
                     cohorts, "tag", "enc-B")

    path = save_records([_record("run-b", 2, 0.93)], paths, preprocess, probe,
                        cohorts, "tag", "enc-A")
    payload = json.loads(path.read_text(encoding="utf-8"))
    assert sorted(item["run_id"] for item in payload["runs"]) == ["run-a", "run-b"]
    assert payload["encoder"]["encoder_tag"] == "enc-A"

def test_save_records_refuses_to_merge_different_preprocess_tags(tmp_path):
    paths = PathsConfig(output_dir=str(tmp_path / "runs"))
    probe = ProbeConfig()
    cohorts = {"icbeb": load_public_splits(get_task("icbeb"))}

    save_records([_record("run-a", 1, 0.90)], paths, PreprocessConfig(), probe,
                 cohorts, "tag", "enc")
    with pytest.raises(RuntimeError, match="different preprocess"):
        save_records([_record("run-b", 2, 0.93)], paths,
                     PreprocessConfig(signal_length=2500), probe,
                     cohorts, "tag", "enc")

def test_dry_run_expands_mode_all():
    assert main(["--task", "icbeb", "--mode", "all", "--dry-run"]) == 0
    assert main(["--matrix", "--tasks", "icbeb", "--modes", "all",
                 "--ratios", "1.0", "--dry-run"]) == 0

def test_unknown_mode_is_rejected():
    with pytest.raises(SystemExit):
        main(["--task", "icbeb", "--mode", "banana", "--dry-run"])