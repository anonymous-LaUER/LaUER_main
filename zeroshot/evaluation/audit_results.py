"""Independent integrity audit for a completed Table 3/Table 4 run."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict

import numpy as np
import pandas as pd

from zeroshot.evaluation.metrics import evaluate_multilabel_scores
from zeroshot.run_experiments import (
    PAPER_SCORE_FAMILIES,
    SCORE_ARRAY_KEYS,
    MATCHING_SCORE_KEYS,
)

EXPECTED_JOB_COUNTS = {
    "table3": {
        "csn": (4599, 38),
        "ptbxl_rhythm": (2098, 12),
        "ptbxl_form": (880, 19),
        "ptbxl_sub": (2158, 23),
        "ptbxl_super": (2158, 5),
        "cpsc2018": (1376, 9),
    },
    "table4": {
        "ptbxl_super_to_cpsc2018": (867, 3),
        "ptbxl_super_to_csn": (1532, 5),
        "cpsc2018_to_ptbxl_super": (1425, 6),
        "cpsc2018_to_csn": (1231, 9),
        "csn_to_ptbxl_super": (2158, 17),
        "csn_to_cpsc2018": (1376, 11),
    },
}

def _load_npz(path: Path) -> Dict:
    with np.load(path, allow_pickle=False) as archive:
        return {key: archive[key] for key in archive.files}

def audit_run(run_root: str | Path) -> Dict:
    run_root = Path(run_root).resolve()
    required_root_files = (
        "run_manifest.json",
        "checkpoint_load_report.json",
        "paper_table3.csv",
        "paper_table4.csv",
        "all_metrics_summary.csv",
    )
    missing = [
        name for name in required_root_files if not (run_root / name).is_file()
    ]
    if missing:
        raise FileNotFoundError(f"Missing run artifacts: {missing}")
    partials = list((run_root / "raw_scores").glob("*.partial.npz"))
    if partials:
        raise RuntimeError(f"Unfinished partial caches remain: {partials}")

    report = {
        "schema_version": 1,
        "run_root": str(run_root),
        "raw_targets": {},
        "jobs": {},
        "checks": [],
    }
    for target in ("ptbxl", "cpsc2018", "csn"):
        raw = _load_npz(run_root / "raw_scores" / f"{target}.npz")
        sample_ids = raw["sample_ids"].astype(str)
        labels = raw["labels"].astype(str)
        rows, columns = len(sample_ids), len(labels)
        if int(raw["completed"]) != rows:
            raise RuntimeError(f"{target}: completed != sample count")
        if len(set(sample_ids)) != rows:
            raise RuntimeError(f"{target}: duplicate sample IDs")
        if len(set(labels)) != columns:
            raise RuntimeError(f"{target}: duplicate labels")
        for key in SCORE_ARRAY_KEYS:
            if raw[key].shape != (rows, columns):
                raise RuntimeError(f"{target}/{key}: wrong shape")
            if not np.isfinite(raw[key]).all():
                raise RuntimeError(f"{target}/{key}: NaN/Inf")
        for key in MATCHING_SCORE_KEYS:
            if raw[key].shape != (rows,) or not np.isfinite(raw[key]).all():
                raise RuntimeError(f"{target}/{key}: invalid fusion array")
        expected_order = np.arange(columns, dtype=np.int16)
        if not np.all(
            np.sort(raw["candidate_order"], axis=1) == expected_order
        ):
            raise RuntimeError(
                f"{target}: candidate_order rows are not permutations"
            )
        report["raw_targets"][target] = {
            "sample_count": rows,
            "union_prompt_count": columns,
            "all_scores_finite": True,
            "candidate_order_complete": True,
        }

    summary = pd.read_csv(run_root / "all_metrics_summary.csv")
    for table, expected_jobs in EXPECTED_JOB_COUNTS.items():
        for job, (expected_rows, expected_classes) in expected_jobs.items():
            job_root = run_root / table / job
            scores = _load_npz(job_root / "scores.npz")
            labels = scores["labels"].astype(str).tolist()
            truth = scores["y_true"]
            if truth.shape != (expected_rows, expected_classes):
                raise RuntimeError(
                    f"{table}/{job}: expected truth "
                    f"{(expected_rows, expected_classes)}, got {truth.shape}"
                )
            if len(labels) != expected_classes:
                raise RuntimeError(f"{table}/{job}: label count mismatch")
            stored_metrics = json.loads(
                (job_root / "metrics.json").read_text(encoding="utf-8")
            )
            maximum_delta = 0.0
            for family, score_key in PAPER_SCORE_FAMILIES.items():
                recomputed = evaluate_multilabel_scores(
                    truth, scores[score_key], labels
                )
                stored = stored_metrics[family]
                delta = abs(
                    recomputed["macro_auroc"] - stored["macro_auroc"]
                )
                maximum_delta = max(maximum_delta, delta)
                if delta > 1e-12:
                    raise RuntimeError(
                        f"{table}/{job}/{family}: macro AUROC mismatch"
                    )
                row = summary[
                    (summary["table"] == table)
                    & (summary["experiment"] == job)
                    & (summary["score_family"] == family)
                ]
                if len(row) != 1:
                    raise RuntimeError(
                        f"{table}/{job}/{family}: summary row missing/duplicate"
                    )
                if (
                    abs(
                        float(row.iloc[0]["macro_auroc"])
                        - recomputed["macro_auroc"]
                    )
                    > 1e-12
                ):
                    raise RuntimeError(
                        f"{table}/{job}/{family}: summary value mismatch"
                    )
            report["jobs"][f"{table}/{job}"] = {
                "sample_count": expected_rows,
                "class_count": expected_classes,
                "macro_auroc_max_recompute_delta": maximum_delta,
            }

    expected_summary_rows = 12 * len(PAPER_SCORE_FAMILIES)
    if len(summary) != expected_summary_rows:
        raise RuntimeError(
            f"Expected {expected_summary_rows} summary rows, got {len(summary)}"
        )
    report["checks"] = [
        "no_partial_score_caches",
        "all_raw_scores_finite",
        "candidate_orders_are_complete_permutations",
        "all_12_job_shapes_match_protocol",
        "all_36_macro_aurocs_recompute_exactly",
        "summary_rows_are_unique_and_consistent",
    ]
    return report

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--run-root",
        type=Path,
        default=(
            Path(__file__).resolve().parents[1]
            / "results"
            / "best_model_low"
        ),
    )
    args = parser.parse_args()
    report = audit_run(args.run_root)
    output = args.run_root / "integrity_audit.json"
    temporary = output.with_suffix(".json.tmp")
    temporary.write_text(
        json.dumps(report, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    temporary.replace(output)
    print(
        f"PASS: {len(report['jobs'])} jobs, "
        f"{len(report['checks'])} integrity checks"
    )

if __name__ == "__main__":
    main()
