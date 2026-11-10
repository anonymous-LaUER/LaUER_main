"""Evaluate semantic alignment and query-guided concept predictions on downstream tasks."""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import random
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Mapping, Sequence

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Subset

from zeroshot.data.inference_dataset import (
    SignalImageInferenceDataset,
    inference_collate,
)
from zeroshot.data.tasks import TABLE3_TASKS, get_task
from zeroshot.evaluation.metrics import evaluate_multilabel_scores
from zeroshot.models.checkpoint import load_zero_shot_model
from zeroshot.models.reranker import LaUERZeroshotScorer, task_cgl_key
from zeroshot.prompts import DEFAULT_PROMPT_PATH, prompts_for_labels
from zeroshot.prompts.loader import official_prompt_path

from zeroshot.data.scheme2 import load_scheme2, grouped_truth, aggregate_scores

LOGGER = logging.getLogger("zero_shot")
PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SPLIT_ROOT = Path(__file__).resolve().parent / "splits"
DEFAULT_MAPPING_PRESET = "melp-table17"
DEFAULT_MAPPING_PATH = Path(__file__).resolve().parent / "presets" / "melp_table17.json"
MAPPING_PRESETS = {DEFAULT_MAPPING_PRESET: DEFAULT_MAPPING_PATH}
TARGET_QUERY_PROTOCOL = "official_local_annotation_v2"
SOURCE_QUERY_PROTOCOL = "official_source_annotation_v3"
DEFAULT_RESULT_ROOT = Path(__file__).resolve().parent / "results"

TARGET_ROOTS = {
    "ptbxl": Path("datasets/PTB-XL"),
    "cpsc2018": Path("datasets/CPSC2018"),
    "csn": Path("datasets/CSN"),
}

SCORE_ARRAY_KEYS = (
    "ecg_sgl",
    "image_sgl",
    "signal_image_sgl",
    "concept_logit",
)
ALL_SCORE_ARRAY_KEYS = SCORE_ARRAY_KEYS
MATCHING_SCORE_KEYS = (
    "fusion_match_logit",
    "fusion_margin",
    "fusion_probability",
)
PAPER_SCORE_FAMILIES = {
    "ECG-SGL": "ecg_sgl",
    "Image-SGL": "image_sgl",
    "Signal+Image-SGL": "signal_image_sgl",
    "Signal+Image-CGL": "concept_logit",
}
DIAGNOSTIC_SCORE_FAMILIES = {
    "CGL-Fusion-MatchLogit": "fusion_match_logit",
    "CGL-Fusion-Margin": "fusion_margin",
    "CGL-Fusion-Probability": "fusion_probability",
}

@dataclass
class EvaluationJob:
    name: str
    table: str
    target_dataset: str
    source_task: str
    target_task: str
    labels: List[str]
    frame: pd.DataFrame
    truth_columns: List[str]
    manifest_path: Path
    additional_manifest_paths: tuple = ()
    score_groups: dict | None = None
    prompt_dataset: str | None = None

    @property
    def score_labels(self) -> List[str]:
        if self.prompt_dataset is None:
            return self.labels
        return [f"{self.prompt_dataset}:{label}" for label in self.labels]

def _json_dump(payload: Mapping, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    temporary.replace(path)

def _npz_dump(payload: Mapping[str, np.ndarray], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("wb") as stream:
        np.savez_compressed(stream, **payload)
    temporary.replace(path)

def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()

def _sequence_hash(values: Sequence[str]) -> str:
    joined = "\n".join(values).encode("utf-8")
    return hashlib.sha256(joined).hexdigest()

def _setup_logging(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    LOGGER.setLevel(logging.INFO)
    LOGGER.handlers.clear()
    formatter = logging.Formatter(
        "%(asctime)s | %(levelname)s | %(message)s"
    )
    console = logging.StreamHandler()
    console.setFormatter(formatter)
    file_handler = logging.FileHandler(path, encoding="utf-8")
    file_handler.setFormatter(formatter)
    LOGGER.addHandler(console)
    LOGGER.addHandler(file_handler)

def _read_manifest(path: Path) -> pd.DataFrame:
    if not path.is_file():
        raise FileNotFoundError(path)
    frame = pd.read_csv(path, dtype={"sample_id": str})
    if frame["sample_id"].duplicated().any():
        raise ValueError(f"Duplicate sample IDs in {path}")
    return frame

def build_jobs(
    split_root: Path,
    mapping_path: Path,
    evaluation_split: str = "test",
) -> List[EvaluationJob]:
    if evaluation_split == "val+test":
        validation_jobs = build_jobs(split_root, mapping_path, "val")
        test_jobs = build_jobs(split_root, mapping_path, "test")
        for validation, test in zip(validation_jobs, test_jobs):
            if (validation.name, validation.labels, validation.truth_columns) != (
                test.name, test.labels, test.truth_columns
            ):
                raise ValueError("Validation/test task definitions do not match")
            combined = pd.concat([validation.frame, test.frame], ignore_index=True)
            if combined["sample_id"].duplicated().any():
                raise ValueError(f"Duplicate sample IDs across val/test for {test.name}")
            test.frame = combined
            test.additional_manifest_paths = (validation.manifest_path,)
        return test_jobs
    if evaluation_split not in {"val", "test"}:
        raise ValueError(
            f"evaluation_split must be 'val', 'test', or 'val+test', got {evaluation_split!r}"
        )
    jobs: List[EvaluationJob] = []
    for task_name in TABLE3_TASKS:
        spec = get_task(task_name)
        path = (
            split_root
            / task_name
            / f"{evaluation_split}_complete_case.csv"
        )
        frame = _read_manifest(path)
        jobs.append(
            EvaluationJob(
                name=task_name,
                table="table3",
                target_dataset=spec.dataset,
                source_task=task_name,
                target_task=task_name,
                labels=list(spec.labels),
                frame=frame,
                truth_columns=list(spec.labels),
                manifest_path=path,
            )
        )

    mappings = load_scheme2(mapping_path)
    prompt_context = mappings.get("prompt_context", "target")
    if prompt_context not in {"source", "target"}:
        raise ValueError(f"Unsupported mapping prompt_context: {prompt_context!r}")
    for pair_name, pair in mappings["pairs"].items():
        source_task, target_task = pair_name.split("_to_")
        path = split_root / target_task / f"{evaluation_split}_complete_case.csv"
        frame = grouped_truth(_read_manifest(path), pair["target_truth_groups"])
        prompt_dataset = None
        if prompt_context == "source":
            source_spec = get_task(source_task)
            prompt_dataset = source_spec.dataset
            # Reuse a source query when it contributes to multiple evaluation groups.
            labels = list(dict.fromkeys(
                label for members in pair["source_prediction_groups"].values()
                for label in members
            ))
            unknown = set(labels).difference(source_spec.labels)
            if unknown:
                raise ValueError(f"{pair_name}: unknown source labels {sorted(unknown)}")
        else:
            labels = list(pair["query_labels"])
        jobs.append(EvaluationJob(
            name=pair_name, table="table4", target_dataset=get_task(target_task).dataset,
            source_task=source_task, target_task=target_task,
            labels=labels, frame=frame,
            truth_columns=[f"zs_label__{label}" for label in pair["target_truth_groups"]],
            manifest_path=path, score_groups=pair, prompt_dataset=prompt_dataset,
        ))
    return jobs

def _target_frame(jobs: Sequence[EvaluationJob], target: str) -> pd.DataFrame:
    frames = [job.frame for job in jobs if job.target_dataset == target]
    if not frames:
        raise ValueError(f"No jobs target {target}")
    combined = pd.concat(frames, ignore_index=True, sort=False)
    path_column = "wfdb_path" if target == "csn" else "signal_path"
    metadata_columns = ["sample_id", path_column, "image_path"]
    for column in metadata_columns:
        if column not in combined:
            raise ValueError(f"{target}: missing metadata column {column}")
    grouped = combined.groupby("sample_id", sort=False)
    for column in (path_column, "image_path"):
        conflicts = grouped[column].nunique(dropna=False)
        if (conflicts > 1).any():
            bad = conflicts[conflicts > 1].index.tolist()[:10]
            raise ValueError(
                f"{target}: conflicting {column} for samples {bad}"
            )
    return (
        combined[metadata_columns]
        .drop_duplicates("sample_id")
        .sort_values("sample_id")
        .reset_index(drop=True)
    )

def _target_prompt_banks(
    jobs: Sequence[EvaluationJob], target: str
) -> Dict[str, Dict[str, str]]:
    """Group score keys and their native labels by prompt registry."""
    banks: Dict[str, Dict[str, str]] = {}
    for job in jobs:
        if job.target_dataset == target:
            bank = banks.setdefault(job.prompt_dataset or target, {})
            bank.update(zip(job.score_labels, job.labels))
    return banks

def _raw_meta(
    target: str,
    sample_ids: Sequence[str],
    labels: Sequence[str],
    run_manifest: Mapping,
) -> Dict:
    return {
        "schema_version": 1,
        "target_dataset": target,
        "sample_count": len(sample_ids),
        "sample_id_sha256": _sequence_hash(sample_ids),
        "labels": list(labels),
        "label_sha256": _sequence_hash(labels),
        "config_sha256": run_manifest["config_sha256"],
        "checkpoint_sha256": run_manifest["checkpoint_sha256"],
        "prompt_sha256": run_manifest["prompt_sha256"],
        "mapping_sha256": run_manifest["mapping_sha256"],
        "top_k": len(labels),
        "score_schema_version": 3,
        "primary_cgl_score": "concept_logit",
    }

def _validate_raw_meta(expected: Mapping, observed: Mapping) -> None:
    keys = (
        "target_dataset",
        "sample_count",
        "sample_id_sha256",
        "labels",
        "config_sha256",
        "checkpoint_sha256",
        "prompt_sha256",
        "mapping_sha256",
        "top_k",
        "score_schema_version",
        "query_protocol",
        "effective_prompt_sha256",
        "prompts",
        "cgl_prompts",
        "task_labels",
    )
    mismatches = {
        key: {"expected": expected.get(key), "observed": observed.get(key)}
        for key in keys
        if expected.get(key) != observed.get(key)
    }
    if mismatches:
        raise RuntimeError(f"Existing score cache mismatch: {mismatches}")

def _empty_raw(sample_ids: Sequence[str], labels: Sequence[str]) -> Dict:
    rows, columns = len(sample_ids), len(labels)
    payload = {
        "sample_ids": np.asarray(sample_ids),
        "labels": np.asarray(labels),
        "completed": np.asarray(0, dtype=np.int64),
        "candidate_order": np.full(
            (rows, columns), -1, dtype=np.int16
        ),
    }
    for key in ALL_SCORE_ARRAY_KEYS:
        payload[key] = np.full(
            (rows, columns), np.nan, dtype=np.float32
        )
    for key in MATCHING_SCORE_KEYS:
        payload[key] = np.full(rows, np.nan, dtype=np.float32)
    return payload

def _load_npz(path: Path) -> Dict:
    with np.load(path, allow_pickle=False) as archive:
        return {key: archive[key] for key in archive.files}

def score_target(
    target: str,
    jobs: Sequence[EvaluationJob],
    reranker: LaUERZeroshotScorer,
    run_root: Path,
    run_manifest: Mapping,
    batch_size: int,
    pair_chunk_size: int,
    checkpoint_every_batches: int,
    log_every_batches: int,
    max_samples: int = 0,
    prompt_path: str | Path = DEFAULT_PROMPT_PATH,
    cgl_prompt_path: str | Path | None = None,
    sgl_max_length: int = 512,
    cgl_max_length: int = 512,
) -> Path:
    frame = _target_frame(jobs, target)
    if max_samples > 0:
        frame = frame.iloc[:max_samples].reset_index(drop=True)
    sample_ids = frame["sample_id"].astype(str).tolist()
    prompt_banks = _target_prompt_banks(jobs, target)
    labels, prompts, cgl_prompts = [], [], []
    for dataset, bank in prompt_banks.items():
        labels.extend(bank)
        native_labels = list(bank.values())
        prompts.extend(prompts_for_labels(native_labels, prompt_path, task=dataset))
        cgl_prompts.extend(prompts_for_labels(
            native_labels, cgl_prompt_path or prompt_path, task=dataset
        ))
    prompt_features = reranker.encode_prompts(
        labels, prompts, cgl_prompts=cgl_prompts,
        sgl_max_length=sgl_max_length, cgl_max_length=cgl_max_length,
    )
    task_labels = {job.name: job.score_labels for job in jobs if job.target_dataset == target}
    # A shared label column cannot store query-context-dependent CGL scores.
    score_keys = tuple(key for key in ALL_SCORE_ARRAY_KEYS if key != "concept_logit")
    score_keys += tuple(task_cgl_key(name) for name in task_labels)

    raw_root = run_root / "raw_scores"
    final_path = raw_root / f"{target}.task_queries.npz"
    partial_path = raw_root / f"{target}.task_queries.partial.npz"
    meta_path = raw_root / f"{target}.task_queries.meta.json"
    expected_meta = _raw_meta(
        target, sample_ids, labels, run_manifest
    )
    expected_meta["prompts"] = dict(zip(labels, prompts))
    expected_meta["query_protocol"] = run_manifest.get("query_protocol", TARGET_QUERY_PROTOCOL)
    if expected_meta["query_protocol"] == SOURCE_QUERY_PROTOCOL:
        expected_meta["effective_prompt_sha256"] = {
            dataset: _sha256(official_prompt_path(dataset)) for dataset in prompt_banks
        }
    else:
        expected_meta["effective_prompt_sha256"] = _sha256(official_prompt_path(target))
    expected_meta["cgl_prompts"] = dict(zip(labels, cgl_prompts))
    expected_meta["task_labels"] = task_labels
    expected_meta["prompt_tokenization"] = prompt_features.truncation

    if final_path.is_file() and meta_path.is_file():
        observed_meta = json.loads(meta_path.read_text(encoding="utf-8"))
        _validate_raw_meta(expected_meta, observed_meta)
        LOGGER.info("%s: validated completed score cache", target)
        return final_path

    if partial_path.is_file() and meta_path.is_file():
        observed_meta = json.loads(meta_path.read_text(encoding="utf-8"))
        _validate_raw_meta(expected_meta, observed_meta)
        raw = _load_npz(partial_path)
        completed = int(raw["completed"])
        LOGGER.info("%s: resuming at sample %d", target, completed)
    else:
        raw = _empty_raw(sample_ids, labels)
        del raw["concept_logit"]
        for name, job_labels in task_labels.items():
            raw[task_cgl_key(name)] = np.full(
                (len(sample_ids), len(job_labels)), np.nan, dtype=np.float32)
        completed = 0
        _json_dump(expected_meta, meta_path)

    if raw["sample_ids"].astype(str).tolist() != sample_ids:
        raise RuntimeError(f"{target}: partial sample IDs are inconsistent")
    if raw["labels"].astype(str).tolist() != labels:
        raise RuntimeError(f"{target}: partial labels are inconsistent")
    if completed > len(frame):
        raise RuntimeError(f"{target}: invalid completed index {completed}")

    dataset = SignalImageInferenceDataset(
        frame=frame, dataset=target, data_root=TARGET_ROOTS[target]
    )
    starting_completed = completed
    subset = Subset(dataset, range(completed, len(dataset)))
    loader = DataLoader(
        subset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=reranker.device.type == "cuda",
        collate_fn=inference_collate,
    )
    start_time = time.perf_counter()
    nonfinite_replaced = 0
    for batch_index, batch in enumerate(loader, start=1):
        position = completed
        count = len(batch["sample_id"])
        expected_ids = sample_ids[position : position + count]
        if batch["sample_id"] != expected_ids:
            raise RuntimeError(
                f"{target}: loader order mismatch at {position}"
            )
        encoded = reranker.encode_signal_image(batch["ecg_signal"], batch["ecg_image"])
        scores = reranker.score_task_sets(encoded, prompt_features, task_labels)
        row_slice = slice(position, position + count)
        for key in score_keys:
            raw[key][row_slice] = scores[key].numpy().astype(
                np.float32, copy=False
            )
        for key in MATCHING_SCORE_KEYS:
            raw[key][row_slice] = scores[key].numpy().astype(
                np.float32, copy=False
            )
        raw["candidate_order"][row_slice] = scores[
            "candidate_order"
        ].numpy().astype(np.int16, copy=False)
        nonfinite_replaced += sum(
            int(metadata.get("nonfinite_replaced", 0))
            for metadata in batch["signal_metadata"]
        )
        completed += count
        raw["completed"] = np.asarray(completed, dtype=np.int64)

        if (
            batch_index % checkpoint_every_batches == 0
            or completed == len(dataset)
        ):
            _npz_dump(raw, partial_path)
        if (
            batch_index % log_every_batches == 0
            or completed == len(dataset)
        ):
            elapsed = time.perf_counter() - start_time
            processed = completed - starting_completed
            rate = processed / max(elapsed, 1e-9)
            remaining = (len(dataset) - completed) / max(rate, 1e-9)
            gpu_peak = (
                torch.cuda.max_memory_allocated() / 1024**2
                if reranker.device.type == "cuda"
                else 0.0
            )
            LOGGER.info(
                "%s: %d/%d (%.1f%%), %.2f samples/s, ETA %.1f min, "
                "GPU peak %.0f MiB",
                target,
                completed,
                len(dataset),
                100.0 * completed / max(len(dataset), 1),
                rate,
                remaining / 60.0,
                gpu_peak,
            )

    for key in score_keys + MATCHING_SCORE_KEYS:
        if not np.isfinite(raw[key]).all():
            raise RuntimeError(f"{target}: non-finite final scores in {key}")
    if (raw["candidate_order"] < 0).any():
        raise RuntimeError(f"{target}: incomplete candidate ordering")
    raw["completed"] = np.asarray(len(dataset), dtype=np.int64)
    _npz_dump(raw, final_path)
    if partial_path.is_file():
        partial_path.unlink()
    final_meta = dict(expected_meta)
    final_meta.update(
        {
            "completed": len(dataset),
            "signal_nonfinite_values_replaced": nonfinite_replaced,
            "elapsed_seconds": time.perf_counter() - start_time,
            "gpu_peak_memory_mib": (
                torch.cuda.max_memory_allocated() / 1024**2
                if reranker.device.type == "cuda"
                else 0.0
            ),
        }
    )
    _json_dump(final_meta, meta_path)
    LOGGER.info("%s: final scores saved to %s", target, final_path)
    return final_path

def _per_class_rows(family: str, metrics: Mapping) -> List[Dict]:
    rows = []
    for label, values in metrics["per_class"].items():
        confusion = values.get("oracle_confusion") or {}
        rows.append(
            {
                "score_family": family,
                "label": label,
                "positives": values["positives"],
                "negatives": values["negatives"],
                "prevalence": values["prevalence"],
                "valid_for_auc": values["valid_for_auc"],
                "auroc": values["auroc"],
                "average_precision": values["average_precision"],
                "oracle_max_f1": values["oracle_max_f1"],
                "oracle_threshold": values["oracle_threshold"],
                "oracle_sensitivity": confusion.get("sensitivity"),
                "oracle_specificity": confusion.get("specificity"),
                "oracle_precision": confusion.get("precision"),
                "oracle_accuracy": confusion.get("accuracy"),
            }
        )
    return rows

def evaluate_jobs(
    jobs: Sequence[EvaluationJob],
    run_root: Path,
    run_manifest: Mapping,
    max_samples: int = 0,
    prompt_path: str | Path = DEFAULT_PROMPT_PATH,
    cgl_prompt_path: str | Path | None = None,
) -> pd.DataFrame:
    summary_rows = []
    raw_cache = {}
    meta_cache = {}
    for job in jobs:
        if job.target_dataset not in raw_cache:
            raw_path = (
                run_root / "raw_scores" / f"{job.target_dataset}.task_queries.npz"
            )
            raw_cache[job.target_dataset] = _load_npz(raw_path)
            meta_cache[job.target_dataset] = json.loads(
                (
                    run_root
                    / "raw_scores"
                    / f"{job.target_dataset}.task_queries.meta.json"
                ).read_text(encoding="utf-8")
            )
        raw = raw_cache[job.target_dataset]
        base_ids = raw["sample_ids"].astype(str).tolist()
        base_index = {sample_id: index for index, sample_id in enumerate(base_ids)}
        job_frame = job.frame
        if max_samples > 0:
            job_frame = job_frame[
                job_frame["sample_id"].astype(str).isin(base_index)
            ].reset_index(drop=True)
        missing = [
            sample_id
            for sample_id in job_frame["sample_id"].astype(str)
            if sample_id not in base_index
        ]
        if missing:
            raise RuntimeError(
                f"{job.name}: samples missing from raw scores: {missing[:10]}"
            )
        row_indices = np.asarray(
            [
                base_index[sample_id]
                for sample_id in job_frame["sample_id"].astype(str)
            ],
            dtype=np.int64,
        )
        raw_labels = raw["labels"].astype(str).tolist()
        raw_label_index = {
            label: index for index, label in enumerate(raw_labels)
        }
        score_labels = job.score_labels
        missing_labels = [
            label for label in score_labels if label not in raw_label_index
        ]
        if missing_labels:
            raise RuntimeError(
                f"{job.name}: labels missing from raw scores {missing_labels}"
            )
        column_indices = np.asarray(
            [raw_label_index[label] for label in score_labels], dtype=np.int64
        )
        truth = job_frame[job.truth_columns].to_numpy(dtype=np.int8)
        sliced_scores = {
            key: raw[key][np.ix_(row_indices, column_indices)]
            for key in ALL_SCORE_ARRAY_KEYS if key != "concept_logit"
        }
        if meta_cache[job.target_dataset]["task_labels"][job.name] != score_labels:
            raise RuntimeError(f"{job.name}: task query labels are inconsistent")
        sliced_scores["concept_logit"] = raw[task_cgl_key(job.name)][row_indices]
        evaluation_labels = job.labels
        if job.score_groups is not None:
            evaluation_labels = list(job.score_groups["target_truth_groups"])
            sliced_scores = {key: aggregate_scores(values, job.labels, job.score_groups)
                             for key, values in sliced_scores.items()}
        fusion = {
            key: raw[key][row_indices] for key in MATCHING_SCORE_KEYS
        }
        metrics = {}
        class_rows = []
        for family, score_key in PAPER_SCORE_FAMILIES.items():
            family_metrics = evaluate_multilabel_scores(
                truth, sliced_scores[score_key], evaluation_labels
            )
            metrics[family] = family_metrics
            class_rows.extend(_per_class_rows(family, family_metrics))
            summary_rows.append(
                {
                    "table": job.table,
                    "experiment": job.name,
                    "source_task": job.source_task,
                    "target_task": job.target_task,
                    "score_family": family,
                    "sample_count": family_metrics["sample_count"],
                    "class_count": family_metrics["class_count"],
                    "valid_auc_class_count": family_metrics[
                        "valid_auc_class_count"
                    ],
                    "macro_auroc": family_metrics["macro_auroc"],
                    "macro_average_precision": family_metrics[
                        "macro_average_precision"
                    ],
                    "micro_auroc": family_metrics["micro_auroc"],
                    "micro_average_precision": family_metrics[
                        "micro_average_precision"
                    ],
                    "lrap": family_metrics["ranking"][
                        "label_ranking_average_precision"
                    ],
                    "mrr": family_metrics["ranking"][
                        "mean_reciprocal_rank"
                    ],
                }
            )
        diagnostic_metrics = {}
        diagnostic_class_rows = []
        for family, score_key in DIAGNOSTIC_SCORE_FAMILIES.items():
            if score_key in fusion:

                score = np.repeat(
                    fusion[score_key][:, None], len(evaluation_labels), axis=1
                )
            else:
                score = sliced_scores[score_key]
            family_metrics = evaluate_multilabel_scores(
                truth, score, evaluation_labels
            )
            diagnostic_metrics[family] = family_metrics
            diagnostic_class_rows.extend(
                _per_class_rows(family, family_metrics)
            )

        job_root = run_root / job.table / job.name
        score_payload = {
            "sample_ids": np.asarray(
                job_frame["sample_id"].astype(str).tolist(), dtype=str
            ),
            "labels": np.asarray(evaluation_labels),
            "y_true": truth,
            **sliced_scores,
            **fusion,
        }
        _npz_dump(score_payload, job_root / "scores.npz")
        prompt_dataset = job.prompt_dataset or job.target_dataset
        _json_dump(
            {
                "schema_version": 1,
                "experiment": job.name,
                "table": job.table,
                "source_task": job.source_task,
                "target_task": job.target_task,
                "target_dataset": job.target_dataset,
                "manifest_path": str(job.manifest_path.resolve()),
                "manifest_paths": [str(path.resolve()) for path in (
                    *job.additional_manifest_paths, job.manifest_path)],
                "sample_count": len(job_frame),
                "sample_id_sha256": _sequence_hash(
                    job_frame["sample_id"].astype(str).tolist()
                ),
                "labels": evaluation_labels,
                "prompt_dataset": prompt_dataset,
                "prompts": dict(zip(job.labels, prompts_for_labels(
                    job.labels, prompt_path, task=prompt_dataset))),
                "cgl_prompts": dict(zip(job.labels, prompts_for_labels(
                    job.labels, cgl_prompt_path or prompt_path, task=prompt_dataset))),
                "truth_columns": job.truth_columns,
                "query_labels": job.labels,
                "score_groups": job.score_groups,
                "sample_policy": "mapped_positive" if job.table == "table4" else "native",
                "paper_primary_score_family": "Signal+Image-CGL",
                "paper_primary_score_key": "concept_logit",
                "checkpoint_sha256": run_manifest["checkpoint_sha256"],
            },
            job_root / "manifest.json",
        )
        _json_dump(metrics, job_root / "metrics.json")
        _json_dump(
            diagnostic_metrics, job_root / "diagnostic_metrics.json"
        )
        pd.DataFrame(class_rows).to_csv(
            job_root / "per_class_metrics.csv", index=False
        )
        pd.DataFrame(diagnostic_class_rows).to_csv(
            job_root / "diagnostic_per_class_metrics.csv", index=False
        )
        LOGGER.info(
            "%s: Table %s metrics complete (%d samples, %d classes)",
            job.name,
            job.table[-1],
            len(job_frame),
            len(evaluation_labels),
        )
    return pd.DataFrame(summary_rows)

def write_paper_summaries(summary: pd.DataFrame, run_root: Path) -> None:
    summary.to_csv(run_root / "all_metrics_summary.csv", index=False)
    table3 = summary[summary["table"] == "table3"].copy()
    table4 = summary[summary["table"] == "table4"].copy()
    table3.to_csv(run_root / "table3_summary_long.csv", index=False)
    table4.to_csv(run_root / "table4_summary_long.csv", index=False)

    table3_order = {
        "csn": "CSN",
        "ptbxl_rhythm": "PTBXL-Rhythm",
        "ptbxl_form": "PTBXL-Form",
        "ptbxl_sub": "PTBXL-Sub",
        "ptbxl_super": "PTBXL-Super",
        "cpsc2018": "CPSC2018",
    }
    table4_order = {
        "ptbxl_super_to_cpsc2018": "PTBXL-Super→CPSC2018",
        "ptbxl_super_to_csn": "PTBXL-Super→CSN",
        "cpsc2018_to_ptbxl_super": "CPSC2018→PTBXL-Super",
        "cpsc2018_to_csn": "CPSC2018→CSN",
        "csn_to_ptbxl_super": "CSN→PTBXL-Super",
        "csn_to_cpsc2018": "CSN→CPSC2018",
    }

    def _paper_table(frame: pd.DataFrame, order: Mapping[str, str]):
        output = pd.DataFrame(index=PAPER_SCORE_FAMILIES)
        for experiment, column in order.items():
            values = (
                frame[frame["experiment"] == experiment]
                .set_index("score_family")["macro_auroc"]
                .reindex(output.index)
            )
            output[column] = values * 100.0
        output["Average"] = output[list(order.values())].mean(axis=1)
        output.index.name = "Method"
        return output.reset_index()

    paper3 = _paper_table(table3, table3_order)
    paper4 = _paper_table(table4, table4_order)
    paper3.to_csv(
        run_root / "paper_table3_full_precision.csv", index=False
    )
    paper4.to_csv(
        run_root / "paper_table4_full_precision.csv", index=False
    )
    paper3.round(2).to_csv(run_root / "paper_table3.csv", index=False)
    paper4.round(2).to_csv(run_root / "paper_table4.csv", index=False)
    _json_dump(
        paper3[
            paper3["Method"] == "Signal+Image-CGL"
        ].iloc[0].to_dict(),
        run_root / "ours_table3_primary.json",
    )
    _json_dump(
        paper4[
            paper4["Method"] == "Signal+Image-CGL"
        ].iloc[0].to_dict(),
        run_root / "ours_table4_primary.json",
    )

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=Path,
        default=PROJECT_ROOT / "configs" / "pretrain_SGL+CGL+TGL.yaml",
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=(
            PROJECT_ROOT
            / "weights"
            / "lauer_sgl_cgl_tgl.pt"
        ),
    )
    parser.add_argument("--split-root", type=Path, default=DEFAULT_SPLIT_ROOT)
    mapping = parser.add_mutually_exclusive_group()
    mapping.add_argument(
        "--mapping-preset", choices=tuple(MAPPING_PRESETS),
        help="Built-in Table 4 mapping (default: melp-table17)",
    )
    mapping.add_argument("--mapping-path", type=Path, help="Custom Table 4 mapping JSON")
    parser.add_argument("--result-root", type=Path, default=DEFAULT_RESULT_ROOT)
    parser.add_argument(
        "--prompt-path",
        type=Path,
        default=Path(__file__).resolve().parent / "prompts" / "melp_ckepe.json",
        help="Default prompt bank shared by SGL and CGL (MELP/MERL CKEPE)",
    )
    parser.add_argument("--sgl-prompt-path", type=Path, default=None,
                        help="SGL Article prompts; defaults to --prompt-path")
    parser.add_argument("--cgl-prompt-path", type=Path, default=None,
                        help="CGL Query prompts; defaults to --prompt-path")
    parser.add_argument("--sgl-max-length", type=int, default=512)
    parser.add_argument("--cgl-max-length", type=int, default=512)
    parser.add_argument("--run-name", default="lauer")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--batch-size", type=int, default=48)
    parser.add_argument("--pair-chunk-size", type=int, default=32)
    parser.add_argument(
        "--checkpoint-every-batches", type=int, default=100
    )
    parser.add_argument("--log-every-batches", type=int, default=25)
    parser.add_argument(
        "--targets",
        default="ptbxl,cpsc2018,csn",
        help="Comma-separated target datasets",
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=0,
        help="Smoke-test limit per target; 0 runs the full paper protocol.",
    )
    parser.add_argument(
        "--score-only",
        action="store_true",
        help="Create raw target scores but do not calculate job metrics.",
    )
    parser.add_argument(
        "--evaluation-split",
        choices=("val", "test", "val+test"),
        default="test",
        help="Split used for Table 3/4; val+test pools samples before computing metrics.",
    )
    args = parser.parse_args()
    if args.mapping_path is None:
        args.mapping_preset = args.mapping_preset or DEFAULT_MAPPING_PRESET
        args.mapping_path = MAPPING_PRESETS[args.mapping_preset]
    return args

def main() -> None:
    args = parse_args()
    if args.batch_size < 1 or args.pair_chunk_size < 1:
        raise ValueError("Batch and pair chunk sizes must be positive")
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    targets = [value.strip() for value in args.targets.split(",") if value.strip()]
    unknown_targets = set(targets).difference(TARGET_ROOTS)
    if unknown_targets:
        raise ValueError(f"Unknown targets: {sorted(unknown_targets)}")
    if args.max_samples > 0 and args.run_name == "lauer":
        raise ValueError(
            "Smoke runs must use a distinct --run-name to avoid mixing caches"
        )

    run_root = args.result_root.resolve() / args.run_name
    mapping_path = args.mapping_path.resolve()
    mapping_sha256 = _sha256(mapping_path)
    manifest_path = run_root / "run_manifest.json"
    if manifest_path.is_file():
        previous_mapping = json.loads(
            manifest_path.read_text(encoding="utf-8")
        ).get("mapping_sha256")
        if previous_mapping is not None and previous_mapping != mapping_sha256:
            raise ValueError(f"{run_root}: mapping changed; choose a different --run-name")

    jobs = build_jobs(args.split_root.resolve(), mapping_path, args.evaluation_split)
    uses_source_prompts = any(job.prompt_dataset is not None for job in jobs)
    prompt_datasets = sorted({
        job.prompt_dataset or job.target_dataset
        for job in jobs if job.target_dataset in targets
    })

    _setup_logging(run_root / "inference.log")
    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)
    torch.backends.cudnn.benchmark = False

    prompt_path = (args.sgl_prompt_path or args.prompt_path).resolve()
    cgl_prompt_path = (args.cgl_prompt_path or args.prompt_path).resolve()
    LOGGER.info("Hashing experiment inputs")
    run_manifest = {
        "schema_version": 1,
        "query_protocol": SOURCE_QUERY_PROTOCOL if uses_source_prompts else TARGET_QUERY_PROTOCOL,
        "effective_dataset_queries": {
            dataset: {"path": str(official_prompt_path(dataset)),
                      "sha256": _sha256(official_prompt_path(dataset))}
            for dataset in prompt_datasets
        },
        "run_name": args.run_name,
        "config_path": str(args.config.resolve()),
        "config_sha256": _sha256(args.config.resolve()),
        "checkpoint_path": str(args.checkpoint.resolve()),
        "checkpoint_sha256": _sha256(args.checkpoint.resolve()),
        "prompt_path": str(prompt_path),
        "prompt_sha256": _sha256(prompt_path),
        "cgl_prompt_path": str(cgl_prompt_path),
        "cgl_prompt_sha256": _sha256(cgl_prompt_path),
        "sgl_max_length": args.sgl_max_length,
        "cgl_max_length": args.cgl_max_length,
        "mapping_preset": args.mapping_preset or "custom",
        "mapping_path": str(mapping_path),
        "mapping_sha256": mapping_sha256,
        "split_root": str(args.split_root.resolve()),
        "device": args.device,
        "batch_size": args.batch_size,
        "pair_chunk_size": args.pair_chunk_size,
        "max_samples": args.max_samples,
        "evaluation_split": args.evaluation_split,
        "targets": targets,
        "seed": 42,
        "score_families": PAPER_SCORE_FAMILIES,
        "protocols": {
            "table3": "target-native full class set",
            "table4": "source-group scores on mapped-positive target records",
            "sgl_top_k": "K equals candidate class count",
            "cgl": "signal-image fusion + Query-guided tracing head",
            "split": args.evaluation_split,
        },
    }
    _json_dump(run_manifest, manifest_path)

    LOGGER.info("Loading strict zero-shot model")
    model, load_report = load_zero_shot_model(
        args.config, args.checkpoint, args.device
    )
    _json_dump(load_report, run_root / "checkpoint_load_report.json")
    reranker = LaUERZeroshotScorer(model, use_amp=True)

    for target in targets:
        score_target(
            target=target,
            jobs=jobs,
            reranker=reranker,
            run_root=run_root,
            run_manifest=run_manifest,
            batch_size=args.batch_size,
            pair_chunk_size=args.pair_chunk_size,
            checkpoint_every_batches=args.checkpoint_every_batches,
            log_every_batches=args.log_every_batches,
            max_samples=args.max_samples,
            prompt_path=prompt_path,
            cgl_prompt_path=cgl_prompt_path,
            sgl_max_length=args.sgl_max_length,
            cgl_max_length=args.cgl_max_length,
        )
    if args.score_only:
        LOGGER.info("Score-only run complete")
        return
    if set(targets) != set(TARGET_ROOTS):
        raise ValueError(
            "Metric aggregation requires all three targets; rerun with "
            "--targets ptbxl,cpsc2018,csn"
        )
    summary = evaluate_jobs(
        jobs, run_root, run_manifest,
        max_samples=args.max_samples, prompt_path=prompt_path,
        cgl_prompt_path=cgl_prompt_path,
    )
    write_paper_summaries(summary, run_root)
    LOGGER.info("All Table 3 and Table 4 results saved to %s", run_root)

if __name__ == "__main__":
    main()
