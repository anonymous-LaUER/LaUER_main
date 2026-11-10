"""Load fixed downstream splits and select training-label budgets.

Named label alignment preserves class identity when CSV column orders differ.
Subsampling uses a fixed seed and leaves validation and test partitions intact."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import numpy as np
import pandas as pd

from .config import TaskSpec
from .label_sets import expected_label_names

SPLITS: Tuple[str, ...] = ("train", "val", "test")

class CohortError(RuntimeError):
    pass

@dataclass
class SplitCohort:

    task: TaskSpec
    label_names: List[str]

    ids: Dict[str, List[str]]

    labels: Dict[str, np.ndarray]

    row_index: Dict[str, Dict[str, int]]

    signal_rel: Dict[str, Dict[str, str]]

    csv_sha256: Dict[str, str]

    fingerprint: str

    label_permutation: Dict[str, List[int]]

    label_order: str
    warnings: List[str]

    @property
    def num_classes(self) -> int:
        return len(self.label_names)

    @property
    def family(self) -> str:
        return self.task.family

    def counts(self) -> Dict[str, int]:
        return {s: len(self.ids[s]) for s in SPLITS}

    def all_ids(self) -> List[str]:
        seen = set()
        out: List[str] = []
        for split in SPLITS:
            for sample_id in self.ids[split]:
                if sample_id not in seen:
                    seen.add(sample_id)
                    out.append(sample_id)
        return out

def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()

def _canonical_id(task: TaskSpec, raw_value, csv_path: Path, row_number: int) -> str:
    if task.family == "ptbxl":
        try:
            return str(int(raw_value))
        except (TypeError, ValueError) as error:
            raise CohortError(
                f"{csv_path}: row {row_number} has non-integer ecg_id {raw_value!r}"
            ) from error
    text = str(raw_value).strip()
    if task.family == "icbeb":
        return text

    return Path(text).stem

def load_public_splits(task: TaskSpec,
                       strict_counts: bool = True,
                       strict_labels: bool = True,
                       label_order: str = "name") -> SplitCohort:
    if label_order not in ("name", "positional"):
        raise ValueError(
            f"label_order must be 'name' or 'positional', got {label_order!r}"
        )
    expected_names = expected_label_names(task.key)
    label_permutation: Dict[str, List[int]] = {}
    ids: Dict[str, List[str]] = {}
    labels: Dict[str, np.ndarray] = {}
    row_index: Dict[str, Dict[str, int]] = {}
    signal_rel: Dict[str, Dict[str, str]] = {}
    csv_sha: Dict[str, str] = {}
    warnings: List[str] = []

    for split in SPLITS:
        csv_path = task.csv_path(split)
        if not csv_path.is_file():
            raise CohortError(f"public split CSV missing: {csv_path}")
        csv_sha[split] = _sha256_file(csv_path)
        frame = pd.read_csv(csv_path)

        if task.label_offset >= frame.shape[1]:
            raise CohortError(
                f"{csv_path}: label_offset {task.label_offset} beyond "
                f"{frame.shape[1]} columns"
            )
        label_cols = list(frame.columns[task.label_offset:])
        missing = [c for c in expected_names if c not in label_cols]
        extra = [c for c in label_cols if c not in expected_names]
        if missing or extra:
            raise CohortError(
                f"{csv_path}: label columns differ from expectation.\n"
                f"  expected ({len(expected_names)}): {expected_names}\n"
                f"  found    ({len(label_cols)}): {label_cols}\n"
                f"  missing={missing} extra={extra}"
            )
        if label_cols != expected_names:
            permutation = [label_cols.index(name) for name in expected_names]
            label_permutation[split] = permutation
            if label_order == "positional":
                warnings.append(
                    f"{task.key}/{split}: CSV label column order != canonical "
                    f"(permutation {permutation}); label_order='positional' reproduces "
                    "MELP/MERL positional loading. Per-class "
                    "keys use canonical names; positional loading can associate "
                    "them with different CSV classes. "
                    "Use named alignment to preserve class identity."
                )
            else:
                warnings.append(
                    f"{task.key}/{split}: CSV label column order != canonical "
                    f"(permutation {permutation}); re-aligned by name. "
                    "MELP/MERL read positionally, so their val_auc/test AUROC for "
                    "this task mixes up these columns."
                )
        else:
            permutation = list(range(len(expected_names)))
        label_permutation[split] = permutation

        if label_order == "positional":

            matrix = frame.iloc[:, task.label_offset:].to_numpy(
                dtype=np.float32, copy=True
            )
            ordered_cols = label_cols
        else:
            matrix = frame[expected_names].to_numpy(dtype=np.float32, copy=True)
            ordered_cols = list(expected_names)
        if not np.all((matrix == 0.0) | (matrix == 1.0)):
            bad = int((~((matrix == 0.0) | (matrix == 1.0))).sum())
            raise CohortError(f"{csv_path}: {bad} label entries are not 0/1")

        # --- id ---
        if task.id_column not in frame.columns:
            raise CohortError(
                f"{csv_path}: id column {task.id_column!r} not found; "
                f"columns={list(frame.columns)}"
            )
        split_ids = [
            _canonical_id(task, value, csv_path, row)
            for row, value in enumerate(frame[task.id_column].tolist())
        ]
        duplicates = sorted({i for i in split_ids if split_ids.count(i) > 1})
        if duplicates:
            raise CohortError(f"{csv_path}: duplicated ids {duplicates[:5]}")

        if strict_counts:
            expected = task.expected_counts[SPLITS.index(split)]
            if len(split_ids) != expected:
                raise CohortError(
                    f"{csv_path}: {len(split_ids)} rows, expected {expected}. "
                    "The public split CSV has changed; metrics would not be comparable."
                )

        positive = matrix.sum(axis=0)
        dead = [ordered_cols[i] for i in range(len(ordered_cols)) if positive[i] == 0]
        if dead:
            message = (
                f"{task.key}/{split}: {len(dead)} class(es) have zero positives: {dead}"
            )
            if split == "train":
                raise CohortError(message + " — the linear head cannot learn these classes; check the CSV")
            warnings.append(message)
        zero_label_rows = int((matrix.sum(axis=1) == 0).sum())
        if zero_label_rows:
            warnings.append(
                f"{task.key}/{split}: {zero_label_rows} row(s) with all-zero labels "
                "(retained as negative examples)"
            )

        ids[split] = split_ids
        labels[split] = matrix
        row_index[split] = {i: r for r, i in enumerate(split_ids)}

        if task.family == "ptbxl":
            if "filename_hr" not in frame.columns:
                raise CohortError(f"{csv_path}: filename_hr column missing")
            signal_rel[split] = {
                i: str(v) for i, v in zip(split_ids, frame["filename_hr"].tolist())
            }
        elif task.family == "chapman":
            signal_rel[split] = {
                i: str(v) for i, v in zip(split_ids, frame[task.id_column].tolist())
            }
        else:
            signal_rel[split] = {i: i for i in split_ids}

    for a, b in (("train", "val"), ("train", "test"), ("val", "test")):
        overlap = set(ids[a]) & set(ids[b])
        if overlap:
            raise CohortError(
                f"{task.key}: {len(overlap)} id(s) shared between {a} and {b}: "
                f"{sorted(overlap)[:5]}"
            )

    fingerprint = _cohort_fingerprint(
        task, ids, len(expected_names), labels, label_order
    )
    return SplitCohort(
        task=task,
        label_names=list(expected_names),
        ids=ids,
        labels=labels,
        row_index=row_index,
        signal_rel=signal_rel,
        csv_sha256=csv_sha,
        fingerprint=fingerprint,
        label_permutation=label_permutation,
        label_order=label_order,
        warnings=warnings,
    )

def _cohort_fingerprint(task: TaskSpec, ids: Dict[str, List[str]],
                        num_classes: int,
                        labels: Dict[str, np.ndarray],
                        label_order: str = "name") -> str:
    digest = hashlib.sha256()
    digest.update(f"{task.key}|C={num_classes}|L={label_order}|".encode("utf-8"))
    for split in SPLITS:
        digest.update(f"[{split}]".encode("utf-8"))
        matrix = labels[split]
        for row, sample_id in enumerate(ids[split]):
            bits = "".join("1" if v else "0" for v in matrix[row])
            digest.update(f"{sample_id}:{bits};".encode("utf-8"))
    return digest.hexdigest()

def subsample_train_melp(train_ids: Sequence[str], ratio: float,
                         seed: int = 42) -> List[str]:
    if ratio >= 1.0:
        return list(train_ids)
    sampled = pd.Series(list(train_ids)).sample(frac=ratio, random_state=seed)
    return [str(v) for v in sampled.tolist()]

def subsample_train_merl(train_ids: Sequence[str], ratio: float,
                         seed: int = 42) -> List[str]:
    if ratio >= 1.0:
        return list(train_ids)
    from sklearn.model_selection import train_test_split

    ids = np.asarray(list(train_ids), dtype=object)
    subset, _ = train_test_split(ids, train_size=ratio, random_state=seed)
    return [str(v) for v in subset.tolist()]

def subsample_train(train_ids: Sequence[str], ratio: float,
                    scheme: str = "melp", seed: int = 42) -> List[str]:
    if scheme == "melp":
        return subsample_train_melp(train_ids, ratio, seed)
    if scheme == "merl":
        return subsample_train_merl(train_ids, ratio, seed)
    raise ValueError(f"Unknown subset scheme {scheme!r}; expected melp|merl")

def assert_no_leakage(train_ids: Sequence[str], eval_ids: Sequence[str],
                      label: str = "split") -> None:
    overlap = set(train_ids) & set(eval_ids)
    if overlap:
        raise CohortError(
            f"{label}: {len(overlap)} id(s) appear in both subsets: "
            f"{sorted(overlap)[:5]}"
        )
