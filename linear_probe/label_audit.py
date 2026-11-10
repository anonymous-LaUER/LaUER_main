"""Audit downstream label identities against source annotations."""

from __future__ import annotations

import ast
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from .config import PathsConfig, get_task
from .label_sets import expected_label_names
from .splits import SPLITS, load_public_splits

class LabelAuditError(RuntimeError):
    pass

@dataclass
class SourceComparison:

    name: str
    n_csv: int
    n_raw: int
    n_common: int
    exact_match_rate: float
    per_class: Dict[str, Dict[str, int]] = field(default_factory=dict)
    dead_classes_in_raw: List[str] = field(default_factory=list)
    csv_only_examples: List[str] = field(default_factory=list)

    def summary(self) -> str:
        return (f"{self.name}: common={self.n_common} exact={self.exact_match_rate:.4f} "
                f"dead_in_raw={self.dead_classes_in_raw}")

def _strip_code(value) -> str:
    return str(value).strip().strip("_") if value is not None else ""

def _normalise_subtask(subtask: str) -> Tuple[str, str]:
    """'super'/'super_class'/'ptbxl_super_class' -> ('super', 'ptbxl_super_class')。"""
    short = str(subtask).replace("ptbxl_", "")
    aggregation = short.replace("_class", "")
    if aggregation in ("super", "sub"):
        label_key = f"ptbxl_{aggregation}_class"
    elif aggregation in ("form", "rhythm"):
        label_key = f"ptbxl_{aggregation}"
    else:
        raise LabelAuditError(f"unknown PTB-XL subtask {subtask!r}")
    return aggregation, label_key

def rebuild_ptbxl_labels(paths: PathsConfig, subtask: str,
                         strip_underscore: bool = True,
                         index_csv: Optional[str] = None,
                         scp_csv: Optional[str] = None
                         ) -> Dict[str, np.ndarray]:
    index_csv = index_csv or str(Path(paths.ptbxl_root) / "ptbxl_database.csv")
    scp_csv = scp_csv or str(Path(paths.ptbxl_root) / "scp_statements.csv")
    if not Path(index_csv).is_file():
        raise LabelAuditError(f"PTB-XL index CSV not found: {index_csv}")
    if not Path(scp_csv).is_file():
        raise LabelAuditError(f"scp_statements.csv not found: {scp_csv}")

    subtask, label_key = _normalise_subtask(subtask)
    scp = pd.read_csv(scp_csv, index_col=0)
    labels = expected_label_names(label_key)
    label_index = {name: i for i, name in enumerate(labels)}

    normalise = _strip_code if strip_underscore else (lambda v: str(v).strip())

    code_to_class: Dict[str, str] = {}
    code_to_subclass: Dict[str, str] = {}
    form_codes = set()
    rhythm_codes = set()
    for code, row in scp.iterrows():
        key = normalise(code)
        if pd.notna(row.get("diagnostic")):
            if pd.notna(row.get("diagnostic_class")):
                code_to_class[key] = normalise(row["diagnostic_class"])
            if pd.notna(row.get("diagnostic_subclass")):
                code_to_subclass[key] = normalise(row["diagnostic_subclass"])
        if pd.notna(row.get("form")):
            form_codes.add(key)
        if pd.notna(row.get("rhythm")):
            rhythm_codes.add(key)

    frame = pd.read_csv(index_csv)
    out: Dict[str, np.ndarray] = {}
    for ecg_id, raw_codes in zip(frame["ecg_id"].tolist(),
                                 frame["scp_codes"].tolist()):
        try:
            codes = ast.literal_eval(raw_codes) if isinstance(raw_codes, str) else {}
        except (ValueError, SyntaxError):
            codes = {}
        vector = np.zeros(len(labels), dtype=np.float32)
        for raw_code in codes:
            key = normalise(raw_code)
            if subtask == "super":
                name = code_to_class.get(key)
            elif subtask == "sub":
                name = code_to_subclass.get(key)
            elif subtask == "form":
                name = key if key in form_codes else None
            elif subtask == "rhythm":
                name = key if key in rhythm_codes else None
            else:
                raise LabelAuditError(f"unknown PTB-XL subtask {subtask!r}")
            if name is not None and name in label_index:
                vector[label_index[name]] = 1.0
        out[str(int(ecg_id))] = vector
    return out

def rebuild_cpsc_labels(paths: PathsConfig,
                        index_csv: Optional[str] = None) -> Dict[str, np.ndarray]:
    index_csv = index_csv or str(Path(paths.icbeb_root) / "TrainingSet3" / "REFERENCE.csv")
    if not Path(index_csv).is_file():
        raise LabelAuditError(f"CPSC index CSV not found: {index_csv}")
    int_to_label = {
        1: "NORM", 2: "AFIB", 3: "1AVB", 4: "CLBBB", 5: "CRBBB",
        6: "PAC", 7: "VPC", 8: "STD", 9: "STE",
    }
    labels = expected_label_names("icbeb")
    label_index = {name: i for i, name in enumerate(labels)}
    frame = pd.read_csv(index_csv)
    id_col = "sample_id" if "sample_id" in frame.columns else "Recording"
    out: Dict[str, np.ndarray] = {}
    for _, row in frame.iterrows():
        vector = np.zeros(len(labels), dtype=np.float32)
        for column in ("First_label", "Second_label", "Third_label"):
            value = row.get(column)
            if pd.notna(value):
                name = int_to_label.get(int(value))
                if name in label_index:
                    vector[label_index[name]] = 1.0
        out[str(row[id_col])] = vector
    return out

def rebuild_csn_labels(paths: PathsConfig,
                       index_csv: Optional[str] = None,
                       mapping_csv: Optional[str] = None
                       ) -> Dict[str, np.ndarray]:
    root = Path(paths.chapman_root)
    mapping_csv = mapping_csv or str(root / "ConditionNames_SNOMED-CT.csv")
    if not Path(mapping_csv).is_file():
        raise LabelAuditError(f"SNOMED mapping CSV not found: {mapping_csv}")
    # Codes are identities. Text aliases and string substrings must never add diagnoses.
    # index_csv remains an unused compatibility parameter; indexes are not label truth.
    snomed = pd.read_csv(mapping_csv, dtype=str)
    acronym_to_code = dict(zip(snomed["Acronym Name"], snomed["Snomed_CT"]))
    labels = expected_label_names("chapman")
    out: Dict[str, np.ndarray] = {}
    for path in sorted((root / "WFDBRecords").rglob("*.hea")):
        lines = path.read_text(encoding="utf-8").splitlines()
        entries = [line.split(":", 1)[1] for line in lines if line.startswith("#Dx:")]
        if len(entries) != 1:
            raise LabelAuditError(f"Missing or duplicate CSN #Dx: {path}")
        codes = {s.strip() for s in entries[0].split(",")}
        if any(not s.isdigit() for s in codes):
            raise LabelAuditError(f"Invalid CSN diagnosis code: {path}")
        if path.stem in out:
            raise LabelAuditError(f"Duplicate CSN source ID: {path.stem}")
        out[path.stem] = np.array([acronym_to_code[label] in codes for label in labels], dtype=np.float32)
    if not out:
        raise LabelAuditError(f"No CSN source headers: {root}")
    return out

def compare(task_key: str,
            csv_labels: Dict[str, np.ndarray],
            raw_labels: Dict[str, np.ndarray],
            name: str,
            max_examples: int = 5) -> SourceComparison:
    label_names = expected_label_names(task_key)
    common = sorted(set(csv_labels) & set(raw_labels))
    if not common:
        return SourceComparison(name=name, n_csv=len(csv_labels), n_raw=len(raw_labels),
                                n_common=0, exact_match_rate=0.0)
    csv_matrix = np.stack([csv_labels[i] for i in common])
    raw_matrix = np.stack([raw_labels[i] for i in common])
    exact = float((csv_matrix == raw_matrix).all(axis=1).mean())

    per_class: Dict[str, Dict[str, int]] = {}
    dead: List[str] = []
    for index, label in enumerate(label_names):
        csv_pos = int(csv_matrix[:, index].sum())
        raw_pos = int(raw_matrix[:, index].sum())
        both = int(((csv_matrix[:, index] == 1) & (raw_matrix[:, index] == 1)).sum())
        per_class[label] = {
            "csv_positives": csv_pos,
            "raw_positives": raw_pos,
            "both": both,
            "csv_only": csv_pos - both,
            "raw_only": raw_pos - both,
        }
        if raw_pos == 0 < csv_pos:
            dead.append(label)

    examples: List[str] = []
    for row, sample_id in enumerate(common):
        if not np.array_equal(csv_matrix[row], raw_matrix[row]):
            examples.append(sample_id)
            if len(examples) >= max_examples:
                break

    return SourceComparison(
        name=name, n_csv=len(csv_labels), n_raw=len(raw_labels),
        n_common=len(common), exact_match_rate=exact, per_class=per_class,
        dead_classes_in_raw=dead, csv_only_examples=examples,
    )

def audit_task(task_key: str, paths: Optional[PathsConfig] = None,
               splits: Sequence[str] = ("train",)) -> Dict[str, object]:
    paths = paths or PathsConfig.from_env()
    task = get_task(task_key)
    cohort = load_public_splits(task)

    csv_labels: Dict[str, np.ndarray] = {}
    for split in splits:
        for row, sample_id in enumerate(cohort.ids[split]):
            csv_labels[sample_id] = cohort.labels[split][row]

    comparisons: List[SourceComparison] = []
    if task.family == "ptbxl":
        comparisons.append(compare(
            task_key, csv_labels,
            rebuild_ptbxl_labels(paths, task.key, strip_underscore=True),
            "ptbxl_raw_stripped (trimmed aggregation)",
        ))
        comparisons.append(compare(
            task_key, csv_labels,
            rebuild_ptbxl_labels(paths, task.key, strip_underscore=False),
            "ptbxl_raw_unstripped (untrimmed aggregation)",
        ))
    elif task.family == "icbeb":
        comparisons.append(compare(task_key, csv_labels,
                                   rebuild_cpsc_labels(paths), "cpsc_int_labels"))
    elif task.family == "chapman":
        comparisons.append(compare(task_key, csv_labels,
                                   rebuild_csn_labels(paths), "csn_raw_header_SNOMED_exact"))

    return {
        "task": task_key,
        "splits_compared": list(splits),
        "n_csv_labels": len(csv_labels),
        "comparisons": [
            {
                "name": item.name,
                "n_csv": item.n_csv,
                "n_raw": item.n_raw,
                "n_common": item.n_common,
                "exact_match_rate": item.exact_match_rate,
                "dead_classes_in_raw": item.dead_classes_in_raw,
                "csv_only_examples": item.csv_only_examples,
                "per_class": item.per_class,
            }
            for item in comparisons
        ],
    }
