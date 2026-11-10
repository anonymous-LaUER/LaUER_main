"""Prepare MELP-aligned PTB-XL and CPSC2018 manifests.

Raw datasets are never modified. Released MELP CSV files are the authority for
split membership and targets; local indexes only resolve image/signal paths.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Dict, Mapping

import pandas as pd
from PIL import Image

from zeroshot.data.tasks import TASKS, TaskSpec

def sha256_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()

def _atomic_csv(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    frame.to_csv(temporary, index=False)
    temporary.replace(path)

def _atomic_json(payload: Mapping, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    temporary.replace(path)

def _validate_binary_labels(frame: pd.DataFrame, spec: TaskSpec) -> None:
    missing = set(spec.labels).difference(frame.columns)
    if missing:
        raise ValueError(f"{spec.name}: missing labels {sorted(missing)}")
    values = frame[list(spec.labels)].to_numpy()
    if not ((values == 0) | (values == 1)).all():
        raise ValueError(f"{spec.name}: labels must be binary")

def _load_ptb_image_map(root: Path) -> Dict[int, str]:
    frame = pd.read_csv(root / "ptbxl_index_with_images.csv")
    if frame["ecg_id"].duplicated().any():
        raise ValueError("PTB-XL image index has duplicate ecg_id")
    result: Dict[int, str] = {}
    for row in frame.itertuples(index=False):
        if pd.isna(row.image_path):
            continue
        relative = Path(str(row.image_path).replace("\\", "/"))
        path = relative if relative.is_absolute() else root / relative
        if path.is_file():
            result[int(row.ecg_id)] = path.relative_to(root).as_posix()
    return result

def _load_cpsc_image_map(root: Path) -> Dict[str, str]:
    frame = pd.read_csv(
        root / "cpsc2018_index_with_images.csv",
        dtype={"sample_id": str},
    )
    if frame["sample_id"].duplicated().any():
        raise ValueError("CPSC2018 image index has duplicate sample_id")
    result: Dict[str, str] = {}
    for row in frame.itertuples(index=False):
        if pd.isna(row.image_path):
            continue
        relative = Path(str(row.image_path).replace("\\", "/"))
        path = relative if relative.is_absolute() else root / relative
        if path.is_file():
            result[str(row.sample_id)] = path.relative_to(root).as_posix()
    return result

def _find_cpsc_signals(root: Path) -> Dict[str, str]:
    result: Dict[str, str] = {}
    for directory in ("TrainingSet1", "TrainingSet2", "TrainingSet3"):
        for path in (root / directory).glob("A*.mat"):
            if path.stem in result:
                raise ValueError(f"Duplicate CPSC record {path.stem}")
            result[path.stem] = path.relative_to(root).as_posix()
    return result

def _base_row(
    sample_id: str,
    patient_id,
    split: str,
    dataset: str,
    task: str,
    signal_path: str,
    image_path: str,
    root: Path,
) -> Dict:
    signal = root / signal_path if signal_path else None
    if dataset == "ptbxl" and signal is not None:
        has_signal = (
            signal.with_suffix(".hea").is_file()
            and signal.with_suffix(".dat").is_file()
        )
    else:
        has_signal = bool(signal) and signal.is_file()
    has_image = bool(image_path) and (root / image_path).is_file()
    return {
        "sample_id": sample_id,
        "patient_id": patient_id,
        "split": split,
        "dataset": dataset,
        "task": task,
        "signal_path": signal_path,
        "image_path": image_path,
        "has_signal": has_signal,
        "has_image": has_image,
        "include_ecg_sgl": has_signal,
        "include_signal_image": has_signal and has_image,
        "drop_reason": (
            "missing_signal"
            if not has_signal
            else "missing_image"
            if not has_image
            else ""
        ),
    }

def _prepare_ptb(
    spec: TaskSpec,
    split: str,
    source_path: Path,
    root: Path,
    image_map: Mapping[int, str],
) -> pd.DataFrame:
    source = pd.read_csv(source_path)
    _validate_binary_labels(source, spec)
    rows = []
    for record in source.to_dict(orient="records"):
        ecg_id = int(record["ecg_id"])
        signal_path = Path(
            str(record["filename_hr"]).replace("\\", "/")
        ).as_posix()
        row = _base_row(
            f"{ecg_id:05d}",
            record["patient_id"],
            split,
            "ptbxl",
            spec.name,
            signal_path,
            image_map.get(ecg_id, ""),
            root,
        )
        for label in spec.labels:
            row[label] = int(record[label])
        row["label_count"] = sum(row[label] for label in spec.labels)
        rows.append(row)
    return pd.DataFrame(rows)

def _prepare_cpsc(
    spec: TaskSpec,
    split: str,
    source_path: Path,
    root: Path,
    image_map: Mapping[str, str],
    signal_map: Mapping[str, str],
) -> pd.DataFrame:
    source = pd.read_csv(source_path, dtype={"filename": str})
    _validate_binary_labels(source, spec)
    rows = []
    for record in source.to_dict(orient="records"):
        sample_id = str(record["filename"])
        row = _base_row(
            sample_id,
            record["patient_id"],
            split,
            "cpsc2018",
            spec.name,
            signal_map.get(sample_id, ""),
            image_map.get(sample_id, ""),
            root,
        )
        for label in spec.labels:
            row[label] = int(record[label])
        row["label_count"] = sum(row[label] for label in spec.labels)
        rows.append(row)
    return pd.DataFrame(rows)

def _audit_images(root: Path, relative_paths) -> Dict:
    sizes: Dict[str, int] = {}
    modes: Dict[str, int] = {}
    corrupt = []
    unique_paths = sorted(set(path for path in relative_paths if path))
    for relative in unique_paths:
        path = root / relative
        try:
            with Image.open(path) as image:
                key = f"{image.width}x{image.height}"
                sizes[key] = sizes.get(key, 0) + 1
                modes[image.mode] = modes.get(image.mode, 0) + 1
                image.verify()
        except Exception as error:
            corrupt.append(
                {
                    "path": str(path),
                    "error": f"{type(error).__name__}: {error}",
                }
            )
    return {
        "checked": len(unique_paths),
        "sizes": sizes,
        "modes": modes,
        "corrupt_count": len(corrupt),
        "corrupt": corrupt,
    }

def prepare_benchmarks(
    ptbxl_root: Path,
    cpsc_root: Path,
    melp_split_root: Path,
    output_root: Path,
    full_image_audit: bool = False,
) -> Dict:
    ptbxl_root = ptbxl_root.resolve()
    cpsc_root = cpsc_root.resolve()
    melp_split_root = melp_split_root.resolve()
    output_root = output_root.resolve()
    ptb_images = _load_ptb_image_map(ptbxl_root)
    cpsc_images = _load_cpsc_image_map(cpsc_root)
    cpsc_signals = _find_cpsc_signals(cpsc_root)
    audit: Dict = {
        "schema_version": 1,
        "sources": {
            "ptbxl_root": str(ptbxl_root),
            "cpsc2018_root": str(cpsc_root),
            "melp_split_root": str(melp_split_root),
        },
        "tasks": {},
        "dataset_image_audit": {},
    }
    dataset_image_paths = {"ptbxl": set(), "cpsc2018": set()}

    for task_name, spec in TASKS.items():
        if spec.dataset == "csn":
            continue
        root = ptbxl_root if spec.dataset == "ptbxl" else cpsc_root
        task_audit = {"labels": list(spec.labels), "splits": {}}
        split_ids = {}
        for split in ("train", "val", "test"):
            source_path = spec.split_path(melp_split_root, split)
            if spec.dataset == "ptbxl":
                frame = _prepare_ptb(
                    spec, split, source_path, root, ptb_images
                )
            else:
                frame = _prepare_cpsc(
                    spec,
                    split,
                    source_path,
                    root,
                    cpsc_images,
                    cpsc_signals,
                )
            if frame["sample_id"].duplicated().any():
                raise ValueError(f"{task_name}/{split}: duplicate sample_id")
            _atomic_csv(frame, output_root / task_name / f"{split}.csv")
            complete = frame.loc[
                frame["include_signal_image"]
            ].reset_index(drop=True)
            _atomic_csv(
                complete,
                output_root / task_name / f"{split}_complete_case.csv",
            )
            split_ids[split] = set(frame["sample_id"])
            task_audit["splits"][split] = {
                "source_csv": str(source_path),
                "source_sha256": sha256_file(source_path),
                "rows": len(frame),
                "positive_labels": int(
                    frame[list(spec.labels)].to_numpy().sum()
                ),
                "zero_label_rows": int((frame["label_count"] == 0).sum()),
                "missing_signal": int((~frame["has_signal"]).sum()),
                "missing_image": int((~frame["has_image"]).sum()),
                "complete_case": len(complete),
            }
            dataset_image_paths[spec.dataset].update(
                path for path in frame["image_path"].tolist() if path
            )
        overlaps = {
            "train_val": len(split_ids["train"] & split_ids["val"]),
            "train_test": len(split_ids["train"] & split_ids["test"]),
            "val_test": len(split_ids["val"] & split_ids["test"]),
        }
        if any(overlaps.values()):
            raise ValueError(f"{task_name}: split overlap detected")
        task_audit["split_overlap"] = overlaps
        audit["tasks"][task_name] = task_audit

    if full_image_audit:
        for dataset, paths in dataset_image_paths.items():
            root = ptbxl_root if dataset == "ptbxl" else cpsc_root
            image_audit = _audit_images(root, paths)
            if image_audit["corrupt_count"]:
                raise ValueError(f"{dataset}: corrupt images detected")
            audit["dataset_image_audit"][dataset] = image_audit

    _atomic_json(audit, output_root / "benchmark_audit.json")
    return audit

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--ptbxl-root", type=Path, default=Path("datasets/PTB-XL"))
    parser.add_argument(
        "--cpsc-root", type=Path, default=Path("datasets/CPSC2018")
    )
    parser.add_argument(
        "--melp-split-root",
        type=Path,
        default=Path("linear_probe/data_split"),
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path(__file__).resolve().parents[1] / "splits",
    )
    parser.add_argument("--full-image-audit", action="store_true")
    return parser.parse_args()

def main() -> None:
    args = parse_args()
    audit = prepare_benchmarks(
        args.ptbxl_root,
        args.cpsc_root,
        args.melp_split_root,
        args.output_root,
        args.full_image_audit,
    )
    for task, values in audit["tasks"].items():
        counts = {
            split: split_values["rows"]
            for split, split_values in values["splits"].items()
        }
        print(f"{task}: {counts}")

if __name__ == "__main__":
    main()
