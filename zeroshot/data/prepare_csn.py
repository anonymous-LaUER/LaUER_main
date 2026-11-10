"""Build audited fixed-cohort CSN manifests without modifying the raw dataset.

The CSN split CSVs retain the original cohort membership but their labels were
rebuilt from exact source-header SNOMED codes. The original MELP label matrices
are not valid reference targets. ``CSN_index_with_images.csv`` resolves image
paths only; its diagnosis-code column must not be used as label truth.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Sequence, Tuple

import pandas as pd
from PIL import Image

CSN_LABELS: Tuple[str, ...] = (
    "AQW",
    "UW",
    "SR",
    "WPW",
    "2AVB",
    "AT",
    "VB",
    "ARS",
    "STTC",
    "SA",
    "STE",
    "VPB",
    "TWO",
    "STTU",
    "ALS",
    "APB",
    "2AVB1",
    "PRIE",
    "CCR",
    "CR",
    "AF",
    "AVB",
    "QTIE",
    "LBBB",
    "VEB",
    "SVT",
    "RBBB",
    "1AVB",
    "STDD",
    "MI",
    "AFIB",
    "TWC",
    "PWC",
    "ERV",
    "RVH",
    "LVH",
    "ST",
    "JEB",
)

SPLIT_FILENAMES: Mapping[str, str] = {
    "train": "CSN_train.csv",
    "val": "CSN_val.csv",
    "test": "CSN_test.csv",
}

def sha256_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    """Return the SHA256 checksum of ``path``."""
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()

def _atomic_write_dataframe(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    frame.to_csv(temporary, index=False)
    temporary.replace(path)

def _atomic_write_json(payload: Mapping, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    temporary.replace(path)

def sample_id_from_melp_path(raw_path: str) -> str:
    """Extract a Chapman/CSN record ID from a MELP ``ecg_path`` value."""
    normalized = str(raw_path).strip().replace("\\", "/")
    if not normalized:
        raise ValueError("Empty ecg_path in MELP split")
    return Path(normalized).stem

def relative_wfdb_base(raw_path: str) -> Path:
    """Convert a MELP path to a CSN-root-relative WFDB record base path."""
    normalized = str(raw_path).strip().replace("\\", "/")
    marker = "/WFDBRecords/"
    if marker not in normalized:
        raise ValueError(f"Unexpected MELP CSN path: {raw_path!r}")
    relative = normalized.split(marker, 1)[1].lstrip("/")
    return Path("WFDBRecords") / Path(relative).with_suffix("")

def load_image_index(csn_root: Path, image_index_csv: Path) -> Dict[str, str]:
    """Load and validate the current CSN image index.

    Returned paths are relative to ``csn_root`` and use forward slashes.
    """
    frame = pd.read_csv(
        image_index_csv,
        dtype={"sample_id": str, "record_id": str},
        skip_blank_lines=True,
    )
    required = {"sample_id", "image_path"}
    missing = required.difference(frame.columns)
    if missing:
        raise ValueError(
            f"Image index is missing required columns: {sorted(missing)}"
        )
    if frame["sample_id"].isna().any():
        raise ValueError("Image index contains empty sample_id values")
    if frame["sample_id"].duplicated().any():
        duplicates = frame.loc[
            frame["sample_id"].duplicated(keep=False), "sample_id"
        ].tolist()
        raise ValueError(f"Duplicate sample IDs in image index: {duplicates[:10]}")

    image_map: Dict[str, str] = {}
    for row in frame.itertuples(index=False):
        sample_id = str(row.sample_id)
        raw_image_path = "" if pd.isna(row.image_path) else str(row.image_path).strip()
        if not raw_image_path:
            continue
        relative = Path(raw_image_path.replace("\\", "/"))
        absolute = relative if relative.is_absolute() else csn_root / relative
        if absolute.is_file():
            image_map[sample_id] = absolute.relative_to(csn_root).as_posix()
    return image_map

def build_split_manifest(
    split: str,
    split_csv: Path,
    csn_root: Path,
    image_map: Mapping[str, str],
) -> pd.DataFrame:
    """Build a manifest from the audited CSN split labels."""
    source = pd.read_csv(split_csv, skip_blank_lines=True)
    required = {"ecg_path", "diagnose", *CSN_LABELS}
    missing = required.difference(source.columns)
    if missing:
        raise ValueError(
            f"{split_csv} is missing MELP columns: {sorted(missing)}"
        )
    observed_labels = tuple(source.columns[3:])
    if observed_labels != CSN_LABELS:
        raise ValueError(
            f"Unexpected CSN label order in {split_csv}.\n"
            f"Expected: {CSN_LABELS}\nObserved: {observed_labels}"
        )

    rows: List[Dict] = []
    for record in source.to_dict(orient="records"):
        sample_id = sample_id_from_melp_path(record["ecg_path"])
        wfdb_base = relative_wfdb_base(record["ecg_path"])
        mat_path = (csn_root / wfdb_base).with_suffix(".mat")
        hea_path = (csn_root / wfdb_base).with_suffix(".hea")
        has_signal = mat_path.is_file() and hea_path.is_file()
        image_path = image_map.get(sample_id, "")
        has_image = bool(image_path) and (csn_root / image_path).is_file()

        if not has_signal:
            drop_reason = "missing_signal"
        elif not has_image:
            drop_reason = "missing_image"
        else:
            drop_reason = ""

        row: Dict = {
            "sample_id": sample_id,
            "split": split,
            "wfdb_path": wfdb_base.as_posix(),
            "image_path": image_path,
            "age": record.get("age", ""),
            "diagnose": record.get("diagnose", ""),
        }
        label_count = 0
        for label in CSN_LABELS:
            value = int(record[label])
            if value not in (0, 1):
                raise ValueError(
                    f"{sample_id}: label {label} must be binary, got {value}"
                )
            row[label] = value
            label_count += value
        row.update(
            {
                "label_count": label_count,
                "has_signal": bool(has_signal),
                "has_image": bool(has_image),
                "include_ecg_sgl": bool(has_signal),
                "include_signal_image": bool(has_signal and has_image),
                "drop_reason": drop_reason,
            }
        )
        rows.append(row)

    manifest = pd.DataFrame(rows)
    if manifest["sample_id"].duplicated().any():
        raise ValueError(f"Duplicate sample IDs in MELP {split} split")
    return manifest

def audit_all_images(csn_root: Path, image_map: Mapping[str, str]) -> Dict:
    """Verify every indexed image and summarize dimensions and modes."""
    sizes: Counter = Counter()
    modes: Counter = Counter()
    corrupt: List[Dict[str, str]] = []
    for sample_id, relative in image_map.items():
        path = csn_root / relative
        try:
            with Image.open(path) as image:
                sizes[f"{image.width}x{image.height}"] += 1
                modes[image.mode] += 1
                image.verify()
        except Exception as error:  # pragma: no cover - integration audit
            corrupt.append(
                {
                    "sample_id": sample_id,
                    "path": str(path),
                    "error": f"{type(error).__name__}: {error}",
                }
            )
    return {
        "checked": len(image_map),
        "sizes": dict(sizes),
        "modes": dict(modes),
        "corrupt_count": len(corrupt),
        "corrupt": corrupt,
    }

def prepare_csn(
    csn_root: Path,
    melp_split_root: Path,
    output_root: Path,
    image_index_csv: Path | None = None,
    full_image_audit: bool = False,
) -> Dict:
    """Prepare all MELP-aligned CSN split manifests and return the audit."""
    csn_root = csn_root.resolve()
    melp_split_root = melp_split_root.resolve()
    output_root = output_root.resolve()
    image_index_csv = (
        image_index_csv.resolve()
        if image_index_csv is not None
        else csn_root / "CSN_index_with_images.csv"
    )

    image_map = load_image_index(csn_root, image_index_csv)
    manifests: Dict[str, pd.DataFrame] = {}
    split_hashes: Dict[str, str] = {}
    for split, filename in SPLIT_FILENAMES.items():
        split_csv = melp_split_root / filename
        if not split_csv.is_file():
            raise FileNotFoundError(f"Missing MELP split: {split_csv}")
        manifests[split] = build_split_manifest(
            split=split,
            split_csv=split_csv,
            csn_root=csn_root,
            image_map=image_map,
        )
        split_hashes[split] = sha256_file(split_csv)

    seen: Dict[str, set] = {
        split: set(frame["sample_id"]) for split, frame in manifests.items()
    }
    overlaps = {
        "train_val": len(seen["train"].intersection(seen["val"])),
        "train_test": len(seen["train"].intersection(seen["test"])),
        "val_test": len(seen["val"].intersection(seen["test"])),
    }
    if any(overlaps.values()):
        raise ValueError(f"MELP CSN splits overlap: {overlaps}")

    output_root.mkdir(parents=True, exist_ok=True)
    for split, frame in manifests.items():
        _atomic_write_dataframe(frame, output_root / f"{split}.csv")

    for split, frame in manifests.items():
        complete_case = frame.loc[
            frame["include_signal_image"]
        ].reset_index(drop=True)
        _atomic_write_dataframe(
            complete_case,
            output_root / f"{split}_complete_case.csv",
        )

    dropped = pd.concat(
        [
            frame.loc[~frame["include_signal_image"]]
            for frame in manifests.values()
        ],
        ignore_index=True,
    )
    _atomic_write_dataframe(dropped, output_root / "dropped_samples.csv")

    audit: Dict = {
        "dataset": "CSN",
        "protocol": "MELP/MERL multi-label split",
        "csn_root": str(csn_root),
        "melp_split_root": str(melp_split_root),
        "image_index_csv": str(image_index_csv),
        "image_index_sha256": sha256_file(image_index_csv),
        "melp_split_sha256": split_hashes,
        "label_count": len(CSN_LABELS),
        "labels": list(CSN_LABELS),
        "split_overlap": overlaps,
        "splits": {},
        "image_index_valid_count": len(image_map),
    }
    for split, frame in manifests.items():
        audit["splits"][split] = {
            "rows": int(len(frame)),
            "unique_ids": int(frame["sample_id"].nunique()),
            "signal_available": int(frame["has_signal"].sum()),
            "image_available": int(frame["has_image"].sum()),
            "complete_case": int(frame["include_signal_image"].sum()),
            "dropped": int((~frame["include_signal_image"]).sum()),
            "all_zero_label_rows": int((frame[list(CSN_LABELS)].sum(axis=1) == 0).sum()),
            "mean_labels_per_sample": float(
                frame[list(CSN_LABELS)].sum(axis=1).mean()
            ),
            "label_support": {
                label: int(frame[label].sum()) for label in CSN_LABELS
            },
            "dropped_sample_ids": frame.loc[
                ~frame["include_signal_image"], "sample_id"
            ].tolist(),
        }
    if full_image_audit:
        audit["full_image_audit"] = audit_all_images(csn_root, image_map)
        if audit["full_image_audit"]["corrupt_count"]:
            raise ValueError(
                f"Corrupt CSN images found: "
                f"{audit['full_image_audit']['corrupt_count']}"
            )

    _atomic_write_json(audit, output_root / "audit.json")
    return audit

def build_parser() -> argparse.ArgumentParser:
    project_root = Path(__file__).resolve().parents[2]
    parser = argparse.ArgumentParser(
        description="Prepare MELP-aligned CSN zero-shot manifests."
    )
    parser.add_argument(
        "--csn-root",
        type=Path,
        default=Path("datasets/CSN"),
    )
    parser.add_argument(
        "--melp-split-root",
        type=Path,
        default=Path("linear_probe/data_split/CSN"),
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=project_root / "splits" / "csn",
    )
    parser.add_argument(
        "--image-index-csv",
        type=Path,
        default=None,
    )
    parser.add_argument(
        "--full-image-audit",
        action="store_true",
        help="Open and verify every indexed PNG (slower, recommended once).",
    )
    return parser

def main() -> None:
    args = build_parser().parse_args()
    audit = prepare_csn(
        csn_root=args.csn_root,
        melp_split_root=args.melp_split_root,
        output_root=args.output_root,
        image_index_csv=args.image_index_csv,
        full_image_audit=args.full_image_audit,
    )
    print(json.dumps(audit, ensure_ascii=False, indent=2))

if __name__ == "__main__":
    main()
