"""Merge task-specific records into dataset-family cohorts for feature extraction."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from .config import PathsConfig
from .splits import SPLITS, SplitCohort

class CoverageError(RuntimeError):
    pass

EXPECTED_IMAGE_SIZE: Tuple[int, int] = (720, 1000)   # (H, W)

@dataclass(frozen=True)
class SamplePaths:
    sample_id: str
    signal: Optional[Path]
    image: Optional[Path]

@dataclass
class FamilyCohort:

    family: str
    label: str
    ids: Dict[str, List[str]]
    signal_rel: Dict[str, Dict[str, str]]
    num_classes: int

    def all_ids(self) -> List[str]:
        seen = set()
        out: List[str] = []
        for split in SPLITS:
            for sample_id in self.ids[split]:
                if sample_id not in seen:
                    seen.add(sample_id)
                    out.append(sample_id)
        return out

def merge_family_cohorts(cohorts: "Sequence[SplitCohort]") -> FamilyCohort:
    if not cohorts:
        raise ValueError("merge_family_cohorts needs at least one cohort")
    family = cohorts[0].family
    for cohort in cohorts:
        if cohort.family != family:
            raise ValueError(
                f"cannot merge cohorts from different families: {family} vs {cohort.family}"
            )
    ids: Dict[str, List[str]] = {split: [] for split in SPLITS}
    seen: Dict[str, set] = {split: set() for split in SPLITS}
    signal_rel: Dict[str, Dict[str, str]] = {split: {} for split in SPLITS}
    for cohort in cohorts:
        for split in SPLITS:
            for sample_id in cohort.ids[split]:
                if sample_id in seen[split]:
                    continue
                seen[split].add(sample_id)
                ids[split].append(sample_id)
                signal_rel[split][sample_id] = cohort.signal_rel[split][sample_id]
    return FamilyCohort(
        family=family,
        label="+".join(cohort.task.key for cohort in cohorts),
        ids=ids,
        signal_rel=signal_rel,
        num_classes=max(cohort.num_classes for cohort in cohorts),
    )

def cohort_label(cohort) -> str:
    task = getattr(cohort, "task", None)
    if task is not None:
        return task.key
    return getattr(cohort, "label", cohort.family)

def _ptbxl_signal_path(root: Path, rel: str) -> Optional[Path]:
    base = root / rel.replace("/", os.sep)
    if Path(str(base) + ".hea").is_file():
        return base
    return None

def _ptbxl_image_path(root: Path, sample_id: str) -> Optional[Path]:
    candidate = root / "images" / f"{int(sample_id):05d}.png"
    return candidate if candidate.is_file() else None

def _icbeb_signal_path(root: Path, sample_id: str) -> Optional[Path]:
    for folder in ("TrainingSet1", "TrainingSet2", "TrainingSet3"):
        candidate = root / folder / f"{sample_id}.mat"
        if candidate.is_file():
            return candidate
    return None

def _chapman_signal_path(root: Path, rel: str) -> Optional[Path]:
    cleaned = rel.replace("/CSN/", "").lstrip("/\\").replace("/", os.sep)
    candidate = root / cleaned
    return candidate if candidate.is_file() else None

def resolve_sample_paths(cohort, paths: PathsConfig,
                         check_exists: bool = True) -> Dict[str, SamplePaths]:
    family = cohort.family
    label = cohort_label(cohort)
    root = Path(paths.root_for(family))
    if not root.is_dir():
        raise CoverageError(f"dataset root not found: {root}")

    resolved: Dict[str, SamplePaths] = {}
    for split in SPLITS:
        for sample_id in cohort.ids[split]:
            if sample_id in resolved:
                continue
            rel = cohort.signal_rel[split][sample_id]
            if family == "ptbxl":
                signal = _ptbxl_signal_path(root, rel)
                image = _ptbxl_image_path(root, sample_id)
            elif family == "icbeb":
                signal = _icbeb_signal_path(root, sample_id)
                image = root / "images" / f"{sample_id}.png"
                image = image if image.is_file() else None
            elif family == "chapman":
                signal = _chapman_signal_path(root, rel)
                image = root / "images" / f"{sample_id}.png"
                image = image if image.is_file() else None
            else:
                raise CoverageError(f"Unknown family {family!r}")
            resolved[sample_id] = SamplePaths(sample_id, signal, image)

    if check_exists:
        for split in SPLITS:
            for sample_id in cohort.ids[split]:
                item = resolved[sample_id]
                if item.signal is None:
                    raise CoverageError(
                        f"{label}/{split}: signal file missing for {sample_id} "
                        f"(rel={cohort.signal_rel[split][sample_id]!r})"
                    )
                if item.image is None:
                    raise CoverageError(
                        f"{label}/{split}: image file missing for {sample_id}"
                    )
    return resolved

def coverage_report(cohort, paths: PathsConfig) -> Dict[str, Dict[str, int]]:
    resolved = resolve_sample_paths(cohort, paths, check_exists=False)
    report: Dict[str, Dict[str, int]] = {}
    for split in SPLITS:
        ids = cohort.ids[split]
        missing_signal = sum(1 for i in ids if resolved[i].signal is None)
        missing_image = sum(1 for i in ids if resolved[i].image is None)
        report[split] = {
            "n": len(ids),
            "missing_signal": missing_signal,
            "missing_image": missing_image,
        }
    report["all"] = {
        "n": len(resolved),
        "missing_signal": sum(1 for v in resolved.values() if v.signal is None),
        "missing_image": sum(1 for v in resolved.values() if v.image is None),
    }
    return report

def probe_image_sizes(resolved: Dict[str, SamplePaths],
                      sample_ids: List[str],
                      expected: Tuple[int, int] = EXPECTED_IMAGE_SIZE) -> Dict[str, object]:
    from PIL import Image

    sizes: Dict[Tuple[int, int], int] = {}
    modes: Dict[str, int] = {}
    alpha_ranges: Dict[str, int] = {}
    band_ok = True
    opaque_ok = True
    for sample_id in sample_ids:
        item = resolved[sample_id]
        if item.image is None:
            continue
        with Image.open(item.image) as handle:
            size = (handle.height, handle.width)
            mode = handle.mode
            alpha_range = None
            if "A" in mode:
                alpha = np.asarray(handle)[:, :, -1]
                alpha_range = f"min{int(alpha.min())}_max{int(alpha.max())}"
        sizes[size] = sizes.get(size, 0) + 1
        modes[mode] = modes.get(mode, 0) + 1
        if alpha_range is not None:
            alpha_ranges[alpha_range] = alpha_ranges.get(alpha_range, 0) + 1
            if alpha_range != "min255_max255":
                opaque_ok = False
        if size != expected:
            band_ok = False
    return {
        "sampled": len(sample_ids),
        "sizes": {f"{h}x{w}": n for (h, w), n in sizes.items()},
        "matches_expected": set(sizes) == {expected},
        "modes": modes,
        "alpha_ranges": alpha_ranges,
        "fully_opaque": opaque_ok,
        "band_height": expected[0] // 12,
        "band_divides_evenly": expected[0] % 12 == 0,
        "band_ok": band_ok,
    }
