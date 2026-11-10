"""Canonical MELP downstream task definitions.

Label order is explicit instead of inferred from CSV column positions. This is
important for PTB-XL subclass, whose released files do not all present columns
in the same order.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Tuple

@dataclass(frozen=True)
class TaskSpec:
    name: str
    dataset: str
    labels: Tuple[str, ...]
    melp_relative_dir: str
    melp_file_prefix: str

    def split_path(self, melp_root: str | Path, split: str) -> Path:
        return (
            Path(melp_root)
            / self.melp_relative_dir
            / f"{self.melp_file_prefix}_{split}.csv"
        )

TASKS: Dict[str, TaskSpec] = {
    "ptbxl_super": TaskSpec(
        "ptbxl_super",
        "ptbxl",
        ("HYP", "NORM", "MI", "CD", "STTC"),
        "ptbxl/super_class",
        "ptbxl_super_class",
    ),
    "ptbxl_sub": TaskSpec(
        "ptbxl_sub",
        "ptbxl",
        (
            "AMI", "LAFB/LPFB", "LVH", "STTC", "IMI", "SEHYP", "CRBBB",
            "WPW", "LAO/LAE", "NORM", "ISC", "AVB", "RAO/RAE", "LMI",
            "ISCI", "ISCA", "NST", "CLBBB", "ILBBB", "IRBBB", "PMI",
            "RVH", "IVCD",
        ),
        "ptbxl/sub_class",
        "ptbxl_sub_class",
    ),
    "ptbxl_form": TaskSpec(
        "ptbxl_form",
        "ptbxl",
        (
            "PAC", "DIG", "HVOLT", "STD", "LPR", "QWAVE", "VCLVH", "NDT",
            "TAB", "LOWT", "ABQRS", "LNGQT", "PRC(S)", "NT", "STE", "NST",
            "PVC", "INVT", "LVOLT",
        ),
        "ptbxl/form",
        "ptbxl_form",
    ),
    "ptbxl_rhythm": TaskSpec(
        "ptbxl_rhythm",
        "ptbxl",
        (
            "SVARR", "BIGU", "STACH", "SARRH", "SBRAD", "TRIGU", "AFIB",
            "SR", "SVTAC", "PSVT", "AFLT", "PACE",
        ),
        "ptbxl/rhythm",
        "ptbxl_rhythm",
    ),
    "cpsc2018": TaskSpec(
        "cpsc2018",
        "cpsc2018",
        ("AFIB", "VPC", "NORM", "1AVB", "CRBBB", "STE", "PAC", "CLBBB", "STD"),
        "CPSC",
        "CPSC",
    ),
    "csn": TaskSpec(
        "csn",
        "csn",
        (
            "AQW", "UW", "SR", "WPW", "2AVB", "AT", "VB", "ARS", "STTC",
            "SA", "STE", "VPB", "TWO", "STTU", "ALS", "APB", "2AVB1",
            "PRIE", "CCR", "CR", "AF", "AVB", "QTIE", "LBBB", "VEB",
            "SVT", "RBBB", "1AVB", "STDD", "MI", "AFIB", "TWC", "PWC",
            "ERV", "RVH", "LVH", "ST", "JEB",
        ),
        "CSN",
        "CSN",
    ),
}

TABLE3_TASKS: Tuple[str, ...] = (
    "csn",
    "ptbxl_rhythm",
    "ptbxl_form",
    "ptbxl_sub",
    "ptbxl_super",
    "cpsc2018",
)

def get_task(name: str) -> TaskSpec:
    try:
        return TASKS[name]
    except KeyError as error:
        raise KeyError(
            f"Unknown task {name!r}; expected one of {sorted(TASKS)}"
        ) from error
