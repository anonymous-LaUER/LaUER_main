"""Task definitions, preprocessing contracts, and linear-probe training budgets."""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Optional, Tuple

PREPROCESS_VERSION = "prep_v2_meeti_aligned_raw_mv_L5000_swap45"

PACKAGE_ROOT = Path(__file__).resolve().parent
PROJECT_ROOT = PACKAGE_ROOT.parent

@dataclass(frozen=True)
class TaskSpec:

    key: str
    family: str              # ptbxl | icbeb | chapman
    split_subdir: str
    file_stem: str
    id_column: str
    label_offset: int
    expected_counts: Tuple[int, int, int]
    display_name: str

    @property
    def split_dir(self) -> Path:
        return PACKAGE_ROOT / "data_split" / self.split_subdir

    def csv_path(self, split: str) -> Path:
        return self.split_dir / f"{self.file_stem}_{split}.csv"

TASKS: Dict[str, TaskSpec] = {
    t.key: t
    for t in (
        TaskSpec(
            key="ptbxl_super_class", family="ptbxl",
            split_subdir="ptbxl/super_class", file_stem="ptbxl_super_class",
            id_column="ecg_id", label_offset=6,
            expected_counts=(17084, 2146, 2158), display_name="PTB-XL (super)",
        ),
        TaskSpec(
            key="ptbxl_sub_class", family="ptbxl",
            split_subdir="ptbxl/sub_class", file_stem="ptbxl_sub_class",
            id_column="ecg_id", label_offset=6,
            expected_counts=(17084, 2146, 2158), display_name="PTB-XL (sub)",
        ),
        TaskSpec(
            key="ptbxl_form", family="ptbxl",
            split_subdir="ptbxl/form", file_stem="ptbxl_form",
            id_column="ecg_id", label_offset=6,
            expected_counts=(7197, 901, 880), display_name="PTB-XL (form)",
        ),
        TaskSpec(
            key="ptbxl_rhythm", family="ptbxl",
            split_subdir="ptbxl/rhythm", file_stem="ptbxl_rhythm",
            id_column="ecg_id", label_offset=6,
            expected_counts=(16832, 2100, 2098), display_name="PTB-XL (rhythm)",
        ),
        TaskSpec(
            key="icbeb", family="icbeb",
            split_subdir="CPSC", file_stem="CPSC",
            id_column="filename", label_offset=7,
            expected_counts=(4950, 551, 1376), display_name="CPSC2018",
        ),
        TaskSpec(
            key="chapman", family="chapman",
            split_subdir="CSN", file_stem="CSN",
            id_column="ecg_path", label_offset=3,
            expected_counts=(16546, 1860, 4620), display_name="CSN",
        ),
    )
}

TASK_ORDER: Tuple[str, ...] = (
    "ptbxl_super_class", "ptbxl_sub_class", "ptbxl_form", "ptbxl_rhythm",
    "icbeb", "chapman",
)

def get_task(key: str) -> TaskSpec:
    try:
        return TASKS[key]
    except KeyError as error:
        raise KeyError(
            f"Unknown task {key!r}; expected one of {list(TASK_ORDER)}"
        ) from error

@dataclass
class PathsConfig:

    ptbxl_root: str = "datasets/PTB-XL"
    icbeb_root: str = "datasets/CPSC2018"
    chapman_root: str = "datasets/CSN"

    cache_dir: str = str(PROJECT_ROOT / "output" / "linear_probe" / "cache")
    output_dir: str = str(PROJECT_ROOT / "output" / "linear_probe" / "runs")

    def root_for(self, family: str) -> str:
        table = {
            "ptbxl": self.ptbxl_root,
            "icbeb": self.icbeb_root,
            "chapman": self.chapman_root,
        }
        try:
            return table[family]
        except KeyError as error:
            raise KeyError(f"Unknown dataset family {family!r}") from error

    @classmethod
    def from_env(cls, **overrides) -> "PathsConfig":
        cfg = cls(
            ptbxl_root=os.environ.get("LAUER_PTBXL_ROOT", cls.ptbxl_root),
            icbeb_root=os.environ.get("LAUER_ICBEB_ROOT", cls.icbeb_root),
            chapman_root=os.environ.get("LAUER_CHAPMAN_ROOT", cls.chapman_root),
            cache_dir=os.environ.get("LAUER_LP_CACHE", cls.cache_dir),
            output_dir=os.environ.get("LAUER_LP_OUTPUT", cls.output_dir),
        )
        for key, value in overrides.items():
            if value:
                setattr(cfg, key, value)
        return cfg

@dataclass
class PreprocessConfig:

    csn_apply_gain: bool = True

    signal_length: int = 5000
    n_leads: int = 12

    signal_swap_leads: bool = True

    image_swap_bands: bool = True

    def tag(self) -> str:
        return (
            f"{PREPROCESS_VERSION}"
            f"_L{self.signal_length}"
            f"_g{int(self.csn_apply_gain)}"
            f"_ls{int(self.signal_swap_leads)}"
            f"_is{int(self.image_swap_bands)}"
        )

@dataclass(frozen=True)
class BudgetSpec:

    name: str
    optimizer: str                 # "adamw" | "adam"
    lr: float
    weight_decay: float
    batch_size: int
    epochs: int
    scheduler: str                 # "cosine" | "multistep"
    eta_min: float = 0.0
    milestones: Tuple[int, ...] = ()
    gamma: float = 0.1

    early_stop_patience: Optional[int] = None

    select: str = "best_val"

    schedule_step: str = "epoch"

    #: merl = sklearn train_test_split(train_size=ratio, rs=42)
    subset_scheme: str = "melp"

#: MELP: AdamW(1e-3, 1e-4) + CosineAnnealingLR(T_max=epochs, eta_min=1e-5)
#:       batch 128, bf16-mixed, EarlyStopping(val_auc, patience=5, min_delta=0),

MELP_BUDGET = BudgetSpec(
    name="melp",
    optimizer="adamw",
    lr=1e-3,
    weight_decay=1e-4,
    batch_size=128,
    epochs=100,
    scheduler="cosine",
    eta_min=1e-5,
    early_stop_patience=5,
    select="best_val",
    subset_scheme="melp",
)

#:   Adam(lr=1e-3, wd=1e-4), MultiStepLR(milestones=[40], gamma=0.1), batch 16,

MERL_BUDGET = BudgetSpec(
    name="merl",
    optimizer="adam",
    lr=1e-3,
    weight_decay=1e-4,
    batch_size=16,
    epochs=100,
    scheduler="multistep",
    milestones=(40,),
    gamma=0.1,
    early_stop_patience=None,
    select="last",
    subset_scheme="merl",
)

BUDGETS: Dict[str, BudgetSpec] = {b.name: b for b in (MELP_BUDGET, MERL_BUDGET)}

def get_budget(name: str) -> BudgetSpec:
    try:
        return BUDGETS[name]
    except KeyError as error:
        raise KeyError(f"Unknown budget {name!r}; expected {sorted(BUDGETS)}") from error

MODES: Tuple[str, ...] = ("signal_only", "image_only", "fusion")
RATIOS: Tuple[float, ...] = (0.01, 0.1, 1.0)

@dataclass
class ProbeConfig:

    budget: BudgetSpec = MELP_BUDGET
    seeds: Tuple[int, ...] = (42,)
    device: str = "cuda"

    amp: str = "bf16"

    num_workers: int = 4
    prefetch_factor: int = 4

    encoder_checkpoint: str = "weights/lauer_sgl_cgl_tgl.pt"
    config_yaml: str = "configs/pretrain_SGL+CGL+TGL.yaml"

    single_modal_feature: str = "cls"

    extract_batch_size: int = 0

    oom_retries: int = 3

@dataclass
class RunConfig:

    task: str = "ptbxl_super_class"
    mode: str = "signal_only"
    ratio: float = 1.0
    paths: PathsConfig = field(default_factory=PathsConfig)
    preprocess: PreprocessConfig = field(default_factory=PreprocessConfig)
    probe: ProbeConfig = field(default_factory=ProbeConfig)

    use_cache: bool = True

    verify_first: bool = True

    note: str = ""

    def __post_init__(self):
        get_task(self.task)
        if self.probe.single_modal_feature not in ("sgl", "cls"):
            raise ValueError(
                "single_modal_feature must be 'sgl' or 'cls', got "
                f"{self.probe.single_modal_feature!r}"
            )
        if self.mode not in MODES:
            raise ValueError(f"mode must be one of {MODES}, got {self.mode!r}")
        if self.ratio not in RATIOS:
            raise ValueError(f"ratio must be one of {RATIOS}, got {self.ratio!r}")

    def run_id(self) -> str:
        seed_part = "-".join(str(s) for s in self.probe.seeds)
        return (
            f"{self.task}__{self.mode}__r{self.ratio:g}"
            f"__{self.probe.budget.name}__seed{seed_part}"
        )
