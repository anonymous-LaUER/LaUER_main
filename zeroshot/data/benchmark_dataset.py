"""Runtime loaders for MELP-aligned PTB-XL and CPSC2018 manifests."""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import pandas as pd
import torch
from PIL import Image
from scipy.io import loadmat
from torch.utils.data import Dataset

from data.meeti_dataset import get_image_transform
from zeroshot.data.lead_order import (
    MIMIC_LEAD_ORDER,
    image_to_mimic_order,
    signal_to_mimic_order,
)
from zeroshot.data.tasks import get_task

EXPECTED_LEADS: Tuple[str, ...] = MIMIC_LEAD_ORDER

def _canonical_lead(name: str) -> str:
    upper = str(name).strip().upper()
    if upper in {"AVR", "AVL", "AVF"}:
        return "a" + upper[1:]
    return upper

def _finite_signal(
    signal: np.ndarray, sample_id: str
) -> Tuple[np.ndarray, Dict]:
    signal = np.asarray(signal, dtype=np.float32)
    if signal.shape != (12, 5000):
        raise ValueError(
            f"{sample_id}: expected signal [12,5000], got {signal.shape}"
        )
    nonfinite = ~np.isfinite(signal)
    metadata = {"nonfinite_replaced": int(nonfinite.sum())}
    if nonfinite.any():
        signal = np.nan_to_num(signal, nan=0.0, posinf=0.0, neginf=0.0)
    return np.ascontiguousarray(signal, dtype=np.float32), metadata

def load_ptbxl_signal(wfdb_base: str | Path) -> Tuple[np.ndarray, Dict]:
    """Load PTB-XL high-resolution data in physical mV."""
    import wfdb

    wfdb_base = Path(wfdb_base)
    physical, fields = wfdb.rdsamp(str(wfdb_base))
    if float(fields["fs"]) != 500.0:
        raise ValueError(f"{wfdb_base.name}: expected 500 Hz, got {fields['fs']}")
    names = tuple(_canonical_lead(name) for name in fields["sig_name"])
    if set(names) != set(EXPECTED_LEADS):
        raise ValueError(f"{wfdb_base.name}: unexpected leads {names}")
    order = [names.index(lead) for lead in EXPECTED_LEADS]
    signal = np.asarray(physical[:5000, order].T, dtype=np.float32)
    signal, metadata = _finite_signal(signal, wfdb_base.name)
    metadata.update({"loader": "wfdb", "sampling_rate": 500.0})
    return signal, metadata

def load_cpsc_signal(mat_path: str | Path) -> Tuple[np.ndarray, Dict]:
    """Load the first ten seconds of a CPSC2018 record in physical mV."""
    mat_path = Path(mat_path)
    payload = loadmat(mat_path)
    if "ECG" not in payload:
        raise KeyError(f"{mat_path}: no ECG structure")
    signal = np.asarray(payload["ECG"][0, 0]["data"], dtype=np.float32)
    if signal.ndim != 2 or signal.shape[0] != 12:
        raise ValueError(f"{mat_path.name}: expected [12,N], got {signal.shape}")
    output = np.zeros((12, 5000), dtype=np.float32)
    length = min(signal.shape[1], 5000)
    output[:, :length] = signal[:, :length]

    output = signal_to_mimic_order(output)
    output, metadata = _finite_signal(output, mat_path.stem)
    metadata.update(
        {
            "loader": "scipy.loadmat",
            "sampling_rate": 500.0,
            "source_length": int(signal.shape[1]),
            "used_length": int(length),
        }
    )
    return output, metadata

def load_ecg_image(
    path: str | Path,
    image_transform=None,
) -> torch.Tensor:
    path = Path(path)
    with Image.open(path) as source:
        if source.size != (1000, 720):
            raise ValueError(
                f"{path.name}: expected image size (1000,720), got {source.size}"
            )
        image = source.convert("RGB")

    image = image_to_mimic_order(image)
    transform = image_transform or get_image_transform(
        image_size=1000, encoder_type="vit2d"
    )
    tensor = transform(image)
    if tensor.shape != (3, 720, 1000):
        raise ValueError(f"{path.name}: unexpected tensor shape {tensor.shape}")
    if not torch.isfinite(tensor).all():
        raise ValueError(f"{path.name}: image tensor contains NaN/Inf")
    return tensor

class BenchmarkZeroShotDataset(Dataset):
    """One PTB-XL or CPSC2018 zero-shot task."""

    def __init__(
        self,
        task: str,
        manifest_csv: str | Path,
        data_root: str | Path,
        require_image: bool = True,
        image_transform=None,
    ) -> None:
        self.spec = get_task(task)
        if self.spec.dataset not in {"ptbxl", "cpsc2018"}:
            raise ValueError(
                f"BenchmarkZeroShotDataset does not load {self.spec.dataset}"
            )
        self.manifest_csv = Path(manifest_csv)
        self.data_root = Path(data_root)
        self.require_image = bool(require_image)
        self.image_transform = image_transform or get_image_transform(
            image_size=1000, encoder_type="vit2d"
        )
        self.frame = pd.read_csv(
            self.manifest_csv, dtype={"sample_id": str}
        )
        required = {
            "sample_id", "signal_path", "image_path", "has_signal",
            "has_image", *self.spec.labels,
        }
        missing = required.difference(self.frame.columns)
        if missing:
            raise ValueError(f"Manifest missing columns {sorted(missing)}")
        if not self.frame["has_signal"].astype(bool).all():
            raise ValueError("Manifest contains unavailable signals")
        if self.require_image and not self.frame["has_image"].astype(bool).all():
            raise ValueError("require_image=True but manifest has missing images")
        self.labels = self.frame[list(self.spec.labels)].to_numpy(
            dtype=np.float32
        )
        if not np.isin(self.labels, (0.0, 1.0)).all():
            raise ValueError("Manifest labels must be binary")

    def __len__(self) -> int:
        return len(self.frame)

    def __getitem__(self, index: int) -> Dict:
        row = self.frame.iloc[index]
        signal_path = self.data_root / str(row["signal_path"])
        if self.spec.dataset == "ptbxl":
            signal, metadata = load_ptbxl_signal(signal_path)
        else:
            signal, metadata = load_cpsc_signal(signal_path)
        result: Dict = {
            "sample_id": str(row["sample_id"]),
            "patient_id": str(row.get("patient_id", "")),
            "ecg_signal": torch.from_numpy(signal),
            "labels": torch.from_numpy(self.labels[index]),
            "signal_metadata": metadata,
        }
        if self.require_image:
            result["ecg_image"] = load_ecg_image(
                self.data_root / str(row["image_path"]),
                image_transform=self.image_transform,
            )
        return result
