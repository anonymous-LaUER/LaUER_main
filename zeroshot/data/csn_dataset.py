"""Runtime CSN loader for MELP-aligned zero-shot evaluation."""

from __future__ import annotations

import logging
import re
from pathlib import Path
from typing import Dict, Sequence, Tuple

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
)

LOGGER = logging.getLogger(__name__)

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

EXPECTED_LEADS: Tuple[str, ...] = MIMIC_LEAD_ORDER

def _fallback_gain_from_header(header_path: Path) -> float:
    """Infer a common ADC gain from a damaged CSN header."""
    text = header_path.read_text(encoding="utf-8", errors="replace")
    gains = [
        float(value)
        for value in re.findall(r"(?<!\d)(\d+(?:\.\d+)?)/mV", text)
    ]
    plausible = [gain for gain in gains if 10.0 <= gain <= 10000.0]
    if not plausible:
        raise ValueError(f"Cannot infer ADC gain from {header_path}")
    return float(np.median(plausible))

def _load_damaged_wfdb_record(wfdb_base: Path) -> np.ndarray:
    """Load a CSN ``mat['val']`` record when its WFDB header is malformed."""
    mat_path = wfdb_base.with_suffix(".mat")
    hea_path = wfdb_base.with_suffix(".hea")
    payload = loadmat(mat_path)
    if "val" not in payload:
        raise KeyError(f"Fallback MAT file has no 'val' array: {mat_path}")
    signal = np.asarray(payload["val"], dtype=np.float32)
    if signal.shape != (12, 5000):
        raise ValueError(
            f"Fallback signal must have shape [12, 5000], got {signal.shape}"
        )
    gain = _fallback_gain_from_header(hea_path)
    return signal / gain

def load_csn_signal(
    wfdb_base: str | Path,
    expected_fs: float = 500.0,
    expected_length: int = 5000,
) -> Tuple[np.ndarray, Dict]:
    """Load one CSN ECG in physical mV as a finite ``[12, 5000]`` array.

    WFDB is the primary reader. A MAT fallback is used only when WFDB parsing
    fails, notably for the known damaged ``JS01052.hea`` record.
    """
    import wfdb

    wfdb_base = Path(wfdb_base)
    metadata: Dict = {
        "sample_id": wfdb_base.name,
        "loader": "wfdb",
        "nonfinite_replaced": 0,
    }
    try:
        physical, fields = wfdb.rdsamp(str(wfdb_base))
        if float(fields["fs"]) != float(expected_fs):
            raise ValueError(
                f"{wfdb_base.name}: expected {expected_fs} Hz, got {fields['fs']}"
            )
        observed_leads = tuple(fields["sig_name"])
        if set(observed_leads) != set(EXPECTED_LEADS):
            raise ValueError(
                f"{wfdb_base.name}: unexpected leads {observed_leads}"
            )
        lead_indices = [observed_leads.index(lead) for lead in EXPECTED_LEADS]
        signal = np.asarray(physical[:, lead_indices].T, dtype=np.float32)
    except Exception as wfdb_error:
        signal = _load_damaged_wfdb_record(wfdb_base)
        metadata.update(
            {
                "loader": "mat_fallback",
                "wfdb_error": f"{type(wfdb_error).__name__}: {wfdb_error}",
            }
        )
        LOGGER.warning(
            "WFDB load failed for %s; used MAT fallback: %s",
            wfdb_base.name,
            wfdb_error,
        )

    expected_shape = (len(EXPECTED_LEADS), int(expected_length))
    if signal.shape != expected_shape:
        raise ValueError(
            f"{wfdb_base.name}: expected signal shape {expected_shape}, "
            f"got {signal.shape}"
        )
    nonfinite = ~np.isfinite(signal)
    metadata["nonfinite_replaced"] = int(nonfinite.sum())
    if metadata["nonfinite_replaced"]:
        signal = np.nan_to_num(
            signal,
            nan=0.0,
            posinf=0.0,
            neginf=0.0,
        )
    return np.ascontiguousarray(signal, dtype=np.float32), metadata

def load_csn_image(
    image_path: str | Path,
    image_transform=None,
) -> torch.Tensor:
    """Load one 1000x720 CSN PNG and apply ViT-2D training preprocessing."""
    image_path = Path(image_path)
    with Image.open(image_path) as source:
        if source.size != (1000, 720):
            raise ValueError(
                f"{image_path.name}: expected image size (1000, 720), "
                f"got {source.size}"
            )
        image = source.convert("RGB")

    image = image_to_mimic_order(image)
    transform = image_transform or get_image_transform(
        image_size=1000,
        encoder_type="vit2d",
    )
    tensor = transform(image)
    if tensor.shape != (3, 720, 1000):
        raise ValueError(
            f"{image_path.name}: expected tensor [3,720,1000], got {tensor.shape}"
        )
    if not torch.isfinite(tensor).all():
        raise ValueError(f"{image_path.name}: image tensor contains NaN/Inf")
    return tensor

class CSNZeroShotDataset(Dataset):
    """CSN dataset backed by a manifest produced by ``prepare_csn.py``."""

    def __init__(
        self,
        manifest_csv: str | Path,
        csn_root: str | Path = "datasets/CSN",
        require_image: bool = True,
        image_transform=None,
    ) -> None:
        self.manifest_csv = Path(manifest_csv)
        self.csn_root = Path(csn_root)
        self.require_image = bool(require_image)
        self.image_transform = image_transform or get_image_transform(
            image_size=1000,
            encoder_type="vit2d",
        )
        self.frame = pd.read_csv(
            self.manifest_csv,
            dtype={"sample_id": str},
            skip_blank_lines=True,
        )
        required = {
            "sample_id",
            "wfdb_path",
            "image_path",
            "has_signal",
            "has_image",
            *CSN_LABELS,
        }
        missing = required.difference(self.frame.columns)
        if missing:
            raise ValueError(
                f"Manifest is missing required columns: {sorted(missing)}"
            )
        if not self.frame["has_signal"].astype(bool).all():
            raise ValueError("Dataset manifest contains unavailable signals")
        if self.require_image and not self.frame["has_image"].astype(bool).all():
            raise ValueError(
                "require_image=True but manifest contains missing images"
            )
        self.labels = self.frame[list(CSN_LABELS)].to_numpy(dtype=np.float32)
        if not np.isin(self.labels, (0.0, 1.0)).all():
            raise ValueError("CSN manifest labels must be binary")

    def __len__(self) -> int:
        return len(self.frame)

    def __getitem__(self, index: int) -> Dict:
        row = self.frame.iloc[index]
        wfdb_base = self.csn_root / str(row["wfdb_path"])
        signal, signal_metadata = load_csn_signal(wfdb_base)
        sample: Dict = {
            "sample_id": str(row["sample_id"]),
            "ecg_signal": torch.from_numpy(signal),
            "labels": torch.from_numpy(self.labels[index]),
            "signal_metadata": signal_metadata,
        }
        if self.require_image:
            sample["ecg_image"] = load_csn_image(
                self.csn_root / str(row["image_path"]),
                image_transform=self.image_transform,
            )
        return sample

