"""Label-agnostic signal/image loader used by the target-centric runner."""

from __future__ import annotations

from pathlib import Path
from typing import Dict

import pandas as pd
import torch
from torch.utils.data import Dataset

from data.meeti_dataset import get_image_transform
from zeroshot.data.benchmark_dataset import (
    load_cpsc_signal,
    load_ecg_image,
    load_ptbxl_signal,
)
from zeroshot.data.csn_dataset import load_csn_image, load_csn_signal

class SignalImageInferenceDataset(Dataset):
    """Load unique target-domain records without binding a label task."""

    def __init__(
        self,
        frame: pd.DataFrame,
        dataset: str,
        data_root: str | Path,
    ) -> None:
        if dataset not in {"ptbxl", "cpsc2018", "csn"}:
            raise ValueError(f"Unsupported target dataset: {dataset}")
        self.dataset = dataset
        self.data_root = Path(data_root)
        self.frame = frame.reset_index(drop=True).copy()
        path_column = "wfdb_path" if dataset == "csn" else "signal_path"
        required = {"sample_id", path_column, "image_path"}
        missing = required.difference(self.frame.columns)
        if missing:
            raise ValueError(f"Inference manifest missing {sorted(missing)}")
        if self.frame["sample_id"].astype(str).duplicated().any():
            raise ValueError("Inference manifest contains duplicate sample IDs")
        self.path_column = path_column
        self.image_transform = get_image_transform(
            image_size=1000, encoder_type="vit2d"
        )

    def __len__(self) -> int:
        return len(self.frame)

    def __getitem__(self, index: int) -> Dict:
        row = self.frame.iloc[index]
        signal_path = self.data_root / str(row[self.path_column])
        if self.dataset == "ptbxl":
            signal, metadata = load_ptbxl_signal(signal_path)
            image = load_ecg_image(
                self.data_root / str(row["image_path"]),
                self.image_transform,
            )
        elif self.dataset == "cpsc2018":
            signal, metadata = load_cpsc_signal(signal_path)
            image = load_ecg_image(
                self.data_root / str(row["image_path"]),
                self.image_transform,
            )
        else:
            signal, metadata = load_csn_signal(signal_path)
            image = load_csn_image(
                self.data_root / str(row["image_path"]),
                self.image_transform,
            )
        return {
            "sample_id": str(row["sample_id"]),
            "ecg_signal": torch.from_numpy(signal),
            "ecg_image": image,
            "signal_metadata": metadata,
        }

def inference_collate(samples):
    """Collate only stable tensor/string fields; metadata may be heterogeneous."""
    return {
        "sample_id": [sample["sample_id"] for sample in samples],
        "ecg_signal": torch.stack(
            [sample["ecg_signal"] for sample in samples]
        ),
        "ecg_image": torch.stack(
            [sample["ecg_image"] for sample in samples]
        ),
        "signal_metadata": [
            sample["signal_metadata"] for sample in samples
        ],
    }
