"""Downstream signal and image preprocessing aligned with LaUER pretraining."""

from __future__ import annotations

from pathlib import Path
from typing import Callable, Dict, List, Optional, Sequence

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from data.lead_order import image_array_to_mimic_order, signal_to_mimic_order

from .config import PreprocessConfig
from .cohort import SamplePaths

class SignalLoadError(RuntimeError):
    pass

def _read_ptbxl_signal(path) -> np.ndarray:
    try:
        import wfdb
    except ImportError as error:  # pragma: no cover
        raise SignalLoadError("wfdb is required to read PTB-XL signals") from error
    record = wfdb.rdrecord(str(path))
    return np.asarray(record.p_signal, dtype=np.float32).T

def _read_icbeb_signal(path) -> np.ndarray:
    """CPSC2018 .mat -> ['ECG'][0,0]['data'] -> [12, N] float。"""
    from scipy.io import loadmat

    data = loadmat(str(path), variable_names=["ECG"])
    if "ECG" not in data:
        raise SignalLoadError(f"{path}: .mat has no ECG field")
    return np.asarray(data["ECG"][0, 0]["data"], dtype=np.float32)

_CSN_DEFAULT_GAIN = 1000.0

def read_wfdb_gain(hea_path) -> Optional[float]:
    lines = [line.strip() for line in
             Path(hea_path).read_text(encoding="utf-8", errors="replace").splitlines()]
    lines = [line for line in lines if line and not line.startswith("#")]
    if not lines:
        return None
    gains = []
    for line in lines[1:]:
        parts = line.split()
        if len(parts) < 3:
            continue
        token = parts[2]
        number = token.split("/")[0] if "/" in token else token
        try:
            gains.append(float(number))
        except ValueError:
            continue
    if not gains:
        return None
    if max(gains) - min(gains) > 1e-9:
        raise SignalLoadError(
            f"{hea_path}: inconsistent per-lead gains {sorted(set(gains))}"
        )
    return gains[0]

def _read_chapman_signal(path, apply_gain: bool = True) -> np.ndarray:
    from scipy.io import loadmat

    data = loadmat(str(path), variable_names=["val"])
    if "val" not in data:
        raise SignalLoadError(f"{path}: .mat has no val field")
    signal = np.asarray(data["val"], dtype=np.float32)
    if not apply_gain:
        return signal

    path = Path(path)
    hea_path = path.with_suffix(".hea")
    gain = read_wfdb_gain(hea_path) if hea_path.is_file() else None
    if gain is None:
        gain = _CSN_DEFAULT_GAIN
    if gain <= 0:
        raise SignalLoadError(f"{hea_path}: non-positive gain {gain}")
    return signal / float(gain)

SIGNAL_READERS: Dict[str, Callable[[object], np.ndarray]] = {
    "ptbxl": _read_ptbxl_signal,
    "icbeb": _read_icbeb_signal,
    "chapman": _read_chapman_signal,
}

def preprocess_signal(raw: np.ndarray,
                      preprocess: PreprocessConfig) -> np.ndarray:
    arr = np.asarray(raw, dtype=np.float32)
    if arr.ndim != 2:
        raise SignalLoadError(f"signal must be 2D [C, N], got shape {arr.shape}")
    if arr.shape[0] != preprocess.n_leads:
        candidate = arr.T
        if candidate.shape[0] != preprocess.n_leads:
            raise SignalLoadError(
                f"signal has {arr.shape[0]} leads (and {candidate.shape[0]} after "
                f"transpose), expected {preprocess.n_leads}"
            )
        arr = candidate
    arr = np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)

    length = int(preprocess.signal_length)
    arr = arr[:, :length]
    if arr.shape[1] < length:
        pad = np.zeros((arr.shape[0], length - arr.shape[1]), dtype=np.float32)
        arr = np.concatenate([arr, pad], axis=1)

    if preprocess.signal_swap_leads:
        arr = signal_to_mimic_order(arr)
    return np.ascontiguousarray(arr, dtype=np.float32)

def read_raw_signal(path, family: str,
                    preprocess: Optional[PreprocessConfig] = None) -> np.ndarray:
    preprocess = preprocess or PreprocessConfig()
    try:
        if family == "chapman":

            return _read_chapman_signal(path, apply_gain=preprocess.csn_apply_gain)
        return SIGNAL_READERS[family](path)
    except SignalLoadError:
        raise
    except Exception as error:
        raise SignalLoadError(f"failed to read {family} signal {path}: {error}") from error

def load_preprocessed_signal(path, family: str,
                             preprocess: PreprocessConfig) -> np.ndarray:
    return preprocess_signal(read_raw_signal(path, family, preprocess), preprocess)

def load_preprocessed_image(path, preprocess: PreprocessConfig) -> np.ndarray:
    from PIL import Image

    try:
        with Image.open(path) as handle:
            arr = np.asarray(handle.convert("RGB"), dtype=np.uint8)
    except Exception as error:
        raise SignalLoadError(f"failed to read image {path}: {error}") from error
    if preprocess.image_swap_bands:
        arr = image_array_to_mimic_order(arr, n_leads=preprocess.n_leads)
    arr = np.ascontiguousarray(arr.transpose(2, 0, 1))     # [3, H, W]
    return arr

# Dataset

class ECGProbeDataset(Dataset):

    def __init__(self,
                 family: str,
                 sample_ids: Sequence[str],
                 labels: np.ndarray,
                 paths: Dict[str, SamplePaths],
                 mode: str,
                 preprocess: PreprocessConfig):
        self.family = family
        self.sample_ids = [str(i) for i in sample_ids]
        self.labels = np.asarray(labels, dtype=np.float32)
        self.paths = paths
        self.mode = mode
        self.preprocess = preprocess
        if self.labels.shape[0] != len(self.sample_ids):
            raise ValueError(
                f"labels rows {self.labels.shape[0]} != {len(self.sample_ids)} ids"
            )
        self.needs_signal = mode in ("signal_only", "fusion")
        self.needs_image = mode in ("image_only", "fusion")

    def __len__(self) -> int:
        return len(self.sample_ids)

    def __getitem__(self, index: int) -> Dict[str, object]:
        sample_id = self.sample_ids[index]
        item = self.paths.get(sample_id)
        if item is None:
            raise SignalLoadError(f"no path entry for {sample_id}")
        out: Dict[str, object] = {
            "sample_id": sample_id,
            "label": torch.from_numpy(self.labels[index]),
        }
        if self.needs_signal:
            if item.signal is None:
                raise SignalLoadError(f"{sample_id}: signal path missing")
            signal = load_preprocessed_signal(item.signal, self.family, self.preprocess)
            out["ecg_signal"] = torch.from_numpy(signal)
        if self.needs_image:
            if item.image is None:
                raise SignalLoadError(f"{sample_id}: image path missing")
            image = load_preprocessed_image(item.image, self.preprocess)
            out["ecg_image"] = torch.from_numpy(image)
        return out

def probe_collate(batch: List[Dict[str, object]]) -> Dict[str, object]:
    out: Dict[str, object] = {
        "sample_id": [b["sample_id"] for b in batch],
        "label": torch.stack([b["label"] for b in batch]),
    }
    if "ecg_signal" in batch[0]:
        out["ecg_signal"] = torch.stack([b["ecg_signal"] for b in batch])
    if "ecg_image" in batch[0]:
        out["ecg_image"] = torch.stack([b["ecg_image"] for b in batch])
    return out

def make_loader(dataset: ECGProbeDataset,
                batch_size: int,
                shuffle: bool = False,
                num_workers: int = 0,
                prefetch_factor: int = 2,
                pin_memory: bool = True,
                seed: int = 42) -> DataLoader:
    generator = None
    if shuffle:
        generator = torch.Generator()
        generator.manual_seed(int(seed))
    kwargs: Dict[str, object] = dict(
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=pin_memory and torch.cuda.is_available(),
        drop_last=False,
        collate_fn=probe_collate,
    )
    if shuffle:
        kwargs["generator"] = generator
    if num_workers > 0:
        kwargs["prefetch_factor"] = prefetch_factor
        kwargs["persistent_workers"] = False
    return DataLoader(dataset, **kwargs)
