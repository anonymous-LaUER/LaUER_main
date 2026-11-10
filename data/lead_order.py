"""Explicit signal-lead and image-band permutations shared by pretraining and downstream tasks."""

from __future__ import annotations

from typing import Tuple

import numpy as np
from PIL import Image

MIMIC_LEAD_ORDER: Tuple[str, ...] = (
    "I", "II", "III", "aVR", "aVF", "aVL", "V1", "V2", "V3", "V4", "V5", "V6",
)

STANDARD_LEAD_ORDER: Tuple[str, ...] = (
    "I", "II", "III", "aVR", "aVL", "aVF", "V1", "V2", "V3", "V4", "V5", "V6",
)

LEAD_SWAP_INDICES: Tuple[int, int] = (4, 5)

N_LEADS: int = 12

def signal_to_mimic_order(signal: np.ndarray) -> np.ndarray:
    out = np.array(signal, copy=True)
    i, j = LEAD_SWAP_INDICES
    if out.shape[0] < max(i, j) + 1:
        raise ValueError(
            f"signal_to_mimic_order expects at least {max(i, j) + 1} leads, "
            f"got shape {out.shape}"
        )
    out[[i, j]] = out[[j, i]]
    return out

def band_bounds(height: int, n_leads: int = N_LEADS) -> int:
    if n_leads <= 0 or height % n_leads != 0:
        raise ValueError(
            f"image height {height} is not divisible by {n_leads} leads; "
            "cannot locate lead bands"
        )
    return height // n_leads

def image_array_to_mimic_order(image: np.ndarray,
                               n_leads: int = N_LEADS) -> np.ndarray:
    arr = np.array(image, copy=True)
    if arr.ndim not in (2, 3):
        raise ValueError(f"image array must be 2D or 3D, got shape {arr.shape}")
    b = band_bounds(arr.shape[0], n_leads)
    i, j = LEAD_SWAP_INDICES
    band_i = slice(i * b, (i + 1) * b)
    band_j = slice(j * b, (j + 1) * b)
    arr[band_i], arr[band_j] = arr[band_j].copy(), arr[band_i].copy()
    return arr

def image_to_mimic_order(image: Image.Image,
                         n_leads: int = N_LEADS) -> Image.Image:
    arr = np.asarray(image.convert("RGB"))
    return Image.fromarray(image_array_to_mimic_order(arr, n_leads=n_leads))
