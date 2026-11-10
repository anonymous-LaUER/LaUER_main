
import os
from pathlib import Path

import numpy as np
import pytest
import torch

from linear_probe.config import PreprocessConfig
from linear_probe.dataset import load_preprocessed_image, preprocess_signal

MEETI_CANDIDATES = (
    Path("datasets/MEETI"),
    Path(" ./datasets/MEETI"),
    Path(os.environ.get("LAUER_MEETI_ROOT", "datasets/MEETI")),
)

def _find_meeti_sample():
    for root in MEETI_CANDIDATES:
        if not root.is_dir():
            continue
        for mat in sorted(root.glob("p*/p*/s*/*.mat"))[:1]:
            png = mat.with_suffix(".png")
            if png.is_file():
                return mat, png
    return None

SAMPLE = _find_meeti_sample()
pytestmark = pytest.mark.skipif(
    SAMPLE is None, reason="MEETI pretraining data not present"
)

def _pretrain_loaders():
    from data.meeti_dataset import MEETIDataset

    dataset = MEETIDataset.__new__(MEETIDataset)
    dataset.image_transform = None
    return dataset

def test_signal_matches_pretraining_bit_exactly():
    mat_path, _ = SAMPLE
    dataset = _pretrain_loaders()
    pre_signal, _, _, _ = dataset._load_mat(str(mat_path))

    down = preprocess_signal(pre_signal, PreprocessConfig(signal_swap_leads=False))

    assert pre_signal.shape == (12, 5000)
    assert down.shape == (12, 5000)
    assert down.dtype == np.float32
    assert np.array_equal(down, pre_signal), (
        f"Downstream/pretraining signal tensors differ: max|diff|="
        f"{float(np.abs(down - pre_signal).max())}"
    )

def test_signal_differs_from_pretraining_only_by_the_lead_swap():
    mat_path, _ = SAMPLE
    dataset = _pretrain_loaders()
    pre_signal, _, _, _ = dataset._load_mat(str(mat_path))

    down = preprocess_signal(pre_signal, PreprocessConfig())
    expected = pre_signal.copy()
    expected[[4, 5]] = expected[[5, 4]]

    assert np.array_equal(down, expected)
    assert not np.array_equal(down, pre_signal) or np.array_equal(pre_signal[4], pre_signal[5])

def test_image_matches_pretraining_bit_exactly_without_band_swap():
    _, png_path = SAMPLE
    dataset = _pretrain_loaders()
    pre_image = dataset._load_image(str(png_path))

    down = load_preprocessed_image(png_path, PreprocessConfig(image_swap_bands=False))
    down_tensor = torch.from_numpy(down)

    assert pre_image.dtype == torch.uint8 and down_tensor.dtype == torch.uint8
    assert tuple(pre_image.shape) == tuple(down_tensor.shape) == (3, 720, 1000)
    assert torch.equal(pre_image, down_tensor), (
        f"Downstream/pretraining image tensors differ: max|diff|="
        f"{int((pre_image.to(torch.int16) - down_tensor.to(torch.int16)).abs().max())}"
    )

def test_image_band_swap_is_the_same_permutation_as_signal():
    mat_path, png_path = SAMPLE
    dataset = _pretrain_loaders()
    pre_image = dataset._load_image(str(png_path))

    down = load_preprocessed_image(png_path, PreprocessConfig())
    expected = pre_image.numpy().copy()

    expected[:, 240:300], expected[:, 300:360] = (
        expected[:, 300:360].copy(), expected[:, 240:300].copy()
    )
    assert np.array_equal(down, expected)
    assert not np.array_equal(down, pre_image.numpy())