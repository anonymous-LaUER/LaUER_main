import numpy as np
import pytest
from PIL import Image

from data.lead_order import (
    LEAD_SWAP_INDICES,
    MIMIC_LEAD_ORDER,
    STANDARD_LEAD_ORDER,
    band_bounds,
    image_array_to_mimic_order,
    image_to_mimic_order,
    signal_to_mimic_order,
)

def _lead_index_signal():
    arr = np.zeros((12, 3), dtype=np.float32)
    for lead in range(12):
        arr[lead] = lead
    return arr

def _lead_index_image():
    arr = np.zeros((720, 4, 3), dtype=np.uint8)
    for lead in range(12):
        arr[lead * 60:(lead + 1) * 60, :, 0] = lead + 1
    return arr

def test_orders_differ_only_at_avl_avf():
    assert STANDARD_LEAD_ORDER[4] == "aVL" and STANDARD_LEAD_ORDER[5] == "aVF"
    assert MIMIC_LEAD_ORDER[4] == "aVF" and MIMIC_LEAD_ORDER[5] == "aVL"
    assert LEAD_SWAP_INDICES == (4, 5)
    for index in range(12):
        if index in LEAD_SWAP_INDICES:
            continue
        assert STANDARD_LEAD_ORDER[index] == MIMIC_LEAD_ORDER[index]

def test_signal_swap_only_touches_4_and_5():
    out = signal_to_mimic_order(_lead_index_signal())
    assert out[4, 0] == 5 and out[5, 0] == 4
    for index in range(12):
        if index in LEAD_SWAP_INDICES:
            continue
        assert out[index, 0] == index

def test_signal_swap_does_not_mutate_input():
    src = _lead_index_signal()
    signal_to_mimic_order(src)
    assert src[4, 0] == 4 and src[5, 0] == 5

def test_image_swap_moves_bands_4_and_5():
    out = image_array_to_mimic_order(_lead_index_image())
    assert out[240, 0, 0] == 6      # band5 (aVF) moved up to band4
    assert out[300, 0, 0] == 5      # band4 (aVL) moved down to band5
    assert out[0, 0, 0] == 1 and out[660, 0, 0] == 12

def test_signal_and_image_use_the_same_permutation():
    signal = signal_to_mimic_order(_lead_index_signal())[:, 0]
    image = image_array_to_mimic_order(_lead_index_image())[:, 0, 0] - 1
    band_height = 720 // 12
    for lead in range(12):
        assert image[lead * band_height] == signal[lead]

def test_image_pil_roundtrip():
    src = Image.fromarray(_lead_index_image())
    out = np.asarray(image_to_mimic_order(src))
    assert out[240, 0, 0] == 6 and out[300, 0, 0] == 5

def test_band_bounds_rejects_non_divisible_height():
    assert band_bounds(720) == 60
    with pytest.raises(ValueError):
        band_bounds(721)

def test_signal_swap_rejects_too_few_leads():
    with pytest.raises(ValueError):
        signal_to_mimic_order(np.zeros((4, 10), dtype=np.float32))
