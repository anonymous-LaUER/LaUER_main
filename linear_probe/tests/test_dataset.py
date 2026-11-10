import numpy as np
import pytest
from PIL import Image

from linear_probe.config import PreprocessConfig
from linear_probe.dataset import (
    SignalLoadError,
    load_preprocessed_image,
    preprocess_signal,
)

def _ramp_signal(n=6000, leads=12, scale=1.0):
    base = np.linspace(0.0, 1.0, n, dtype=np.float32) * scale
    return np.stack([base * (lead + 1) for lead in range(leads)])

@pytest.mark.parametrize("family", ["ptbxl", "icbeb", "chapman"])
def test_same_window_for_every_family(family):
    out = preprocess_signal(_ramp_signal(6000), PreprocessConfig(signal_swap_leads=False))
    assert out.shape == (12, 5000)
    assert out[0, 4999] > 0 and out[0, 3000] > 0

    assert np.all(out[:, 2500:] != 0)

def test_pads_short_records_to_signal_length():
    out = preprocess_signal(_ramp_signal(3000), PreprocessConfig())
    assert out.shape == (12, 5000)
    assert np.all(out[:, 3000:] == 0.0)

def test_truncates_long_records_to_signal_length():
    out = preprocess_signal(_ramp_signal(9000), PreprocessConfig())
    assert out.shape == (12, 5000)
    assert out[0, -1] == pytest.approx(4999 / 8999, abs=1e-4)

def test_signal_length_is_configurable():
    out = preprocess_signal(_ramp_signal(9000), PreprocessConfig(signal_length=2500))
    assert out.shape == (12, 2500)
    assert out[0, -1] == pytest.approx(2499 / 8999, abs=1e-4)

def test_no_amplitude_normalization():
    raw = _ramp_signal(5000, scale=10.0)
    out = preprocess_signal(raw, PreprocessConfig(signal_swap_leads=False))
    assert float(out.min()) == pytest.approx(float(raw.min()))
    assert float(out.max()) == pytest.approx(float(raw.max()))
    assert np.array_equal(out, raw)

def test_output_is_bit_identical_to_input_when_swap_disabled():
    rng = np.random.default_rng(0)
    raw = (rng.standard_normal((12, 5000)) * 0.2).astype(np.float32)
    out = preprocess_signal(raw, PreprocessConfig(signal_swap_leads=False))
    assert out.dtype == np.float32
    assert np.array_equal(out, raw)
    assert out.flags["C_CONTIGUOUS"]

def test_constant_record_is_untouched():
    raw = np.full((12, 5000), 2.0, dtype=np.float32)
    out = preprocess_signal(raw, PreprocessConfig(signal_swap_leads=False))
    assert np.all(np.isfinite(out))
    assert np.allclose(out, 2.0)

def test_nan_and_inf_are_replaced():
    raw = _ramp_signal(5000)
    raw[0, 10] = np.nan
    raw[1, 10] = np.inf
    raw[2, 10] = -np.inf
    out = preprocess_signal(raw, PreprocessConfig())
    assert np.all(np.isfinite(out))
    assert out[0, 10] == 0.0 and out[1, 10] == 0.0 and out[2, 10] == 0.0

def test_lead_swap_can_be_disabled():
    raw = np.zeros((12, 5000), dtype=np.float32)
    raw[4] = 4.0
    raw[5] = 5.0
    swapped = preprocess_signal(raw, PreprocessConfig())
    raw_out = preprocess_signal(raw, PreprocessConfig(signal_swap_leads=False))
    assert swapped[4, 0] == 5.0 and swapped[5, 0] == 4.0
    assert raw_out[4, 0] == 4.0 and raw_out[5, 0] == 5.0

def test_transposed_input_is_accepted():
    raw = _ramp_signal(5000).T                     # [N, 12]
    out = preprocess_signal(raw, PreprocessConfig(signal_swap_leads=False))
    assert out.shape == (12, 5000)
    assert np.array_equal(out, np.ascontiguousarray(raw.T))

def test_wrong_lead_count_raises():
    with pytest.raises(SignalLoadError):
        preprocess_signal(np.zeros((8, 5000), dtype=np.float32), PreprocessConfig())

def test_1d_input_raises():
    with pytest.raises(SignalLoadError):
        preprocess_signal(np.zeros(5000, dtype=np.float32), PreprocessConfig())

def test_image_loader_returns_uint8_channels_first_and_swaps_bands(tmp_path):
    arr = np.zeros((720, 1000, 3), dtype=np.uint8)
    for lead in range(12):
        arr[lead * 60:(lead + 1) * 60, :, 0] = lead + 1
    path = tmp_path / "sample.png"
    Image.fromarray(arr).save(path)

    out = load_preprocessed_image(path, PreprocessConfig(image_swap_bands=True))
    assert out.dtype == np.uint8
    assert out.shape == (3, 720, 1000)
    assert out[0, 240, 0] == 6 and out[0, 300, 0] == 5

    raw = load_preprocessed_image(path, PreprocessConfig(image_swap_bands=False))
    assert raw[0, 240, 0] == 5 and raw[0, 300, 0] == 6

def test_image_loader_missing_file_raises(tmp_path):
    with pytest.raises(SignalLoadError):
        load_preprocessed_image(tmp_path / "nope.png", PreprocessConfig())

_HEA_TEMPLATE = (
    "JS00001 12 500 5000\n"
    "JS00001.mat 16+24 1000/mV 16 0 371 15767 0 I\n"
    "JS00001.mat 16+24 1000/mV 16 0 366 -22837 0 II\n"
    "#Age: 59\n"
    "#Sex: Female\n"
)

def test_read_wfdb_gain_skips_the_record_line(tmp_path):
    from linear_probe.dataset import read_wfdb_gain

    path = tmp_path / "JS00001.hea"
    path.write_text(_HEA_TEMPLATE, encoding="utf-8")
    assert read_wfdb_gain(path) == 1000.0

def test_read_wfdb_gain_rejects_mixed_gains(tmp_path):
    from linear_probe.dataset import read_wfdb_gain

    path = tmp_path / "mixed.hea"
    path.write_text(
        "X 2 500 5000\n"
        "X.mat 16+24 1000/mV 16 0 0 0 0 I\n"
        "X.mat 16+24 500/mV 16 0 0 0 0 II\n",
        encoding="utf-8",
    )
    with pytest.raises(SignalLoadError):
        read_wfdb_gain(path)

def test_read_wfdb_gain_returns_none_without_signal_lines(tmp_path):
    from linear_probe.dataset import read_wfdb_gain

    path = tmp_path / "empty.hea"
    path.write_text("#only comments\n", encoding="utf-8")
    assert read_wfdb_gain(path) is None

def test_csn_reader_divides_by_header_gain(tmp_path):
    scipy_io = pytest.importorskip("scipy.io")
    from linear_probe.dataset import _read_chapman_signal

    raw = (np.arange(12 * 500, dtype=np.int16).reshape(12, 500) - 2500)
    scipy_io.savemat(str(tmp_path / "rec.mat"), {"val": raw})
    lead_lines = "".join(
        f"rec.mat 16+24 1000/mV 16 0 0 0 0 {name}\n"
        for name in ("I", "II", "III", "aVR", "aVL", "aVF",
                     "V1", "V2", "V3", "V4", "V5", "V6")
    )
    (tmp_path / "rec.hea").write_text(
        "rec 12 500 500\n" + lead_lines, encoding="utf-8"
    )
    converted = _read_chapman_signal(tmp_path / "rec.mat", apply_gain=True)
    assert np.allclose(converted, raw.astype(np.float32) / 1000.0, atol=1e-4)
    unchanged = _read_chapman_signal(tmp_path / "rec.mat", apply_gain=False)
    assert np.array_equal(unchanged, raw.astype(np.float32))

def test_csn_reader_falls_back_to_default_gain_without_header(tmp_path):
    scipy_io = pytest.importorskip("scipy.io")
    from linear_probe.dataset import _CSN_DEFAULT_GAIN, _read_chapman_signal

    raw = np.full((12, 100), 1000, dtype=np.int16)
    scipy_io.savemat(str(tmp_path / "nohdr.mat"), {"val": raw})
    converted = _read_chapman_signal(tmp_path / "nohdr.mat", apply_gain=True)
    assert np.allclose(converted, 1000.0 / _CSN_DEFAULT_GAIN)

def test_csn_gain_flag_changes_preprocess_tag():
    base = PreprocessConfig()
    assert base.tag() != PreprocessConfig(csn_apply_gain=False).tag()

def test_read_raw_signal_honours_the_csn_gain_switch(tmp_path):
    scipy_io = pytest.importorskip("scipy.io")
    from linear_probe.dataset import read_raw_signal

    raw = (np.arange(12 * 500, dtype=np.int16).reshape(12, 500) - 2500)
    scipy_io.savemat(str(tmp_path / "rec.mat"), {"val": raw})
    lead_lines = "".join(
        f"rec.mat 16+24 1000/mV 16 0 0 0 0 {name}\n"
        for name in ("I", "II", "III", "aVR", "aVL", "aVF",
                     "V1", "V2", "V3", "V4", "V5", "V6")
    )
    (tmp_path / "rec.hea").write_text("rec 12 500 500\n" + lead_lines,
                                      encoding="utf-8")

    counts = read_raw_signal(tmp_path / "rec.mat", "chapman",
                             PreprocessConfig(csn_apply_gain=False))
    milli = read_raw_signal(tmp_path / "rec.mat", "chapman",
                            PreprocessConfig(csn_apply_gain=True))

    assert np.array_equal(counts, raw.astype(np.float32))
    assert np.allclose(milli, raw.astype(np.float32) / 1000.0, atol=1e-4)
    assert not np.array_equal(counts, milli)
