
import pytest

from linear_probe.features import (
    FeatureError,
    _loader_mode_for,
    feature_cache_path,
)

def test_loader_mode_covers_every_requested_mode():
    assert _loader_mode_for({"signal_only"}) == "signal_only"
    assert _loader_mode_for({"image_only"}) == "image_only"
    assert _loader_mode_for({"fusion"}) == "fusion"

    assert _loader_mode_for({"signal_only", "image_only"}) == "fusion"
    assert _loader_mode_for({"signal_only", "fusion"}) == "fusion"
    assert _loader_mode_for({"image_only", "fusion"}) == "fusion"
    assert _loader_mode_for({"signal_only", "image_only", "fusion"}) == "fusion"

def test_loader_mode_covers_every_subset():
    modes = ("signal_only", "image_only", "fusion")
    covered = {
        "signal_only": {"signal_only"},
        "image_only": {"image_only"},
        "fusion": {"signal_only", "image_only", "fusion"},
    }
    for mask in range(1, 1 << len(modes)):
        subset = {modes[i] for i in range(len(modes)) if mask & (1 << i)}
        loader_mode = _loader_mode_for(subset)
        assert subset <= covered[loader_mode], (subset, loader_mode)

def test_loader_mode_rejects_empty_modes():
    with pytest.raises(FeatureError):
        _loader_mode_for(set())

@pytest.mark.parametrize("mode", ["signal_only", "image_only"])
def test_single_modal_cache_filename_is_feature_kind_specific(mode):
    cls = feature_cache_path("cache", "icbeb", mode, "tag", "prep", "cls")
    sgl = feature_cache_path("cache", "icbeb", mode, "tag", "prep", "sgl")
    assert cls != sgl
    assert "cls" in cls.name and "sgl" not in cls.name

def test_fusion_cache_filename_ignores_feature_kind():
    cls = feature_cache_path("cache", "icbeb", "fusion", "tag", "prep", "cls")
    sgl = feature_cache_path("cache", "icbeb", "fusion", "tag", "prep", "sgl")
    assert cls == sgl
