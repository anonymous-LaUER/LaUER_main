"""Strict checkpoint loading for zero-shot evaluation (v3).

Only tgl parameters may be unused. Every encoder, adapter, projection,
SGL temperature, signal-image fusion, matching head, and Query-guided tracing head must
match exactly.
"""

from __future__ import annotations

from collections import OrderedDict
from pathlib import Path
from typing import Dict, Mapping, MutableMapping, Tuple

import torch

from configs.config import load_config_from_yaml
from models.lauer import LaUER

ALLOWED_UNUSED_PREFIXES: Tuple[str, ...] = (
    "tgl.",
    "tgl_proj.",
    "text_adapter.proj1_cgl.",
)
CASCADE_PREFIXES: Tuple[str, ...] = (
    "fusion_sit.",
    "fusion_sit2.",
    "fusion_ts.",
)

def _torch_load(path: Path):
    try:
        return torch.load(path, map_location="cpu", weights_only=True)
    except TypeError:  # torch < 2.0
        return torch.load(path, map_location="cpu")
    except Exception:
        # Some legacy training artifacts contain safe non-tensor metadata that
        # is rejected by weights_only. Loading a user-supplied local checkpoint
        # is the intended operation here.
        return torch.load(path, map_location="cpu", weights_only=False)

def _is_state_dict(value) -> bool:
    return (
        isinstance(value, Mapping)
        and bool(value)
        and all(isinstance(key, str) for key in value)
        and all(isinstance(tensor, torch.Tensor) for tensor in value.values())
    )

def extract_state_dict(artifact) -> OrderedDict:
    """Extract a tensor state dictionary from common checkpoint containers."""
    if _is_state_dict(artifact):
        state = artifact
    elif isinstance(artifact, Mapping):
        state = None
        for key in (
            "model_state_dict",
            "state_dict",
            "model",
            "module",
            "network",
        ):
            candidate = artifact.get(key)
            if _is_state_dict(candidate):
                state = candidate
                break
        if state is None:
            raise ValueError(
                "Checkpoint does not contain a recognized tensor state_dict"
            )
    else:
        raise TypeError(
            f"Unsupported checkpoint artifact type: {type(artifact).__name__}"
        )

    normalized = OrderedDict(state)
    for prefix in ("module.", "_orig_mod."):
        if normalized and all(key.startswith(prefix) for key in normalized):
            normalized = OrderedDict(
                (key[len(prefix):], value)
                for key, value in normalized.items()
            )
    return normalized

def _uses_independent_projection(state: Mapping[str, torch.Tensor]) -> bool:
    return any(".proj1_cgl." in key for key in state)

def _validate_architecture(config, state: Mapping[str, torch.Tensor]) -> Dict:

    legacy_stage_keys = [
        key
        for key in state
        if key.startswith(CASCADE_PREFIXES)
        or key.startswith(("fusion_branch1.", "fusion_branch2."))
    ]
    if legacy_stage_keys:
        raise ValueError(
            "Checkpoint contains legacy cascade/branch CGL parameters, "
            f"which are not supported by LaUER: {legacy_stage_keys[:5]}"
        )

    checkpoint_independent = _uses_independent_projection(state)
    config_shared = bool(config.projection.use_shared_proj1)
    if checkpoint_independent == config_shared:
        checkpoint_mode = (
            "10_projection_independent"
            if checkpoint_independent
            else "7_projection_shared"
        )
        config_mode = (
            "7_projection_shared"
            if config_shared
            else "10_projection_independent"
        )
        raise ValueError(
            "Projection topology mismatch: checkpoint is "
            f"{checkpoint_mode}, config is {config_mode}"
        )
    return {
        "cgl_structure": "fusion_fusion+multilabel",
        "projection_topology": (
            "10_projection_independent"
            if checkpoint_independent
            else "7_projection_shared"
        ),
    }

def _is_allowed_unused(key: str) -> bool:
    return key.startswith(ALLOWED_UNUSED_PREFIXES)

def load_zero_shot_model(
    config_path: str | Path,
    checkpoint_path: str | Path,
    device: str | torch.device = "cpu",
    mode: str = "sgl+cgl",
) -> Tuple[LaUER, Dict]:
    """Construct and strictly load the zero-shot-required model subgraph."""
    config_path = Path(config_path).resolve()
    checkpoint_path = Path(checkpoint_path).resolve()
    if not config_path.is_file():
        raise FileNotFoundError(config_path)
    if not checkpoint_path.is_file():
        raise FileNotFoundError(checkpoint_path)

    state = extract_state_dict(_torch_load(checkpoint_path))
    config = load_config_from_yaml(str(config_path))
    if mode not in {"sgl", "sgl+cgl"}:
        raise ValueError(f"Unsupported zero-shot mode: {mode}")
    architecture = (_validate_architecture(config, state) if mode != "sgl"
                    else {"cgl_structure": "not applicable", "mode": "sgl"})
    model = LaUER(config, mode=mode)

    # TextEncoder registers its transformer lazily; materialize it before
    # comparing state dictionaries so fine-tuned text weights cannot be
    # silently skipped.
    model.load_pretrained_encoders(device="cpu")

    expected = OrderedDict(
        (key, value) for key, value in model.state_dict().items()
        if not key.startswith("query_encoder.")
        and not key.startswith("text_encoder.")
    )

    shape_mismatches = []
    for key in set(expected).intersection(state):
        if tuple(expected[key].shape) != tuple(state[key].shape):
            shape_mismatches.append(
                {
                    "key": key,
                    "expected": tuple(expected[key].shape),
                    "checkpoint": tuple(state[key].shape),
                }
            )
    missing = sorted(set(expected).difference(state))
    unused = sorted(set(state).difference(expected))
    illegal_unused = [key for key in unused if not _is_allowed_unused(key)]

    if shape_mismatches or missing or illegal_unused:
        pieces = []
        if shape_mismatches:
            pieces.append(f"shape_mismatches={shape_mismatches[:10]}")
        if missing:
            pieces.append(f"missing={missing[:20]}")
        if illegal_unused:
            pieces.append(f"unexpected={illegal_unused[:20]}")
        raise RuntimeError(
            "Checkpoint is not an exact match for the zero-shot subgraph: "
            + "; ".join(pieces)
        )

    required_state = OrderedDict(
        (key, state[key]) for key in expected
    )

    incompatibility = model.load_state_dict(required_state, strict=False)
    illegal_missing = [
        key for key in incompatibility.missing_keys
        if not key.startswith("query_encoder.")
        and not key.startswith("text_encoder.")
    ]
    if illegal_missing or incompatibility.unexpected_keys:
        raise RuntimeError(f"Unexpected load result: {incompatibility}")

    model.requires_grad_(False)
    model.eval()
    target_device = torch.device(device)
    model.to(target_device)
    report = {
        "config_path": str(config_path),
        "checkpoint_path": str(checkpoint_path),
        **architecture,
        "checkpoint_tensor_count": len(state),
        "loaded_tensor_count": len(required_state),
        "unused_tensor_count": len(unused),
        "unused_prefixes": list(ALLOWED_UNUSED_PREFIXES),
        "unused_tensors": unused,
        "shape_mismatches": [],
        "missing_tensors": [],
        "device": str(target_device),
    }
    return model, report
