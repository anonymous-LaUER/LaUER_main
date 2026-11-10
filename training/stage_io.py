"""LaUER stage metadata, architecture signatures, and checked checkpoint loading."""

from __future__ import annotations

from dataclasses import asdict, is_dataclass
import hashlib
import json
import os
from typing import Any, Dict, Mapping, Optional

import torch

STAGE_FORMAT_VERSION = 1
VALID_MODES = ("sgl", "sgl+cgl", "sgl+cgl+tgl")
MODE_RANK = {mode: idx + 1 for idx, mode in enumerate(VALID_MODES)}
MODE_DIR = {
    "sgl": "sgl",
    "sgl+cgl": "sgl_cgl",
    "sgl+cgl+tgl": "sgl_cgl_tgl",
}
MODE_COMPONENTS = {
    "sgl": ["encoders", "sgl_adapters", "sgl_loss"],
    "sgl+cgl": [
        "encoders", "sgl_adapters", "sgl_loss",
        "cgl_adapters", "fusion_encoders", "cgl_heads",
    ],
    "sgl+cgl+tgl": [
        "encoders", "sgl_adapters", "sgl_loss",
        "cgl_adapters", "fusion_encoders", "cgl_heads",
        "tgl", "tgl_projection",
    ],
}

def mode_output_dir(output_dir: str, run_name: str, mode: str) -> str:
    if mode not in MODE_DIR:
        raise ValueError(f"Unsupported training mode: {mode!r}")
    return os.path.join(output_dir, run_name, MODE_DIR[mode])

def _config_dict(config: Any) -> Dict[str, Any]:
    if is_dataclass(config):
        return asdict(config)
    if isinstance(config, Mapping):
        return dict(config)
    raise TypeError(f"Unsupported config type: {type(config).__name__}")

_NON_ARCH_KEYS = {
    "freeze",
    "checkpoint",
    "checkpoint_path",
    "model_name_or_path",
    "tokenizer_name_or_path",
    "query_model_name_or_path",
    "medcpt_path",

    "loss_weights",
    "temperature",
    "learnable_temp",
    "temp_max",
    "sgl_in_global",
    "cgl_in_global",
    "dropout",
    "cgl_head_dropout",
    "drop_path_rate",
    "layernorm",
    "layernorm_position",
    "activation",
    "hidden_dim",
    "max_length",
    "report_mask_rate",
    "mlm_loss_weight",
    "ar_loss_weight",
    "full_mask_prob",
    "full_mask_rampup",
    "lm_weight",
    "max_generation_len",
    "keyword_threshold",

    "loss_mode",
    "mode",
    "region_weights",
    "tgl_report_marker",
    "tgl_explanation_marker",
    "max_tgl_length",

    "loss_type",
    "asl_gamma_pos",
    "asl_gamma_neg",
    "asl_clip",
    "asl_eps",
}

LEGACY_IGNORABLE_PREFIXES = ("text_adapter.proj1_cgl.",)

FROZEN_EXCLUDED_PREFIXES = ("text_encoder.",)

def _without_runtime_fields(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {
            key: _without_runtime_fields(item)
            for key, item in value.items()
            if key not in _NON_ARCH_KEYS
        }
    if isinstance(value, (list, tuple)):
        return [_without_runtime_fields(item) for item in value]
    return value

def stage_architecture_payload(config: Any, mode: str) -> Dict[str, Any]:
    """Return only architecture fields that already exist in ``mode``.

    Later-stage-only configuration is deliberately excluded.  For example, a
    TGL config difference must not prevent a SGL checkpoint from
    initializing SGL+CGL.
    """
    if mode not in MODE_RANK:
        raise ValueError(f"Unsupported training mode: {mode!r}")
    cfg = _config_dict(config)
    rank = MODE_RANK[mode]

    projection = cfg["projection"]
    projection_payload: Dict[str, Any] = {}
    for modality in ("ecg", "image", "text"):
        projection_payload[modality] = {
            "sgl": projection[modality]["sgl"],
        }
        if rank >= 2:
            projection_payload[modality]["cgl"] = projection[modality]["cgl"]
    if rank >= 2:
        projection_payload["use_shared_proj1"] = projection["use_shared_proj1"]
    if rank >= 3:
        projection_payload["tgl_proj"] = projection["tgl_proj"]

    payload: Dict[str, Any] = {
        "ecg_encoder": cfg["ecg_encoder"],
        "image_encoder": cfg["image_encoder"],
        "text_encoder": cfg["text_encoder"],
        "projection": projection_payload,
        "sgl": cfg["sgl"],
    }
    if rank >= 2:
        payload["cgl"] = cfg["cgl"]
    if rank >= 3:
        payload["tgl"] = cfg["tgl"]
    return _without_runtime_fields(payload)

def architecture_signature(config: Any, mode: str) -> str:
    payload = stage_architecture_payload(config, mode)
    encoded = json.dumps(
        payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()

def build_stage_artifact(model, epoch: Optional[int] = None,
                         metrics: Optional[Mapping[str, Any]] = None) -> Dict[str, Any]:
    mode = getattr(model, "mode", None)
    config = getattr(model, "config", None)
    if mode not in MODE_RANK or config is None:
        raise ValueError("Model must expose valid .mode and .config for stage saving")
    return {
        "format_version": STAGE_FORMAT_VERSION,
        "artifact_type": "stage_weights",
        "source_mode": mode,
        "components": list(MODE_COMPONENTS[mode]),
        "epoch": epoch,
        "metrics": dict(metrics or {}),
        "config": _config_dict(config),
        "architecture_signature": architecture_signature(config, mode),

        "model_state_dict": {
            key: value for key, value in model.state_dict().items()
            if not key.startswith(FROZEN_EXCLUDED_PREFIXES)
        },
    }

def extract_model_state(artifact: Any) -> Mapping[str, torch.Tensor]:
    if isinstance(artifact, Mapping):
        if "model_state_dict" in artifact:
            state = artifact["model_state_dict"]
        elif "state_dict" in artifact:
            state = artifact["state_dict"]
        else:
            state = artifact
        if isinstance(state, Mapping):
            return state
    raise RuntimeError("Weight file does not contain a valid model state_dict")

def _strip_module_prefix(state: Mapping[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    if any(key.startswith("module.") for key in state):
        return {
            key.replace("module.", "", 1): value
            for key, value in state.items()
        }
    return dict(state)

def key_stage(key: str) -> int:
    """Return the first stage that owns a state_dict key."""
    key = key.replace("module.", "", 1) if key.startswith("module.") else key
    if key.startswith("tgl.") or key.startswith("tgl_proj."):
        return 3
    if ".proj1_cgl." in key or key.startswith((
        "fusion_si.",
        "fusion_si_layers.",
        "fusion_sit.",
        "fusion_sit2.",
        "fusion_ts.",
        "fusion_branch1.",
        "fusion_branch2.",
        "cgl_loss_module.",
        "query_guided_tracing_head.",
        "query_encoder.",
    )):
        return 2
    return 1

def infer_source_mode(state: Mapping[str, torch.Tensor]) -> str:
    normalized = [
        key.replace("module.", "", 1) if key.startswith("module.") else key
        for key in state
    ]
    if any(key.startswith(("tgl.", "tgl_proj.")) for key in normalized):
        return "sgl+cgl+tgl"
    # Do not use proj1_cgl alone as evidence of CGL.  Legacy SGL models
    # accidentally constructed and saved those random, never-used projections.
    if any(key.startswith((
        "fusion_si.",
        "fusion_si_layers.",
        "fusion_sit.",
        "fusion_sit2.",
        "fusion_ts.",
        "fusion_branch1.",
        "fusion_branch2.",
        "cgl_loss_module.",
        "query_guided_tracing_head.",
        "query_encoder.",
    )) for key in normalized):
        return "sgl+cgl"
    if normalized:
        return "sgl"
    raise RuntimeError("Cannot infer the source stage from an empty weight file")

def artifact_source_mode(artifact: Any,
                         state: Optional[Mapping[str, torch.Tensor]] = None) -> str:
    if isinstance(artifact, Mapping):
        mode = artifact.get("source_mode") or artifact.get("mode")
        if mode is not None:
            if mode not in MODE_RANK:
                raise RuntimeError(f"Weight file declares unsupported mode: {mode!r}")
            return mode
    return infer_source_mode(state if state is not None else extract_model_state(artifact))

def _log(logger, level: str, message: str) -> None:
    if logger is None:
        print(message)
    else:
        getattr(logger, level)(message)

def load_stage_initialization(model, checkpoint_path: str, target_mode: str,
                              allow_partial: bool = False, device: str = "cpu",
                              logger=None) -> Dict[str, Any]:
    """Strictly initialize ``target_mode`` from its completed predecessor.

    Every key belonging to the source stage must exist and have the same shape.
    Only modules introduced after the source stage may remain randomly initialized.
    """
    if target_mode not in MODE_RANK:
        raise ValueError(f"Unsupported target mode: {target_mode!r}")
    if target_mode == "sgl":
        raise RuntimeError("SGL is the first stage and must not use --init-from")
    if not os.path.isfile(checkpoint_path):
        raise FileNotFoundError(f"Stage weight file not found: {checkpoint_path}")

    artifact = torch.load(checkpoint_path, map_location=device)
    source_state = _strip_module_prefix(extract_model_state(artifact))
    source_mode = artifact_source_mode(artifact, source_state)
    source_rank = MODE_RANK[source_mode]
    target_rank = MODE_RANK[target_mode]
    required_rank = target_rank - 1

    if source_rank == target_rank:
        raise RuntimeError(
            f"--init-from received a same-stage {source_mode!r} weight file. "
            "Use --resume with checkpoint.pt to continue the same training stage."
        )
    if source_rank > target_rank:
        raise RuntimeError(
            f"Cannot initialize {target_mode!r} from later-stage {source_mode!r} weights."
        )
    if source_rank != required_rank and not allow_partial:
        required_mode = VALID_MODES[required_rank - 1]
        raise RuntimeError(
            f"Target mode {target_mode!r} requires {required_mode!r} weights, "
            f"but the source is {source_mode!r}. Re-run with --allow-partial-init "
            "only if randomly initializing every skipped stage is intentional."
        )

    saved_signature = artifact.get("architecture_signature") if isinstance(artifact, Mapping) else None
    current_signature = architecture_signature(model.config, source_mode)
    if saved_signature is not None and saved_signature != current_signature:
        raise RuntimeError(
            "Stage architecture signature mismatch. The source model configuration "
            "does not match the corresponding modules in the target configuration.\n"
            f"  source: {saved_signature}\n  target: {current_signature}"
        )

    current_state = model.state_dict()

    expected_keys = {
        key for key in current_state
        if key_stage(key) <= source_rank
        and not key.startswith(FROZEN_EXCLUDED_PREFIXES)
    }
    load_state = {
        key: value for key, value in source_state.items()
        if key_stage(key) <= source_rank
    }
    load_keys = set(load_state)
    missing = sorted(expected_keys - load_keys)
    unexpected = sorted(
        key for key in load_keys - set(current_state)
        if not key.startswith(LEGACY_IGNORABLE_PREFIXES)
    )
    shape_mismatches = sorted(
        (key, tuple(load_state[key].shape), tuple(current_state[key].shape))
        for key in expected_keys & load_keys
        if tuple(load_state[key].shape) != tuple(current_state[key].shape)
    )
    if missing or unexpected or shape_mismatches:
        lines = [
            f"Incomplete or incompatible {source_mode!r} stage weights: {checkpoint_path}"
        ]
        if missing:
            lines.append(f"Missing required keys: {len(missing)} (first: {missing[:8]})")
        if unexpected:
            lines.append(f"Unexpected source keys: {len(unexpected)} (first: {unexpected[:8]})")
        if shape_mismatches:
            lines.append(f"Shape mismatches: {shape_mismatches[:8]}")
        raise RuntimeError("\n".join(lines))

    model.load_state_dict(load_state, strict=False)
    random_keys = sorted(
        key for key in current_state
        if key_stage(key) > source_rank
    )
    _log(logger, "info", "=" * 60)
    _log(
        logger, "info",
        f"[Stage Init] source={source_mode} -> target={target_mode}",
    )
    _log(logger, "info", f"[Stage Init] Loaded required keys: {len(expected_keys)}")
    if source_rank != required_rank:
        skipped = VALID_MODES[source_rank:required_rank]
        _log(
            logger, "warning",
            f"[Stage Init] PARTIAL initialization explicitly allowed; skipped stages "
            f"{list(skipped)} are random",
        )
    _log(
        logger, "info",
        f"[Stage Init] Randomly initialized later-stage keys: {len(random_keys)}",
    )
    _log(logger, "info", "=" * 60)
    return {
        "source_mode": source_mode,
        "target_mode": target_mode,
        "loaded_keys": len(expected_keys),
        "random_keys": len(random_keys),
        "partial": source_rank != required_rank,
    }

def validate_resume_artifact(artifact: Any, model, checkpoint_path: str) -> str:
    state = _strip_module_prefix(extract_model_state(artifact))
    saved_mode = artifact_source_mode(artifact, state)
    current_mode = getattr(model, "mode", None)
    if saved_mode != current_mode:
        raise RuntimeError(
            f"Resume mode mismatch for {checkpoint_path}: checkpoint={saved_mode!r}, "
            f"current={current_mode!r}. Use --init-from for stage transitions."
        )
    saved_signature = artifact.get("architecture_signature") if isinstance(artifact, Mapping) else None
    if saved_signature is not None:
        current_signature = architecture_signature(model.config, current_mode)
        if saved_signature != current_signature:
            raise RuntimeError(
                f"Resume architecture mismatch for {checkpoint_path}. Resume requires "
                "the exact model configuration that created the checkpoint."
            )
    return saved_mode
