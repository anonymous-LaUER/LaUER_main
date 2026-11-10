"""Extract and cache frozen raw encoder CLS or fused CLS features for linear evaluation."""

from __future__ import annotations

import hashlib
import json
from contextlib import nullcontext
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Set

import numpy as np
import torch

from .config import MODES, PROJECT_ROOT, PathsConfig, PreprocessConfig, ProbeConfig
from .dataset import ECGProbeDataset, make_loader
from .splits import SplitCohort

class FeatureError(RuntimeError):
    pass

def sha256_file(path: Path, chunk: int = 1 << 22) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(chunk), b""):
            digest.update(block)
    return digest.hexdigest()

def _cached_file_digest(path: Path, sidecar_dir: Path) -> str:
    stat = path.stat()
    sidecar_dir.mkdir(parents=True, exist_ok=True)
    sidecar = sidecar_dir / (hashlib.sha1(str(path).encode("utf-8")).hexdigest() + ".json")
    try:
        record = json.loads(sidecar.read_text(encoding="utf-8"))
        if (record.get("size") == stat.st_size
                and record.get("mtime") == stat.st_mtime
                and record.get("path") == str(path)):
            return str(record["sha256"])
    except Exception:
        pass
    digest = sha256_file(path)
    sidecar.write_text(json.dumps({
        "path": str(path),
        "size": stat.st_size,
        "mtime": stat.st_mtime,
        "sha256": digest,
    }), encoding="utf-8")
    return digest

def resolve_project_path(path_like: str) -> Path:
    path = Path(path_like)
    return path if path.is_absolute() else PROJECT_ROOT / path

def encoder_tag(probe: ProbeConfig, paths: PathsConfig) -> str:
    checkpoint = resolve_project_path(probe.encoder_checkpoint)
    if not checkpoint.is_file():
        raise FeatureError(f"encoder checkpoint not found: {checkpoint}")
    yaml_path = resolve_project_path(probe.config_yaml)
    if not yaml_path.is_file():
        raise FeatureError(f"config YAML not found: {yaml_path}")
    sidecar_dir = Path(paths.cache_dir) / "digests"
    ckpt_sha = _cached_file_digest(checkpoint, sidecar_dir)[:16]
    yaml_sha = sha256_file(yaml_path)[:12]
    return f"{checkpoint.stem}-{ckpt_sha}-cfg{yaml_sha}-amp{probe.amp}"

@dataclass
class EncoderBundle:

    modes: Set[str]
    device: str
    config: object
    sig_encoder: object = None
    img_encoder: object = None
    model: object = None
    dims: Dict[str, int] = field(default_factory=dict)

    def close(self) -> None:
        self.sig_encoder = None
        self.img_encoder = None
        self.model = None
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

def _artifact_state_dict(ckpt: dict):
    for key in ("state_dict", "model_state_dict"):
        if key in ckpt:
            return ckpt[key], key
    return ckpt, None

def load_single_encoder(encoder_type: str, probe: ProbeConfig, config, device: str):
    checkpoint = resolve_project_path(probe.encoder_checkpoint)
    ckpt = torch.load(checkpoint, map_location="cpu", weights_only=True)
    state_dict, wrapper = _artifact_state_dict(ckpt)

    prefix = "ecg_encoder." if encoder_type == "vit1d" else "image_encoder."
    enc_state = {
        k[len(prefix):]: v for k, v in state_dict.items() if k.startswith(prefix)
    }
    if not enc_state:
        prefixes = sorted({k.split(".")[0] for k in state_dict})
        raise FeatureError(
            f"no {prefix!r} keys in {checkpoint} (wrapper={wrapper}, "
            f"top-level prefixes={prefixes})"
        )

    if encoder_type == "vit1d":
        from models.encoders.vit1d_encoder import ViT1DEncoder
        encoder = ViT1DEncoder(config.ecg_encoder)
    else:
        from models.encoders.vit2d_encoder import ViT2DEncoder
        encoder = ViT2DEncoder(config.image_encoder)

    try:
        encoder.load_state_dict(enc_state, strict=True)
    except RuntimeError as error:
        raise FeatureError(
            f"{encoder_type} state_dict mismatch for {checkpoint}: {error}"
        ) from error

    encoder.to(device).eval()
    for param in encoder.parameters():
        param.requires_grad = False
    return encoder

_TOLERATED_MISSING_PREFIXES = ("text_encoder.", "query_encoder.", "tgl.",
                             "tgl_proj.")
_TOLERATED_UNEXPECTED_PREFIXES = ("tgl.", "tgl_proj.", "text_encoder.",
                                  "query_encoder.", "cgl_loss_module.")

def load_fusion_model(probe: ProbeConfig, config, device: str):
    from models.lauer import LaUER
    from training.stage_io import MODE_RANK, artifact_source_mode

    checkpoint = resolve_project_path(probe.encoder_checkpoint)
    ckpt = torch.load(checkpoint, map_location="cpu", weights_only=True)
    state_dict, _ = _artifact_state_dict(ckpt)

    try:
        source_mode = artifact_source_mode(ckpt, state_dict)
    except Exception as error:
        raise FeatureError(
            f"cannot infer checkpoint source mode from {checkpoint}: {error}"
        ) from error
    if MODE_RANK.get(source_mode, 0) < MODE_RANK["sgl+cgl"]:
        raise FeatureError(
            f"fusion probe requires sgl+cgl (or later) weights, got {source_mode!r}. "
            "Fusion (SGL-only) weights would run fusion layers on random init."
        )

    model = LaUER(config, mode="sgl+cgl")
    model.load_pretrained_encoders(device=device)
    missing, unexpected = model.load_state_dict(state_dict, strict=False)

    bad_missing = [k for k in missing if not k.startswith(_TOLERATED_MISSING_PREFIXES)]
    if bad_missing:
        raise FeatureError(
            f"{checkpoint}: {len(bad_missing)} unexpected missing keys, e.g. "
            f"{bad_missing[:8]}"
        )
    bad_unexpected = [
        k for k in unexpected if not k.startswith(_TOLERATED_UNEXPECTED_PREFIXES)
    ]
    if bad_unexpected:
        raise FeatureError(
            f"{checkpoint}: {len(bad_unexpected)} unrecognised keys, e.g. "
            f"{bad_unexpected[:8]}"
        )
    if missing:
        print(f"[features] frozen keys absent from checkpoint (kept as init): {len(missing)}")

    freed = 0
    for attr in ("cgl_loss_module", "query_guided_tracing_head", "query_encoder",
                 "text_encoder", "tgl", "tgl_proj"):
        component = getattr(model, attr, None)
        if component is None:
            continue
        freed += sum(p.numel() for p in component.parameters())
        setattr(model, attr, None)
    if freed:
        print(f"[features] freed {freed / 1e6:.1f}M unused params (text/tgl/head)")

    model.to(device).eval()
    for param in model.parameters():
        param.requires_grad = False
    return model

def build_bundle(probe: ProbeConfig, modes: Iterable[str], device: str) -> EncoderBundle:
    modes = set(modes)
    unknown = modes - set(MODES)
    if unknown:
        raise FeatureError(f"unknown modes {sorted(unknown)}; expected {list(MODES)}")

    from configs.config import load_config_from_yaml

    yaml_path = resolve_project_path(probe.config_yaml)
    config = load_config_from_yaml(str(yaml_path))
    print(f"[features] LaUERConfig <- {yaml_path.name}")

    bundle = EncoderBundle(modes=modes, device=device, config=config)
    needs_signal = bool(modes & {"signal_only", "fusion"})
    needs_image = bool(modes & {"image_only", "fusion"})

    needs_adapter = (probe.single_modal_feature == "sgl"
                     and bool(modes & {"signal_only", "image_only"}))

    if "fusion" in modes or needs_adapter:
        bundle.model = load_fusion_model(probe, config, device)
        bundle.sig_encoder = bundle.model.ecg_encoder
        bundle.img_encoder = bundle.model.image_encoder
    else:
        if needs_signal:
            bundle.sig_encoder = load_single_encoder("vit1d", probe, config, device)
        if needs_image:
            bundle.img_encoder = load_single_encoder("vit2d", probe, config, device)

    for name, encoder in (("signal_only", bundle.sig_encoder),
                          ("image_only", bundle.img_encoder)):
        if encoder is None:
            continue
        sub = getattr(encoder, "config", None)
        bundle.dims[name] = int(
            getattr(encoder, "width", 0) or getattr(sub, "width", 0) or 0
        )
    return bundle

@dataclass
class FeatureSet:

    family: str
    mode: str
    ids: List[str]
    feats: torch.Tensor
    meta: Dict[str, object] = field(default_factory=dict)

    @property
    def dim(self) -> int:
        return int(self.feats.shape[1])

    def rows_for(self, sample_ids: Sequence[str]) -> torch.Tensor:
        index = {sid: row for row, sid in enumerate(self.ids)}
        missing = [sid for sid in sample_ids if sid not in index]
        if missing:
            raise FeatureError(
                f"{self.mode}: {len(missing)} id(s) missing from feature cache, "
                f"e.g. {missing[:5]}"
            )
        rows = torch.as_tensor([index[sid] for sid in sample_ids], dtype=torch.long)
        return self.feats.index_select(0, rows)

def feature_cache_path(cache_dir: str, family: str, mode: str, tag: str,
                       prep_tag: str, feature_kind: str = "sgl") -> Path:
    suffix = "" if mode == "fusion" else f"__{feature_kind}"
    return (Path(cache_dir) / "features"
            / f"{family}__{mode}{suffix}__{tag}__{prep_tag}.pt")

def save_feature_set(path: Path, feature_set: FeatureSet) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "family": feature_set.family,
        "mode": feature_set.mode,
        "ids": list(feature_set.ids),
        "feats": feature_set.feats.to(torch.float32).contiguous(),
        "meta": feature_set.meta,
    }
    tmp = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, tmp)
    path.unlink(missing_ok=True)
    tmp.replace(path)

def load_feature_set(path: Path) -> Optional[FeatureSet]:
    if not path.is_file():
        return None
    try:
        payload = torch.load(path, map_location="cpu", weights_only=True)
    except Exception as error:
        print(f"[features] ignoring unreadable cache {path}: {error}")
        return None
    return FeatureSet(
        family=str(payload["family"]),
        mode=str(payload["mode"]),
        ids=list(payload["ids"]),
        feats=payload["feats"].to(torch.float32),
        meta=dict(payload.get("meta", {})),
    )

def _amp_context(device: str, amp: str):
    if amp == "none" or torch.device(device).type != "cuda":
        return nullcontext()
    if amp == "bf16":
        if not getattr(torch.cuda, "is_bf16_supported", lambda: False)():
            raise FeatureError("bf16 requested but the CUDA device does not support it")
        return torch.amp.autocast("cuda", dtype=torch.bfloat16)
    if amp == "fp16":
        return torch.amp.autocast("cuda", dtype=torch.float16)
    raise FeatureError(f"amp must be one of bf16|fp16|none, got {amp!r}")

def _loader_mode_for(modes: Iterable[str]) -> str:
    modes = set(modes)
    needs_signal = bool(modes & {"signal_only", "fusion"})
    needs_image = bool(modes & {"image_only", "fusion"})
    if needs_signal and needs_image:
        return "fusion"
    if needs_signal:
        return "signal_only"
    if needs_image:
        return "image_only"
    raise FeatureError(f"nothing to extract from modes={sorted(modes)}")

def _assert_bundle_covers(bundle: EncoderBundle, modes: Iterable[str],
                          probe: ProbeConfig) -> None:
    modes = set(modes)
    if modes & {"signal_only", "fusion"} and bundle.sig_encoder is None:
        raise FeatureError(
            f"signal encoder missing from the feature bundle (requested {sorted(modes)})"
        )
    if modes & {"image_only", "fusion"} and bundle.img_encoder is None:
        raise FeatureError(
            f"image encoder missing from the feature bundle (requested {sorted(modes)})"
        )
    if (probe.single_modal_feature == "sgl"
            and modes & {"signal_only", "image_only"} and bundle.model is None):
        raise FeatureError(
            "single_modal_feature='sgl' needs the SGL adapters (bundle.model)"
        )

@torch.inference_mode()
def _run_pass(bundle: EncoderBundle, dataset: ECGProbeDataset, modes: Set[str],
              batch_size: int, probe: ProbeConfig, device: str,
              prep: PreprocessConfig) -> Dict[str, torch.Tensor]:
    _assert_bundle_covers(bundle, modes, probe)
    loader = make_loader(
        dataset, batch_size=batch_size, shuffle=False,
        num_workers=probe.num_workers, prefetch_factor=probe.prefetch_factor,
        pin_memory=True,
    )
    collected: Dict[str, List[torch.Tensor]] = {m: [] for m in modes}
    expected_ids = dataset.sample_ids
    cursor = 0
    amp_ctx = _amp_context(device, probe.amp)
    use_sgl = probe.single_modal_feature == "sgl"

    for batch in loader:
        batch_ids = list(batch["sample_id"])
        if batch_ids != expected_ids[cursor:cursor + len(batch_ids)]:
            raise FeatureError(
                "extraction order mismatch: loader returned unexpected sample ids "
                "(cache would be silently misaligned)"
            )
        cursor += len(batch_ids)

        signal = batch.get("ecg_signal")
        image = batch.get("ecg_image")
        signal = signal.to(device, non_blocking=True) if signal is not None else None
        image = image.to(device, non_blocking=True) if image is not None else None

        with amp_ctx:
            sig_tokens = (bundle.sig_encoder.encode_tokens(signal)
                          if signal is not None else None)
            img_tokens = (bundle.img_encoder.encode_tokens(image)
                          if image is not None else None)

            if "signal_only" in modes and sig_tokens is None:
                raise FeatureError(
                    "signal batch missing: loader mode does not load signals"
                )
            if "image_only" in modes and img_tokens is None:
                raise FeatureError(
                    "image batch missing: loader mode does not load images"
                )

            signal_feat = None
            image_feat = None
            if "signal_only" in modes:
                signal_feat = (bundle.model.adapters.ecg_adapter.forward_sgl(sig_tokens)
                               if use_sgl else sig_tokens[:, 0])
            if "image_only" in modes:
                image_feat = (bundle.model.adapters.image_adapter.forward_sgl(img_tokens)
                              if use_sgl else img_tokens[:, 0])
            fused = None
            if "fusion" in modes:
                sig_proj, img_proj, _ = bundle.model.adapters.project_tokens(
                    sig_tokens, img_tokens, None
                )
                fused = bundle.model.extract_fusion_si_cls(sig_proj, img_proj)

        if signal_feat is not None:
            collected["signal_only"].append(signal_feat.float().cpu())
        if image_feat is not None:
            collected["image_only"].append(image_feat.float().cpu())
        if "fusion" in modes:
            collected["fusion"].append(fused.float().cpu())

    if cursor != len(expected_ids):
        raise FeatureError(
            f"extraction covered {cursor} of {len(expected_ids)} samples"
        )
    return {mode: torch.cat(chunks, dim=0) for mode, chunks in collected.items()}

def extract_family_features(cohort,
                            paths: PathsConfig,
                            preprocess: PreprocessConfig,
                            probe: ProbeConfig,
                            modes: Iterable[str],
                            device: str,
                            use_cache: bool = True,
                            bundle: Optional[EncoderBundle] = None) -> Dict[str, FeatureSet]:
    modes = set(modes)
    family = cohort.family
    ids = cohort.all_ids()
    tag = encoder_tag(probe, paths)
    prep_tag = preprocess.tag()

    cached: Dict[str, FeatureSet] = {}
    to_compute: Set[str] = set()
    for mode in sorted(modes):
        feature_set = None
        if use_cache:
            feature_set = load_feature_set(
                feature_cache_path(paths.cache_dir, family, mode, tag, prep_tag,
                                   probe.single_modal_feature)
            )
        if feature_set is not None and feature_set.ids == ids:
            print(f"[features] cache hit: {family}/{mode} "
                  f"({feature_set.feats.shape[0]} x {feature_set.dim})")
            cached[mode] = feature_set
        else:
            if feature_set is not None:
                print(f"[features] cache id-set mismatch for {family}/{mode}; recomputing")
            to_compute.add(mode)

    if not to_compute:
        return cached

    from .cohort import resolve_sample_paths

    resolved = resolve_sample_paths(cohort, paths, check_exists=True)

    placeholder_labels = np.zeros((len(ids), cohort.num_classes), dtype=np.float32)
    loader_mode = _loader_mode_for(to_compute)
    dataset = ECGProbeDataset(
        family=family, sample_ids=ids, labels=placeholder_labels,
        paths=resolved, mode=loader_mode, preprocess=preprocess,
    )

    if bundle is None:
        bundle = build_bundle(probe, to_compute, device)

    batch_size = int(probe.extract_batch_size) or _default_batch_size(loader_mode)
    attempt = 0
    while True:
        try:
            print(f"[features] extracting {family} ({len(ids)} samples, "
                  f"mode={dataset.mode}, batch={batch_size}, amp={probe.amp})")
            tensors = _run_pass(bundle, dataset, to_compute, batch_size, probe,
                                device, preprocess)
            break
        except torch.cuda.OutOfMemoryError:
            attempt += 1
            if attempt > probe.oom_retries:
                raise FeatureError(
                    f"CUDA OOM persisted after {probe.oom_retries} batch halvings "
                    f"(batch={batch_size})"
                )
            batch_size = max(1, batch_size // 2)
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            print(f"[features] CUDA OOM -> retry with batch_size={batch_size}")

    meta_common = {
        "family": family,
        "encoder_tag": tag,
        "preprocess_tag": prep_tag,
        "amp": probe.amp,
        "single_modal_feature": probe.single_modal_feature,
        "n_samples": len(ids),
    }
    out: Dict[str, FeatureSet] = dict(cached)
    for mode, tensor in tensors.items():
        feature_set = FeatureSet(
            family=family, mode=mode, ids=list(ids), feats=tensor,
            meta={**meta_common, "mode": mode, "dim": int(tensor.shape[1])},
        )
        if use_cache:
            save_feature_set(
                feature_cache_path(paths.cache_dir, family, mode, tag, prep_tag,
                                   probe.single_modal_feature),
                feature_set,
            )
        out[mode] = feature_set
        print(f"[features] {family}/{mode}: {tuple(tensor.shape)}")
    return out

def _default_batch_size(mode: str) -> int:
    if mode == "fusion":
        return 16
    if mode == "image_only":
        return 32
    return 128
