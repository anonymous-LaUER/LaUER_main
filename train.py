"""Single-device cumulative LaUER training: SGL, SGL+CGL, and SGL+CGL+TGL."""
import os
import sys
import gc
import time
from contextlib import nullcontext

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import argparse
import logging
import math
import threading

import torch
import torch.nn as nn

from tqdm import tqdm

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from configs.config import load_config_from_yaml
from models.lauer import LaUER
from data.meeti_dataset import create_meeti_dataloaders
from evaluation.zero_shot import (
    zero_shot_retrieval_eval,
    format_eval_results,
    retrieval_results_from_embeddings,
)
from evaluation.cgl_eval import cgl_accuracy_eval
from evaluation.utils import (
    average_retrieval_metrics,
    average_eval_results,
    format_full_eval_results,
)
from training.nan_resilience import (
    NaNResilienceManager,
    diagnose_nan_loss,
    _format_nan_batch_log,
)
from training.stage_io import (
    FROZEN_EXCLUDED_PREFIXES,
    LEGACY_IGNORABLE_PREFIXES,
    build_stage_artifact,
    extract_model_state,
    load_stage_initialization,
    mode_output_dir,
    validate_resume_artifact,
)
from training.reproducibility import (
    capture_rng_state,
    configure_reproducibility,
    loader_generators,
    restore_rng_state,
    seed_rng,
)
from training.run_logging import (
    build_progress_postfix,
    capture_startup_prints,
    format_train_epoch,
    format_validation_epoch,
    log_dataloader_summary,
    log_run_configuration,
    log_runtime_environment,
)

def setup_logging(output_dir):
    os.makedirs(output_dir, exist_ok=True)
    log_file = os.path.join(output_dir, 'train.log')
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s [%(levelname)s] %(message)s',
        handlers=[
            logging.FileHandler(log_file, encoding='utf-8'),
            logging.StreamHandler(),
        ]
    )
    return logging.getLogger(__name__)

def set_seed(seed, deterministic=False, benchmark=True):
    """Apply both the configured backend policy and global seed."""
    return configure_reproducibility(seed, deterministic, benchmark)

def build_optimizer(model, config):
    wd = float(config.train.weight_decay)
    configured_norm_wd = getattr(config.train, 'norm_weight_decay', None)
    norm_wd = wd if configured_norm_wd is None else float(configured_norm_wd)
    if wd < 0 or norm_wd < 0:
        raise ValueError(
            f"weight_decay and norm_weight_decay must be >= 0, got {wd} and {norm_wd}"
        )
    wd_sw = config.train.wd_switches  # WeightDecaySwitches

    is_image_unfrozen = not config.image_encoder.freeze
    is_signal_unfrozen = not config.ecg_encoder.freeze

    norm_param_ids = {
        id(param)
        for module in model.modules()
        if isinstance(module, nn.LayerNorm)
        for param in module.parameters(recurse=False)
    }
    no_decay_keywords = {
        'pos_embed', 'position_embeddings', 'lead_embeddings', 'cls_token',
        'logit_scale',
    }
    no_decay_exact = {
        'tgl.embedding.weight',
    }

    # {category: {'matrix': [params], 'norm': [params], 'none': [params]}}
    groups = {}

    def _add_param(category, p, decay_kind):
        if category not in groups:
            groups[category] = {'matrix': [], 'norm': [], 'none': []}
        groups[category][decay_kind].append(p)

    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue

        force_no_decay = (
            name.endswith('.bias')
            or name in no_decay_exact
            or any(kw in name for kw in no_decay_keywords)
        )
        is_weight = ('weight' in name or name.endswith('.proj'))

        if 'logit_scale' in name:
            category, wd_enabled = 'sgl', False

        # proj2 → wd_switches[2], category=sgl
        elif 'adapter.proj2' in name:
            category, wd_enabled = 'sgl', wd_sw.proj2

        elif 'adapter.proj1_sgl' in name:
            category, wd_enabled = 'sgl', wd_sw.proj1_sgl

        elif ('adapter.attn_pooler' in name or 'adapter.lead_pooler' in name
              or 'adapter.lead_proj' in name):
            category, wd_enabled = 'sgl', wd_sw.proj2

        elif 'adapter.proj1_cgl' in name:
            category, wd_enabled = 'cgl', wd_sw.proj1_cgl

        # fusion_encoder → wd_switches[3], category=cgl
        elif name.startswith('fusion_'):
            category, wd_enabled = 'cgl', wd_sw.fusion_encoder

        # cgl_head → wd_switches[4], category=cgl
        elif 'cgl_loss_module.heads' in name:
            category, wd_enabled = 'cgl', wd_sw.cgl_head

        elif name.startswith('query_guided_tracing_head.'):
            category, wd_enabled = 'cgl', wd_sw.cgl_head

        # tgl_proj → wd_switches[7], category=tgl
        elif 'tgl_proj' in name:
            category, wd_enabled = 'tgl', wd_sw.tgl

        # tgl → wd_switches[7], category=tgl
        elif name.startswith('tgl.') or name.startswith('tgl_loss.'):
            category, wd_enabled = 'tgl', wd_sw.tgl

        # image_encoder → wd_switches[5], category=encoder
        elif is_image_unfrozen and name.startswith('image_encoder.'):
            category, wd_enabled = 'encoder', wd_sw.image_encoder

        # signal_encoder → wd_switches[6], category=encoder
        elif is_signal_unfrozen and name.startswith('ecg_encoder.'):
            category, wd_enabled = 'encoder', wd_sw.signal_encoder

        # text_encoder → category=other
        elif name.startswith('text_encoder.'):
            category, wd_enabled = 'other', False

        else:
            category, wd_enabled = 'other', True

        if not wd_enabled or not is_weight or force_no_decay:
            decay_kind = 'none'
        elif id(p) in norm_param_ids:
            decay_kind = 'norm'
        else:
            decay_kind = 'matrix'
        _add_param(category, p, decay_kind)

    lr_scales = {
        'sgl':    getattr(config.train, 'lr_scale_sgl', 0.01),
        'cgl':    getattr(config.train, 'lr_scale_cgl', 1.0),
        'tgl': getattr(config.train, 'lr_scale_tgl', 1.0),
        'encoder': getattr(config.train, 'lr_scale_encoder', 0.001),
        'other':   1.0,
    }

    param_groups = []
    n_total = 0
    decay_values = {'matrix': wd, 'norm': norm_wd, 'none': 0.0}
    group_suffixes = {'matrix': 'wd', 'norm': 'norm_wd', 'none': 'no_wd'}
    for category in ['sgl', 'cgl', 'tgl', 'encoder', 'other']:
        if category not in groups:
            continue
        for decay_kind in ['matrix', 'norm', 'none']:
            params = groups[category][decay_kind]
            if not params:
                continue
            actual_wd = decay_values[decay_kind]
            actual_lr = config.train.lr * lr_scales[category]
            param_groups.append({
                'params': params,
                'weight_decay': actual_wd,
                'lr': actual_lr,
                'name': f'{category}_{group_suffixes[decay_kind]}',
                'grouping_version': 4,
            })
            n = sum(p.numel() for p in params)
            n_total += n
            decay_label = group_suffixes[decay_kind]
            print(
                f"  [{category:8s} | {decay_label:7s}={actual_wd:.3f} | "
                f"lr={actual_lr:.2e}] {n:>10,} params"
            )

    optimizer = torch.optim.AdamW(
        param_groups,
        lr=config.train.lr,
        betas=tuple(config.train.adam_betas),
        eps=config.train.adam_eps,
    )

    switches_str = f"[{','.join(str(int(s)) for s in wd_sw.switches)}]"
    print(f"[Optimizer] Total: {n_total:,} params, base_lr={config.train.lr:.2e}")
    print(f"[Optimizer] Weight decay: matrix={wd:.6f}, LayerNorm={norm_wd:.6f}")
    print(f"[Optimizer] WD switches: {switches_str}")
    print(f"[Optimizer] LR scales: sgl={lr_scales['sgl']}, cgl={lr_scales['cgl']}, "
          f"tgl={lr_scales['tgl']}, encoder={lr_scales['encoder']}")

    return optimizer

def build_scheduler(optimizer, config, num_training_steps):

    if config.train.warmup_steps > 0:
        warmup_steps = config.train.warmup_steps
    elif config.train.warmup_ratio > 0:
        warmup_steps = int(config.train.warmup_ratio * num_training_steps)
    else:
        warmup_steps = 0

    if config.train.min_lr > 0:
        min_lr_ratio = config.train.min_lr / config.train.lr
    else:
        min_lr_ratio = 0.0

    scheduler_type = config.train.scheduler

    def lr_lambda(step):
        if step < warmup_steps:
            return step / max(1, warmup_steps)

        progress = (step - warmup_steps) / max(1, num_training_steps - warmup_steps)

        if scheduler_type == "cosine":
            multiplier = 0.5 * (1.0 + math.cos(math.pi * progress))
        elif scheduler_type == "linear":
            multiplier = 1.0 - (1.0 - min_lr_ratio) * progress
        elif scheduler_type == "step":

            if progress < 0.5:
                multiplier = 1.0
            elif progress < 0.75:
                multiplier = 0.5
            else:
                multiplier = 0.25
        else:  # "constant"
            multiplier = 1.0

        return max(min_lr_ratio, multiplier)

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

def _amp_autocast(config, device):
    device_type = torch.device(device).type
    if device_type != 'cuda':
        return nullcontext()
    if config.train.fp16 and config.train.bf16:
        raise ValueError("train.fp16 and train.bf16 cannot both be true")
    if config.train.bf16:
        supported = getattr(torch.cuda, 'is_bf16_supported', lambda: False)()
        if not supported:
            raise RuntimeError("BF16 requested but the current CUDA device does not support BF16")
        return torch.amp.autocast('cuda', dtype=torch.bfloat16)
    if config.train.fp16:
        return torch.amp.autocast('cuda', dtype=torch.float16)
    return nullcontext()

def build_grad_scaler(config):
    if not torch.cuda.is_available() or not config.train.fp16 or config.train.bf16:
        return None
    kwargs = {
        'init_scale': float(getattr(config.train, 'amp_init_scale', 4096.0)),
        'growth_interval': int(getattr(config.train, 'amp_growth_interval', 8000)),
        'backoff_factor': float(getattr(config.train, 'amp_backoff_factor', 0.5)),
    }

    try:
        return torch.amp.GradScaler('cuda', **kwargs)
    except (AttributeError, TypeError):
        from torch.cuda.amp import GradScaler as CudaGradScaler
        return CudaGradScaler(**kwargs)

def _tensor_list_l2_norm(tensors):
    if not tensors:
        return torch.tensor(0.0)
    detached = [t.detach() for t in tensors]
    try:

        norms = [n.float() for n in torch._foreach_norm(detached, 2.0)]
    except (RuntimeError, TypeError):
        norms = [torch.linalg.vector_norm(t.float()) for t in detached]
    if not norms:
        return torch.tensor(0.0)
    return torch.linalg.vector_norm(torch.stack(norms))

def _locate_nonfinite(model, use_grad):
    bad = []
    if model is None:
        return bad
    for name, param in model.named_parameters():
        tensor = param.grad if use_grad else param.data
        if not param.requires_grad or tensor is None:
            continue
        if not torch.isfinite(tensor).all().item():
            n_nan = torch.isnan(tensor).sum().item()
            n_inf = torch.isinf(tensor).sum().item()
            bad.append(f"{name} (NaN={n_nan}/{tensor.numel()}, Inf={n_inf}/{tensor.numel()})")
    return bad

def _optimizer_step(optimizer, scheduler, scaler, trainable_params, grad_clip_norm,
                    scaler_enabled, model=None, log_interval=100,
                    sync_nonfinite=False):
    _logger = logging.getLogger(__name__)
    attempt = getattr(optimizer, '_lauer_attempt', 0) + 1
    optimizer._lauer_attempt = attempt
    old_scale = float(scaler.get_scale()) if scaler_enabled and scaler is not None else None

    if scaler_enabled and scaler is not None:
        scaler.unscale_(optimizer)

    grads = [p.grad for p in trainable_params if p.grad is not None]
    if grads:
        max_norm = float(grad_clip_norm) if grad_clip_norm and grad_clip_norm > 0 else float('inf')

        grad_norm_tensor = torch.nn.utils.clip_grad_norm_(
            trainable_params, max_norm, error_if_nonfinite=False,
        )
    else:
        device = trainable_params[0].device if trainable_params else 'cpu'
        grad_norm_tensor = torch.zeros((), device=device)

    # enum {0=ok, 1=inf, 2=nan} (isnan/isinf exclusive), one MAX
    # all_reduce in DDP, then read norm+code back in a single D2H.
    # Normal path: 2 syncs -> 1; non-finite path: 3 syncs -> 1.
    # norm itself is NOT reduced across ranks (log meaning kept).
    finite_flag = (
        torch.isnan(grad_norm_tensor).to(torch.int32) * 2
        + torch.isinf(grad_norm_tensor).to(torch.int32)
    )
    if sync_nonfinite:
        import torch.distributed as dist
        if dist.is_available() and dist.is_initialized() and dist.get_world_size() > 1:
            dist.all_reduce(finite_flag, op=dist.ReduceOp.MAX)
    _norm, _code = torch.stack([
        grad_norm_tensor.detach().to(torch.float32),
        finite_flag.to(torch.float32),
    ]).tolist()
    grad_norm = float(_norm)
    finite_code = int(_code)
    has_nonfinite = finite_code > 0

    if has_nonfinite:
        bad_params = _locate_nonfinite(model, use_grad=True)

        grad_is_nan = (finite_code == 2)
        status_type = 'grad_nonfinite' if grad_is_nan else (
            'amp_overflow' if scaler_enabled and scaler is not None else 'grad_nonfinite')
        if scaler_enabled and scaler is not None and status_type == 'amp_overflow':

            backoff = scaler.get_backoff_factor() if hasattr(scaler, 'get_backoff_factor') else 0.5
            scaler.update(new_scale=old_scale * backoff)
        new_scale = float(scaler.get_scale()) if scaler_enabled and scaler is not None else None
        optimizer.zero_grad(set_to_none=True)
        return {
            'type': status_type,
            'params': bad_params,
            'grad_norm': grad_norm,
            'old_scale': old_scale,
            'new_scale': new_scale,
            'attempt': attempt,
        }

    if scaler_enabled and scaler is not None:
        scaler.step(optimizer)
        scaler.update()
    else:
        optimizer.step()

    params = [p.data for p in trainable_params]
    weight_norm_tensor = _tensor_list_l2_norm(params)
    weight_norm = float(weight_norm_tensor.item())
    weight_bad = not math.isfinite(weight_norm)
    bad_params = _locate_nonfinite(model, use_grad=False) if weight_bad else []

    if not weight_bad:
        scheduler.step()
    optimizer.zero_grad(set_to_none=True)

    new_scale = float(scaler.get_scale()) if scaler_enabled and scaler is not None else None
    if log_interval > 0 and attempt % log_interval == 0:
        scale_msg = f" scale={new_scale:.1f}" if new_scale is not None else ""

        print(f"[Numerics] optimizer_step={attempt} grad_norm={grad_norm:.4e}{scale_msg}", flush=True)

    return {
        'type': 'weight_nonfinite' if weight_bad else 'ok',
        'params': bad_params,
        'grad_norm': grad_norm,
        'old_scale': old_scale,
        'new_scale': new_scale,
        'attempt': attempt,
    }

def _renormalize_accumulated_grads(trainable_params, nominal_count, valid_count):
    if valid_count <= 0 or nominal_count <= 0 or valid_count == nominal_count:
        return
    factor = float(nominal_count) / float(valid_count)
    for p in trainable_params:
        if p.grad is not None:
            p.grad.mul_(factor)

def _handle_optimizer_status(status, nan_mgr, logger):
    nan_mgr.last_optimizer_status = dict(status)
    status_type = status['type']
    if status_type == 'ok':
        nan_mgr.register_clean_optimizer_step()
        return True

    params = status.get('params') or []
    detail = "\n  ".join(params[:8]) if params else "<not localized>"
    if status_type == 'amp_overflow':
        nan_mgr.register_amp_overflow(params, module_name=status.get('nan_signal'))
        logger.warning(
            f"[AMP OVERFLOW] step={status['attempt']} grad_norm={status['grad_norm']} "
            f"scale={status['old_scale']}→{status['new_scale']} | skipped optimizer/scheduler step\n  {detail}"
        )
        if nan_mgr.is_fatal:
            logger.error(f"[NaN FATAL] {nan_mgr.diagnosis()}")
            return False
        return True

    if status_type == 'grad_nonfinite':
        nan_mgr.register_grad_nan(params)
        logger.error(f"[NaN TRACE] Non-finite gradients before optimizer.step:\n  {detail}")
    elif status_type == 'weight_nonfinite':
        nan_mgr.register_weight_nan(params)
        logger.error(f"[NaN FATAL] Non-finite weights after optimizer.step:\n  {detail}")
    logger.error(f"[NaN FATAL] {nan_mgr.diagnosis()}")
    return False

_async_save_lock = threading.Lock()
_async_save_thread = None

def _to_cpu_snapshot(obj):
    if isinstance(obj, torch.Tensor):
        return obj.detach().to('cpu')
    if isinstance(obj, dict):
        return {k: _to_cpu_snapshot(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return type(obj)(_to_cpu_snapshot(v) for v in obj)
    return obj

def _async_save(path, obj, sync=False):
    global _async_save_thread
    if sync:
        torch.save(obj, path)
        return
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp_path = path + '.tmp'
    if torch.cuda.is_available():
        torch.cuda.synchronize()

    obj = _to_cpu_snapshot(obj)

    def _do():
        torch.save(obj, tmp_path)
        os.replace(tmp_path, path)

    with _async_save_lock:
        if _async_save_thread is not None and _async_save_thread.is_alive():
            _async_save_thread.join()
        _async_save_thread = threading.Thread(target=_do, daemon=True)
        _async_save_thread.start()

def wait_async_save():
    global _async_save_thread
    with _async_save_lock:
        if _async_save_thread is not None and _async_save_thread.is_alive():
            _async_save_thread.join()

def _validation_marker_path(checkpoint_path):
    return checkpoint_path + '.validated'

def _read_validation_marker(checkpoint_path):
    try:
        with open(_validation_marker_path(checkpoint_path), 'r') as f:
            return int(f.read().strip())
    except (OSError, ValueError):
        return None

def _write_validation_marker(checkpoint_path, epoch):
    marker = _validation_marker_path(checkpoint_path)
    tmp = marker + '.tmp'
    try:
        with open(tmp, 'w') as f:
            f.write(str(epoch))
        os.replace(tmp, marker)
    except OSError:
        pass

def _flush_log_handlers(logger):
    for _h in getattr(logger, 'handlers', []):
        try:
            _h.flush()
        except Exception:
            pass

def save_model(model, path, epoch=None, metrics=None, async_save=True):
    os.makedirs(os.path.dirname(path), exist_ok=True)

    artifact = build_stage_artifact(model, epoch=epoch, metrics=metrics)
    if async_save:
        _async_save(path, artifact)
    else:
        torch.save(artifact, path)

def save_checkpoint(model, optimizer, scheduler, scaler, epoch, best_loss, path,
                   best_composite=None, best_val_loss=None, epochs_no_improve=0,
                   nan_diagnostics=None, data_generators=None, async_save=True):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    ckpt = build_stage_artifact(
        model,
        epoch=epoch,
        metrics={
            'best_loss': best_loss,
            'best_composite': best_composite if best_composite is not None else -float('inf'),
            'best_val_loss': best_val_loss if best_val_loss is not None else float('inf'),
        },
    )
    ckpt.update({
        'artifact_type': 'training_checkpoint',
        'mode': model.mode,
        'best_loss': best_loss,
        'best_composite': best_composite if best_composite is not None else -float('inf'),
        'best_val_loss': best_val_loss if best_val_loss is not None else float('inf'),
        'epochs_no_improve': epochs_no_improve,
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict(),
    })
    if nan_diagnostics is not None:
        ckpt['nan_diagnostics'] = nan_diagnostics
    ckpt['rng_state'] = capture_rng_state(data_generators)
    if scaler is not None:
        ckpt['scaler_state_dict'] = scaler.state_dict()
    _async_save(path, ckpt, sync=not async_save)

def _load_model_state_checked(model, state_dict, checkpoint_path,
                              allow_partial=False, logger=None):
    current = model.state_dict()
    current_keys = set(current)
    saved_keys = set(state_dict)

    missing = sorted(
        key for key in current_keys - saved_keys
        if not key.startswith(FROZEN_EXCLUDED_PREFIXES)
    )

    unexpected = sorted(
        key for key in saved_keys - current_keys
        if not key.startswith(LEGACY_IGNORABLE_PREFIXES)
    )
    shape_mismatches = sorted(
        (key, tuple(state_dict[key].shape), tuple(current[key].shape))
        for key in current_keys & saved_keys
        if tuple(state_dict[key].shape) != tuple(current[key].shape)
    )

    incompatible = bool(shape_mismatches or (not allow_partial and (missing or unexpected)))
    if incompatible:
        lines = [
            f"Checkpoint architecture is incompatible with the current config: {checkpoint_path}",
            "Resume requires the exact projection/fusion/tgl architecture that created the checkpoint.",
        ]
        if shape_mismatches:
            lines.append("Shape mismatches (checkpoint -> current):")
            lines.extend(
                f"  {key}: {saved_shape} -> {current_shape}"
                for key, saved_shape, current_shape in shape_mismatches[:8]
            )
        if missing:
            lines.append(f"Missing keys: {len(missing)} (first: {missing[:5]})")
        if unexpected:
            lines.append(f"Unexpected keys: {len(unexpected)} (first: {unexpected[:5]})")
        lines.append(
            "Use the original YAML with an explicit --resume path, or start a new run with "
            "--no_auto_resume / a new run_name. Do not partially resume optimizer state across architectures."
        )
        raise RuntimeError("\n".join(lines))

    loadable = {
        key: value for key, value in state_dict.items()
        if not key.startswith(LEGACY_IGNORABLE_PREFIXES)
    }

    if any(k.startswith(FROZEN_EXCLUDED_PREFIXES)
           for k in current_keys - set(loadable)):
        for key, value in current.items():
            if key.startswith(FROZEN_EXCLUDED_PREFIXES):
                loadable.setdefault(key, value)
    if allow_partial:
        result = model.load_state_dict(loadable, strict=False)
        if logger and (result.missing_keys or result.unexpected_keys):
            logger.warning(
                f"[Checkpoint] Partial legacy load: missing={len(result.missing_keys)}, "
                f"unexpected={len(result.unexpected_keys)}; optimizer state will be reset."
            )
        return result
    return model.load_state_dict(loadable, strict=True)

def load_checkpoint(model, optimizer, scheduler, scaler, path, device, logger,
                    data_generators=None):
    ckpt = torch.load(path, map_location='cpu')
    validate_resume_artifact(ckpt, model, path)
    nan_diagnostics = ckpt.get('nan_diagnostics')

    was_remapped = False
    _load_model_state_checked(
        model, ckpt['model_state_dict'], path,
        allow_partial=False, logger=logger,
    )

    if not was_remapped:
        ckpt_pg = ckpt.get('optimizer_state_dict', {}).get('param_groups', [])
        cur_pg = optimizer.param_groups
        opt_incompat = (len(ckpt_pg) != len(cur_pg))
        if not opt_incompat:
            for cg, ng in zip(ckpt_pg, cur_pg):
                if (len(cg['params']) != len(ng['params']) or
                        cg.get('grouping_version') != ng.get('grouping_version')):
                    opt_incompat = True
                    break
        if opt_incompat:
            saved_versions = sorted({
                pg.get('grouping_version', 'legacy') for pg in ckpt_pg
            }, key=str)
            current_versions = sorted({
                pg.get('grouping_version', 'legacy') for pg in cur_pg
            }, key=str)
            was_remapped = True
            logger.warning(
                "[Optimizer Migration] Checkpoint param_group structure is incompatible "
                f"(saved version={saved_versions}, current version={current_versions}). "
                "Model weights were loaded, optimizer state will be reset, and scheduler "
                "progress will be preserved."
            )
    if was_remapped:

        if 'scheduler_state_dict' in ckpt:
            saved_last_epoch = ckpt['scheduler_state_dict'].get('last_epoch', 0)
            scheduler.last_epoch = saved_last_epoch
            logger.info(
                f"[Checkpoint] Scheduler last_epoch restored to {saved_last_epoch}. "
                f"LR schedule continues from step {saved_last_epoch}."
            )

        for i, pg in enumerate(optimizer.param_groups):
            pg['lr'] = scheduler.base_lrs[i] * scheduler.lr_lambdas[i](scheduler.last_epoch)

        if scaler is not None and 'scaler_state_dict' in ckpt:
            scaler.load_state_dict(ckpt['scaler_state_dict'])
        logger.warning(
            "[Checkpoint] Compatibility migration applied. Model weights loaded; "
            "optimizer state reset while scheduler/scaler state is preserved."
        )
        start_epoch = ckpt['epoch'] + 1
        best_loss = ckpt['best_loss']
        best_composite = ckpt.get('best_composite', -float('inf'))
        best_val_loss = ckpt.get('best_val_loss', float('inf'))
        epochs_no_improve = ckpt.get('epochs_no_improve', 0)
        restore_rng_state(ckpt.get('rng_state'), data_generators, logger)
        del ckpt
        if device.type == 'cuda':
            torch.cuda.empty_cache()
        return (start_epoch, best_loss, best_composite, best_val_loss,
                epochs_no_improve, nan_diagnostics)
    optimizer.load_state_dict(ckpt['optimizer_state_dict'])
    scheduler.load_state_dict(ckpt['scheduler_state_dict'])
    if scaler is not None and 'scaler_state_dict' in ckpt:
        scaler.load_state_dict(ckpt['scaler_state_dict'])
    start_epoch = ckpt['epoch'] + 1
    best_loss = ckpt['best_loss']
    best_composite = ckpt.get('best_composite', -float('inf'))
    best_val_loss = ckpt.get('best_val_loss', float('inf'))
    epochs_no_improve = ckpt.get('epochs_no_improve', 0)
    restore_rng_state(ckpt.get('rng_state'), data_generators, logger)
    del ckpt
    if device.type == 'cuda':
        torch.cuda.empty_cache()
    logger.info(f"Resumed from epoch {start_epoch}, best_loss={best_loss:.4f}"
                f", best_composite={best_composite:.4f}")
    return (start_epoch, best_loss, best_composite, best_val_loss,
            epochs_no_improve, nan_diagnostics)

def _val(v, default=0.0):
    if isinstance(v, torch.Tensor):
        return v.item()
    return float(v) if v is not None else default

PROGRESS_REFRESH_EVERY = 10

def _accumulate_loss_details(accumulator, details, device):
    for key, value in details.items():
        if isinstance(value, torch.Tensor):
            scalar = value.detach().float()
        elif isinstance(value, (int, float)):
            scalar = torch.tensor(float(value), device=device)
        else:
            continue
        accumulator[key] = accumulator.get(key, 0.0) + scalar

def _average_loss_details(accumulator, count):
    divisor = max(count, 1)
    return {key: _val(value / divisor) for key, value in accumulator.items()}

def _release_dataloader_workers(loader):
    if loader is None:
        return
    it = getattr(loader, '_iterator', None)
    if it is None:
        return
    try:
        shutdown = getattr(it, '_shutdown_workers', None)
        if shutdown is not None:
            shutdown()
    except Exception:
        pass

    try:
        loader._iterator = None
    except Exception:
        pass

def _maybe_release_train_workers_before_validation(config, train_loader, logger=None):
    tl = getattr(getattr(config, 'data', None), 'train_loader', None)
    flag = getattr(tl, 'release_workers_before_val', True)
    if flag is None:
        flag = True
    if flag:
        _release_dataloader_workers(train_loader)
        if logger is not None:
            logger.info("[DataLoader] Released train workers before validation.")
    else:
        if logger is not None:
            logger.info("[DataLoader] train workers stay alive during "
                        "validation (release_workers_before_val=false).")

def train_one_epoch_sgl(model, dataloader, optimizer, scheduler, scaler, device, config, epoch,
                         trainable_params, nan_mgr=None):
    model.train()
    total_loss = 0.0
    loss_accum = {}
    num_steps = len(dataloader)
    pbar = tqdm(dataloader, desc=f"Train {epoch + 1:03d}/{config.train.epochs:03d}", unit="batch")
    accum = getattr(config.train, 'accumulation_steps', 1) or 1
    scaler_enabled = scaler is not None
    valid_steps = 0
    valid_in_group = 0
    real_accum = 1

    if nan_mgr is None:
        nan_mgr = NaNResilienceManager()

    for i, batch in enumerate(pbar):
        if (i % accum) == 0:
            real_accum = min(accum, num_steps - i)
            valid_in_group = 0

        with _amp_autocast(config, device):
            ecg_signal = batch['ecg_signal'].to(device, non_blocking=True)
            ecg_image = batch['ecg_image'].to(device, non_blocking=True)
            text_ids = batch['text_input_ids'].to(device, non_blocking=True)
            text_mask = batch['text_attention_mask'].to(device, non_blocking=True)

            aug_sig = batch.get('ecg_signal_aug')
            aug_img = batch.get('ecg_image_aug')
            if aug_sig is not None:
                aug_sig = aug_sig.to(device, non_blocking=True)
            if aug_img is not None:
                aug_img = aug_img.to(device, non_blocking=True)
            loss, loss_dict, _ = model(
                ecg_signal, ecg_image, text_ids, text_mask,
                ecg_signal_aug=aug_sig, ecg_image_aug=aug_img,
            )

        loss_value = float(loss.detach().float().item())
        if not math.isfinite(loss_value):

            nan_accum = getattr(model, '_nan_accum', None)
            if nan_accum:
                sources = [k for k, v in nan_accum.items()
                           if v is not None and not bool(v.item())]
                if sources:
                    model._nan_signal = '+'.join(sources)
            nan_source = getattr(model, '_nan_signal', None)
            if nan_source is not None:
                model._nan_signal = None

            logger = logging.getLogger(__name__)
            logger.warning(_format_nan_batch_log(loss, loss_dict, nan_source, epoch, i))

            nan_mgr.register_nan_batch(loss_dict, module_name=nan_source, step=i)

            if nan_mgr.is_fatal:
                logger.error(f"[NaN FATAL] {nan_mgr.diagnosis()}")
                optimizer.zero_grad()
                return None, None, nan_mgr

            if nan_mgr.action == 'reduce_lr':
                lr_changes = nan_mgr.reduce_lr(optimizer, scheduler)
                logger.warning(
                    f"[NaN ESCALATE] {nan_mgr.max_consecutive_skip} consecutive NaN. "
                    f"LR reduced: {lr_changes[0][0]:.2e} → {lr_changes[0][1]:.2e}. "
                    f"First NaN: {nan_mgr.first_nan_module}. Resuming..."
                )

            optimizer.zero_grad(set_to_none=True)
            valid_in_group = 0
            continue

        nan_mgr.register_clean_batch()

        if scaler_enabled:
            scaler.scale(loss / real_accum).backward()
        else:
            (loss / real_accum).backward()
        valid_in_group += 1

        if ((i + 1) % accum == 0) or ((i + 1) == num_steps):
            _renormalize_accumulated_grads(
                trainable_params, real_accum, valid_in_group
            )
            status = _optimizer_step(
                optimizer, scheduler, scaler, trainable_params,
                config.train.grad_clip_norm, scaler_enabled, model=model,
                log_interval=getattr(config.train, 'nan_log_interval', 100),
            )
            status.update({'epoch': epoch, 'batch_step': i,
                           'sample_ids': batch.get('sample_id', batch.get('sample_ids')),
                           'nan_signal': getattr(model, '_nan_signal', None)})
            model._nan_signal = None
            if not _handle_optimizer_status(status, nan_mgr, logging.getLogger(__name__)):
                return None, None, nan_mgr

        total_loss += loss_value
        valid_steps += 1
        _accumulate_loss_details(loss_accum, loss_dict, device)

        if (i % PROGRESS_REFRESH_EVERY == 0) or ((i + 1) == num_steps):
            postfix = build_progress_postfix(
                loss_value, loss_dict, 'sgl',
                max(pg['lr'] for pg in optimizer.param_groups),
            )
            pbar.set_postfix(postfix)

    avg_loss = total_loss / max(valid_steps, 1)
    avg_details = _average_loss_details(loss_accum, valid_steps)
    return avg_loss, avg_details, nan_mgr

def train_one_epoch_full(model, dataloader, optimizer, scheduler, scaler, device, config, epoch, trainable_params, nan_mgr=None):
    model.train()
    total_loss = 0.0
    loss_accum = {}
    num_steps = len(dataloader)
    pbar = tqdm(dataloader, desc=f"Train {epoch + 1:03d}/{config.train.epochs:03d}", unit="batch")
    accum = getattr(config.train, 'accumulation_steps', 1) or 1
    scaler_enabled = scaler is not None
    valid_steps = 0
    valid_in_group = 0
    real_accum = 1

    if nan_mgr is None:
        nan_mgr = NaNResilienceManager()

    for i, batch in enumerate(pbar):
        if (i % accum) == 0:
            real_accum = min(accum, num_steps - i)
            valid_in_group = 0

        ecg_signal = batch['ecg_signal'].to(device, non_blocking=True)
        ecg_image = batch['ecg_image'].to(device, non_blocking=True)
        text_input_ids = batch['text_input_ids'].to(device, non_blocking=True)
        text_attention_mask = batch['text_attention_mask'].to(device, non_blocking=True)
        tgl_text_ids = batch.get('tgl_text_ids')
        if tgl_text_ids is not None:
            tgl_text_ids = tgl_text_ids.to(device, non_blocking=True)
        labels = batch.get('labels')
        if labels is not None:
            labels = labels.to(device)
        aug_sig = batch.get('ecg_signal_aug')
        aug_img = batch.get('ecg_image_aug')
        if aug_sig is not None:
            aug_sig = aug_sig.to(device, non_blocking=True)
        if aug_img is not None:
            aug_img = aug_img.to(device, non_blocking=True)

        with _amp_autocast(config, device):
            loss, loss_dict, _ = model(ecg_signal, ecg_image, text_input_ids, text_attention_mask,
                                       labels=labels, tgl_text_ids=tgl_text_ids,
                                       ecg_signal_aug=aug_sig, ecg_image_aug=aug_img)

        loss_value = float(loss.detach().float().item())
        if not math.isfinite(loss_value):

            nan_accum = getattr(model, '_nan_accum', None)
            if nan_accum:
                sources = [k for k, v in nan_accum.items()
                           if v is not None and not bool(v.item())]
                if sources:
                    model._nan_signal = '+'.join(sources)
            nan_source = getattr(model, '_nan_signal', None)
            if nan_source is not None:
                model._nan_signal = None

            logger = logging.getLogger(__name__)
            logger.warning(_format_nan_batch_log(loss, loss_dict, nan_source, epoch, i))

            nan_mgr.register_nan_batch(loss_dict, module_name=nan_source, step=i)

            if nan_mgr.is_fatal:
                logger.error(f"[NaN FATAL] {nan_mgr.diagnosis()}")
                optimizer.zero_grad()
                return None, None, nan_mgr

            if nan_mgr.action == 'reduce_lr':
                lr_changes = nan_mgr.reduce_lr(optimizer, scheduler)
                logger.warning(
                    f"[NaN ESCALATE] {nan_mgr.max_consecutive_skip} consecutive NaN. "
                    f"LR reduced: {lr_changes[0][0]:.2e} → {lr_changes[0][1]:.2e}. "
                    f"First NaN: {nan_mgr.first_nan_module}. Resuming..."
                )

            optimizer.zero_grad(set_to_none=True)
            valid_in_group = 0
            continue

        nan_mgr.register_clean_batch()

        if scaler_enabled:
            scaler.scale(loss / real_accum).backward()
        else:
            (loss / real_accum).backward()
        valid_in_group += 1

        if ((i + 1) % accum == 0) or ((i + 1) == num_steps):
            _renormalize_accumulated_grads(
                trainable_params, real_accum, valid_in_group
            )
            status = _optimizer_step(
                optimizer, scheduler, scaler, trainable_params,
                config.train.grad_clip_norm, scaler_enabled, model=model,
                log_interval=getattr(config.train, 'nan_log_interval', 100),
            )
            status.update({'epoch': epoch, 'batch_step': i,
                           'sample_ids': batch.get('sample_id', batch.get('sample_ids')),
                           'nan_signal': getattr(model, '_nan_signal', None)})
            model._nan_signal = None
            if not _handle_optimizer_status(status, nan_mgr, logging.getLogger(__name__)):
                return None, None, nan_mgr

        total_loss += loss_value
        valid_steps += 1
        _accumulate_loss_details(loss_accum, loss_dict, device)

        if (i % PROGRESS_REFRESH_EVERY == 0) or ((i + 1) == num_steps):
            postfix = build_progress_postfix(
                loss_value, loss_dict, model.mode,
                max(pg['lr'] for pg in optimizer.param_groups),
            )
            pbar.set_postfix(postfix)

    avg_loss = total_loss / max(valid_steps, 1)
    avg_details = _average_loss_details(loss_accum, valid_steps)
    return avg_loss, avg_details, nan_mgr

@torch.no_grad()
def validate_sgl(model, dataloader, device, logger, config=None):
    if dataloader is None:
        return {}
    model.eval()
    results = zero_shot_retrieval_eval(
        model, dataloader, device,
        bf16=bool(config and config.train.bf16),
        fp16=bool(config and config.train.fp16),
    )
    logger.info(f"\n[Zero-Shot]\n{format_eval_results(results)}")
    model.train()
    return results

@torch.no_grad()
def validate_full(model, dataloader, device, logger, config=None):
    if dataloader is None:
        return {}
    model.eval()

    # Zero-shot retrieval
    bf16 = bool(config and config.train.bf16)
    fp16 = bool(config and config.train.fp16)
    results = zero_shot_retrieval_eval(
        model, dataloader, device, bf16=bf16, fp16=fp16,
    )
    logger.info(f"\n[Zero-Shot]\n{format_eval_results(results)}")

    cgl_results = cgl_accuracy_eval(
        model, dataloader, device, bf16=bf16, fp16=fp16,
    )
    logger.info(
        f"[CGL] Fusion Match Acc={cgl_results['acc_matching']:.4f}"
    )

    model.train()
    return {'zero_shot': results, 'cgl': cgl_results}

@torch.no_grad()
def compute_val_loss(model, dataloader, device, config, is_full):

    if dataloader is None or len(dataloader) == 0:
        return float('inf'), {}, None, None

    model.eval()

    total_loss = 0.0
    loss_accum = {}
    mlm_weighted_sum = 0.0
    mlm_total_sum = 0
    nan_val_count = 0
    total_val_batches = len(dataloader)
    all_embeddings = {'ecg_embed': [], 'image_embed': [], 'text_embed': []}

    mlc_logits_list = []
    mlc_labels_list = []

    def _forward_val_batch(batch):
        with _amp_autocast(config, device):
            labels = batch.get('labels')
            if labels is not None:
                labels = labels.to(device)
            tgl_text_ids = batch.get('tgl_text_ids')
            if tgl_text_ids is not None:
                tgl_text_ids = tgl_text_ids.to(device, non_blocking=True)
            return model(
                batch['ecg_signal'].to(device, non_blocking=True),
                batch['ecg_image'].to(device, non_blocking=True),
                batch['text_input_ids'].to(device, non_blocking=True),
                batch['text_attention_mask'].to(device, non_blocking=True),
                labels=labels,
                tgl_text_ids=tgl_text_ids,
            )

    for batch in tqdm(dataloader, desc="Validating", unit="batch",
                      leave=False, dynamic_ncols=True):
        loss, loss_dict, embeddings = _forward_val_batch(batch)

        loss_value = float(loss.detach().float().item())
        if not math.isfinite(loss_value):
            nan_val_count += 1

            if hasattr(model, '_nan_signal'):
                model._nan_signal = None
            continue

        if is_full and getattr(model, 'query_guided_tracing_head', None) is not None:
            b_labels = batch.get('labels')
            if b_labels is not None:
                mlc_lg = getattr(model, '_last_mlc_logits', None)
                if mlc_lg is not None:
                    mlc_logits_list.append(mlc_lg.detach().float().cpu())
                    mlc_labels_list.append(b_labels.detach().float().cpu())

        total_loss += loss_value
        for key in all_embeddings:
            all_embeddings[key].append(embeddings[key].detach().float().cpu())
        for k, v in loss_dict.items():
            if isinstance(v, torch.Tensor):
                metric = v.detach().float()
            elif isinstance(v, (int, float)):
                metric = torch.tensor(float(v), device=device)
            else:
                continue
            loss_accum[k] = loss_accum.get(k, 0.0) + metric

        if 'loss_mlm' in loss_dict:
            mlm_t = int(loss_dict.get('mlm_total', 0))
            mlm_weighted_sum += loss_dict['loss_mlm'].detach().float() * max(mlm_t, 1)
            mlm_total_sum += mlm_t

    if nan_val_count > 0:
        _logger = logging.getLogger(__name__)
        _logger.warning(
            f"[Val NaN] {nan_val_count}/{total_val_batches} batches NaN. "
            f"Val metrics may be unreliable."
        )
        if nan_val_count > total_val_batches * 0.5:
            _logger.error(
                f"[Val NaN] >50% batches NaN ({nan_val_count}/{total_val_batches}). "
                f"Skipping validation this epoch."
            )
            return float('inf'), {}, None, None

    n = max(total_val_batches - nan_val_count, 1)
    avg_losses = {k: _val(v / n) for k, v in loss_accum.items()}
    if mlm_total_sum > 0:
        avg_losses['loss_mlm'] = _val(mlm_weighted_sum / mlm_total_sum)
    compact_embeddings = {
        key: torch.cat(parts, dim=0) for key, parts in all_embeddings.items()
    }

    mlc_metrics = None
    if mlc_logits_list:
        from evaluation.utils import multilabel_metrics
        mlc_metrics = multilabel_metrics(
            torch.cat(mlc_logits_list, 0), torch.cat(mlc_labels_list, 0),
        )
    return total_loss / n, avg_losses, compact_embeddings, mlc_metrics

@torch.no_grad()
def _run_ablation(model, val_loader, device, loss_details, logger):
    ar_mode = bool(getattr(model, 'loss_mode_ar', False))
    zero_correct, zero_total = 0, 0

    for batch in val_loader:

        sgl_ids = batch.get('text_input_ids')
        if sgl_ids is None:
            continue
        sgl_ids = sgl_ids.to(device) if isinstance(sgl_ids, torch.Tensor) else sgl_ids

        ecg_signal = batch['ecg_signal'].to(device)
        ecg_image = batch['ecg_image'].to(device)
        text_mask = batch['text_attention_mask'].to(device)
        ecg_tokens, image_tokens, text_tokens, _ = model._encode_all(
            ecg_signal, ecg_image, sgl_ids, text_mask,
        )
        _, (ecg_proj, img_proj, _) = model.adapters.forward_combined(
            ecg_tokens, image_tokens, text_tokens, text_mask,
        )

        # CGL fusion (single forward, no triplet needed for ablation)
        if model._use_alternating_fusion:
            fused = model._run_alternating_fusion(ecg_proj, img_proj)
            fused_si_pos = fused[:, 0, :].unsqueeze(1)  # just CLS as 1-token cross_kv
        else:
            fused = model.fusion_si(primary=ecg_proj, cross_kv=img_proj)
            fused_si_pos = fused  # [B, N, 768]

        if ar_mode:

            dec_ids = batch.get('tgl_text_ids')
            if dec_ids is None:
                continue
            dec_ids = dec_ids.to(device) if isinstance(dec_ids, torch.Tensor) else dec_ids
            zc, zt = model.ablation_ar_acc(dec_ids, fused_si_pos, zero_cross_attn=True)
        else:
            zc, zt = model.ablation_mlm_acc(sgl_ids, fused_si_pos, zero_cross_attn=True)
        zero_correct += zc
        zero_total += zt

    if ar_mode:
        rep_t = int(loss_details.get('report_total', 0) or 0)
        expl_t = int(loss_details.get('expl_total', 0) or 0)
        normal_total = rep_t + expl_t
        normal_correct = (int(loss_details.get('report_correct', 0) or 0)
                          + int(loss_details.get('expl_correct', 0) or 0))
    else:
        normal_total = int(loss_details.get('mlm_total', 0) or 0)
        normal_correct = int(loss_details.get('mlm_correct', 0) or 0)

    if normal_total > 0:
        normal_acc = normal_correct / normal_total * 100
        zero_acc = zero_correct / zero_total * 100 if zero_total > 0 else 0.0
        gap = normal_acc - zero_acc
        metric = "AR Acc" if ar_mode else "MLM Acc"
        logger.info(
            f"[Ablation] Normal {metric}: {normal_acc:.1f}%  |  "
            f"Zero Cross-Attn: {zero_acc:.1f}%  |  Gap: {gap:.1f}%  "
            f"{'✅ ECG-dependent' if gap > 5 else '❌ memory-reliant'}"
        )
        return gap
    return None

@torch.no_grad()
def validate_epoch(model, val_loader, device, config, is_full, logger, epoch=0):
    if val_loader is None:
        return None

    model.eval()

    val_loss, loss_details, embeddings, mlc_metrics = compute_val_loss(
        model, val_loader, device, config, is_full,
    )

    if math.isnan(val_loss) or math.isinf(val_loss) or val_loss == float('inf'):
        logger.error(
            f"[Validate] Validation loss is NaN/Inf ({val_loss}). "
            f"Skipping evaluation this epoch to avoid corrupted metrics."
        )
        model.train()
        return None

    zs_results = retrieval_results_from_embeddings(
        embeddings['ecg_embed'], embeddings['image_embed'], embeddings['text_embed'],
        model.sgl_loss.logit_scale.exp(),
    )

    cgl_results = None
    if is_full and 'acc_matching' in loss_details:
        cgl_results = {
            key: value for key, value in loss_details.items()
            if key.startswith('acc_matching')
        }

    ablation_gap = None
    ablation_every = getattr(config.train, 'tgl_ablation_every_n', 0)
    if ablation_every > 0 and (epoch + 1) % ablation_every == 0:
        if hasattr(model, 'has_tgl') and model.has_tgl:
            ablation_gap = _run_ablation(model, val_loader, device, loss_details, logger)

    model.train()

    tgl_metrics = None
    ar_tokens = int(loss_details.get('ar_tokens', 0) or 0)
    ar_loss = float(loss_details.get('loss_ar', 0.0) or 0.0)
    if ar_tokens > 0:
        ar_ppl = math.exp(min(ar_loss, 20.0))
        tgl_metrics = {'ar_ppl': ar_ppl, 'ar_loss': ar_loss}
        mlm_total = int(loss_details.get('mlm_total', 0) or 0)
        if mlm_total > 0:
            tgl_metrics['mlm_accuracy'] = (
                float(loss_details.get('mlm_correct', 0.0) or 0.0) / mlm_total)
            tgl_metrics['mlm_loss'] = float(loss_details.get('loss_mlm', 0.0) or 0.0)

        rep_t = int(loss_details.get('report_total', 0) or 0)
        expl_t = int(loss_details.get('expl_total', 0) or 0)
        if rep_t + expl_t > 0:
            rep_c = float(loss_details.get('report_correct', 0.0) or 0.0)
            expl_c = float(loss_details.get('expl_correct', 0.0) or 0.0)
            tgl_metrics['ar_accuracy'] = (rep_c + expl_c) / (rep_t + expl_t)
            tgl_metrics['report_accuracy'] = rep_c / rep_t if rep_t > 0 else 0.0
            tgl_metrics['expl_accuracy'] = expl_c / expl_t if expl_t > 0 else 0.0

    #    SGL-only: composite = avg_r1 / 100.0
    #    SGL+CGL: composite = w_zs·(R@1/100) + w_cgl·avg_itm

    w_zs = getattr(config.train, 'composite_zeroshot_weight', 0.5)
    w_cgl = getattr(config.train, 'composite_cgl_weight', 0.5)
    w_mlm = getattr(config.train, 'composite_mlm_weight', 0.0)
    w_ar = getattr(config.train, 'composite_ar_weight', 0.0)
    w_mlc = getattr(config.train, 'composite_mlc_weight', 0.0)

    avg_r1 = float(np.mean([zs_results[d]['R@1'] for d in zs_results])) if zs_results else 0.0

    avg_itm = 0.0
    if is_full and cgl_results:
        cgl_accs = [v for k, v in cgl_results.items() if k.startswith('acc_matching') and isinstance(v, (int, float))]
        avg_itm = float(np.mean(cgl_accs)) if cgl_accs else 0.0

    mlm_component = 0.0
    ar_component = 0.0
    if tgl_metrics:
        if 'mlm_accuracy' in tgl_metrics:
            mlm_component = w_mlm * tgl_metrics['mlm_accuracy']
        ar_component = w_ar * (1.0 / (1.0 + tgl_metrics['ar_loss']))
    elif is_full and (w_mlm > 0 or w_ar > 0):
        logging.getLogger(__name__).warning(
            "[Composite] TGL metrics missing -- mlm/ar components treated as 0 "
            "(composite upper bound shrinks)."
        )

    mlc_component = 0.0
    if mlc_metrics is not None and w_mlc > 0:
        mlc_component = w_mlc * mlc_metrics['mlc_map']

    if not is_full:

        composite = avg_r1 / 100.0
    else:
        composite = (w_zs * (avg_r1 / 100.0) + w_cgl * avg_itm
                     + mlc_component + mlm_component + ar_component)

    return {
        'val_loss': val_loss,
        'loss_details': loss_details,
        'zero_shot': zs_results,
        'cgl': cgl_results,
        'composite': composite,
        'avg_r1': avg_r1,
        'avg_itm': avg_itm,
        'tgl_metrics': tgl_metrics,
        'multilabel': mlc_metrics,
    }

def _format_train_log(epoch, train_loss, train_details, config, mode, lr=None,
                      elapsed_seconds=None):
    return format_train_epoch(
        epoch, config.train.epochs, train_loss, train_details, config, mode,
        lr=lr, elapsed_seconds=elapsed_seconds,
    )

def _format_val_log(epoch, val_results, config, mode):
    return format_validation_epoch(
        epoch, config.train.epochs, val_results, config, mode,
    )

def main():
    parser = argparse.ArgumentParser(description='lauer Training')
    parser.add_argument('--mode', type=str, default='sgl',
                        choices=['sgl', 'sgl+cgl', 'sgl+cgl+tgl'],
                        help='Training mode: sgl | sgl+cgl | sgl+cgl+tgl')
    parser.add_argument('--config', type=str,
                        default='configs/pretrain_SGL.yaml')
    parser.add_argument('--data_root', type=str, default=None,
                        help='Paired-record root; defaults to data.data_root in the YAML')
    parser.add_argument('--output_dir', type=str, default='')
    parser.add_argument('--init-from', type=str, default='',
                        help='Completed predecessor-stage weights used to initialize a new stage')
    parser.add_argument('--allow-partial-init', action='store_true',
                        help='Allow a fresh start (no --init-from) or skipping intermediate '
                             'stages; missing modules stay randomly initialized')
    parser.add_argument('--batch_size', type=int, default=0)
    parser.add_argument('--epochs', type=int, default=0)
    parser.add_argument('--device', type=str,
                        default='cuda' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--resume', type=str, default='',
                        help='Resume from checkpoint.pt')
    parser.add_argument('--resume_lr_factor', type=float, default=1.0,
                        help='One-time LR multiplier applied after loading a resume checkpoint')
    parser.add_argument('--reset_amp_scaler', action='store_true',
                        help='Discard resumed FP16 GradScaler state and restart from config scale')
    parser.add_argument('--no_auto_resume', action='store_true',
                        help='Ignore checkpoint.pt in the output run directory and start a new run')
    args = parser.parse_args()

    init_from = args.init_from
    if args.resume and init_from:
        parser.error("--resume and --init-from are mutually exclusive")

    if not os.path.exists(args.config):
        raise FileNotFoundError(f"Config file not found: {args.config}")
    config = load_config_from_yaml(args.config)

    if args.data_root:
        config.data.data_root = args.data_root
    if args.output_dir:
        config.train.output_dir = args.output_dir
    if args.batch_size > 0:
        config.train.batch_size = args.batch_size
    if args.epochs > 0:
        config.train.epochs = args.epochs

    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    mode = args.mode
    has_cgl = (mode in ('sgl+cgl', 'sgl+cgl+tgl'))
    has_tgl = (mode == 'sgl+cgl+tgl')

    is_full = has_cgl
    expected_mode = getattr(config.train, 'expected_mode', '')
    if expected_mode and expected_mode != mode:
        raise RuntimeError(
            f"Config {args.config!r} is for mode {expected_mode!r}, "
            f"but --mode is {mode!r}. Use the matching stage YAML."
        )

    set_seed(
        config.train.seed,
        deterministic=config.train.deterministic,
        benchmark=config.train.benchmark,
    )
    run_dir = mode_output_dir(
        config.train.output_dir, config.train.run_name, mode
    )
    os.makedirs(run_dir, exist_ok=True)
    logger = setup_logging(run_dir)
    device = torch.device(args.device)

    log_run_configuration(
        logger, config, mode,
        config_path=args.config,
        data_root=config.data.data_root,
        run_dir=run_dir,
        device=device,
        init_from=init_from,
        resume=args.resume,
        auto_resume=not args.no_auto_resume,
        world_size=1,
    )
    log_runtime_environment(logger, device)

    from transformers import AutoTokenizer
    tokenizer_path = config.text_encoder.tokenizer_name_or_path
    if not tokenizer_path:
        logger.error("text_encoder.tokenizer_name_or_path not set!")
        return
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, local_files_only=True)
    with capture_startup_prints(logger, "BUILDING RAW DATALOADERS"):
        dataloaders = create_meeti_dataloaders(
            config.data, config.train, tokenizer,
            max_text_length=config.text_encoder.max_length,
            image_size=config.image_encoder.image_size,
            encoder_type=config.image_encoder.encoder_type,
            load_labels=has_cgl,

            include_tgl_ids=has_tgl,
        )

    train_loader = dataloaders.get('train')
    val_loader = dataloaders.get('val')
    test_loader = dataloaders.get('test')
    checkpoint_generators = loader_generators(dataloaders)
    log_dataloader_summary(logger, dataloaders, config, world_size=1)

    if train_loader is None:
        logger.error("No training data!")

        return 1

    auto_checkpoint = os.path.join(run_dir, 'checkpoint.pt')
    checkpoint_path = args.resume or (None if args.no_auto_resume else auto_checkpoint)
    checkpoint_exists = bool(checkpoint_path and os.path.exists(checkpoint_path))
    if args.resume and not checkpoint_exists:
        raise FileNotFoundError(f"Explicit --resume checkpoint not found: {args.resume}")

    if mode == 'sgl' and init_from:
        raise RuntimeError("SGL is the first stage and must not use --init-from")
    if has_cgl and not checkpoint_exists and not init_from:
        predecessor = 'sgl' if mode == 'sgl+cgl' else 'sgl+cgl'
        if not args.allow_partial_init:
            raise RuntimeError(
                f"Fresh {mode!r} training requires completed {predecessor!r} weights. "
                "Pass them with --init-from, or pass --allow-partial-init to start "
                "from scratch with all missing modules randomly initialized."
            )
        logger.warning(
            f"[Init] Fresh {mode!r} start via --allow-partial-init: no {predecessor!r} "
            "weights provided, all trainable modules (predecessor-stage adapters, "
            "fusion/heads, ViT encoders) start randomly initialized and train jointly "
            "(frozen MedCPT text encoder is still loaded from weights/)."
        )

    # Dataset/tokenizer construction must not shift the model initialization RNG.
    seed_rng(config.train.seed)
    with capture_startup_prints(logger, "BUILDING MODEL AND LOADING ENCODERS"):
        model = LaUER(config, mode=mode)
        model.load_pretrained_encoders(device=str(device))
        model = model.to(device)

    if hasattr(model, 'query_guided_tracing_head') and model.query_guided_tracing_head is not None:
        train_ds = getattr(train_loader, 'dataset', None)
        label_names = getattr(train_ds, 'label_names', None) if train_ds is not None else None
        if label_names:
            model.prepare_prompt_bank(list(label_names), device=str(device))
            logger.info(f"[MultiLabel] {len(label_names)} disease queries encoded & "
                        f"query encoder released "
                        f"({getattr(train_ds, 'annotations', None) is not None} annotated samples)")
        else:
            logger.warning("[MultiLabel] No annotation CSV labels found; "
                           "multi-label classification loss will be zero.")
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    if checkpoint_exists and init_from:
        logger.info(
            f"[Resume] Existing same-stage checkpoint selected; --init-from is not "
            f"re-applied: {checkpoint_path}"
        )
    elif init_from:
        load_stage_initialization(
            model,
            init_from,
            target_mode=mode,
            allow_partial=args.allow_partial_init,
            device='cpu',
            logger=logger,
        )

    with capture_startup_prints(logger, "BUILDING OPTIMIZER PARAMETER GROUPS"):
        optimizer = build_optimizer(model, config)

    module_lrs = {}
    for pg in optimizer.param_groups:
        cat = pg['name'].split('_')[0]  # e.g. "sgl_wd" → "sgl"
        if cat not in module_lrs and 'lr' in pg:
            module_lrs[cat] = pg['lr']
    lr_summary = '  '.join(f"{k}={v:.2e}" for k, v in module_lrs.items())
    n_trainable = sum(p.numel() for p in trainable_params)
    n_total = sum(p.numel() for p in model.parameters())
    logger.info(
        "\n[RUNTIME] TRAINABLE PARAMETER SUMMARY\n"
        f"  trainable_parameters={n_trainable:,}, total_parameters={n_total:,}, "
        f"trainable_ratio={n_trainable / max(n_total, 1) * 100:.2f}%\n"
        f"  module_learning_rates | base={config.train.lr:.3e}, {lr_summary}"
    )

    num_training_steps = config.train.epochs * math.ceil(len(train_loader) / getattr(config.train, 'accumulation_steps', 1))
    scheduler = build_scheduler(optimizer, config, num_training_steps)
    scaler = build_grad_scaler(config)

    start_epoch = 0
    best_loss = float('inf')
    best_composite = -float('inf')
    best_val_loss = float('inf')

    nan_mgr = NaNResilienceManager()

    if checkpoint_exists:
        logger.info(
            f"[Resume] Selected checkpoint: {checkpoint_path} "
            f"({'explicit' if args.resume else 'auto-discovered'})"
        )
        start_epoch, best_loss, best_composite, best_val_loss, ckpt_no_improve, nan_diagnostics =             load_checkpoint(model, optimizer, scheduler, scaler, checkpoint_path, device, logger,
                            data_generators=checkpoint_generators)

        nan_mgr.restore_from_state(nan_diagnostics)
        if args.resume_lr_factor <= 0:
            raise ValueError("--resume_lr_factor must be > 0")
        if args.resume_lr_factor != 1.0:
            for pg in optimizer.param_groups:
                pg['lr'] *= args.resume_lr_factor
            scheduler.base_lrs = [lr * args.resume_lr_factor for lr in scheduler.base_lrs]
            if hasattr(scheduler, '_last_lr'):
                scheduler._last_lr = [lr * args.resume_lr_factor for lr in scheduler._last_lr]
            logger.warning(
                f"[Resume] Applied one-time LR factor {args.resume_lr_factor:g}; "
                f"current max LR={max(pg['lr'] for pg in optimizer.param_groups):.2e}"
            )

        if args.reset_amp_scaler and scaler is not None:
            scaler = build_grad_scaler(config)
            logger.warning(
                f"[Resume] GradScaler reset to scale={scaler.get_scale():.1f} "
                f"growth_interval={getattr(config.train, 'amp_growth_interval', 8000)}"
            )
    elif args.resume_lr_factor != 1.0:

        logger.warning(
            f"[Resume] --resume_lr_factor={args.resume_lr_factor:g} ignored: "
            f"no checkpoint to resume."
        )

        logger.info(f"Restored LR: {max(pg['lr'] for pg in optimizer.param_groups):.2e}")
        if start_epoch >= config.train.epochs:
            logger.info("Training already complete. Running final eval.")

    if checkpoint_exists and val_loader is not None:
        _val_every = getattr(config.train, 'val_every_n_epochs', 1) or 1
        trained_epoch = start_epoch - 1
        validated_epoch = _read_validation_marker(checkpoint_path)
        was_val_epoch = ((trained_epoch + 1) % _val_every == 0)
        if was_val_epoch and (validated_epoch is None or validated_epoch < trained_epoch):
            logger.info(
                f"\n[Resume] Epoch {trained_epoch + 1} trained but NOT validated "
                f"(marker={validated_epoch if validated_epoch is not None else 'missing'}). "
                f"Running catch-up validation before continuing..."
            )
            try:
                catchup = validate_epoch(
                    model, val_loader, device, config, is_full, logger,
                    epoch=start_epoch,
                )
            except Exception as _cv_err:
                logger.error(f"[Resume] Catch-up validation FAILED: {_cv_err}; continuing without it.")
                catchup = None
            if catchup is not None:

                logger.info(_format_val_log(trained_epoch + 1, catchup, config, mode))
                c_composite = catchup['composite']
                c_val_loss = catchup['val_loss']
                c_r1 = catchup['avg_r1']
                if c_composite > best_composite:
                    best_composite = c_composite
                    if c_val_loss < best_val_loss:
                        best_val_loss = c_val_loss
                    save_model(
                        model, os.path.join(run_dir, 'best_weights.pt'),
                        epoch=trained_epoch,
                        metrics={'composite': c_composite, 'val_loss': c_val_loss, 'avg_r1': c_r1},
                    )
                    logger.info(
                        f"[Resume] Catch-up validation updated best | epoch={trained_epoch + 1}, "
                        f"composite={c_composite:.4f}, val_loss={c_val_loss:.4f}, avg_R@1={c_r1:.2f}%"
                    )
                    ckpt_no_improve = 0
                else:
                    ckpt_no_improve = ckpt_no_improve + 1
                _write_validation_marker(checkpoint_path, trained_epoch)
                _flush_log_handlers(logger)
            else:
                logger.warning("[Resume] Catch-up validation produced no results; continuing without it.")
                _flush_log_handlers(logger)

    if train_loader is None or len(train_loader) == 0:
        logger.error("Training data loader is empty -- refusing to train.")
        return 1
    patience = getattr(config.train, 'early_stop_patience', 10)
    val_every = getattr(config.train, 'val_every_n_epochs', 1)
    epochs_no_improve = ckpt_no_improve if checkpoint_exists else 0

    epoch = start_epoch
    while epoch < config.train.epochs:
        logger.info(
            f"\n{'=' * 100}\n"
            f"[Epoch {epoch + 1:03d}/{config.train.epochs:03d}] START\n"
            f"{'=' * 100}"
        )

        total_epochs = config.train.epochs
        if hasattr(model, 'has_tgl') and model.has_tgl:
            model.epoch_progress = epoch / max(1, total_epochs - 1)

        model._nan_signal = None
        nan_mgr.reset_epoch()

        train_started_at = time.time()
        if is_full:
            train_loss, train_details, _ = train_one_epoch_full(
                model, train_loader, optimizer, scheduler, scaler, device, config, epoch, trainable_params, nan_mgr=nan_mgr)
        else:
            train_loss, train_details, _ = train_one_epoch_sgl(
                model, train_loader, optimizer, scheduler, scaler, device, config, epoch, trainable_params, nan_mgr=nan_mgr)
        train_elapsed = time.time() - train_started_at

        if train_loss is None:
            nan_diag = nan_mgr.diagnosis()
            logger.error(
                f"\n  ╔══════════════════════════════════════════════════╗\n"
                f"  ║  [NaN FATAL] Training halted mid-epoch {epoch+1}       ║\n"
                f"  ║  {nan_diag[:48]:48s} ║\n"
                f"  ╚══════════════════════════════════════════════════╝"
            )

            if nan_mgr.action == 'fatal_stop':
                logger.error(
                    f"[NaN FATAL] lr_reductions exceeded the cap "
                    f"({nan_mgr.max_lr_reductions}); cannot continue. Stopping."
                )
                break

            ckpt_path = os.path.join(run_dir, 'checkpoint.pt')
            if not os.path.exists(ckpt_path):
                logger.error(
                    "[FATAL] No checkpoint.pt found — first epoch NaN, cannot rollback."
                )

                crash_path = os.path.join(run_dir, f'nan_crash_epoch{epoch+1}.pt')
                save_checkpoint(model, optimizer, scheduler, scaler, epoch, best_loss,
                              crash_path, best_composite=best_composite,
                              best_val_loss=best_val_loss, epochs_no_improve=epochs_no_improve,
                              nan_diagnostics=nan_mgr.diagnostic_state(),
                              data_generators=checkpoint_generators,
                              async_save=False)
                logger.info(f"[NaN DEBUG] Crash dump saved to: {crash_path}")
                break

            crash_path = os.path.join(run_dir, f'nan_crash_epoch{epoch+1}.pt')
            save_checkpoint(model, optimizer, scheduler, scaler, epoch, best_loss,
                          crash_path, best_composite=best_composite,
                          best_val_loss=best_val_loss, epochs_no_improve=epochs_no_improve,
                          nan_diagnostics=nan_mgr.diagnostic_state(),
                          data_generators=checkpoint_generators,
                          async_save=False)
            logger.info(f"[NaN DEBUG] Crash dump saved to: {crash_path}")

            logger.warning(f"[ROLLBACK] Loading checkpoint from: {ckpt_path}")
            rollback_start, rollback_best_loss, best_composite, best_val_loss, ckpt_no_improve, nan_diagnostics = \
                load_checkpoint(model, optimizer, scheduler, scaler, ckpt_path, device, logger,
                                data_generators=checkpoint_generators)
            nan_mgr.restore_from_state(nan_diagnostics)

            if scaler is not None:
                scaler = build_grad_scaler(config)
                logger.warning(f"[ROLLBACK] GradScaler reset to {scaler.get_scale():.1f}")

            old_lrs = [pg['lr'] for pg in optimizer.param_groups]
            for pg in optimizer.param_groups:
                pg['lr'] = max(pg['lr'] * 0.5, 1e-8)
            new_lrs = [pg['lr'] for pg in optimizer.param_groups]
            nan_mgr.register_rollback_lr_reduction()

            for i in range(len(optimizer.param_groups)):
                ratio = new_lrs[i] / max(old_lrs[i], 1e-8)
                scheduler.base_lrs[i] = scheduler.base_lrs[i] * ratio

            logger.warning(
                f"[ROLLBACK] Restored epoch {rollback_start} weights. "
                f"LR: {old_lrs[0]:.2e} → {new_lrs[0]:.2e}. "
                f"Re-running epoch {epoch+1}..."
            )

            min_lr = min(pg['lr'] for pg in optimizer.param_groups)
            if min_lr < 1e-7:
                logger.error(
                    f"[FATAL] LR too low ({min_lr:.1e}) after rollback. "
                    f"Cannot continue. Stopping."
                )
                break

            best_loss = rollback_best_loss
            epochs_no_improve = ckpt_no_improve
            epoch = rollback_start
            continue

        if train_loss < best_loss:
            best_loss = train_loss

        save_checkpoint(model, optimizer, scheduler, scaler, epoch, best_loss,
                        os.path.join(run_dir, 'checkpoint.pt'),
                        best_composite=best_composite, best_val_loss=best_val_loss,
                        epochs_no_improve=epochs_no_improve,
                        nan_diagnostics=nan_mgr.diagnostic_state(),
                        data_generators=checkpoint_generators)
        logger.info(
            f"[Checkpoint] Updated checkpoint.pt | completed_epoch={epoch + 1}, "
            f"next_epoch={epoch + 2}, best_train_loss={best_loss:.4f}, "
            f"best_composite={best_composite:.4f}, patience={epochs_no_improve}/{patience}"
        )

        _maybe_release_train_workers_before_validation(config, train_loader, logger)

        should_val = ((epoch + 1) % val_every == 0) and val_loader is not None

        if should_val:

            try:
                val_results = validate_epoch(
                    model, val_loader, device, config, is_full, logger,
                    epoch=epoch + 1,
                )
            except Exception as val_err:
                _release_dataloader_workers(val_loader)
                logger.error(
                    f"[Validation] FAILED at epoch {epoch + 1}: {val_err}"
                )
                logger.error(
                    "[Validation] Skipping validation this epoch; continuing "
                    "training (checkpoint already saved before validation)."
                )
                val_results = None
            if val_results is not None:
                composite = val_results['composite']
                val_loss = val_results['val_loss']
                avg_r1 = val_results['avg_r1']

                current_lr = max(pg['lr'] for pg in optimizer.param_groups)
                logger.info(_format_train_log(
                    epoch + 1, train_loss, train_details, config, mode,
                    lr=current_lr, elapsed_seconds=train_elapsed,
                ))
                logger.info(_format_val_log(epoch + 1, val_results, config, mode))

                composite_ok = False
                previous_best_composite = best_composite
                previous_best_val_loss = best_val_loss

                if composite > best_composite:
                    best_composite = composite
                    if val_loss < best_val_loss:
                        best_val_loss = val_loss
                    composite_ok = True
                    save_model(
                        model,
                        os.path.join(run_dir, 'best_weights.pt'),
                        epoch=epoch,
                        metrics={
                            'composite': composite,
                            'val_loss': val_loss,
                            'avg_r1': avg_r1,
                        },
                    )
                    logger.info(
                        f"[Checkpoint] Saved new best_weights.pt | epoch={epoch + 1}, "
                        f"composite={composite:.4f}, val_loss={val_loss:.4f}, "
                        f"avg_R@1={avg_r1:.2f}%"
                    )

                if composite_ok:
                    epochs_no_improve = 0
                else:
                    epochs_no_improve += 1
                logger.info(
                    f"[Selection] improved={composite_ok} | "
                    f"current(composite={composite:.4f}, val_loss={val_loss:.4f}) | "
                    f"previous_best(composite={previous_best_composite:.4f}, "
                    f"val_loss={previous_best_val_loss:.4f}) | "
                    f"patience={epochs_no_improve}/{patience}"
                )

                _write_validation_marker(
                    os.path.join(run_dir, 'checkpoint.pt'), epoch)
                _flush_log_handlers(logger)
            else:

                _flush_log_handlers(logger)
                current_lr = max(pg['lr'] for pg in optimizer.param_groups)
                logger.info(_format_train_log(
                    epoch + 1, train_loss, train_details, config, mode,
                    lr=current_lr, elapsed_seconds=train_elapsed,
                ))
        else:

            current_lr = max(pg['lr'] for pg in optimizer.param_groups)
            logger.info(_format_train_log(
                epoch + 1, train_loss, train_details, config, mode,
                lr=current_lr, elapsed_seconds=train_elapsed,
            ))

        _release_dataloader_workers(val_loader)

        torch.cuda.empty_cache()

        gc.collect()
        logger.info(
            f"[VRAM] epoch {epoch + 1}: allocated="
            f"{torch.cuda.memory_allocated() / 2 ** 30:.1f}GB, "
            f"reserved={torch.cuda.memory_reserved() / 2 ** 30:.1f}GB"
        )

        if should_val and epochs_no_improve >= patience:
            reason_parts = [f"no improvement for {patience} epochs"]
            reason_parts.append(f"best_composite={best_composite:.4f}")
            reason_parts.append(f"best_val_loss={best_val_loss:.4f}")
            logger.info(f"\nEarly stopping at epoch {epoch+1}: " + ", ".join(reason_parts))
            break

        epoch += 1

    save_model(
        model,
        os.path.join(run_dir, 'last_weights.pt'),
        epoch=max(epoch - 1, 0),
        metrics={'best_loss': best_loss},
    )
    logger.info("✓ Last model saved")

    logger.info(f"\n{'='*40} Final Evaluation {'='*40}")
    best_path = os.path.join(run_dir, 'best_weights.pt')
    if not os.path.exists(best_path):
        best_path = os.path.join(run_dir, 'last_weights.pt')
        logger.warning("best_weights.pt was not created; evaluating last_weights.pt")
    best_artifact = torch.load(best_path, map_location=device)
    best_sd = dict(extract_model_state(best_artifact))

    _load_model_state_checked(
        model, best_sd, best_path, allow_partial=False, logger=logger,
    )
    model = model.to(device)
    model.eval()

    eval_results = {}
    for loader_name, loader in [('Val', val_loader), ('Test', test_loader)]:
        if loader is None:
            continue
        logger.info(f"\n--- {loader_name} Set ---")
        if is_full:
            eval_results[loader_name] = validate_full(
                model, loader, device, logger, config=config,
            )
        else:
            eval_results[loader_name] = validate_sgl(
                model, loader, device, logger, config=config,
            )

    if len(eval_results) == 2:
        logger.info(f"\n{'='*40} Val + Test Averaged {'='*40}")
        avg_result = average_eval_results(eval_results['Val'], eval_results['Test'])
        if isinstance(avg_result, dict) and 'ecg2text' in avg_result:

            logger.info(format_eval_results(avg_result))
        elif isinstance(avg_result, dict):
            logger.info(f"\n[Averaged]\n{format_full_eval_results(avg_result)}")

    logger.info("\nTraining complete!")

if __name__ == '__main__':

    sys.exit(main())
