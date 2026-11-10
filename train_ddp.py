"""Distributed cumulative LaUER training with gradient accumulation and synchronized validation."""
import os
import sys

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
import gc
import time
import math
from contextlib import nullcontext
import argparse
import logging
from datetime import timedelta

import torch
import torch.nn as nn
import torch.distributed as dist
from torch.utils.data.distributed import DistributedSampler
from tqdm import tqdm
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from configs.config import load_config_from_yaml
from models.lauer import LaUER
from data.meeti_dataset import create_meeti_dataloaders
from evaluation.zero_shot import (
    format_eval_results,
    retrieval_results_from_embeddings,
)
from evaluation.utils import (
    average_eval_results,
    format_full_eval_results,
)
from training.reproducibility import (
    capture_rng_state,
    configure_reproducibility,
    loader_generators,
    make_data_generator,
    restore_rng_state,
    seed_rng,
    seed_worker,
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

def init_distributed():
    if 'RANK' not in os.environ:
        return 0, 1, 0, torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    rank = int(os.environ['RANK'])
    world_size = int(os.environ['WORLD_SIZE'])
    local_rank = int(os.environ['LOCAL_RANK'])

    backend = 'nccl' if (torch.cuda.is_available()
                         and dist.is_nccl_available()) else 'gloo'

    try:
        dist.init_process_group(
            backend=backend,
            device_id=torch.device(f'cuda:{local_rank}'),
        )
    except RuntimeError:
        master_addr = os.environ.get('MASTER_ADDR', '127.0.0.1')
        master_port = int(os.environ.get('MASTER_PORT', '29500'))
        store = dist.TCPStore(
            host_name=master_addr,
            port=master_port,
            world_size=world_size,
            is_master=(rank == 0),
            timeout=timedelta(seconds=30),
            use_libuv=False,
        )
        dist.init_process_group(
            backend=backend, store=store,
            rank=rank, world_size=world_size,
            device_id=torch.device(f'cuda:{local_rank}'),
        )

    torch.cuda.set_device(local_rank)

    return rank, world_size, local_rank, torch.device(f'cuda:{local_rank}')

def setup_logging(output_dir, rank=0):
    os.makedirs(output_dir, exist_ok=True)
    log_file = os.path.join(output_dir, 'train.log')
    logging.basicConfig(
        level=logging.INFO if rank == 0 else logging.WARNING,
        format='%(asctime)s [%(levelname)s] %(message)s',
        handlers=[
            logging.FileHandler(log_file, encoding='utf-8'),
            logging.StreamHandler() if rank == 0 else logging.NullHandler(),
        ]
    )
    return logging.getLogger(__name__)

def set_seed(seed, deterministic=False, benchmark=True):
    """Apply both the configured backend policy and global seed."""
    return configure_reproducibility(seed, deterministic, benchmark)

from train import (
    build_optimizer, build_scheduler, build_grad_scaler, _amp_autocast,
    _optimizer_step, _handle_optimizer_status, save_model,
    save_checkpoint, load_checkpoint, _val,
    _load_model_state_checked,
    _renormalize_accumulated_grads, _accumulate_loss_details,
    PROGRESS_REFRESH_EVERY, _async_save, wait_async_save,
    _release_dataloader_workers, _maybe_release_train_workers_before_validation,
    _write_validation_marker, _read_validation_marker,
    _flush_log_handlers,
    _average_loss_details,
)
from training.nan_resilience import (
    NaNResilienceManager,
    _format_nan_batch_log,
)
from training.stage_io import (
    build_stage_artifact,
    extract_model_state,
    load_stage_initialization,
    mode_output_dir,
    validate_resume_artifact,
)

def save_checkpoint_ddp(model, optimizer, scheduler, scaler, epoch,
                        best_loss, path, rank, best_composite=None, best_val_loss=None,
                        train_sampler=None, epochs_no_improve=0, nan_diagnostics=None,
                        data_generators=None, gather_rng_across_ranks=True,
                        async_save=True):
    local_rng_state = capture_rng_state(data_generators)
    rng_state_by_rank = None
    if (gather_rng_across_ranks and dist.is_initialized()
            and dist.get_world_size() > 1):
        rng_state_by_rank = [None] * dist.get_world_size()
        dist.all_gather_object(rng_state_by_rank, local_rng_state)
    elif rank == 0:
        rng_state_by_rank = [local_rng_state]

    if rank != 0:
        return
    os.makedirs(os.path.dirname(path), exist_ok=True)
    raw_model = model.module if hasattr(model, 'module') else model
    ckpt = build_stage_artifact(
        raw_model,
        epoch=epoch,
        metrics={
            'best_loss': best_loss,
            'best_composite': best_composite if best_composite is not None else -float('inf'),
            'best_val_loss': best_val_loss if best_val_loss is not None else float('inf'),
        },
    )
    ckpt.update({
        'artifact_type': 'training_checkpoint',
        'mode': raw_model.mode,
        'best_loss': best_loss,
        'best_composite': best_composite if best_composite is not None else -float('inf'),
        'best_val_loss': best_val_loss if best_val_loss is not None else float('inf'),
        'epochs_no_improve': epochs_no_improve,
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict(),
    })
    if nan_diagnostics is not None:
        ckpt['nan_diagnostics'] = nan_diagnostics
    ckpt['rng_state_by_rank'] = rng_state_by_rank
    # Keep the old key so new checkpoints remain readable by older code.
    ckpt['rng_state'] = rng_state_by_rank[0]
    if scaler is not None:
        ckpt['scaler_state_dict'] = scaler.state_dict()
    if train_sampler is not None and hasattr(train_sampler, 'state_dict'):
        ckpt['train_sampler_state_dict'] = train_sampler.state_dict()
    _async_save(path, ckpt, sync=not async_save)

def load_checkpoint_ddp(model, optimizer, scheduler, scaler, path, device, logger,
                        data_generators=None):
    ckpt = torch.load(path, map_location='cpu',weights_only=False)
    raw_model = model.module if hasattr(model, 'module') else model
    validate_resume_artifact(ckpt, raw_model, path)
    nan_diagnostics = ckpt.get('nan_diagnostics')
    model_state = ckpt['model_state_dict']

    was_remapped = False

    has_module_prefix = any(k.startswith('module.') for k in model_state.keys())

    if has_module_prefix:
        model_state = {k.replace('module.', '', 1): v for k, v in model_state.items()}

    _load_model_state_checked(
        raw_model, model_state, path,
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
            was_remapped = True
            if is_rank0 := ((not dist.is_initialized()) or (dist.get_rank() == 0)):
                logger.warning(
                    "[Checkpoint] Optimizer param_group structure changed. "
                    "Optimizer state reset; scheduler position is preserved."
                )
    if was_remapped:

        if 'scheduler_state_dict' in ckpt:
            saved_last_epoch = ckpt['scheduler_state_dict'].get('last_epoch', 0)
            scheduler.last_epoch = saved_last_epoch
            if is_rank0 := ((not dist.is_initialized()) or (dist.get_rank() == 0)):
                logger.info(
                    f"[Checkpoint] Scheduler last_epoch restored to {saved_last_epoch}. "
                    f"LR schedule continues from step {saved_last_epoch}."
                )

        for i, pg in enumerate(optimizer.param_groups):
            pg['lr'] = scheduler.base_lrs[i] * scheduler.lr_lambdas[i](scheduler.last_epoch)

        if scaler is not None and 'scaler_state_dict' in ckpt:
            scaler.load_state_dict(ckpt['scaler_state_dict'])
        if is_rank0 := ((not dist.is_initialized()) or (dist.get_rank() == 0)):
            logger.warning(
                "[Checkpoint] Compatibility migration applied. Model weights loaded; "
                "optimizer state reset while scheduler/scaler state is preserved."
            )
        start_epoch = ckpt['epoch'] + 1
        best_loss = ckpt['best_loss']
        best_composite = ckpt.get('best_composite', -float('inf'))
        best_val_loss = ckpt.get('best_val_loss', float('inf'))
        epochs_no_improve = ckpt.get('epochs_no_improve', 0)
        sampler_state = ckpt.get('train_sampler_state_dict', None)
        current_rank = dist.get_rank() if dist.is_initialized() else 0
        states_by_rank = ckpt.get('rng_state_by_rank')
        if states_by_rank is not None and current_rank < len(states_by_rank):
            restore_rng_state(states_by_rank[current_rank], data_generators, logger)
        elif current_rank == 0:
            restore_rng_state(ckpt.get('rng_state'), data_generators, logger)
        del ckpt
        if device.type == 'cuda':
            torch.cuda.empty_cache()
        return (start_epoch, best_loss, best_composite, best_val_loss,
                sampler_state, epochs_no_improve, nan_diagnostics)
    optimizer.load_state_dict(ckpt['optimizer_state_dict'])
    scheduler.load_state_dict(ckpt['scheduler_state_dict'])
    if scaler is not None and 'scaler_state_dict' in ckpt:
        scaler.load_state_dict(ckpt['scaler_state_dict'])

    sampler_state = ckpt.get('train_sampler_state_dict', None)
    start_epoch = ckpt['epoch'] + 1
    best_loss = ckpt['best_loss']
    best_composite = ckpt.get('best_composite', -float('inf'))
    best_val_loss = ckpt.get('best_val_loss', float('inf'))
    epochs_no_improve = ckpt.get('epochs_no_improve', 0)
    current_rank = dist.get_rank() if dist.is_initialized() else 0
    states_by_rank = ckpt.get('rng_state_by_rank')
    if states_by_rank is not None and current_rank < len(states_by_rank):
        restore_rng_state(states_by_rank[current_rank], data_generators, logger)
    elif current_rank == 0:
        restore_rng_state(ckpt.get('rng_state'), data_generators, logger)
    del ckpt
    if device.type == 'cuda':
        torch.cuda.empty_cache()
    is_rank0 = (not dist.is_initialized()) or (dist.get_rank() == 0)
    if is_rank0:
        logger.info(f"Resumed from epoch {start_epoch}, best_loss={best_loss:.4f}"
                    f", best_composite={best_composite:.4f}")
    return (start_epoch, best_loss, best_composite, best_val_loss,
            sampler_state, epochs_no_improve, nan_diagnostics)

def _model_forward(model, batch, device):
    return model(batch=batch)

def train_one_epoch_ddp(model, dataloader, optimizer, scheduler, scaler,
                        device, config, epoch, rank, world_size, nan_mgr=None):
    model.train()
    accum = getattr(config.train, 'accumulation_steps', 1) or 1
    num_steps = len(dataloader)
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    valid_steps = 0
    valid_in_group = 0
    real_accum = 1
    total_loss = 0.0
    loss_accum = {}
    scaler_enabled = scaler is not None
    raw_model = model.module if hasattr(model, 'module') else model

    if nan_mgr is None:
        nan_mgr = NaNResilienceManager()

    pbar = tqdm(
        dataloader,
        desc=f"Train {epoch + 1:03d}/{config.train.epochs:03d}",
        unit="batch",
        disable=(rank != 0),
    )

    for i, batch in enumerate(pbar):
        is_last = ((i + 1) % accum == 0) or ((i + 1) == num_steps)

        if (i % accum) == 0:
            remaining = num_steps - i
            real_accum = min(accum, remaining)
            valid_in_group = 0

        sync_ctx = model.no_sync() if not is_last and hasattr(model, 'no_sync') else nullcontext()

        with sync_ctx:
            with _amp_autocast(config, device):
                loss, loss_dict, _ = _model_forward(model, batch, device)

            if dist.is_initialized() and dist.get_world_size() > 1:
                world_size = dist.get_world_size()
                status_tensor = torch.stack([
                    (~torch.isfinite(loss.detach())).float(),
                    torch.nan_to_num(loss.detach().float()),
                ])
                dist.all_reduce(status_tensor, op=dist.ReduceOp.SUM)
                any_nan, global_loss_sum = status_tensor.cpu().tolist()
                any_nan = any_nan > 0
                loss_value = global_loss_sum / world_size
            else:
                loss_value = float(loss.detach().float().item())
                any_nan = not math.isfinite(loss_value)

            if any_nan:

                nan_accum = getattr(raw_model, '_nan_accum', None)
                if nan_accum:
                    names = [k for k, v in nan_accum.items() if v is not None]
                    local_flags = torch.tensor(
                        [0 if not bool(nan_accum[n].item()) else 1 for n in names],
                        dtype=torch.int32, device=loss.device,
                    )
                    if dist.is_initialized() and dist.get_world_size() > 1:
                        dist.all_reduce(local_flags, op=dist.ReduceOp.MIN)
                    sources = [n for n, f in zip(names, local_flags.tolist()) if f == 0]
                    if sources:
                        raw_model._nan_signal = '+'.join(sources)
                nan_source = getattr(raw_model, '_nan_signal', None)
                if nan_source is not None:
                    raw_model._nan_signal = None

                nan_mgr.register_nan_batch(loss_dict, module_name=nan_source, step=i)

                if rank == 0:
                    logger = logging.getLogger(__name__)
                    logger.warning(_format_nan_batch_log(loss, loss_dict, nan_source, epoch, i))

                if nan_mgr.is_fatal:
                    if rank == 0:
                        logging.getLogger(__name__).error(f"[NaN FATAL] {nan_mgr.diagnosis()}")
                    optimizer.zero_grad(set_to_none=True)
                    return None, None, nan_mgr

                if nan_mgr.action == 'reduce_lr':
                    lr_changes = nan_mgr.reduce_lr(optimizer, scheduler)
                    if rank == 0:
                        logging.getLogger(__name__).warning(
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

        if is_last:
            _renormalize_accumulated_grads(
                trainable_params, real_accum, valid_in_group
            )
            status = _optimizer_step(
                optimizer, scheduler, scaler, trainable_params,
                config.train.grad_clip_norm, scaler_enabled, model=raw_model,
                log_interval=getattr(config.train, 'nan_log_interval', 100),
                sync_nonfinite=True,
            )
            status.update({
                'epoch': epoch,
                'batch_step': i,
                'rank': rank,
                'sample_ids': batch.get('sample_id', batch.get('sample_ids')),
                'nan_signal': getattr(raw_model, '_nan_signal', None),
            })
            raw_model._nan_signal = None

            can_continue_local = _handle_optimizer_status(
                status, nan_mgr, logging.getLogger(__name__)
            )
            fatal_tensor = torch.tensor(
                0 if can_continue_local else 1, device=device, dtype=torch.int32
            )
            if dist.is_initialized() and dist.get_world_size() > 1:
                dist.all_reduce(fatal_tensor, op=dist.ReduceOp.MAX)
            if fatal_tensor.item() > 0:
                return None, None, nan_mgr

        total_loss += loss_value
        valid_steps += 1
        _accumulate_loss_details(loss_accum, loss_dict, device)

        if rank == 0 and ((i % PROGRESS_REFRESH_EVERY == 0) or ((i + 1) == num_steps)):
            postfix = build_progress_postfix(
                loss_value, loss_dict, raw_model.mode,
                max(pg['lr'] for pg in optimizer.param_groups),
            )
            pbar.set_postfix(postfix)

    avg_loss = total_loss / max(valid_steps, 1)
    avg_details = _average_loss_details(loss_accum, valid_steps)
    return avg_loss, avg_details, nan_mgr

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

@torch.no_grad()
def validate_unified_ddp(model, val_loader, device, config, is_full, rank):
    if val_loader is None:
        return None

    from models.gather_layer import gather_features_nograd
    raw_model = model.module if hasattr(model, 'module') else model
    is_ddp = dist.is_initialized() and dist.get_world_size() > 1
    model.eval()

    all_ecg, all_img, all_txt = [], [], []
    # [matching, reserved, reserved, cgl_count, mlm_correct, mlm_total,
    #  mlm_loss_sum, ar_loss_sum, ar_tokens, dec_count, val_loss, val_count,

    stats = torch.zeros(16, device=device, dtype=torch.float32)
    nan_val_count = 0

    mlc_logits_list = []
    mlc_labels_list = []

    for batch in tqdm(val_loader, desc="Val", unit="batch",
                      disable=(is_ddp and rank != 0)):
        with _amp_autocast(config, device):
            loss, loss_dict, embeddings = _model_forward(model, batch, device)

        if not torch.isfinite(loss.detach()).item():
            nan_val_count += 1
            if hasattr(raw_model, '_nan_signal'):
                raw_model._nan_signal = None
            continue

        if is_full and getattr(raw_model, 'query_guided_tracing_head', None) is not None:
            b_labels = batch.get('labels')
            if b_labels is not None:
                mlc_lg = getattr(raw_model, '_last_mlc_logits', None)
                if mlc_lg is not None:
                    mlc_logits_list.append(mlc_lg.detach().float())
                    mlc_labels_list.append(
                        b_labels.to(device, non_blocking=True).detach().float())

        stats[10] += loss.detach().float()
        stats[11] += 1

        if embeddings is not None:
            all_ecg.append(embeddings['ecg_embed'])
            all_img.append(embeddings['image_embed'])
            all_txt.append(embeddings['text_embed'])

        if 'acc_matching' in loss_dict:
            stats[0] += loss_dict['acc_matching']
            stats[3] += 1

        if 'mlm_correct' in loss_dict:
            mlm_tokens = float(loss_dict['mlm_total'])
            ar_tokens = float(loss_dict.get('ar_tokens', 0))
            stats[4] += loss_dict['mlm_correct']
            stats[5] += mlm_tokens

            stats[6] += loss_dict.get('loss_mlm', 0.0) * max(mlm_tokens, 1)

            stats[7] += loss_dict.get('loss_ar', 0.0)
            stats[8] += ar_tokens
            stats[9] += 1

            if 'report_total' in loss_dict:
                stats[12] += float(loss_dict.get('report_correct', 0.0))
                stats[13] += float(loss_dict.get('report_total', 0.0))
                stats[14] += float(loss_dict.get('expl_correct', 0.0))
                stats[15] += float(loss_dict.get('expl_total', 0.0))

    # DDP all_reduce
    if is_ddp:
        dist.all_reduce(stats, op=dist.ReduceOp.SUM)

    model.train()

    if nan_val_count > 0 and rank == 0:
        _logger = logging.getLogger(__name__)
        _logger.warning(
            f"[Val NaN] {nan_val_count}/{max(len(val_loader), 1)} batches NaN. "
            f"Val metrics may be unreliable."
        )

    skip_val = int(
        (nan_val_count > max(len(val_loader), 1) * 0.5) or (not all_ecg)
    )
    if is_ddp:
        skip_flag = torch.tensor(skip_val, device=stats.device, dtype=torch.int32)
        dist.all_reduce(skip_flag, op=dist.ReduceOp.MAX)
        skip_val = int(skip_flag.item())
    if skip_val:
        if rank == 0:
            if nan_val_count > max(len(val_loader), 1) * 0.5:
                logging.getLogger(__name__).error(
                    f"[Val NaN] >50% batches NaN (or empty val loader). "
                    f"Skipping validation this epoch."
                )
            else:
                logging.getLogger(__name__).warning(
                    "[Val] No valid embedding batches collected; "
                    "skipping zero-shot metrics."
                )
        return None

    if is_ddp:
        all_ecg = [gather_features_nograd(torch.cat(all_ecg))]
        all_img = [gather_features_nograd(torch.cat(all_img))]
        all_txt = [gather_features_nograd(torch.cat(all_txt))]

        has_mlc_t = torch.tensor(1 if mlc_logits_list else 0, device=device)
        dist.all_reduce(has_mlc_t, op=dist.ReduceOp.MAX)
        if bool(has_mlc_t.item()):
            if mlc_logits_list:
                mlc_logits_cat = gather_features_nograd(torch.cat(mlc_logits_list, 0))
                mlc_labels_cat = gather_features_nograd(torch.cat(mlc_labels_list, 0))
            else:
                n_class = (len(raw_model.label_names)
                           if raw_model.label_names is not None else 0)
                _empty = torch.zeros(0, n_class, device=device)
                mlc_logits_cat = gather_features_nograd(_empty)
                mlc_labels_cat = gather_features_nograd(_empty)
        else:
            mlc_logits_cat = mlc_labels_cat = None
    else:
        mlc_logits_cat = torch.cat(mlc_logits_list, 0) if mlc_logits_list else None
        mlc_labels_cat = torch.cat(mlc_labels_list, 0) if mlc_labels_list else None

    if rank != 0:
        return None

    multilabel = None
    if mlc_logits_cat is not None and mlc_labels_cat is not None:
        from evaluation.utils import multilabel_metrics
        multilabel = multilabel_metrics(mlc_logits_cat, mlc_labels_cat)

    (matching_sum, _reserved_matching_1, _reserved_matching_2, cgl_count,
     mlm_correct, mlm_total, mlm_loss_sum, ar_loss_sum,
     ar_tok_sum, dec_count, val_loss_sum, val_count,
     report_correct, report_total, expl_correct, expl_total) = stats.cpu().tolist()

    global_ecg = torch.cat(all_ecg).cpu()
    global_img = torch.cat(all_img).cpu()
    global_txt = torch.cat(all_txt).cpu()
    zs_results = retrieval_results_from_embeddings(
        global_ecg, global_img, global_txt,
        raw_model.sgl_loss.logit_scale.exp(),
    )

    cgl_results = {}
    if cgl_count > 0:
        cgl_results['acc_matching'] = matching_sum / cgl_count

    tgl_metrics = None
    if dec_count > 0:

        ar_loss_global = ar_loss_sum / max(dec_count, 1)
        tgl_metrics = {
            'ar_ppl': math.exp(min(ar_loss_global, 20.0)) if ar_tok_sum > 0 else 0.0,
            'ar_loss': ar_loss_global,
        }

        if mlm_total > 0:
            tgl_metrics['mlm_accuracy'] = mlm_correct / mlm_total
            tgl_metrics['mlm_loss'] = mlm_loss_sum / max(mlm_total, 1)

        if report_total + expl_total > 0:
            tgl_metrics['ar_accuracy'] = (
                (report_correct + expl_correct) / (report_total + expl_total))
            tgl_metrics['report_accuracy'] = (
                report_correct / report_total if report_total > 0 else 0.0)
            tgl_metrics['expl_accuracy'] = (
                expl_correct / expl_total if expl_total > 0 else 0.0)

    w_zs = getattr(config.train, 'composite_zeroshot_weight', 0.5)
    w_cgl = getattr(config.train, 'composite_cgl_weight', 0.5)
    w_mlm = getattr(config.train, 'composite_mlm_weight', 0.0)
    w_ar = getattr(config.train, 'composite_ar_weight', 0.0)
    w_mlc = getattr(config.train, 'composite_mlc_weight', 0.0)
    avg_r1 = float(np.mean([zs_results[d]['R@1'] for d in zs_results]))

    avg_itm = 0.0
    if cgl_results:
        cgl_accs = [v for k, v in cgl_results.items() if k.startswith('acc_matching')]
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
    if multilabel is not None and w_mlc > 0:
        mlc_component = w_mlc * multilabel['mlc_map']

    if not is_full:
        composite = avg_r1 / 100.0
    else:
        composite = (w_zs * (avg_r1 / 100.0) + w_cgl * avg_itm
                     + mlc_component + mlm_component + ar_component)

    val_loss = val_loss_sum / max(val_count, 1)

    return {
        'val_loss': val_loss,
        'zero_shot': zs_results,
        'cgl': cgl_results,
        'composite': composite,
        'avg_r1': avg_r1,
        'avg_itm': avg_itm,
        'tgl_metrics': tgl_metrics,
        'multilabel': multilabel,
        'loss_details': {},
    }

def _build_ddp_dataloaders(config, data_root, rank, world_size,
                           logger, tokenizer, load_labels=False,
                           include_tgl_ids=False):

    dataloaders = create_meeti_dataloaders(
        config.data, config.train, tokenizer,
        max_text_length=config.text_encoder.max_length,
        image_size=config.image_encoder.image_size,
        encoder_type=config.image_encoder.encoder_type,
        load_labels=load_labels,

        include_tgl_ids=include_tgl_ids,
    )
    return _replace_samplers(dataloaders, rank, world_size, config)

def _replace_samplers(dataloaders, rank, world_size, config):
    train_loader = dataloaders.get('train')
    val_loader = dataloaders.get('val')
    test_loader = dataloaders.get('test')
    train_sampler = None

    val_dataset = val_loader.dataset if val_loader is not None else None
    test_dataset = test_loader.dataset if test_loader is not None else None

    if train_loader is not None:
        train_dataset = train_loader.dataset
        train_sampler = DistributedSampler(
            train_dataset, num_replicas=world_size, rank=rank,
            shuffle=True, drop_last=True, seed=config.train.seed,
        )

        t_nw, t_pm, t_pf, t_persist, _, t_dl = config.data.get_train_loader_cfg()
        train_loader = torch.utils.data.DataLoader(
            train_dataset, batch_size=config.train.batch_size,
            sampler=train_sampler,
            num_workers=t_nw,
            pin_memory=t_pm,
            collate_fn=getattr(train_loader, 'collate_fn', None),
            worker_init_fn=seed_worker,
            generator=make_data_generator(config.train.seed, split='train', rank=rank),
            drop_last=t_dl,

            persistent_workers=t_persist and t_nw > 0,

            prefetch_factor=t_pf if t_nw > 0 else None,
        )

    if val_loader is not None:
        val_dataset = val_loader.dataset
        val_sampler = DistributedSampler(
            val_dataset, num_replicas=world_size, rank=rank,
            shuffle=False, drop_last=False, seed=config.train.seed,
        )

        v_bs, v_nw, v_pm, v_persist, _, v_dl = config.data.get_val_loader_cfg()
        val_loader = torch.utils.data.DataLoader(
            val_dataset, batch_size=v_bs,
            sampler=val_sampler,
            num_workers=v_nw,
            pin_memory=v_pm,
            collate_fn=getattr(val_loader, 'collate_fn', None),
            worker_init_fn=seed_worker,
            generator=make_data_generator(config.train.seed, split='val', rank=rank),
            drop_last=v_dl,
            persistent_workers=v_persist,
        )

    if test_loader is not None and rank != 0:
        test_loader = None

    return train_loader, val_loader, test_loader, train_sampler, val_dataset, test_dataset

def main():
    parser = argparse.ArgumentParser(description='lauer DDP Training')
    parser.add_argument('--mode', type=str, default='sgl',
                        choices=['sgl', 'sgl+cgl', 'sgl+cgl+tgl'],
                        help='Training mode: sgl | sgl+cgl | sgl+cgl+tgl')
    parser.add_argument('--config', type=str,
                        default='configs/pretrain_SGL.yaml')
    parser.add_argument('--data_root', type=str, default='./datasets/MEETI')
    parser.add_argument('--output_dir', type=str, default='')
    parser.add_argument('--init-from', type=str, default='',
                        help='Completed predecessor-stage weights used to initialize a new stage')
    parser.add_argument('--allow-partial-init', action='store_true',
                        help='Allow a fresh start (no --init-from) or skipping intermediate '
                             'stages; missing modules stay randomly initialized')
    parser.add_argument('--batch_size', type=int, default=0,
                        help='Per-GPU batch size. Global = bs x nproc x accum_steps')
    parser.add_argument('--epochs', type=int, default=0)
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
    # This must happen after configure_reproducibility so deterministic CUDA
    # environment settings are present before any CUDA context is created.
    rank, world_size, local_rank, device = init_distributed()
    is_ddp = world_size > 1
    run_dir = mode_output_dir(
        config.train.output_dir, config.train.run_name, mode
    )
    logger = setup_logging(run_dir, rank=rank)

    log_run_configuration(
        logger, config, mode,
        config_path=args.config,
        data_root=config.data.data_root,
        run_dir=run_dir,
        device=device,
        init_from=init_from,
        resume=args.resume,
        auto_resume=not args.no_auto_resume,
        world_size=world_size,
    )
    log_runtime_environment(logger, device)

    from transformers import AutoTokenizer
    tokenizer_path = config.text_encoder.tokenizer_name_or_path
    if not tokenizer_path:
        if rank == 0:
            logger.error("text_encoder.tokenizer_name_or_path not set!")
        dist.destroy_process_group() if is_ddp else None
        return
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, local_files_only=True)

    with capture_startup_prints(logger, "BUILDING DDP DATALOADERS"):
        train_loader, val_loader, test_loader, train_sampler, val_dataset, test_dataset = \
            _build_ddp_dataloaders(
                config, config.data.data_root,
                rank, world_size, logger, tokenizer,
                load_labels=has_cgl,
                include_tgl_ids=has_tgl,
            )

    if train_loader is None:
        if rank == 0:
            logger.error("No training data!")

        dist.destroy_process_group() if is_ddp else None
        return 1

    checkpoint_generators = loader_generators({
        'train': train_loader,
        'val': val_loader,
        'test': test_loader,
    })

    log_dataloader_summary(
        logger,
        {'train': train_loader, 'val': val_loader, 'test': test_loader},
        config,
        world_size=world_size,
    )

    auto_checkpoint = os.path.join(run_dir, 'checkpoint.pt')
    checkpoint_path = args.resume or (None if args.no_auto_resume else auto_checkpoint)
    checkpoint_exists = bool(checkpoint_path and os.path.exists(checkpoint_path))
    if args.resume and not checkpoint_exists:
        if is_ddp:
            dist.destroy_process_group()
        raise FileNotFoundError(f"Explicit --resume checkpoint not found: {args.resume}")
    if mode == 'sgl' and init_from:
        if is_ddp:
            dist.destroy_process_group()
        raise RuntimeError("SGL is the first stage and must not use --init-from")
    if has_cgl and not checkpoint_exists and not init_from:
        predecessor = 'sgl' if mode == 'sgl+cgl' else 'sgl+cgl'
        if not args.allow_partial_init:
            if is_ddp:
                dist.destroy_process_group()
            raise RuntimeError(
                f"Fresh {mode!r} training requires completed {predecessor!r} weights. "
                "Pass them with --init-from, or pass --allow-partial-init to start "
                "from scratch with all missing modules randomly initialized."
            )
        if rank == 0:
            logger.warning(
                f"[Init] Fresh {mode!r} start via --allow-partial-init: no {predecessor!r} "
                "weights provided, all trainable modules (predecessor-stage adapters, "
                "fusion/heads, ViT encoders) start randomly initialized and train jointly "
                "(frozen MedCPT text encoder is still loaded from weights/)."
            )

    # Dataset/tokenizer construction must not shift model initialization, and
    # every rank must start from bit-identical model parameters.
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
                if rank == 0:
                    logger.info(f"[MultiLabel] {len(label_names)} disease queries encoded & "
                                f"query encoder released")

    if checkpoint_exists and init_from:
        if rank == 0:
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

    if rank == 0:
        module_lrs = {}
        for pg in optimizer.param_groups:
            cat = pg['name'].split('_')[0]
            if cat not in module_lrs and 'lr' in pg:
                module_lrs[cat] = pg['lr']
        lr_summary = '  '.join(f"{k}={v:.2e}" for k, v in module_lrs.items())
        logger.info(
            f"[RUNTIME] MODULE LEARNING RATES | base={config.train.lr:.2e} | {lr_summary}"
        )

    trainable_params = [p for p in model.parameters() if p.requires_grad]
    if rank == 0:
        n_trainable = sum(p.numel() for p in trainable_params)
        n_total = sum(p.numel() for p in model.parameters())
        logger.info(
            f"[RUNTIME] PARAMETERS | trainable={n_trainable:,} | total={n_total:,} | "
            f"trainable_ratio={n_trainable / max(n_total, 1) * 100:.2f}%"
        )

    if is_ddp:
        model = nn.parallel.DistributedDataParallel(
            model, device_ids=[local_rank],
            find_unused_parameters=False,
        )

    # Model initialization used the same base seed on every rank.  From this
    # point onward dropout/masking RNG streams must be independent per rank.
    runtime_seed = seed_rng(config.train.seed + rank)
    if rank == 0:
        logger.info(f"[Seed] Per-rank runtime RNG: base_seed + rank (rank 0={runtime_seed})")

    num_training_steps = config.train.epochs * math.ceil(
        len(train_loader) / getattr(config.train, 'accumulation_steps', 1)
    )
    scheduler = build_scheduler(optimizer, config, num_training_steps)
    scaler = build_grad_scaler(config)

    start_epoch = 0
    best_loss = float('inf')
    best_composite = -float('inf')
    best_val_loss = float('inf')

    nan_mgr = NaNResilienceManager()

    if checkpoint_exists:
        if rank == 0:
            logger.info(
                f"[Resume] Selected checkpoint: {checkpoint_path} "
                f"({'explicit' if args.resume else 'auto-discovered'})"
            )
        try:
            start_epoch, best_loss, best_composite, best_val_loss, sampler_state, ckpt_no_improve, nan_diagnostics = \
                load_checkpoint_ddp(
                model, optimizer, scheduler, scaler, checkpoint_path, device, logger,
                data_generators=checkpoint_generators)

            nan_mgr.restore_from_state(nan_diagnostics)
        except RuntimeError as exc:
            if rank == 0:
                logger.error(f"[Resume] Checkpoint load rejected:\n{exc}")
            if is_ddp:
                dist.destroy_process_group()
            raise
        if args.resume_lr_factor <= 0:
            raise ValueError("--resume_lr_factor must be > 0")
        if args.resume_lr_factor != 1.0:
            for pg in optimizer.param_groups:
                pg['lr'] *= args.resume_lr_factor
            scheduler.base_lrs = [lr * args.resume_lr_factor for lr in scheduler.base_lrs]
            if hasattr(scheduler, '_last_lr'):
                scheduler._last_lr = [lr * args.resume_lr_factor for lr in scheduler._last_lr]
            if rank == 0:
                logger.warning(
                    f"[Resume] Applied one-time LR factor {args.resume_lr_factor:g}; "
                    f"current max LR={max(pg['lr'] for pg in optimizer.param_groups):.2e}"
                )
        if args.reset_amp_scaler and scaler is not None:
            scaler = build_grad_scaler(config)
            if rank == 0:
                logger.warning(
                    f"[Resume] GradScaler reset to scale={scaler.get_scale():.1f} "
                    f"growth_interval={getattr(config.train, 'amp_growth_interval', 8000)}"
                )

        if sampler_state is not None and train_sampler is not None and hasattr(train_sampler, 'load_state_dict'):
            train_sampler.load_state_dict(sampler_state)
            if rank == 0:
                logger.info(f"  >> Restored sampler state from checkpoint")

        if start_epoch >= config.train.epochs:
            if rank == 0:
                logger.info("Training already complete. Running final eval.")

    if checkpoint_exists and val_loader is not None:
        _val_every = getattr(config.train, 'val_every_n_epochs', 1) or 1
        trained_epoch = start_epoch - 1
        validated_epoch = _read_validation_marker(checkpoint_path)
        was_val_epoch = ((trained_epoch + 1) % _val_every == 0)
        if was_val_epoch and (validated_epoch is None or validated_epoch < trained_epoch):
            if rank == 0:
                logger.info(
                    f"\n[Resume] Epoch {trained_epoch + 1} trained but NOT validated "
                    f"(marker={validated_epoch if validated_epoch is not None else 'missing'}). "
                    f"Running catch-up validation before continuing..."
                )

            catchup = validate_unified_ddp(
                model, val_loader, device, config, is_full, rank,
            )
            if catchup is not None and rank == 0:

                logger.info(_format_val_log(trained_epoch + 1, catchup, config, mode))
                raw_model = model.module if hasattr(model, 'module') else model
                c_composite = catchup['composite']
                c_val_loss = catchup['val_loss']
                c_r1 = catchup['avg_r1']
                if c_composite > best_composite:
                    best_composite = c_composite
                    if c_val_loss < best_val_loss:
                        best_val_loss = c_val_loss
                    save_model(
                        raw_model, os.path.join(run_dir, 'best_weights.pt'),
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
            elif catchup is None and rank == 0:
                logger.warning(
                    "[Resume] Catch-up validation produced no results; continuing without it.")
                _flush_log_handlers(logger)

    patience = getattr(config.train, 'early_stop_patience', 10)
    val_every = getattr(config.train, 'val_every_n_epochs', 1)
    epochs_no_improve = ckpt_no_improve if checkpoint_exists else 0

    epoch = start_epoch
    while epoch < config.train.epochs:
        if rank == 0:
            logger.info(
                f"\n{'=' * 100}\n"
                f"[Epoch {epoch + 1:03d}/{config.train.epochs:03d}] START\n"
                f"{'=' * 100}"
            )

        train_sampler.set_epoch(epoch)

        raw_model = model.module if hasattr(model, 'module') else model
        if hasattr(raw_model, 'has_tgl') and raw_model.has_tgl:
            raw_model.epoch_progress = epoch / max(1, config.train.epochs - 1)

        raw_model._nan_signal = None
        nan_mgr.reset_epoch()

        train_started_at = time.time()
        train_loss, train_details, _ = train_one_epoch_ddp(
            model, train_loader, optimizer, scheduler, scaler,
            device, config, epoch, rank, world_size, nan_mgr=nan_mgr,
        )
        train_elapsed = time.time() - train_started_at

        if train_loss is None:
            if rank == 0:
                nan_diag = nan_mgr.diagnosis()
                logger.error(
                    f"\n╔════════════════════════════════════════════════════╗\n"
                    f"  ║  [NaN FATAL] Training halted mid-epoch {epoch+1}   ║\n"
                    f"  ║  {nan_diag[:48]:48s}                               ║\n"
                    f"  ╚════════════════════════════════════════════════════╝"
                )

            if is_ddp:
                _fatal_flag = torch.tensor(
                    int(nan_mgr.action == 'fatal_stop'),
                    device=device, dtype=torch.int32,
                )
                dist.all_reduce(_fatal_flag, op=dist.ReduceOp.MAX)
                if bool(_fatal_flag.item()):
                    nan_mgr.action = 'fatal_stop'
            if nan_mgr.action == 'fatal_stop':
                if rank == 0:
                    logger.error(
                        f"[NaN FATAL] lr_reductions exceeded the cap "
                        f"({nan_mgr.max_lr_reductions}); cannot continue. Stopping."
                    )
                break

            ckpt_path = os.path.join(run_dir, 'checkpoint.pt')

            if is_ddp:
                dist.barrier()

            if not os.path.exists(ckpt_path):
                if rank == 0:
                    logger.error(
                        "[FATAL] No checkpoint.pt found — first epoch NaN, cannot rollback."
                    )

                crash_path = os.path.join(run_dir, f'nan_crash_epoch{epoch+1}.pt')
                save_checkpoint_ddp(model, optimizer, scheduler, scaler, epoch, best_loss,
                                  crash_path, rank, best_composite=best_composite,
                                  best_val_loss=best_val_loss, train_sampler=train_sampler,
                                  epochs_no_improve=epochs_no_improve,
                                  nan_diagnostics=nan_mgr.diagnostic_state(),
                                  data_generators=checkpoint_generators,
                                  gather_rng_across_ranks=True,
                                  async_save=False)
                if rank == 0:
                    logger.info(f"[NaN DEBUG] Crash dump saved to: {crash_path}")
                break

            crash_path = os.path.join(run_dir, f'nan_crash_epoch{epoch+1}.pt')
            save_checkpoint_ddp(model, optimizer, scheduler, scaler, epoch, best_loss,
                              crash_path, rank, best_composite=best_composite,
                              best_val_loss=best_val_loss, train_sampler=train_sampler,
                              epochs_no_improve=epochs_no_improve,
                              nan_diagnostics=nan_mgr.diagnostic_state(),
                              data_generators=checkpoint_generators,
                              gather_rng_across_ranks=True,
                              async_save=False)
            if rank == 0:
                logger.info(f"[NaN DEBUG] Crash dump saved to: {crash_path}")

            rollback_start, rollback_best_loss, best_composite, best_val_loss, sampler_state, ckpt_no_improve, nan_diagnostics = \
                load_checkpoint_ddp(model, optimizer, scheduler, scaler, ckpt_path, device, logger,
                                    data_generators=checkpoint_generators)
            nan_mgr.restore_from_state(nan_diagnostics)

            if scaler is not None:
                scaler = build_grad_scaler(config)
                if rank == 0:
                    logger.warning(f"[ROLLBACK] GradScaler reset to {scaler.get_scale():.1f}")

            old_lrs = [pg['lr'] for pg in optimizer.param_groups]
            for pg in optimizer.param_groups:
                pg['lr'] = max(pg['lr'] * 0.5, 1e-8)
            new_lrs = [pg['lr'] for pg in optimizer.param_groups]
            nan_mgr.register_rollback_lr_reduction()

            for i in range(len(optimizer.param_groups)):
                ratio = new_lrs[i] / max(old_lrs[i], 1e-8)
                scheduler.base_lrs[i] = scheduler.base_lrs[i] * ratio

            if rank == 0:
                logger.warning(
                    f"[ROLLBACK] Restored epoch {rollback_start} weights. "
                    f"LR: {old_lrs[0]:.2e} → {new_lrs[0]:.2e}. "
                    f"Re-running epoch {epoch+1}..."
                )

            min_lr = min(pg['lr'] for pg in optimizer.param_groups)
            if min_lr < 1e-7:
                if rank == 0:
                    logger.error(
                        f"[FATAL] LR too low ({min_lr:.1e}) after rollback. Stopping."
                    )
                break

            best_loss = rollback_best_loss
            epochs_no_improve = ckpt_no_improve
            epoch = rollback_start

            if is_ddp:
                dist.barrier()
            continue

        should_val = ((epoch + 1) % val_every == 0) and val_loader is not None
        if should_val:

            _maybe_release_train_workers_before_validation(
                config, train_loader, logger if rank == 0 else None)
        val_results = None
        if should_val:
            val_results = validate_unified_ddp(
                model, val_loader, device, config, is_full, rank,
            )
        if should_val:
            _release_dataloader_workers(val_loader)

        torch.cuda.empty_cache()

        gc.collect()
        if rank == 0:
            logger.info(
                f"[VRAM] epoch {epoch + 1}: allocated="
                f"{torch.cuda.memory_allocated() / 2 ** 30:.1f}GB, "
                f"reserved={torch.cuda.memory_reserved() / 2 ** 30:.1f}GB"
            )

        if rank == 0:
            raw_model = model.module if hasattr(model, 'module') else model

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
                        raw_model,
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

                _write_validation_marker(checkpoint_path, epoch)
                _flush_log_handlers(logger)
            else:
                current_lr = max(pg['lr'] for pg in optimizer.param_groups)
                logger.info(_format_train_log(
                    epoch + 1, train_loss, train_details, config, mode,
                    lr=current_lr, elapsed_seconds=train_elapsed,
                ))
                _flush_log_handlers(logger)

        if train_loss < best_loss:
            best_loss = train_loss

        save_checkpoint_ddp(model, optimizer, scheduler, scaler, epoch,
                           best_loss, os.path.join(run_dir, 'checkpoint.pt'),
                           rank, best_composite=best_composite, best_val_loss=best_val_loss,
                           train_sampler=train_sampler, epochs_no_improve=epochs_no_improve,
                           nan_diagnostics=nan_mgr.diagnostic_state(),
                           data_generators=checkpoint_generators)
        if rank == 0:
            logger.info(
                f"[Checkpoint] Updated checkpoint.pt | completed_epoch={epoch + 1}, "
                f"next_epoch={epoch + 2}, best_train_loss={best_loss:.4f}, "
                f"best_composite={best_composite:.4f}, "
                f"patience={epochs_no_improve}/{patience}"
            )

        if is_ddp:
            wait_async_save()
            dist.barrier()

        should_stop = bool(rank == 0 and should_val and epochs_no_improve >= patience)
        if is_ddp:
            stop_flag = torch.tensor(
                int(should_stop), device=device, dtype=torch.int32
            )
            dist.broadcast(stop_flag, src=0)
            should_stop = bool(stop_flag.item())

        if should_stop:
            if rank == 0:
                reason_parts = [f"no improvement for {patience} epochs"]
                reason_parts.append(f"best_composite={best_composite:.4f}")
                logger.info(f"\nEarly stopping at epoch {epoch+1}: " + ", ".join(reason_parts))
            break

        epoch += 1

    if rank == 0:
        raw_model = model.module if hasattr(model, 'module') else model
        save_model(
            raw_model,
            os.path.join(run_dir, 'last_weights.pt'),
            epoch=max(epoch - 1, 0),
            metrics={'best_loss': best_loss},
        )
        logger.info("✓ Last model saved: stage->last_weights.pt")

    if is_ddp:
        dist.barrier()

    if rank == 0:
        try:
            logger.info(f"\n{'='*40} Final Evaluation {'='*40}")

            del model, optimizer, scheduler
            if scaler is not None:
                del scaler
            torch.cuda.empty_cache()

            best_path = os.path.join(run_dir, 'best_weights.pt')
            if not os.path.exists(best_path):
                best_path = os.path.join(run_dir, 'last_weights.pt')
                logger.warning("best_weights.pt was not created; evaluating last_weights.pt")
            best_artifact = torch.load(best_path, map_location=device)
            best_sd = dict(extract_model_state(best_artifact))

            eval_model = LaUER(config, mode=mode)
            eval_model.load_pretrained_encoders(device=str(device))
            eval_model = eval_model.to(device)

            eval_model.query_encoder = None

            _load_model_state_checked(
                eval_model, best_sd, best_path, allow_partial=False, logger=logger,
            )
            eval_model = eval_model.to(device)
            eval_model.eval()

            from evaluation.zero_shot import zero_shot_retrieval_eval
            from evaluation.cgl_eval import cgl_accuracy_eval

            def _build_nonddp_loader(dataset, bs, collate_fn, split):
                if dataset is None:
                    return None

                _, v_nw, v_pm, v_persist, _, v_dl = config.data.get_val_loader_cfg()
                return torch.utils.data.DataLoader(
                    dataset, batch_size=bs, shuffle=False,
                    num_workers=v_nw,
                    pin_memory=v_pm,
                    collate_fn=collate_fn, drop_last=v_dl,
                    worker_init_fn=seed_worker,
                    generator=make_data_generator(config.train.seed, split=split),
                    persistent_workers=v_persist,
                )

            eval_collate = (getattr(val_loader, 'collate_fn', None)
                            if val_loader is not None else None)
            _eval_bs, _, _, _, _, _ = config.data.get_val_loader_cfg()
            eval_val_loader = _build_nonddp_loader(
                val_dataset, _eval_bs, eval_collate, 'val')
            eval_test_loader = _build_nonddp_loader(
                test_dataset, _eval_bs, eval_collate, 'test')

            eval_results = {}
            for loader_name, loader in [('Val', eval_val_loader), ('Test', eval_test_loader)]:
                if loader is None:
                    continue
                logger.info(f"\n--- {loader_name} Set ---")
                if is_full:
                    zs = zero_shot_retrieval_eval(
                        eval_model, loader, device,
                        bf16=config.train.bf16, fp16=config.train.fp16,
                    )
                    logger.info(f"\n[Zero-Shot]\n{format_eval_results(zs)}")
                    cgl = cgl_accuracy_eval(
                        eval_model, loader, device,
                        bf16=config.train.bf16, fp16=config.train.fp16,
                    )
                    cgl_accs = ', '.join(
                        f"{k}={v:.4f}" for k, v in cgl.items()
                        if k.startswith('acc_matching') and isinstance(v, (int, float))
                    )
                    logger.info(f"[ITM] {cgl_accs}")
                    eval_results[loader_name] = {'zero_shot': zs, 'cgl': cgl}
                else:
                    zs = zero_shot_retrieval_eval(
                        eval_model, loader, device,
                        bf16=config.train.bf16, fp16=config.train.fp16,
                    )
                    logger.info(f"\n[Zero-Shot]\n{format_eval_results(zs)}")
                    eval_results[loader_name] = zs

            if len(eval_results) == 2:
                logger.info(f"\n{'='*40} Val + Test Averaged {'='*40}")
                avg_result = average_eval_results(eval_results['Val'], eval_results['Test'])
                if isinstance(avg_result, dict) and 'ecg2text' in avg_result:
                    logger.info(format_eval_results(avg_result))
                elif isinstance(avg_result, dict):
                    logger.info(f"\n[Averaged]\n{format_full_eval_results(avg_result)}")

            logger.info("\nTraining complete!")
        except Exception as _final_eval_err:
            logger.error("[Final Eval] FAILED: %r", _final_eval_err)
            logger.error("Final evaluation failed; training artifacts intact.")

    if is_ddp:
        dist.barrier()
        dist.destroy_process_group()

if __name__ == '__main__':

    sys.exit(main())
