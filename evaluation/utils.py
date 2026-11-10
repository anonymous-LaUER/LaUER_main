"""Checkpoint loading and shared retrieval, matching, and multilabel evaluation utilities."""
import os
from contextlib import nullcontext
import numpy as np
import torch

def evaluation_autocast(device, bf16=False, fp16=False):
    if torch.device(device).type != 'cuda':
        return nullcontext()
    if bf16 and fp16:
        raise ValueError("fp16 and bf16 cannot both be enabled")
    if bf16:
        if not getattr(torch.cuda, 'is_bf16_supported', lambda: False)():
            raise RuntimeError("BF16 evaluation requested but the CUDA device does not support it")
        return torch.amp.autocast('cuda', dtype=torch.bfloat16)
    if fp16:
        return torch.amp.autocast('cuda', dtype=torch.float16)
    return nullcontext()

def average_retrieval_metrics(result_a, result_b):
    keys = ['R@1', 'R@5', 'R@10', 'MedianRank', 'MeanRank']
    return {key: (result_a[key] + result_b[key]) / 2 for key in keys}

def average_eval_results(res_val, res_test):
    """Average available validation/test metrics using their returned metric keys."""
    avg = {}
    for direction in ['ecg2text', 'text2ecg', 'image2text', 'text2image',
                       'ecg2image', 'image2ecg']:
        if direction in res_val and direction in res_test:
            avg[direction] = average_retrieval_metrics(
                res_val[direction], res_test[direction])
    if 'cgl' in res_val and 'cgl' in res_test:
        cgl_avg = {
            'acc_matching': (res_val['cgl']['acc_matching'] + res_test['cgl']['acc_matching']) / 2,
        }
        avg['cgl'] = cgl_avg
    return avg

def format_full_eval_results(result):
    lines = ["-" * 40]
    if 'cgl' in result:
        lines.append(f"  CGL Fusion Match Acc: {result['cgl']['acc_matching']:.4f}")
    lines.append("-" * 40)
    return "\n".join(lines)

def _check_eval_mode_compatibility(model, checkpoint, state_dict, logger):
    from training.stage_io import MODE_RANK, artifact_source_mode
    model_mode = getattr(model, 'mode', None)
    if model_mode is None or model_mode not in MODE_RANK:
        logger.warning(f"[Checkpoint] Model mode {model_mode!r} not recognized; skip mode check")
        return
    try:
        ckpt_mode = artifact_source_mode(checkpoint, state_dict)
    except Exception as error:
        logger.warning(f"[Checkpoint] Cannot infer checkpoint source mode: {error}")
        return
    if MODE_RANK[model_mode] > MODE_RANK[ckpt_mode]:
        raise RuntimeError(
            f"Checkpoint mode mismatch: checkpoint source={ckpt_mode!r} is a "
            f"predecessor stage, but evaluation mode={model_mode!r} requires "
            f"{MODE_RANK[model_mode]}-stage weights. CGL/metrics would run on "
            f"randomly initialized weights. Use a {model_mode!r} checkpoint "
            "(or --mode sgl for the SGL-only subset)."
        )
    logger.info(f"[Checkpoint] source_mode={ckpt_mode} compatible with eval mode={model_mode}")

def load_checkpoint_weights(model, checkpoint_path, device, logger):
    checkpoint = torch.load(checkpoint_path, map_location=device)

    if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
        state_dict = checkpoint['model_state_dict']
        logger.info("Unwrapped model_state_dict from training checkpoint")
    else:
        state_dict = checkpoint

    _check_eval_mode_compatibility(model, checkpoint, state_dict, logger)

    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    logger.info(f"Loaded weights: {len(state_dict)} params, missing={len(missing)}, unexpected={len(unexpected)}")
    cgl_missing = [k for k in missing if 'cgl' in k.lower() or 'fusion' in k.lower()]

    from training.stage_io import FROZEN_EXCLUDED_PREFIXES
    non_cgl_missing = [
        k for k in missing
        if k not in cgl_missing and not k.startswith(FROZEN_EXCLUDED_PREFIXES)
    ]
    if non_cgl_missing:
        logger.warning(f"Unexpected missing keys: {non_cgl_missing}")

def compute_multilabel_map(scores, labels):
    if torch.is_tensor(scores):
        scores = scores.detach().cpu().numpy()
    if torch.is_tensor(labels):
        labels = labels.detach().cpu().numpy()
    aps = []
    for c in range(labels.shape[1]):
        lc = labels[:, c]
        n_pos = int(lc.sum())
        if n_pos == 0:
            continue
        order = np.argsort(-scores[:, c])
        lc = lc[order]
        tp = np.cumsum(lc)
        prec = tp / np.arange(1, len(lc) + 1)
        aps.append(float((prec * lc).sum() / n_pos))
    return float(np.mean(aps)) if aps else 0.0

def multilabel_metrics(logits, labels):
    logits = logits.float()
    labels = labels.float()
    if float(labels.sum()) <= 0:
        return None
    probs = torch.sigmoid(logits)
    map_score = compute_multilabel_map(probs, labels)
    preds = (probs > 0.5).float()
    tp = (preds * labels).sum(0)
    fp = (preds * (1 - labels)).sum(0)
    fn = ((1 - preds) * labels).sum(0)
    prec = tp / (tp + fp + 1e-8)
    rec = tp / (tp + fn + 1e-8)
    f1 = 2 * prec * rec / (prec + rec + 1e-8)
    return {
        'mlc_map': map_score,
        'mlc_macro_f1': float(f1.mean()),
        'mlc_n': int(labels.shape[0]),
    }
