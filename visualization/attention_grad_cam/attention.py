"""Capture real attention probabilities; ALBEF A * clamp(dS/dA, min=0)."""
from contextlib import contextmanager
import numpy as np
import torch
from models.attention import _mha_fp32_legacy

# Below this many effective cells (out of the 480 patch bins) the map is a handful of
# isolated dots rather than a distributed attribution; reported, never silently hidden.
SATURATED_CELLS = 60

@contextmanager
def capture_attention(target, owner=None):
    captured = {}
    if owner is None:
        def hook(module, inputs, output):
            if not output.requires_grad:
                output.requires_grad_(True)
            captured['attention'] = output
        handle = target.register_forward_hook(hook)
        try:
            yield captured
        finally:
            handle.remove()
    else:
        original = owner.mha_fp32
        def traced(module, query, key, value, *args, **kwargs):
            if module is not target:
                return original(module, query, key, value, *args, **kwargs)
            # Enable a graph for frozen inference weights without altering values.
            if not query.requires_grad:
                query = query.detach().requires_grad_(True)
            output, attention = _mha_fp32_legacy(module, query, key, value, *args, **kwargs)
            captured['attention'] = attention
            return output, attention
        owner.mha_fp32 = traced
        try:
            yield captured
        finally:
            owner.mha_fp32 = original

def _concentration(positive):
    """Effective number of contributing patches (exp of entropy) out of the 480 grid bins."""
    flat = np.clip(positive, 0, None).ravel()
    total = float(flat.sum())
    if total <= 0:
        return 0.0, 0.0
    p = flat / total
    return float(np.exp(-(p * np.log(p + 1e-12)).sum())), float(np.sort(flat)[-8:].sum() / total)

def attention_cam(score, captured, rows):
    attention = captured['attention']
    gradient = torch.autograd.grad(score, attention)[0]
    if attention.shape[-1] != 481:
        raise ValueError('Expected CLS + 480 memory patches')
    if not torch.isfinite(attention).all() or not torch.isfinite(gradient).all():
        raise ValueError('Non-finite attention or gradient')
    a = attention.detach().float()[0, :, rows, 1:]
    g = gradient.detach().float()[0, :, rows, 1:]
    # Per-head mass parked on the memory CLS token. A head whose row is dominated by CLS
    # carries a near-delta distribution over the remaining patches once column 0 is dropped.
    cls_mass = attention.detach().float()[0, :, rows, 0].mean(1)
    per_query = (a * g.clamp_min(0)).mean(0)
    positive = per_query.mean(0).reshape(12, 40)
    peak = positive.max().item()
    normalized = positive / peak if peak > 0 else positive
    status = 'ok' if peak > 0 else ('zero_gradient' if g.abs().max().item() == 0 else 'no_positive_cam')
    effective_cells, top8_mass = _concentration(positive.cpu().numpy())
    return {'positive': positive.cpu().numpy(), 'normalized': normalized.cpu().numpy(),
            'per_query': per_query.cpu().numpy(), 'attention': a.cpu().numpy(),
            'attention_gradient': g.cpu().numpy()}, {
        'status': status, 'positive_max': peak, 'query_rows': list(rows),
        'effective_cells': round(effective_cells, 1), 'top8_mass': round(top8_mass, 4),
        'saturated': bool(peak > 0 and effective_cells < SATURATED_CELLS),
        'head_cls_mass': [round(float(v), 4) for v in cls_mass.cpu().tolist()],
        'head_patch_sum': [round(float(v), 4) for v in (1 - cls_mass).cpu().tolist()],
        'method': 'ALBEF attention * positive gradient; mean heads then selected queries',
        'normalization': 'positive maximum; cubic resize and Gaussian blur for display only'}
