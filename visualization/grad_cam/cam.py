"""Token Grad-CAM with explicit diagnostics (CLS is never a spatial bin)."""
import numpy as np
import torch

class Capture:
    def __init__(self, module):
        self.value = None
        self.handle = module.register_forward_hook(self.hook)

    def hook(self, module, inputs, output):
        # Frozen models still need activation gradients, not parameter updates.
        if not output.requires_grad:
            output.requires_grad_(True)
        self.value = output

    def close(self):
        self.handle.remove()
        self.value = None

def token_cam(score, activation):
    if activation is None or activation.shape[1] != 481:
        raise ValueError('CAM requires CLS + 12 x 40 tokens')
    grad = torch.autograd.grad(score, activation)[0].float()
    act = activation.detach().float()
    if not torch.isfinite(grad).all() or not torch.isfinite(act).all():
        raise ValueError('Non-finite activation/gradient')
    weights = grad[:, 1:].mean(dim=1, keepdim=True)
    signed = (weights * act[:, 1:]).sum(-1)[0].reshape(12, 40)
    positive = signed.clamp_min(0)
    peak = positive.max().item()
    normalized = positive / peak if peak > 0 else positive
    status = ('zero_gradient' if grad[:, 1:].abs().max().item() == 0 else
              'no_positive_cam' if peak == 0 else 'ok')
    return {'signed': signed.cpu().numpy(), 'positive': positive.cpu().numpy(),
            'normalized': normalized.cpu().numpy()}, {
                'status': status, 'positive_max': peak,
                'patch_gradient_max': grad[:, 1:].abs().max().item(),
                'cls_gradient_max': grad[:, :1].abs().max().item(),
                'normalization': 'global positive maximum; no per-lead normalization'}

def generation_regions(ids, sep_id, report_marker, explanation_marker):
    """Return absolute target-token positions; logits at position t-1 predict t."""
    values = list(ids)
    boundaries = [i for i, value in enumerate(values) if value == sep_id]
    first = boundaries[0] if boundaries else len(values)
    second = boundaries[1] if len(boundaries) > 1 else len(values)
    start = first + 1
    if values[start:start + len(explanation_marker)] == list(explanation_marker):
        start += len(explanation_marker)
    return {'report': list(range(1 + len(report_marker), first)),
            'explanation': list(range(start, second))}, len(boundaries) >= 2

def text_score(logits, ids, positions):
    if not positions or min(positions) < 1 or max(positions) >= ids.shape[1]:
        raise ValueError('Empty/invalid generated token selection')
    positions = torch.tensor(positions, device=logits.device)
    return logits[0, positions - 1].float().log_softmax(-1).gather(
        1, ids[0, positions, None]).mean()
