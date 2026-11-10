"""Optional asymmetric multilabel loss; the paper configuration uses binary cross-entropy."""
import torch
import torch.nn as nn

class AsymmetricLoss(nn.Module):

    def __init__(self, gamma_pos: float = 0.0, gamma_neg: float = 2.0,
                 clip: float = 0.05, eps: float = 1e-8):
        super().__init__()
        if not 0.0 <= clip < 1.0:
            raise ValueError(f"clip(m) must be in [0, 1): got {clip}")
        if eps <= 0.0:
            raise ValueError(f"eps must be positive: got {eps}")
        if gamma_pos < 0.0 or gamma_neg < 0.0:
            raise ValueError(f"focusing exponents must be nonnegative: got gamma_pos={gamma_pos}, gamma_neg={gamma_neg}")
        self.gamma_pos = float(gamma_pos)
        self.gamma_neg = float(gamma_neg)
        self.clip = float(clip)
        self.eps = float(eps)

    def forward(self, logits: torch.Tensor, targets: torch.Tensor,
                label_weights: torch.Tensor = None) -> torch.Tensor:
        targets = targets.to(logits.dtype)
        p = torch.sigmoid(logits)                        # [B, C] ∈ (0, 1)

        p_tilde = (p - self.clip).clamp(min=0.0)         # [B, C] ∈ [0, 1)

        pos_term = targets * (1.0 - p).pow(self.gamma_pos) * torch.log(p + self.eps)

        neg_term = (1.0 - targets) * p_tilde.pow(self.gamma_neg) * torch.log(1.0 - p_tilde + self.eps)

        loss = -(pos_term + neg_term)
        if label_weights is None:
            return loss.mean()
        if label_weights.ndim != 1 or label_weights.numel() != logits.shape[-1]:
            raise ValueError(
                "label_weights must be one-dimensional with length equal to C"
            )
        weights = label_weights.to(device=loss.device, dtype=loss.dtype)
        denominator = loss.shape[0] * weights.sum().clamp_min(self.eps)
        return (loss * weights.unsqueeze(0)).sum() / denominator

    def extra_repr(self) -> str:
        return (f"gamma_pos={self.gamma_pos}, gamma_neg={self.gamma_neg}, "
                f"clip={self.clip}, eps={self.eps}")
