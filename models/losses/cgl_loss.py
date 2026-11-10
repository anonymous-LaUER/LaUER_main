"""Signal-image matching head, hard-negative sampling, and the matching component of CGL."""
import torch
import torch.nn as nn
import torch.nn.functional as F

class MatchingHead(nn.Module):
    """Two-logit classifier over the fused signal-image CLS token."""

    def __init__(self, hidden_size=768, dropout=0.1):
        super().__init__()
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        self.fc = nn.Linear(hidden_size, 2)

    def forward(self, cls_embedding):
        """
        Args:
            cls_embedding: [N, 768]

        Returns:
            logits: [N, 2]
        """
        return self.fc(self.dropout(cls_embedding))

def hard_negative_mining(sim_matrix):
    """Sample off-diagonal negatives; return a safe fallback for a singleton batch."""
    with torch.no_grad():
        B = sim_matrix.size(0)
        if B <= 1:

            return torch.zeros(B, dtype=torch.long, device=sim_matrix.device)
        weights = F.softmax(sim_matrix, dim=1)  # [B, B]
        weights.fill_diagonal_(0)

        row_sum = weights.sum(dim=1)
        degenerate = ~(torch.isfinite(row_sum) & (row_sum > 0.0))
        bad = weights.masked_fill(
            torch.eye(B, dtype=torch.bool, device=weights.device),
            float('-inf'),
        )

        nan_rows = torch.isnan(bad).masked_fill(
            torch.eye(B, dtype=torch.bool, device=weights.device), True,
        ).all(dim=1)
        fallback = torch.where(
            nan_rows,
            torch.arange(B, device=weights.device).add(1).remainder(B),
            bad.argmax(dim=1),
        )                                          # [B] long
        fallback_dist = F.one_hot(fallback, num_classes=B).to(weights.dtype)
        sampling = torch.where(degenerate[:, None], fallback_dist, weights)
        neg = torch.multinomial(sampling, 1).squeeze(-1)
        return neg

class CGLMatchingLoss(nn.Module):
    """Cross-entropy matching supervision for one positive and two hard-negative pairs."""

    def __init__(self, hidden_size=768, num_stages=2, dropout=0.1,
                 nan_debug=False):
        super().__init__()
        self.num_stages = num_stages
        self.nan_debug = bool(nan_debug)
        self.heads = nn.ModuleList([
            MatchingHead(hidden_size, dropout) for _ in range(num_stages)
        ])

    def compute_stage_loss(self, stage, cls_pos, cls_neg1, cls_neg2):
        """Compute matching loss from positive and two negative fused CLS features."""
        if stage < 1 or stage > self.num_stages:
            raise ValueError(f"Invalid CGL stage: {stage} (num_stages={self.num_stages})")
        head = self.heads[stage - 1]
        return self._compute_stage_loss(cls_pos, cls_neg1, cls_neg2, head)

    def _compute_stage_loss(self, cls_pos, cls_neg1, cls_neg2, head):
        import logging
        _logger = logging.getLogger(__name__)

        B = cls_pos.size(0)
        cls_all = torch.cat([cls_pos, cls_neg1, cls_neg2], dim=0)  # [3B, D]
        logits = head(cls_all)  # [3B, 2]

        if self.nan_debug and not torch.isfinite(logits).all():
            n_nan = torch.isnan(logits).sum().item()
            n_inf = torch.isinf(logits).sum().item()
            _logger.error(
                f"[CGL NaN] Stage head logits [{list(logits.shape)}]: "
                f"NaN={n_nan} Inf={n_inf} | "
                f"CLS stats: pos_mean={cls_pos.mean():.4f} pos_std={cls_pos.std():.4f} "
                f"neg1_mean={cls_neg1.mean():.4f} neg2_mean={cls_neg2.mean():.4f}"
            )

        labels = torch.zeros(3 * B, dtype=torch.long, device=cls_pos.device)
        labels[:B] = 1
        loss = F.cross_entropy(logits, labels)

        if self.nan_debug and not torch.isfinite(loss):
            _logger.error(
                f"[CGL NaN] Stage CrossEntropy loss is NaN/Inf: "
                f"logits range=[{logits.min():.4f}, {logits.max():.4f}] "
                f"mean={logits.mean():.4f} std={logits.std():.4f}"
            )

        acc = (logits.argmax(dim=1) == labels).float().mean().detach()
        return loss, acc
