"""Token-granularity language-model loss.

The paper uses causal teacher forcing and an equal weighted average of report
and explanation token losses. Masked-language utilities support optional
hybrid settings; they are not used by the supplied autoregressive configuration."""
import torch
import torch.nn as nn
import torch.nn.functional as F

class TGLLoss(nn.Module):
    """Region-balanced autoregressive loss and optional hybrid masking utilities."""

    def __init__(self, config, mask_token_id=None, sep_token_id=None,
                 pad_token_id=None, nan_debug=False):
        super().__init__()
        self.report_mask_rate = config.report_mask_rate    # 0.95
        self.full_mask_prob = config.full_mask_prob        # 0.2
        self.full_mask_rampup = getattr(config, 'full_mask_rampup', False)
        self.mlm_loss_weight = config.mlm_loss_weight      # 1.0 (β)
        self.ar_loss_weight = config.ar_loss_weight        # 0.5 (α)
        self.vocab_size = config.vocab_size                # 30522
        self.mask_token_id = mask_token_id
        self.sep_token_id = sep_token_id
        self.pad_token_id = pad_token_id
        self.nan_debug = bool(nan_debug)

        if hasattr(config, 'effective_loss_mode'):
            self.loss_mode = config.effective_loss_mode
        else:
            self.loss_mode = getattr(config, 'loss_mode', 'hybrid')
        self.ar_only = (self.loss_mode == 'ar')
        rw = getattr(config, 'region_weights', None)
        # Equal regional weights prevent the longer explanation from dominating the report.
        self.region_weights = (float(rw[0]), float(rw[1])) if rw is not None else (0.5, 0.5)

    # BERT Masking

    def bert_mask_report(self, input_ids, sep_positions, epoch_progress=None):
        B, L = input_ids.shape
        device = input_ids.device
        masked_ids = input_ids.clone()

        if self.full_mask_rampup and epoch_progress is not None:
            fm_prob = min(0.5, 0.2 + 0.3 * epoch_progress)
        else:
            fm_prob = self.full_mask_prob

        pos_idx = torch.arange(L, device=device).unsqueeze(0)     # [1, L]
        sep = sep_positions.unsqueeze(-1)                          # [B, 1]
        is_report = (pos_idx >= 1) & (pos_idx <= sep)             # [B, L]

        if self.training:
            full_mask_sample = torch.rand(B, device=device) < fm_prob  # [B]
        else:
            full_mask_sample = torch.zeros(B, dtype=torch.bool, device=device)

        full_mask_ids = is_report & full_mask_sample.unsqueeze(-1)  # [B, L]
        masked_ids[full_mask_ids] = self.mask_token_id

        rand_mask = torch.rand(B, L, device=device)
        rand_choice = torch.rand(B, L, device=device)

        partial = (is_report & (~full_mask_sample.unsqueeze(-1))
                   & (rand_mask < self.report_mask_rate))

        has_report = is_report.any(dim=-1)                          # [B]
        needs_partial = has_report & (~full_mask_sample)             # [B]
        partial_count = partial.sum(dim=-1)                          # [B]
        needs_force = needs_partial & (partial_count == 0)           # [B]

        force_mask = torch.zeros(B, L, dtype=torch.bool, device=device)
        if L > 1:
            force_mask[:, 1] = needs_force
        pos1_valid = (sep.squeeze(-1) > 1)                          # [B]
        partial = partial | (force_mask & pos1_valid.unsqueeze(-1))

        is_mask_token = partial & (rand_choice < 0.8)
        is_random_token = partial & (rand_choice >= 0.8) & (rand_choice < 0.9)

        masked_ids[is_mask_token] = self.mask_token_id

        random_words = torch.randint(
            self.mask_token_id + 1, self.vocab_size, (B, L), device=device,
        )
        masked_ids[is_random_token] = random_words[is_random_token]

        return masked_ids

    # Loss Computation

    def compute_loss(self, logits, input_ids, masked_ids, sep_positions,
                     mlm_weight=None, ar_weight=None):
        import logging
        _logger = logging.getLogger(__name__)

        if mlm_weight is None:
            mlm_weight = self.mlm_loss_weight
        if ar_weight is None:
            ar_weight = self.ar_loss_weight

        _, L, _ = logits.shape

        # --- MLM Loss + Accuracy ---
        mlm_mask = (masked_ids == self.mask_token_id)  # [B, L]
        mlm_logits = logits[mlm_mask]      # [N_mlm, V]
        mlm_labels = input_ids[mlm_mask]   # [N_mlm]
        mlm_correct = torch.zeros((), device=logits.device, dtype=torch.long)
        mlm_total = mlm_labels.numel()

        sep_mask = mlm_mask & (input_ids == self.sep_token_id)  # [B, L]

        sep_total = sep_mask.sum()
        sep_correct = torch.zeros((), device=logits.device, dtype=torch.long)
        if mlm_total > 0:

            if self.nan_debug and not torch.isfinite(mlm_logits).all():
                n_nan = torch.isnan(mlm_logits).sum().item()
                n_inf = torch.isinf(mlm_logits).sum().item()

                valid_mlm = mlm_logits[~torch.isnan(mlm_logits).any(dim=-1)]
                vmin = valid_mlm.min().item() if valid_mlm.numel() > 0 else float('nan')
                vmax = valid_mlm.max().item() if valid_mlm.numel() > 0 else float('nan')
                _logger.error(
                    f"[MLM NaN] mlm_logits [{list(mlm_logits.shape)}]: "
                    f"NaN={n_nan} Inf={n_inf} | "
                    f"valid_range=[{vmin:.4f}, {vmax:.4f}]"
                )

            L_mlm = F.cross_entropy(mlm_logits.float(), mlm_labels, reduction='mean')

            if self.nan_debug and not torch.isfinite(L_mlm):
                _logger.error(
                    f"[MLM NaN] MLM loss is NaN/Inf: "
                    f"logits range=[{mlm_logits.min():.4f}, {mlm_logits.max():.4f}] "
                    f"labels unique={mlm_labels.unique().tolist()[:10]}..."
                )

            mlm_correct = (mlm_logits.argmax(dim=-1) == mlm_labels).sum().detach()

            sep_correct = (
                logits[sep_mask].argmax(dim=-1) == input_ids[sep_mask]
            ).sum().detach()
        else:
            L_mlm = torch.tensor(0.0, device=logits.device)

        row = torch.arange(L - 1, device=logits.device).view(1, L - 1)
        sep = sep_positions.view(-1, 1)
        ar_mask = (row >= sep)
        pad_mask = (input_ids[:, 1:] != self.pad_token_id)
        valid = ar_mask & pad_mask
        selected_ar_logits = logits[:, :-1, :][valid]
        selected_ar_labels = input_ids[:, 1:][valid]
        ar_tokens = selected_ar_labels.numel()
        if ar_tokens > 0:

            if self.nan_debug and not torch.isfinite(selected_ar_logits).all():
                n_nan = torch.isnan(selected_ar_logits).sum().item()
                n_inf = torch.isinf(selected_ar_logits).sum().item()
                _logger.error(
                    f"[AR NaN] ar_logits[valid] [{list(selected_ar_logits.shape)}]: "
                    f"NaN={n_nan} Inf={n_inf} | "
                    f"ar_tokens={ar_tokens}"
                )

            L_ar = F.cross_entropy(
                selected_ar_logits.float(), selected_ar_labels, reduction='sum'
            )

            if self.nan_debug and not torch.isfinite(L_ar):
                _logger.error(
                    f"[AR NaN] AR loss is NaN/Inf: "
                    f"logits range=[{selected_ar_logits.min():.4f}, {selected_ar_logits.max():.4f}] "
                    f"ar_tokens={ar_tokens}"
                )
        else:
            L_ar = torch.tensor(0.0, device=logits.device)

        return L_mlm, L_ar, mlm_correct, mlm_total, ar_tokens, sep_correct, sep_total

    # Combined Forward

    def forward(self, input_ids, sep_positions, cross_kv, tgl,
                epoch_progress=None):
        import logging
        _logger = logging.getLogger(__name__)

        if self.mask_token_id is None or self.sep_token_id is None or self.pad_token_id is None:
            raise RuntimeError(
                "[TGLLoss] Special tokens not injected. "
                "Call model._detect_special_tokens() or model.load_pretrained_encoders() first."
            )

        if self.ar_only:
            return self._forward_ar(input_ids, sep_positions, cross_kv, tgl,
                                    epoch_progress=epoch_progress)
        # 1. BERT mask
        masked_ids = self.bert_mask_report(
            input_ids, sep_positions, epoch_progress=epoch_progress,
        )

        non_pad = (input_ids != self.pad_token_id)           # [B, L]
        last_valid = non_pad.any(dim=0).nonzero()             # [K]
        if last_valid.numel() > 0:
            max_len = last_valid[-1].item() + 1
            if max_len < input_ids.size(1):
                input_ids = input_ids[:, :max_len]
                masked_ids = masked_ids[:, :max_len]

        # 3. Decoder forward
        logits = tgl(masked_ids, sep_positions, cross_kv)  # [B, max_len, V]

        if self.nan_debug:
            if torch.isnan(logits).any() or torch.isinf(logits).any():
                n_nan = torch.isnan(logits).sum().item()
                n_inf = torch.isinf(logits).sum().item()

                nan_per_sample = torch.isnan(logits).any(dim=-1).any(dim=-1)  # [B]
                bad_samples = nan_per_sample.nonzero(as_tuple=True)[0].tolist()
                _logger.error(
                    f"[Decoder NaN] Decoder logits [{list(logits.shape)}]: "
                    f"NaN={n_nan} Inf={n_inf} | "
                    f"affected samples: {bad_samples[:5]}... | "
                    f"cross_kv stats: mean={cross_kv.mean():.4f} std={cross_kv.std():.4f}"
                )

        # 4. Compute losses + accuracy
        L_mlm, L_ar, mlm_correct, mlm_total, ar_tokens, sep_correct, sep_total = \
            self.compute_loss(logits, input_ids, masked_ids, sep_positions)

        return (L_mlm, L_ar, mlm_correct, mlm_total, ar_tokens, sep_correct, sep_total,
                torch.zeros((), device=logits.device, dtype=torch.long),
                torch.zeros((), device=logits.device, dtype=torch.long),
                torch.zeros((), device=logits.device, dtype=torch.long),
                torch.zeros((), device=logits.device, dtype=torch.long))

    def _forward_ar(self, input_ids, sep_positions, cross_kv, tgl,
                    epoch_progress=None):
        import logging
        _logger = logging.getLogger(__name__)

        masked_ids = input_ids

        non_pad = (input_ids != self.pad_token_id)          # [B, L]
        last_valid = non_pad.any(dim=0).nonzero()
        if last_valid.numel() > 0:
            max_len = last_valid[-1].item() + 1
            if max_len < input_ids.size(1):
                input_ids = input_ids[:, :max_len]
                masked_ids = input_ids

        causal_sep = torch.zeros(input_ids.size(0), device=input_ids.device,
                                 dtype=torch.long)
        logits = tgl(masked_ids, causal_sep, cross_kv)  # [B, L, V]

        boundary = (masked_ids == self.sep_token_id).long().argmax(dim=1).clamp(min=1)

        L_mlm = torch.tensor(0.0, device=logits.device)
        L_ar, ar_tokens, rep_c, rep_t, expl_c, expl_t = self._ar_region_loss(
            logits, masked_ids, boundary,
        )
        return (L_mlm, L_ar, 0, 0, ar_tokens, 0, 0,
                rep_c, rep_t, expl_c, expl_t)

    def _ar_region_loss(self, logits, input_ids, boundary):
        shift_logits = logits[:, :-1, :]
        shift_labels = input_ids[:, 1:]                        # [B, L-1]
        target_pos = torch.arange(1, logits.size(1), device=logits.device).view(1, -1)
        valid = (shift_labels != self.pad_token_id)
        rep_mask = valid & (target_pos <= boundary.unsqueeze(-1))
        expl_mask = valid & (target_pos > boundary.unsqueeze(-1))

        rep_logits = shift_logits[rep_mask].float()            # [n_rep, V] fp32
        expl_logits = shift_logits[expl_mask].float()          # [n_expl, V] fp32
        rep_labels = shift_labels[rep_mask]
        expl_labels = shift_labels[expl_mask]

        n_rep = rep_mask.sum().detach()
        n_expl = expl_mask.sum().detach()

        L_rep = F.cross_entropy(rep_logits, rep_labels, reduction='mean')
        L_expl = F.cross_entropy(expl_logits, expl_labels, reduction='mean')
        w0, w1 = self.region_weights
        L_ar = torch.where(
            (n_rep > 0) & (n_expl > 0), w0 * L_rep + w1 * L_expl,
            torch.where(n_expl > 0, L_expl, L_rep),
        )
        L_ar = torch.where(
            (n_rep <= 0) & (n_expl <= 0),
            torch.zeros((), device=logits.device, dtype=L_ar.dtype), L_ar,
        )
        ar_tokens = n_rep + n_expl                              # detached 0-dim tensor

        pred = shift_logits.argmax(dim=-1)
        rep_correct = ((pred == shift_labels) & rep_mask).sum().detach()
        expl_correct = ((pred == shift_labels) & expl_mask).sum().detach()
        rep_total = rep_mask.sum().detach()
        expl_total = expl_mask.sum().detach()
        return L_ar, ar_tokens, rep_correct, rep_total, expl_correct, expl_total

    @torch.no_grad()
    def ablation_zero_cross_attn(self, input_ids, sep_positions, cross_kv, tgl):

        masked_ids = self._static_mask(input_ids, sep_positions)

        # Decoder forward with zero cross_kv
        zero_kv = torch.zeros_like(cross_kv)
        logits = tgl(masked_ids, sep_positions, zero_kv)  # [B, L, V]

        # MLM accuracy only
        mlm_mask = (masked_ids == self.mask_token_id)
        if mlm_mask.any():
            pred = logits[mlm_mask].argmax(dim=-1)
            labels = input_ids[mlm_mask]
            correct = (pred == labels).sum().item()
            total = labels.size(0)
        else:
            correct, total = 0, 0
        return correct, total

    def _static_mask(self, input_ids, sep_positions):
        B, L = input_ids.shape
        device = input_ids.device
        masked_ids = input_ids.clone()

        pos_idx = torch.arange(L, device=device).unsqueeze(0)
        sep = sep_positions.unsqueeze(-1)

        is_report = (pos_idx >= 1) & (pos_idx <= sep)

        rand_mask = torch.rand(B, L, device=device)
        rand_choice = torch.rand(B, L, device=device)
        mask_here = is_report & (rand_mask < self.report_mask_rate)

        is_mask_token = mask_here & (rand_choice < 0.8)
        is_random_token = mask_here & (rand_choice >= 0.8) & (rand_choice < 0.9)

        masked_ids[is_mask_token] = self.mask_token_id
        random_words = torch.randint(
            self.mask_token_id + 1, self.vocab_size, (B, L), device=device
        )
        masked_ids[is_random_token] = random_words[is_random_token]

        return masked_ids
