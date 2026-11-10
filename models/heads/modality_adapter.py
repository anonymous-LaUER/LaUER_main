"""Modality-specific SGL adapters and token-preserving CGL adapters.

SGL uses CLS pooling in the supplied configurations. CGL projects signal and
image tokens independently; clinical-text tokens never enter ECG fusion."""
import torch
import torch.nn as nn
import torch.nn.functional as F

# MLP Builder

def build_mlp(input_dim: int, output_dim: int, num_layers: int,
              hidden_dim: int = 0, activation: str = "none",
              layernorm: float = 0.0, dropout: float = 0.0,
              layernorm_position: str = "hidden_pre") -> nn.Sequential:
    """Build a configurable MLP with explicit activation and LayerNorm placement."""
    if num_layers < 1:
        raise ValueError(f"num_layers must be >= 1, got {num_layers}")

    valid_norm_positions = {
        "hidden_pre", "hidden_post", "output", "both_pre", "both_post",
    }
    if layernorm_position not in valid_norm_positions:
        raise ValueError(
            f"Unsupported layernorm_position={layernorm_position!r}; "
            f"expected one of {sorted(valid_norm_positions)}"
        )

    def _add_hidden_layers(seq, dim):
        if layernorm > 0 and layernorm_position in {"hidden_pre", "both_pre"}:
            seq.append(nn.LayerNorm(dim, eps=layernorm))
        if activation == "gelu":
            seq.append(nn.GELU())
        if layernorm > 0 and layernorm_position in {"hidden_post", "both_post"}:
            seq.append(nn.LayerNorm(dim, eps=layernorm))
        if dropout > 0:
            seq.append(nn.Dropout(dropout))

    def _add_output_norm(seq):
        if layernorm > 0 and layernorm_position in {"output", "both_pre", "both_post"}:
            seq.append(nn.LayerNorm(output_dim, eps=layernorm))

    if num_layers == 1:
        layers = [nn.Linear(input_dim, output_dim, bias=False)]
        _add_hidden_layers(layers, output_dim)
        _add_output_norm(layers)
        return nn.Sequential(*layers)

    if hidden_dim <= 0:
        hidden_dim = (input_dim + output_dim) // 2

    layers = [nn.Linear(input_dim, hidden_dim, bias=False)]
    _add_hidden_layers(layers, hidden_dim)

    for _ in range(num_layers - 2):
        layers.append(nn.Linear(hidden_dim, hidden_dim, bias=False))
        _add_hidden_layers(layers, hidden_dim)

    layers.append(nn.Linear(hidden_dim, output_dim, bias=False))
    _add_output_norm(layers)

    return nn.Sequential(*layers)

# Pooling Functions

class TokenPooler(nn.Module):
    """Pool token features using the configured strategy and optional padding mask."""

    def __init__(self, pooling: str = "cls"):
        super().__init__()
        if pooling not in ("cls", "mean", "attn", "cls+attn", "lead_attn"):
            raise ValueError(
                f"Unknown pooling type: {pooling}. "
                f"Use 'cls', 'mean', 'attn', 'cls+attn' or 'lead_attn'."
            )
        self.pooling = pooling

    def forward(self, tokens, mask=None):
        if self.pooling == "cls":
            return tokens[:, 0, :]

        if mask is not None:
            masked_sum = (tokens * mask.unsqueeze(-1).float()).sum(dim=1)  # [B, D]
            count = mask.sum(dim=1).clamp(min=1).unsqueeze(-1).float()     # [B, 1]
            return masked_sum / count
        return tokens.mean(dim=1)

class AttnPooler(nn.Module):
    """Learnable attention pooling over non-CLS tokens."""

    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim
        self.query = nn.Parameter(torch.zeros(1, 1, dim))
        nn.init.trunc_normal_(self.query, std=0.02)
        self.scale = dim ** -0.5

    def forward(self, tokens, mask=None):
        x = tokens[:, 1:, :]
        scores = (self.query * self.scale) @ x.transpose(-1, -2)   # [B, 1, N-1]
        if mask is not None:
            m = mask[:, 1:].unsqueeze(1)                           # [B, 1, N-1]
            scores = scores.masked_fill(~m, float('-inf'))
        attn = F.softmax(scores, dim=-1)
        return (attn @ x).squeeze(1)                               # [B, D]

class LeadAttnPooler(nn.Module):
    """Pool each lead independently using its own learnable query."""

    def __init__(self, dim: int, num_leads: int = 12):
        super().__init__()
        self.dim = dim
        self.num_leads = num_leads

        self.query = nn.Parameter(torch.zeros(num_leads, 1, dim))
        nn.init.trunc_normal_(self.query, std=0.02)
        self.scale = dim ** -0.5
        self._mask_cache = None   # (P, mask)

    def _lead_mask(self, num_patches, device, dtype):
        if self._mask_cache is not None and self._mask_cache[0] == num_patches:
            return self._mask_cache[1]
        mask = torch.full(
            (self.num_leads, num_patches), float('-inf'), device=device, dtype=dtype
        )
        p = num_patches // self.num_leads
        for k in range(self.num_leads):
            mask[k, k * p:(k + 1) * p] = 0.0
        self._mask_cache = (num_patches, mask)
        return mask

    def forward(self, tokens):
        x = tokens[:, 1:, :]
        B, N1, D = x.shape
        assert N1 % self.num_leads == 0,             f"[LeadAttnPooler] {N1} patches not divisible by {self.num_leads} leads"
        scores = torch.einsum('ld,bnd->bln', self.query.squeeze(1), x)  # [B, L, N-1]
        scores = scores * self.scale
        mask = self._lead_mask(N1, x.device, x.dtype)          # [L, N-1]
        scores = scores + mask.unsqueeze(0)
        attn = F.softmax(scores, dim=-1)
        return torch.einsum('bln,bnd->bld', attn, x)           # [B, L, D]

class ModalityAdapter(nn.Module):

    def __init__(self, sgl_cfg, cgl_cfg=None, use_shared_proj1=False,
                 enable_cgl=True):
        super().__init__()

        self.proj1_sgl = build_mlp(
            sgl_cfg.proj1.input_dim, sgl_cfg.proj1.output_dim,
            sgl_cfg.proj1.num_layers,
            hidden_dim=sgl_cfg.proj1.hidden_dim,
            activation=sgl_cfg.proj1.activation,
            layernorm=sgl_cfg.proj1.layernorm,
            layernorm_position=sgl_cfg.proj1.layernorm_position,
            dropout=sgl_cfg.proj1.dropout,
        )

        self._cgl_enabled = bool(enable_cgl)
        if self._cgl_enabled and not use_shared_proj1 and cgl_cfg is not None:
            self.proj1_cgl = build_mlp(
                cgl_cfg.proj1.input_dim, cgl_cfg.proj1.output_dim,
                cgl_cfg.proj1.num_layers,
                hidden_dim=cgl_cfg.proj1.hidden_dim,
                activation=cgl_cfg.proj1.activation,
                layernorm=cgl_cfg.proj1.layernorm,
                layernorm_position=cgl_cfg.proj1.layernorm_position,
                dropout=cgl_cfg.proj1.dropout,
            )
        else:
            self.proj1_cgl = None

        self.proj2 = build_mlp(
            sgl_cfg.proj2.input_dim, sgl_cfg.proj2.output_dim,
            sgl_cfg.proj2.num_layers,
            hidden_dim=sgl_cfg.proj2.hidden_dim,
            activation=sgl_cfg.proj2.activation,
            layernorm=sgl_cfg.proj2.layernorm,
            layernorm_position=sgl_cfg.proj2.layernorm_position,
            dropout=sgl_cfg.proj2.dropout,
        )
        self.pooler = TokenPooler(sgl_cfg.pooling)

        #   cls+attn: proj1.input = 2×enc       → enc = input // 2
        #   lead_attn: proj1.input = enc + L×k  → enc = input - L×k  (k = lead_pooling_dim)
        if sgl_cfg.pooling == "lead_attn":
            enc_dim = (
                sgl_cfg.proj1.input_dim
                - sgl_cfg.lead_num * sgl_cfg.lead_pooling_dim
            )
        elif sgl_cfg.pooling == "cls+attn":
            enc_dim = sgl_cfg.proj1.input_dim // 2
        else:
            enc_dim = sgl_cfg.proj1.input_dim

        self.attn_pooler = None
        if sgl_cfg.pooling in ("attn", "cls+attn"):
            self.attn_pooler = AttnPooler(enc_dim)

        self.lead_pooler = None
        self.lead_proj = None
        if sgl_cfg.pooling == "lead_attn":
            num_leads = sgl_cfg.lead_num
            lead_k = sgl_cfg.lead_pooling_dim
            assert enc_dim > 0 and lead_k > 0,                 f"[ModalityAdapter] bad lead_attn dims: enc={enc_dim} k={lead_k}"
            self.lead_pooler = LeadAttnPooler(enc_dim, num_leads)
            self.lead_proj = nn.Linear(enc_dim, lead_k, bias=False)
        self._shared = use_shared_proj1

    def forward_sgl(self, raw_tokens, mask=None):
        p = self.pooler.pooling
        if p == "cls":
            pooled = self.proj1_sgl(raw_tokens[:, 0, :])                 # [B, D_mid]
        elif p == "mean":
            x = self.proj1_sgl(raw_tokens)                               # [B, N, D_mid]
            pooled = self.pooler(x, mask)                                 # [B, D_mid]
        elif p == "attn":
            pooled = self.proj1_sgl(self.attn_pooler(raw_tokens, mask))  # [B, D_mid]
        elif p == "cls+attn":
            pooled = torch.cat([                                        # [B, 2D]
                raw_tokens[:, 0, :],                                    # [B, D] CLS
                self.attn_pooler(raw_tokens, mask),
            ], dim=-1)
            pooled = self.proj1_sgl(pooled)                            # [B, D_mid]
        elif p == "lead_attn":
            lead = self.lead_pooler(raw_tokens)                         # [B, L, enc_dim]  (L=12)
            lead_flat = self.lead_proj(lead)                            # [B, L, lead_k]  (lead_k=lead_pooling_dim=48)
            lead_flat = lead_flat.reshape(lead_flat.size(0), -1)        # [B, L*lead_k] = [B, 576]
            pooled = torch.cat([                                        # [B, enc_dim + L*lead_k] = [B, 768]
                raw_tokens[:, 0, :], lead_flat,
            ], dim=-1)
            pooled = self.proj1_sgl(pooled)                            # [B, D_mid]
        else:
            raise ValueError(f"Unknown pooling mode: {p!r}")
        embed = self.proj2(pooled)          # [B, D_mid] → [B, out]
        return F.normalize(embed, dim=-1)

    def forward_sgl_from_pooled(self, pooled_feat):
        if pooled_feat.size(-1) != self.proj2[0].in_features:
            raise RuntimeError(
                f"[ModalityAdapter.forward_sgl_from_pooled] Input dim "
                f"{pooled_feat.size(-1)} != proj2 input_dim {self.proj2[0].in_features}. "
                f"Use forward_sgl(raw_tokens) instead, or adjust proj2 config."
            )
        return F.normalize(self.proj2(pooled_feat), dim=-1)

    def forward_cgl(self, raw_tokens):
        if not self._cgl_enabled:
            raise RuntimeError(
                "CGL projection was not created for this SGL-only model. "
                "Construct LaUER with mode='sgl+cgl' or "
                "mode='sgl+cgl+tgl'."
            )
        if self.proj1_cgl is not None:
            return self.proj1_cgl(raw_tokens)
        return self.proj1_sgl(raw_tokens)

# LaUERAdapters

class LaUERAdapters(nn.Module):
    """Independent semantic embedding and concept-token projections for each modality."""

    def __init__(self, config, enable_cgl=True):
        super().__init__()
        shared = getattr(config, 'use_shared_proj1', False)
        self.enable_cgl = bool(enable_cgl)
        self.ecg_adapter = ModalityAdapter(
            config.ecg.sgl, config.ecg.cgl, use_shared_proj1=shared,
            enable_cgl=self.enable_cgl,
        )
        self.image_adapter = ModalityAdapter(
            config.image.sgl, config.image.cgl, use_shared_proj1=shared,
            enable_cgl=self.enable_cgl,
        )

        self.text_adapter = ModalityAdapter(
            config.text.sgl, config.text.cgl, use_shared_proj1=shared,
            enable_cgl=False,
        )

        self._print_stats()

    def _print_stats(self):
        total = 0
        if not self.enable_cgl:
            n_layers = 6
        elif self.ecg_adapter._shared:
            n_layers = 7
        else:
            n_layers = 6 + sum(
                1 for adapter in (self.ecg_adapter, self.image_adapter, self.text_adapter)
                if adapter.proj1_cgl is not None
            )
        lines = [f"[LaUERAdapters] {n_layers} projection layers"
                 + (" (SGL only)" if not self.enable_cgl else
                    " (shared proj1)" if self.ecg_adapter._shared else "")]
        for name, adapter in [('ECG', self.ecg_adapter), ('Image', self.image_adapter),
                               ('Text', self.text_adapter)]:
            n_sgl = sum(p.numel() for p in adapter.proj1_sgl.parameters())
            n_cgl = sum(p.numel() for p in adapter.proj1_cgl.parameters()) \
                     if adapter.proj1_cgl is not None else 0
            n_proj2 = sum(p.numel() for p in adapter.proj2.parameters())
            n_total = n_sgl + n_cgl + n_proj2
            total += n_total
            line = f"  {name:6s}: proj1_sgl={n_sgl:,}"
            if adapter.proj1_cgl is not None:
                line += f"  proj1_cgl={n_cgl:,}"
            line += f"  proj2={n_proj2:,}  (total={n_total:,})"
            lines.append(line)
        lines.append(f"  {'Total':6s}: {total:,}")
        print('\n'.join(lines))

    def forward(self, ecg_tokens, image_tokens, text_tokens, text_mask=None):
        return (
            self.ecg_adapter.forward_sgl(ecg_tokens),
            self.image_adapter.forward_sgl(image_tokens),
            self.text_adapter.forward_sgl(text_tokens, text_mask),
        )

    def forward_combined(self, ecg_tokens, image_tokens, text_tokens, text_mask=None):
        """Return normalized SGL embeddings and CGL token projections in one pass."""
        if not self.enable_cgl:
            raise RuntimeError(
                "forward_combined() requires CGL projections, but this adapter "
                "container was created for SGL only."
            )

        def _combined_one(adapter, raw_tokens, mask=None, need_tokens=True):

            embed = adapter.forward_sgl(raw_tokens, mask=mask)
            if adapter.proj1_cgl is None:

                projected_tokens = (
                    adapter.proj1_sgl(raw_tokens) if need_tokens else None
                )
            else:
                projected_tokens = adapter.proj1_cgl(raw_tokens)
            return embed, projected_tokens

        ecg_embed, ecg_proj = _combined_one(self.ecg_adapter, ecg_tokens, need_tokens=True)
        img_embed, img_proj = _combined_one(self.image_adapter, image_tokens, need_tokens=True)
        txt_embed, txt_proj = _combined_one(
            self.text_adapter, text_tokens, mask=text_mask, need_tokens=False,
        )

        return (ecg_embed, img_embed, txt_embed), (ecg_proj, img_proj, txt_proj)

    def project_tokens(self, ecg_tokens, image_tokens, text_tokens):
        """Project signal and image tokens for CGL without passing text through fusion."""
        return (
            self.ecg_adapter.forward_cgl(ecg_tokens),
            self.image_adapter.forward_cgl(image_tokens),
            self.text_adapter.forward_cgl(text_tokens) if text_tokens is not None else None,
        )

    def ecg_proj(self, tokens):
        """ECG tokens → SGL embedding [B, 256] L2-normed。"""
        return self.ecg_adapter.forward_sgl(tokens)

    def image_proj(self, tokens):
        """Image tokens → SGL embedding [B, 256] L2-normed。"""
        return self.image_adapter.forward_sgl(tokens)

    def text_proj(self, tokens):
        """Text tokens → SGL embedding [B, 256] L2-normed。"""
        return self.text_adapter.forward_sgl(tokens)
