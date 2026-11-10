"""ViT signal encoder for 12-lead ECG recordings with lead-aware positional embeddings."""
import torch
import torch.nn as nn
from einops import rearrange
from einops.layers.torch import Rearrange

class DropPath(nn.Module):
    def __init__(self, drop_prob: float, scale_by_keep: bool = True):
        super().__init__()
        self.drop_prob = drop_prob
        self.scale_by_keep = scale_by_keep

    def forward(self, x):
        if self.drop_prob <= 0. or not self.training:
            return x
        keep_prob = 1 - self.drop_prob
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)
        random_tensor = x.new_empty(shape).bernoulli_(keep_prob)
        if keep_prob > 0.0 and self.scale_by_keep:
            random_tensor.div_(keep_prob)
        return x * random_tensor

class PreNorm(nn.Module):
    def __init__(self, dim: int, fn: nn.Module):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fn = fn

    def forward(self, x, **kwargs):
        return self.fn(self.norm(x), **kwargs)

class FeedForward(nn.Module):
    """FFN: Linear → GELU → Dropout → Linear → Dropout"""
    def __init__(self, input_dim: int, output_dim: int, hidden_dim: int, drop_out_rate: float = 0.):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(drop_out_rate),
            nn.Linear(hidden_dim, output_dim),
            nn.Dropout(drop_out_rate),
        )

    def forward(self, x):
        return self.net(x)

class Attention(nn.Module):
    def __init__(self, input_dim: int, output_dim: int, heads: int = 8,
                 dim_head: int = 64, qkv_bias: bool = True,
                 drop_out_rate: float = 0., attn_drop_out_rate: float = 0.):
        super().__init__()
        inner_dim = dim_head * heads
        project_out = not (heads == 1 and dim_head == input_dim)

        self.heads = heads
        self.scale = dim_head ** -0.5

        self.attend = nn.Softmax(dim=-1)
        self.dropout = nn.Dropout(attn_drop_out_rate)
        self.to_qkv = nn.Linear(input_dim, inner_dim * 3, bias=qkv_bias)

        if project_out:
            self.to_out = nn.Sequential(nn.Linear(inner_dim, output_dim),
                                        nn.Dropout(drop_out_rate))
        else:
            self.to_out = nn.Identity()

    def forward(self, x):
        qkv = self.to_qkv(x).chunk(3, dim=-1)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h=self.heads), qkv)

        with torch.autocast(device_type=x.device.type, enabled=False):
            q_fp32 = q.float()
            k_fp32 = k.float()
            v_fp32 = v.float()
            dots = torch.matmul(q_fp32, k_fp32.transpose(-1, -2)) * self.scale
            attn = self.attend(dots)
            attn = self.dropout(attn)
            out = torch.matmul(attn, v_fp32)

        out = out.to(q.dtype)
        out = rearrange(out, 'b h n d -> b n (h d)')
        out = self.to_out(out)
        return out

class TransformerBlock(nn.Module):
    def __init__(self, input_dim: int, output_dim: int, hidden_dim: int,
                 heads: int = 8, dim_head: int = 32, qkv_bias: bool = True,
                 drop_out_rate: float = 0., attn_drop_out_rate: float = 0.,
                 drop_path_rate: float = 0.):
        super().__init__()
        self.attn = PreNorm(
            dim=input_dim,
            fn=Attention(input_dim=input_dim, output_dim=output_dim,
                         heads=heads, dim_head=dim_head, qkv_bias=qkv_bias,
                         drop_out_rate=drop_out_rate, attn_drop_out_rate=attn_drop_out_rate),
        )
        self.droppath1 = DropPath(drop_path_rate) if drop_path_rate > 0 else nn.Identity()

        self.ff = PreNorm(
            dim=output_dim,
            fn=FeedForward(input_dim=output_dim, output_dim=output_dim,
                           hidden_dim=hidden_dim, drop_out_rate=drop_out_rate),
        )
        self.droppath2 = DropPath(drop_path_rate) if drop_path_rate > 0 else nn.Identity()

    def forward(self, x):
        x = self.droppath1(self.attn(x)) + x
        x = self.droppath2(self.ff(x)) + x
        return x

class ViT1DCore(nn.Module):
    def __init__(self, num_leads: int, seq_len: int, patch_size: int,
                 width: int = 192, depth: int = 12, mlp_dim: int = 768,
                 heads: int = 3, dim_head: int = 64, qkv_bias: bool = True,
                 drop_out_rate: float = 0.1, attn_drop_out_rate: float = 0.,
                 drop_path_rate: float = 0.1,
                 pos_embedding_type: str = "additive"):
        super().__init__()
        self.num_leads = num_leads
        self.width = width
        self.depth = depth
        self.pos_embedding_type = pos_embedding_type
        assert seq_len % patch_size == 0, \
            f'seq_len ({seq_len}) must be divisible by patch_size ({patch_size})'
        self.num_patches = seq_len // patch_size

        self.to_patch_embedding = nn.Sequential(
            Rearrange('b c (n p) -> b n (p c)', p=patch_size),
            nn.LayerNorm(patch_size),
            nn.Linear(patch_size, width),
            nn.LayerNorm(width),
        )

        self.cls_token = nn.Parameter(torch.randn(1, 1, width) * 0.02)

        if pos_embedding_type == "additive":

            self.lead_embeddings = nn.ParameterList(
                nn.Parameter(torch.randn(width)) for _ in range(num_leads)
            )
            self.pos_embedding = nn.Parameter(torch.randn(1, self.num_patches, width))
            self.pos_proj = None
        elif pos_embedding_type == "concat":

            half = width // 2
            self.lead_embeddings = nn.ParameterList(
                nn.Parameter(torch.randn(half)) for _ in range(num_leads)
            )
            self.pos_embedding = nn.Parameter(torch.randn(1, self.num_patches, half))
            self.pos_proj = nn.Linear(width, width)
        else:
            raise ValueError(f"Unknown pos_embedding_type: {pos_embedding_type}")

        self.dropout = nn.Dropout(drop_out_rate)

        drop_path_list = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]
        for i in range(depth):
            block = TransformerBlock(
                input_dim=width, output_dim=width, hidden_dim=mlp_dim,
                heads=heads, dim_head=dim_head, qkv_bias=qkv_bias,
                drop_out_rate=drop_out_rate, attn_drop_out_rate=attn_drop_out_rate,
                drop_path_rate=drop_path_list[i],
            )
            self.add_module(f'block{i}', block)

        self.norm = nn.LayerNorm(width)

    def forward_encoding(self, series):
        B, L, S = series.shape  # [batch, leads, seq_len]
        assert L == self.num_leads, \
            f'Expected {self.num_leads} leads, got {L}'

        x = self.to_patch_embedding(series.reshape(B * L, 1, S))
        x = x.reshape(B, L, self.num_patches, self.width)
        lead_embeddings = torch.stack(tuple(self.lead_embeddings), dim=0)

        if self.pos_embedding_type == "additive":
            x = (x + lead_embeddings.view(1, L, 1, self.width)
                 + self.pos_embedding.view(1, 1, self.num_patches, self.width))
        elif self.pos_embedding_type == "concat":
            half = lead_embeddings.size(-1)
            lead = lead_embeddings.view(1, L, 1, half).expand(
                B, L, self.num_patches, half
            )
            pos = self.pos_embedding.view(1, 1, self.num_patches, half).expand(
                B, L, self.num_patches, half
            )
            x = x + self.pos_proj(torch.cat([lead, pos], dim=-1))

        x = x.reshape(B, L * self.num_patches, self.width)

        cls_tokens = self.cls_token.expand(B, -1, -1)
        x = torch.cat([cls_tokens, x], dim=1)

        # 5. Dropout
        x = self.dropout(x)

        # 6. Transformer blocks
        for i in range(self.depth):
            x = getattr(self, f'block{i}')(x)

        x = self.norm(x)
        return x

    def forward(self, series):
        return self.forward_encoding(series)

class ViT1DEncoder(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.config = config
        self.vit1d_cfg = config.vit1d

        self.encoder = ViT1DCore(
            num_leads=self.vit1d_cfg.lead_num,
            seq_len=self.vit1d_cfg.seq_length,
            patch_size=self.vit1d_cfg.patch_size,
            width=self.vit1d_cfg.width,
            depth=self.vit1d_cfg.depth,
            mlp_dim=self.vit1d_cfg.mlp_dim,
            heads=self.vit1d_cfg.heads,
            dim_head=64,
            qkv_bias=True,
            drop_out_rate=self.vit1d_cfg.dropout,
            attn_drop_out_rate=0.,
            drop_path_rate=self.vit1d_cfg.drop_path_rate,
            pos_embedding_type=self.vit1d_cfg.pos_embedding_type,
        )

        self._loaded = False
        self.width = self.vit1d_cfg.width

        ckpt_path = self.vit1d_cfg.checkpoint_path
        if ckpt_path:
            self.load_checkpoint(ckpt_path)

        if config.freeze:
            self._freeze()

    def load_checkpoint(self, checkpoint_path: str, device: str = 'cpu'):
        import os
        if not os.path.isabs(checkpoint_path):
            from configs.config import get_project_root
            checkpoint_path = os.path.join(get_project_root(), checkpoint_path)

        print(f"[ViT1DEncoder] Loading checkpoint: {checkpoint_path}")
        ckpt = torch.load(checkpoint_path, map_location=device)

        if 'state_dict' in ckpt:
            state_dict = ckpt['state_dict']
        elif 'model_state_dict' in ckpt:
            state_dict = ckpt['model_state_dict']
        else:
            state_dict = ckpt

        prefix_variants = ['', 'encoder.', 'ecg_encoder.', 'ecg_encoder.encoder.']
        matched = False
        for prefix in prefix_variants:
            stripped = {k[len(prefix):] if k.startswith(prefix) else k: v
                        for k, v in state_dict.items()}

            model_keys = set(self.encoder.state_dict().keys())
            filtered = {k: v for k, v in stripped.items() if k in model_keys}
            if len(filtered) > 0:
                missing, unexpected = self.encoder.load_state_dict(filtered, strict=False)
                print(f"[ViT1DEncoder] Loaded with prefix='{prefix}': "
                      f"{len(filtered)} matched, {len(missing)} missing, {len(unexpected)} unexpected")
                if missing:
                    print(f"  Missing keys (first 10): {missing[:10]}")
                matched = True
                break

        if not matched:
            print(f"[ViT1DEncoder] WARNING: No matching keys found in checkpoint!")

        self._loaded = True

    def load_from_checkpoint(self, checkpoint_path: str, device: str = 'cpu'):
        return self.load_checkpoint(checkpoint_path, device=device)

    def _freeze(self):
        for param in self.encoder.parameters():
            param.requires_grad = False
        self.encoder.eval()
        print(f"[ViT1DEncoder] Frozen all parameters")

    def train(self, mode=True):
        super().train(mode)
        if self.config.freeze:
            self.encoder.eval()
        return self

    def forward(self, ecg_signal):
        pooled, _ = self.encode_both(ecg_signal)
        return pooled

    def encode_tokens(self, ecg_signal):
        _, tokens = self.encode_both(ecg_signal)
        return tokens

    def encode_both(self, ecg_signal):
        tokens = self.encoder(ecg_signal)  # [B, 481, width]
        pooled = tokens[:, 0]              # [B, width]  CLS token
        return pooled, tokens

    @property
    def device(self):
        return next(self.encoder.parameters()).device

    @property
    def dtype(self):
        return next(self.encoder.parameters()).dtype
