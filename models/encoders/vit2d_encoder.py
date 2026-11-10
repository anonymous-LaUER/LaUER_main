"""Lead-aware ViT image encoder with token layout aligned to the ECG signal encoder."""
import torch
import torch.nn as nn
from einops import rearrange

from models.encoders.vit1d_encoder import TransformerBlock

_IMAGENET_MEAN = (0.485, 0.456, 0.406)
_IMAGENET_STD = (0.229, 0.224, 0.225)

def _to_float_normalized(images: torch.Tensor) -> torch.Tensor:
    if images.dtype == torch.uint8:
        x = images.float().div(255.0)
        mean = torch.tensor(_IMAGENET_MEAN, device=x.device)
        std = torch.tensor(_IMAGENET_STD, device=x.device)
        if x.dim() == 3:
            mean = mean.view(3, 1, 1)
            std = std.view(3, 1, 1)
        else:                                               # batch [B, 3, H, W]
            mean = mean.view(1, 3, 1, 1)
            std = std.view(1, 3, 1, 1)
        return (x - mean) / std
    return images

class ViT2DCore(nn.Module):

    def __init__(self, image_height: int, image_width: int,
                 lead_num: int, patch_height: int, patch_width: int,
                 width: int = 192, depth: int = 12, mlp_dim: int = 768,
                 heads: int = 3, dim_head: int = 64, qkv_bias: bool = True,
                 drop_out_rate: float = 0.1, attn_drop_out_rate: float = 0.,
                 drop_path_rate: float = 0.1,
                 pos_embedding_type: str = "additive"):
        super().__init__()

        assert image_height % lead_num == 0, \
            f'image_height ({image_height}) must be divisible by lead_num ({lead_num})'
        assert image_height // lead_num == patch_height, \
            f'patch_height ({patch_height}) must equal image_height/lead_num ({image_height // lead_num})'
        assert image_width % patch_width == 0, \
            f'image_width ({image_width}) must be divisible by patch_width ({patch_width})'

        self.image_height = image_height
        self.image_width = image_width
        self.lead_num = lead_num
        self.patch_height = patch_height
        self.patch_width = patch_width
        self.width = width
        self.depth = depth
        self.pos_embedding_type = pos_embedding_type

        self.num_patches_per_lead = image_width // patch_width
        self.num_patches_total = lead_num * self.num_patches_per_lead

        patch_dim = 3 * patch_height * patch_width  # 3*60*25 = 4500
        self.patch_proj = nn.Sequential(
            nn.LayerNorm(patch_dim),
            nn.Linear(patch_dim, width),
            nn.LayerNorm(width),
        )

        self.cls_token = nn.Parameter(torch.randn(1, 1, width) * 0.02)

        if pos_embedding_type == "additive":

            self.lead_embeddings = nn.ParameterList(
                nn.Parameter(torch.randn(width)) for _ in range(lead_num)
            )
            self.pos_embedding = nn.Parameter(
                torch.randn(1, self.num_patches_per_lead, width)
            )
            self.pos_proj = None
        elif pos_embedding_type == "concat":

            half = width // 2
            self.lead_embeddings = nn.ParameterList(
                nn.Parameter(torch.randn(half)) for _ in range(lead_num)
            )
            self.pos_embedding = nn.Parameter(
                torch.randn(1, self.num_patches_per_lead, half)
            )
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

    def _patchify_lead(self, images, lead_idx):
        B, C, H, W = images.shape
        ph = self.patch_height  # 60
        pw = self.patch_width   # 25

        y_start = lead_idx * ph
        lead_row = images[:, :, y_start:y_start + ph, :]  # [B, 3, 60, 1000]

        patches = rearrange(
            lead_row,
            'b c h (n p) -> b n (c h p)',
            p=pw,
            n=self.num_patches_per_lead,
        )  # [B, 40, 3*60*25=4500]

        return patches

    def forward_encoding(self, images):

        images = _to_float_normalized(images)
        B, C, H, W = images.shape
        expected_h = self.lead_num * self.patch_height
        if H != expected_h or W != self.num_patches_per_lead * self.patch_width:
            raise ValueError(
                f"Expected image [B, C, {expected_h}, "
                f"{self.num_patches_per_lead * self.patch_width}], got {list(images.shape)}"
            )

        patches = rearrange(
            images,
            'b c (l ph) (n pw) -> (b l) n (c ph pw)',
            l=self.lead_num, ph=self.patch_height,
            n=self.num_patches_per_lead, pw=self.patch_width,
        )
        x = self.patch_proj(patches).reshape(
            B, self.lead_num, self.num_patches_per_lead, self.width
        )
        lead_embeddings = torch.stack(tuple(self.lead_embeddings), dim=0)

        if self.pos_embedding_type == "additive":
            x = (x + lead_embeddings.view(1, self.lead_num, 1, self.width)
                 + self.pos_embedding.view(
                     1, 1, self.num_patches_per_lead, self.width
                 ))
        elif self.pos_embedding_type == "concat":
            half = lead_embeddings.size(-1)
            lead = lead_embeddings.view(1, self.lead_num, 1, half).expand(
                B, self.lead_num, self.num_patches_per_lead, half
            )
            pos = self.pos_embedding.view(
                1, 1, self.num_patches_per_lead, half
            ).expand(B, self.lead_num, self.num_patches_per_lead, half)
            x = x + self.pos_proj(torch.cat([lead, pos], dim=-1))

        x = x.reshape(B, self.lead_num * self.num_patches_per_lead, self.width)

        cls_tokens = self.cls_token.expand(B, -1, -1)
        x = torch.cat([cls_tokens, x], dim=1)

        # 6. Dropout
        x = self.dropout(x)

        # 7. Transformer blocks
        for i in range(self.depth):
            x = getattr(self, f'block{i}')(x)

        x = self.norm(x)
        return x

    def forward(self, images):
        return self.forward_encoding(images)

class ViT2DEncoder(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.config = config
        self.vit2d_cfg = config.vit2d

        self.encoder = ViT2DCore(
            image_height=self.vit2d_cfg.image_height,
            image_width=self.vit2d_cfg.image_width,
            lead_num=self.vit2d_cfg.lead_num,
            patch_height=self.vit2d_cfg.patch_height,
            patch_width=self.vit2d_cfg.patch_width,
            width=self.vit2d_cfg.width,
            depth=self.vit2d_cfg.depth,
            mlp_dim=self.vit2d_cfg.mlp_dim,
            heads=self.vit2d_cfg.heads,
            dim_head=64,
            qkv_bias=True,
            drop_out_rate=self.vit2d_cfg.dropout,
            attn_drop_out_rate=0.,
            drop_path_rate=self.vit2d_cfg.drop_path_rate,
            pos_embedding_type=self.vit2d_cfg.pos_embedding_type,
        )

        self._loaded = False
        self.width = self.vit2d_cfg.width

        ckpt_path = self.vit2d_cfg.checkpoint_path
        if ckpt_path:
            self.load_checkpoint(ckpt_path)

        if config.freeze:
            self._freeze()

    def load_checkpoint(self, checkpoint_path: str, device: str = 'cpu'):
        import os
        if not os.path.isabs(checkpoint_path):
            from configs.config import get_project_root
            checkpoint_path = os.path.join(get_project_root(), checkpoint_path)

        print(f"[ViT2DEncoder] Loading checkpoint: {checkpoint_path}")
        ckpt = torch.load(checkpoint_path, map_location=device)

        if 'state_dict' in ckpt:
            state_dict = ckpt['state_dict']
        elif 'model_state_dict' in ckpt:
            state_dict = ckpt['model_state_dict']
        else:
            state_dict = ckpt

        prefix_variants = ['', 'encoder.', 'image_encoder.', 'image_encoder.encoder.']
        matched = False
        for prefix in prefix_variants:
            stripped = {k[len(prefix):] if k.startswith(prefix) else k: v
                        for k, v in state_dict.items()}
            model_keys = set(self.encoder.state_dict().keys())
            filtered = {k: v for k, v in stripped.items() if k in model_keys}
            if len(filtered) > 0:
                missing, unexpected = self.encoder.load_state_dict(filtered, strict=False)
                print(f"[ViT2DEncoder] Loaded with prefix='{prefix}': "
                      f"{len(filtered)} matched, {len(missing)} missing, {len(unexpected)} unexpected")
                if missing:
                    print(f"  Missing keys (first 10): {missing[:10]}")
                matched = True
                break

        if not matched:
            print(f"[ViT2DEncoder] WARNING: No matching keys found in checkpoint!")

        self._loaded = True

    def load_model(self):
        if not self._loaded and self.vit2d_cfg.checkpoint_path:
            self.load_checkpoint(self.vit2d_cfg.checkpoint_path)
        elif self._loaded:
            print(f"[ViT2DEncoder] Already loaded, skipping.")
        else:
            print(f"[ViT2DEncoder] No checkpoint specified, training from scratch.")

    def _freeze(self):
        for param in self.encoder.parameters():
            param.requires_grad = False
        self.encoder.eval()
        print(f"[ViT2DEncoder] Frozen all parameters")

    def train(self, mode=True):
        super().train(mode)
        if self.config.freeze:
            self.encoder.eval()
        return self

    def forward(self, images):
        pooled, _ = self.encode_both(images)
        return pooled

    def encode_tokens(self, images):
        _, tokens = self.encode_both(images)
        return tokens

    def encode_both(self, images):
        tokens = self.encoder(images)  # [B, 481, width]
        pooled = tokens[:, 0]          # [B, width]  CLS token
        return pooled, tokens

    @property
    def device(self):
        return next(self.encoder.parameters()).device

    @property
    def dtype(self):
        return next(self.encoder.parameters()).dtype
