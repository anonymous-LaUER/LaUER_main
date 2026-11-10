"""LaUER: Language-anchored Unified ECG Representation Learning.

SGL aligns signal, image, and clinical-text embeddings. CGL learns a shared
signal-image memory through matching and query-guided disease tracing. TGL
adds autoregressive report and explanation generation from that same memory.
Training is cumulative: sgl -> sgl+cgl -> sgl+cgl+tgl."""
import logging
import torch
import torch.nn as nn
import torch.distributed as dist
import torch.nn.functional as F
from models.gather_layer import gather_features, gather_features_nograd

from models.heads.modality_adapter import LaUERAdapters
from models.losses.sgl_loss import SGLLoss
from models.losses.cgl_loss import CGLMatchingLoss, hard_negative_mining
from models.fusion_encoder import FusionEncoder, FusionLayer
from models.query_guided_tracing_head import QueryGuidedTracingHead

logger = logging.getLogger(__name__)

# These code identities are retained for downstream compatibility but describe
# the same clinical concept.  Their CGL/TGL loss contribution is divided
# within each present group so a synonym-rich concept is not over-counted.
_MLC_SYNONYM_CODE_GROUPS = (
    ("VPC", "PVC", "VPB"),
    ("PAC", "APB"),
    ("LNGQT", "QTIE"),
    ("LPR", "PRIE"),
    ("IVCD", "IDC"),
    ("STACH", "ST"),
    ("SBRAD", "SB"),
    ("SARRH", "SA"),
    ("AFLT", "AF"),
    ("SVTAC", "SVT"),
    ("WAVN", "SAAWR"),
    ("INVT", "TWO"),
    ("LAFB", "LFBBB"),
)

def _safe_item(v):
    """Convert a tensor or numeric value to a Python scalar; map None to zero."""
    if isinstance(v, torch.Tensor):
        return v.item()
    return float(v) if v is not None else 0.0

class LaUER(nn.Module):
    """Unified ECG model with cumulative SGL, CGL, and TGL objectives."""

    def __init__(self, config, mode: str = "sgl"):
        super().__init__()
        if mode not in ("sgl", "sgl+cgl", "sgl+cgl+tgl"):
            raise ValueError(f"Unsupported LaUER training mode: {mode!r}")
        self.config = config
        self.mode = mode
        self._device = None
        self._nan_signal = None

        # Accumulate device-side finite checks without adding GPU synchronizations.
        self._nan_accum = None

        self.sgl_only = (mode == 'sgl')
        self.has_cgl = (mode in ('sgl+cgl', 'sgl+cgl+tgl'))
        self.has_tgl = (mode == 'sgl+cgl+tgl')
        if self.has_tgl:
            mode_desc = 'SGL+CGL+TGL'
        elif self.has_cgl:
            mode_desc = 'SGL+CGL'
        else:
            mode_desc = 'SGL'
        print(f"[LaUER] Mode: {mode_desc}")

        from models.encoders.vit1d_encoder import ViT1DEncoder
        from models.encoders.vit2d_encoder import ViT2DEncoder
        from models.encoders.text_encoder import TextEncoder

        self.ecg_encoder = ViT1DEncoder(config.ecg_encoder)
        self.image_encoder = ViT2DEncoder(config.image_encoder)
        self.text_encoder = TextEncoder(config.text_encoder)

        self.adapters = LaUERAdapters(config.projection, enable_cgl=self.has_cgl)
        self.sgl_loss = SGLLoss(config.sgl)
        self.sgl_weight = config.sgl.sgl_in_global

        if not self.has_cgl:
            self.cgl_weight = 0.0
            return

        cgl_cfg = config.cgl
        self.cgl_weight = cgl_cfg.cgl_in_global
        self.w_matching, self.w_multilabel = cgl_cfg.loss_weights

        # Begin with image tokens; each block reads an original projected memory.
        fusion_cfg = cgl_cfg.fusion
        if fusion_cfg.num_layers > 0:
            if (len(fusion_cfg.cross_kv_sequence) != fusion_cfg.num_layers
                    or len(fusion_cfg.detach_cross_kv) != fusion_cfg.num_layers):
                raise ValueError(
                    f"fusion.cross_kv_sequence ({len(fusion_cfg.cross_kv_sequence)} entries) "
                    f"and fusion.detach_cross_kv ({len(fusion_cfg.detach_cross_kv)} entries) "
                    f"lengths must equal fusion.num_layers ({fusion_cfg.num_layers})"
                )
            self._use_alternating_fusion = True
            self.fusion_layer1_primary = fusion_cfg.layer1_primary
            self.fusion_cross_kv_seq = fusion_cfg.cross_kv_sequence
            self.fusion_detach = fusion_cfg.detach_cross_kv
            self.fusion_si_layers = nn.ModuleList([
                FusionLayer(
                    cgl_cfg.hidden_size, cgl_cfg.num_heads,
                    cgl_cfg.intermediate_size, cgl_cfg.dropout,
                )
                for _ in range(fusion_cfg.num_layers)
            ])
        else:
            self._use_alternating_fusion = False
            self.fusion_si = FusionEncoder(
                hidden_size=cgl_cfg.hidden_size,
                num_heads=cgl_cfg.num_heads,
                intermediate_size=cgl_cfg.intermediate_size,
                num_layers=fusion_cfg.simple_num_layers,
                dropout=cgl_cfg.dropout,
            )

        if self._use_alternating_fusion:
            print(f"[LaUER] Fusion alternating: {fusion_cfg.num_layers}L "
                  f"primary={self.fusion_layer1_primary}, "
                  f"cross_kv={list(self.fusion_cross_kv_seq)}")
        else:
            print(f"[LaUER] Fusion simple FusionEncoder: {fusion_cfg.simple_num_layers}L")

        self.cgl_loss_module = CGLMatchingLoss(
            hidden_size=cgl_cfg.hidden_size, num_stages=1,
            dropout=cgl_cfg.cgl_head_dropout,
            nan_debug=getattr(self.config.train, 'nan_debug', False),
        )

        ml_cfg = cgl_cfg.multilabel
        self.query_guided_tracing_head = QueryGuidedTracingHead(
            text_dim=self.config.text_encoder.hidden_size,
            hidden_size=cgl_cfg.hidden_size,
            num_layers=ml_cfg.num_layers,
            num_heads=ml_cfg.num_heads,
            intermediate_size=ml_cfg.intermediate_size,
            dropout=ml_cfg.dropout,
            query_proj_hidden=ml_cfg.query_proj_hidden,
            query_self_attn=ml_cfg.query_self_attn,
            query_proj=ml_cfg.query_proj,
        )

        if ml_cfg.loss_type == 'asl':
            from models.losses.asl_loss import AsymmetricLoss
            self.asl_loss = AsymmetricLoss(
                gamma_pos=ml_cfg.asl_gamma_pos, gamma_neg=ml_cfg.asl_gamma_neg,
                clip=ml_cfg.asl_clip, eps=ml_cfg.asl_eps)
            self.bce_loss = None
            print(f"[LaUER] MLC loss = ASL(gamma_pos={self.asl_loss.gamma_pos}, "
                  f"gamma_neg={self.asl_loss.gamma_neg}, clip={self.asl_loss.clip})")
        else:
            self.asl_loss = None
            self.bce_loss = nn.BCEWithLogitsLoss()
        self.label_names = None
        self.register_buffer(
            "_mlc_label_weights", torch.empty(0, dtype=torch.float32),
            persistent=False,
        )
        self._prompt_features = None

        self.query_encoder = TextEncoder(self._make_query_encoder_config())
        self._query_encoder_released = False

        if self.has_tgl:
            from models.tgl import AutoregressiveDecoder
            from models.losses.tgl_loss import TGLLoss
            from models.heads.modality_adapter import build_mlp
            self.tgl = AutoregressiveDecoder(config.tgl)
            self.tgl_loss = TGLLoss(
                config.tgl, nan_debug=getattr(config.train, 'nan_debug', False)
            )
            self.lm_weight = config.tgl.lm_weight

            self.loss_mode_ar = (config.tgl.effective_loss_mode == 'ar')
            n_tgl = sum(p.numel() for p in self.tgl.parameters())
            print(f"[LaUER] TGL text decoder: {config.tgl.num_layers}L, "
                  f"{n_tgl/1e6:.1f}M params, "
                  f"lm_weight={self.lm_weight}, "
                  f"mask_rate={config.tgl.report_mask_rate}, "
                  f"ar_weight={config.tgl.ar_loss_weight}, "
                  f"mode={config.tgl.effective_loss_mode}")

            # TGL adapter: fused_si → MLP → tgl cross-attn
            dproj_cfg = config.projection.tgl_proj
            self.allow_tgl_grad_to_encoder = dproj_cfg.allow_tgl_grad_to_encoder
            if dproj_cfg.enable:
                if dproj_cfg.num_layers == 0:
                    self.tgl_proj = nn.Identity()
                else:
                    self.tgl_proj = build_mlp(
                        dproj_cfg.input_dim, dproj_cfg.output_dim,
                        dproj_cfg.num_layers,
                        hidden_dim=dproj_cfg.hidden_dim,
                        activation=dproj_cfg.activation,
                        layernorm=dproj_cfg.layernorm,
                        layernorm_position=dproj_cfg.layernorm_position,
                        dropout=dproj_cfg.dropout,
                    )
                n_dproj = sum(p.numel() for p in self.tgl_proj.parameters()
                             if p.requires_grad)
                print(f"[LaUER] TGLAdapter: {dproj_cfg.num_layers}L, "
                      f"{dproj_cfg.input_dim}→{dproj_cfg.hidden_dim}→{dproj_cfg.output_dim}, "
                      f"{n_dproj/1e6:.3f}M params")
            else:
                self.tgl_proj = nn.Identity()
        else:
            self.tgl = None
            self.tgl_loss = None
            self.tgl_proj = nn.Identity()
            self.lm_weight = 0.0
            self.allow_tgl_grad_to_encoder = False
            self.loss_mode_ar = False

    @property
    def device(self):

        # Computing this on demand also avoids a stale cached CPU value after
        # ``model.to(cuda)``.
        trainable = next((p for p in self.parameters() if p.requires_grad), None)
        if trainable is not None:
            return trainable.device
        parameter = next(iter(self.parameters()), None)
        if parameter is not None:
            return parameter.device
        buffer = next(iter(self.buffers()), None)
        if buffer is not None:
            return buffer.device
        raise RuntimeError("LaUER has no parameters or buffers to determine its device")

    @staticmethod
    def _make_embedding_dict(ecg_embed, image_embed, text_embed):
        return dict(
            ecg_embed=ecg_embed.detach(), image_embed=image_embed.detach(),
            text_embed=text_embed.detach(),
        )

    @staticmethod
    def _pad_text_tokens(text_tokens_list, dim, device, dtype):
        B = len(text_tokens_list)
        max_len = max(t.size(0) for t in text_tokens_list)
        tokens = torch.zeros(B, max_len, dim, device=device, dtype=dtype)
        mask = torch.zeros(B, max_len, dtype=torch.bool, device=device)
        for i, t in enumerate(text_tokens_list):
            L = t.size(0)
            tokens[i, :L] = t.to(device)
            mask[i, :L] = True
        return tokens, mask

    @staticmethod
    def _build_cgl_triplet(primary, cross_kv, sim_matrix, mask=None):
        """Construct matched pairs plus one incorrect image and one incorrect signal per record."""
        neg_p = hard_negative_mining(sim_matrix.T)
        neg_c = hard_negative_mining(sim_matrix)
        p_triplet = torch.cat([primary, primary, primary[neg_p]], dim=0)
        c_triplet = torch.cat([cross_kv, cross_kv[neg_c], cross_kv], dim=0)
        m_triplet = torch.cat([mask, mask, mask[neg_p]], dim=0) if mask is not None else None
        return p_triplet, c_triplet, m_triplet

    def _run_alternating_fusion(self, primary_tokens, cross_kv_tokens):
        """Fuse projected ECG views while alternating cross-attention to original signal/image memories."""
        if self.fusion_layer1_primary == "image":
            fused = cross_kv_tokens
        else:
            fused = primary_tokens

        for i, layer in enumerate(self.fusion_si_layers):
            cross_kv_src = self.fusion_cross_kv_seq[i]
            cross_kv = cross_kv_tokens if cross_kv_src == "image" else primary_tokens
            if self.fusion_detach[i]:
                cross_kv = cross_kv.detach()
            fused = layer(primary=fused, cross_kv=cross_kv)

        return fused

    def _build_loss_dict(self, sgl_dict, total_loss, matching_loss, matching_accuracy,
                          lm_losses=None, lm_stats=None, ar_tokens=None,
                          sep_stats=None, region_stats=None):
        d = {
            'loss_total': total_loss.detach(),
            'loss_sgl': sgl_dict['loss_sgl'].detach(),
            'loss_ecg_text': sgl_dict['loss_ecg_text'].detach(),
            'loss_image_text': sgl_dict['loss_image_text'].detach(),
            'loss_ecg_image': sgl_dict['loss_ecg_image'].detach(),
            'temperature': sgl_dict['temperature'].detach(),
        }
        d['loss_matching'] = matching_loss.detach()
        d['acc_matching'] = (matching_accuracy.detach()
                             if isinstance(matching_accuracy, torch.Tensor)
                             else matching_accuracy)
        if lm_losses is not None:
            d['loss_mlm'] = lm_losses[0].detach()
            d['loss_ar'] = lm_losses[1].detach()
        if lm_stats is not None:
            d['mlm_correct'] = lm_stats[0]
            d['mlm_total'] = lm_stats[1]
        if sep_stats is not None:
            d['sep_correct'] = sep_stats[0]
            d['sep_total'] = sep_stats[1]
        if ar_tokens is not None:
            d['ar_tokens'] = ar_tokens
        if region_stats is not None:
            d['report_correct'] = region_stats[0]
            d['report_total'] = region_stats[1]
            d['expl_correct'] = region_stats[2]
            d['expl_total'] = region_stats[3]
        return d

    def _compute_tgl_loss(self, text_input_ids, fused_si_pos):
        device = text_input_ids.device
        sep_id = getattr(self, 'sep_token_id', 102)
        sep_mask = (text_input_ids == sep_id)
        sep_positions = sep_mask.float().argmax(dim=1)

        sep_positions = sep_positions.clamp(min=1)

        if not hasattr(self, '_sep_diag_done'):
            self._sep_diag_done = True
            n_found = sep_mask.any(dim=1).sum().item()
            if n_found < sep_mask.size(0):
                logger.warning(
                    f"[TGL] SEP token (id={sep_id}) not found in "
                    f"{sep_mask.size(0) - n_found}/{sep_mask.size(0)} samples. "
                    f"Sample token IDs (first 50): {text_input_ids[0, :50].tolist()}. "
                    f"Verify text_separator='[SEP]' and tokenizer SEP id."
                )

        epoch_progress = getattr(self, 'epoch_progress', None)
        if not self.allow_tgl_grad_to_encoder:
            fused_si_pos = fused_si_pos.detach()
        fused_si_proj = self.tgl_proj(fused_si_pos)  # [B, N, 768] → [B, N, D]
        (L_mlm, L_ar, mlm_correct, mlm_total, ar_tokens,
         sep_correct, sep_total, rep_correct, rep_total,
         expl_correct, expl_total) = self.tgl_loss(
            text_input_ids, sep_positions, fused_si_proj, self.tgl,
            epoch_progress=epoch_progress,
        )

        return (L_mlm.to(device), L_ar.to(device), mlm_correct, mlm_total,
                ar_tokens, sep_correct, sep_total,
                rep_correct, rep_total, expl_correct, expl_total)

    def ablation_mlm_acc(self, text_input_ids, fused_si_pos, zero_cross_attn=False):
        cross_kv = torch.zeros_like(fused_si_pos) if zero_cross_attn else fused_si_pos
        cross_kv = self.tgl_proj(cross_kv)

        sep_positions = (text_input_ids == self.sep_token_id).float().argmax(dim=1)
        sep_positions = sep_positions.clamp(min=1)
        result = self.tgl_loss.forward(
            text_input_ids,
            sep_positions,
            cross_kv,
            self.tgl,
        )
        return result[2], result[3]  # mlm_correct, mlm_total

    @torch.no_grad()
    def ablation_ar_acc(self, input_ids, fused_si_pos, zero_cross_attn=False):
        cross_kv = torch.zeros_like(fused_si_pos) if zero_cross_attn else fused_si_pos
        cross_kv = self.tgl_proj(cross_kv)
        sep_positions = (input_ids == self.sep_token_id).float().argmax(dim=1).clamp(min=1)
        result = self.tgl_loss.forward(
            input_ids, sep_positions, cross_kv, self.tgl,
        )

        rep_c, rep_t, expl_c, expl_t = result[7], result[8], result[9], result[10]
        return int(rep_c) + int(expl_c), int(rep_t) + int(expl_t)

    def _make_query_encoder_config(self):
        from dataclasses import replace
        return replace(
            self.config.text_encoder,
            model_name_or_path=self.config.text_encoder.query_model_name_or_path,
            tokenizer_name_or_path=self.config.text_encoder.query_model_name_or_path,
        )

    def _rebuild_query_encoder(self):
        if not getattr(self, 'has_cgl', False):
            return None
        from models.encoders.text_encoder import TextEncoder
        self.query_encoder = TextEncoder(self._make_query_encoder_config())
        self.query_encoder.to(self.device)
        self._query_encoder_released = False
        return self.query_encoder

    def prepare_prompt_bank(self, label_names, device=None):
        """Encode disease queries once, preserve CSV label order, and release the frozen query backbone."""
        if not label_names:
            raise ValueError("label_names must be provided for multi-label classification")
        self.label_names = list(label_names)
        if device is None:
            device = self.device
        self._prepare_mlc_label_weights(self.label_names, device)
        if getattr(self, 'query_encoder', None) is None:
            self._rebuild_query_encoder()
        self.encode_prompt_bank(self.label_names, device)

        # Disease features remain cached; the frozen language backbone can be released.
        self.query_encoder = None
        self._query_encoder_released = True
        print(f"[LaUER] Prompt bank cached [{len(self.label_names)}, 768]; "
              "query encoder released (VRAM freed)")

    def _prepare_mlc_label_weights(self, label_names, device=None):
        """Assign concept weights, sharing weight only when equivalent label codes coexist."""
        weights = torch.ones(len(label_names), dtype=torch.float32)
        grouped = 0
        registered_groups = _MLC_SYNONYM_CODE_GROUPS
        for code_group in registered_groups:
            indices = [
                index
                for index, name in enumerate(label_names)
                if any(name.endswith(f"[{code}]") for code in code_group)
            ]
            if len(indices) > 1:
                weights[indices] = 1.0 / len(indices)
                grouped += 1
        target_device = device if device is not None else self.device
        self._mlc_label_weights = weights.to(target_device)
        print(
            f"[LaUER] MLC label weights: {grouped} synonym groups "
            f"share unit contribution"
        )
        return self._mlc_label_weights

    @torch.no_grad()
    def encode_prompt_bank(self, label_names, device=None):
        """Return frozen disease-query features in the order supplied by the caller."""
        target_device = device if device is not None else self.device
        if self._prompt_features is not None:
            if self._prompt_features.device == target_device:
                return self._prompt_features
            if getattr(self, 'query_encoder', None) is None:

                self._prompt_features = self._prompt_features.to(target_device)
                return self._prompt_features
        if not label_names:
            raise ValueError("label_names must be provided for multi-label classification")
        tokenizer = self.query_encoder.tokenizer
        encoded = tokenizer(
            label_names,
            add_special_tokens=True,
            padding="longest",
            truncation=True,
            max_length=512,
            return_tensors="pt",
        )
        input_ids = encoded["input_ids"].to(target_device)
        attention_mask = encoded["attention_mask"].to(target_device)
        pooled = self.query_encoder(input_ids, attention_mask)  # [C, 768]
        self._prompt_features = pooled
        return pooled

    def _detect_special_tokens(self):
        """Read tokenizer special IDs without expanding the pretrained vocabulary."""
        tokenizer = self.text_encoder.tokenizer

        self.sep_token_id  = getattr(tokenizer, 'sep_token_id', 102)
        self.cls_token_id  = getattr(tokenizer, 'cls_token_id', 101)
        self.mask_token_id = getattr(tokenizer, 'mask_token_id', 103)
        self.pad_token_id  = getattr(tokenizer, 'pad_token_id', 0)

        print(f"[LaUER] Special tokens: CLS={self.cls_token_id}, "
              f"SEP={self.sep_token_id}, MASK={self.mask_token_id}, PAD={self.pad_token_id}")

        if self.has_tgl:
            self.tgl_loss.mask_token_id = self.mask_token_id
            self.tgl_loss.sep_token_id = self.sep_token_id
            self.tgl_loss.pad_token_id = self.pad_token_id

    def load_pretrained_encoders(self, device='cpu'):
        """Initialize ECG encoders, frozen language backbones, and optional text embeddings."""
        print("[LaUER] ViT1D/ViT2D encoders already loaded in __init__")

        self.text_encoder.load_model()
        self.text_encoder.to(device)

        if self.has_cgl and hasattr(self, 'query_encoder'):
            self.query_encoder.load_model()
            self.query_encoder.to(device)

        self._detect_special_tokens()

        print(f"\n[LaUER] Encoder output dimensions:")
        print(f"  ECG (vit1d): {self.config.ecg_encoder.vit1d.width}d")
        print(f"  Image (vit2d): {self.config.image_encoder.vit2d.width}d")
        print(f"  MedCPT:   {self.config.text_encoder.hidden_size}d")
        print(f"  Projection → proj1(token-level) + pooling + proj2 → SGL embedding\n")

        mode_desc_parts = []
        if self.has_tgl:
            mode_desc_parts.append('TGL')
        if self.has_cgl:
            mode_desc_parts.append('CGL(fusion+multilabel)')
        mode_desc_parts.append('ECG=vit1d, Image=vit2d')
        print(f"[LaUER] All encoders loaded ({', '.join(mode_desc_parts)}).")

        if self.has_tgl:
            medcpt_path = self.config.tgl.medcpt_path
            self.tgl.init_embedding_from_medcpt(medcpt_path)

        self._print_param_stats()
        if getattr(self.config.train, 'nan_debug', False):
            self._register_nan_hooks()

    def _register_nan_hooks(self):
        if getattr(self, '_nan_hooks_registered', False):
            return
        self._nan_hooks_registered = True
        import logging
        logger = logging.getLogger(__name__)

        targets = []

        for encoder_name in ['ecg_encoder', 'image_encoder']:
            wrapper = getattr(self, encoder_name, None)
            core = getattr(wrapper, 'encoder', None) if wrapper is not None else None
            if core is not None:
                targets.append((f'encoder.{encoder_name}.core', core))

        if self.has_tgl:
            targets += [
                ('tgl.embedding',  self.tgl.embedding),
                ('tgl.final_norm', self.tgl.final_norm),
                ('tgl.lm_head',    self.tgl.lm_head),
            ]
            for idx, layer in enumerate(self.tgl.layers):
                targets.append((f'tgl.layer.{idx}.self_attn',  layer.self_attn))
                targets.append((f'tgl.layer.{idx}.cross_attn', layer.cross_attn))
                targets.append((f'tgl.layer.{idx}.ffn',        layer.ffn))

        if self.has_cgl:
            if hasattr(self, 'fusion_si') and self.fusion_si is not None:
                targets.append(('fusion_si', self.fusion_si))
            if hasattr(self, 'fusion_si_layers') and self.fusion_si_layers is not None:
                for idx, layer in enumerate(self.fusion_si_layers):
                    targets.append((f'fusion_si.layer.{idx}', layer))
            if hasattr(self, 'cgl_loss_module') and self.cgl_loss_module is not None:
                for idx, head in enumerate(getattr(self.cgl_loss_module, 'heads', [])):
                    targets.append((f'cgl_head.{idx}', head))

        if hasattr(self, 'adapters'):
            for mod_name in ['ecg_adapter', 'image_adapter', 'text_adapter']:
                adapter = getattr(self.adapters, mod_name, None)
                if adapter is None:
                    continue
                for proj_name in ['proj1_sgl', 'proj2']:
                    proj = getattr(adapter, proj_name, None)
                    if proj is not None:
                        targets.append((f'{mod_name}.{proj_name}', proj))

                if getattr(adapter, 'proj1_cgl', None) is not None:
                    targets.append((f'{mod_name}.proj1_cgl', adapter.proj1_cgl))

        root_model = self

        def _make_hook(name):
            def hook(module, inp, out):

                out_tensor = out
                if isinstance(out, (tuple, list)):
                    out_tensor = next(
                        (item for item in out if isinstance(item, torch.Tensor)),
                        None,
                    )
                if not isinstance(out_tensor, torch.Tensor):
                    return
                if torch.isnan(out_tensor).any() or torch.isinf(out_tensor).any():
                    nan_count = torch.isnan(out_tensor).sum().item()
                    inf_count = torch.isinf(out_tensor).sum().item()
                    inp_shapes = [t.shape for t in inp if isinstance(t, torch.Tensor)]
                    logger.error(
                        f"[NaN TRACE] Detected in: {name} | "
                        f"output={list(out_tensor.shape)} NaN={nan_count} Inf={inf_count} | "
                        f"inputs={inp_shapes}"
                    )

                    if getattr(root_model, '_nan_signal', None) is None:
                        root_model._nan_signal = name

                if name.startswith('encoder.') and out_tensor.requires_grad:
                    def _grad_hook(grad):
                        if not torch.isfinite(grad).all():
                            nan_count = torch.isnan(grad).sum().item()
                            inf_count = torch.isinf(grad).sum().item()
                            logger.error(
                                f"[NaN TRACE] Backward gradient at: {name} | "
                                f"shape={list(grad.shape)} NaN={nan_count} Inf={inf_count}"
                            )
                            signal_name = f'{name}.backward'
                            if getattr(root_model, '_nan_signal', None) is None:
                                root_model._nan_signal = signal_name
                        return grad
                    out_tensor.register_hook(_grad_hook)
            return hook

        for name, mod in targets:
            mod.register_forward_hook(_make_hook(name))

        n = len(targets)
        print(f"[LaUER] NaN hooks registered on {n} modules "
              f"({'tgl' if self.has_tgl else 'no'} + "
              f"{'cgl' if self.has_cgl else 'no'} + "
              f"{'adapters' if hasattr(self, 'adapters') else 'no'})")

    def load_stage_weights(self, checkpoint_path, allow_partial=False,
                           device='cpu', logger=None):
        """Load the preceding cumulative stage with explicit architecture and mode checks."""
        from training.stage_io import load_stage_initialization
        return load_stage_initialization(
            self,
            checkpoint_path,
            target_mode=self.mode,
            allow_partial=allow_partial,
            device=device,
            logger=logger,
        )

    def _print_param_stats(self):
        total = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        print(f"[LaUER] Total: {total:,}  Trainable: {trainable:,}  "
              f"({trainable/total*100:.1f}%)")

    def _encode_all(self, ecg_signal, ecg_image, text_input_ids, text_attention_mask):
        ecg_tokens = self.ecg_encoder.encode_tokens(ecg_signal)       # [B, 481, 192]
        image_tokens = self.image_encoder.encode_tokens(ecg_image)     # [B, N, D]
        text_tokens = self.text_encoder.encode_tokens(text_input_ids, text_attention_mask)  # [B, L, 768]

        if text_attention_mask is None:
            text_mask = torch.ones(text_tokens.size(0), text_tokens.size(1),
                                   device=text_tokens.device, dtype=torch.bool)
        else:
            text_mask = text_attention_mask.bool()

        return ecg_tokens, image_tokens, text_tokens, text_mask

    def _encode_aug_embeds(self, ecg_signal_aug, ecg_image_aug):
        ecg_embed_aug = None
        img_embed_aug = None
        if ecg_signal_aug is not None:
            tok = self.ecg_encoder.encode_tokens(ecg_signal_aug)       # [B, 481, 192]
            ecg_embed_aug = self.adapters.ecg_adapter.forward_sgl(tok)
        if ecg_image_aug is not None:
            tok = self.image_encoder.encode_tokens(ecg_image_aug)       # [B, 481, 192]
            img_embed_aug = self.adapters.image_adapter.forward_sgl(tok)
        return ecg_embed_aug, img_embed_aug

    @torch.no_grad()
    def validate_multilabel_logits(self, ecg_signal=None, ecg_image=None, batch=None):
        """Return disease-query logits [B, C] for the current concept bank."""
        if not self.has_cgl or self.query_guided_tracing_head is None or self.label_names is None:
            return None
        if batch is not None:
            device = self.device
            ecg_signal = batch['ecg_signal'].to(device, non_blocking=True)
            ecg_image = batch['ecg_image'].to(device, non_blocking=True)
        ecg_tokens = self.ecg_encoder.encode_tokens(ecg_signal)      # [B, 481, 192]
        image_tokens = self.image_encoder.encode_tokens(ecg_image)    # [B, 481, 192]
        ecg_proj, img_proj, _ = self.adapters.project_tokens(
            ecg_tokens, image_tokens, None,
        )
        if self._use_alternating_fusion:
            fused_si = self._run_alternating_fusion(ecg_proj, img_proj)
        else:
            fused_si = self.fusion_si(primary=ecg_proj, cross_kv=img_proj)
        prompt_features = self.encode_prompt_bank(self.label_names, fused_si.device)
        return self.query_guided_tracing_head(prompt_features, fused_si)       # [B, C]

    def encode_ecg(self, ecg_signal):
        return self.ecg_encoder.encode_tokens(ecg_signal)

    def encode_image(self, ecg_image):
        return self.image_encoder.encode_tokens(ecg_image)

    def encode_text(self, input_ids, attention_mask=None):
        return self.text_encoder.encode_tokens(input_ids, attention_mask)

    def project_features(self, ecg_tokens, image_tokens, text_tokens):
        if isinstance(text_tokens, list):
            if self.has_cgl:
                text_tokens, _ = self._pad_text_tokens(
                    text_tokens, 768, ecg_tokens.device, ecg_tokens.dtype
                )
            else:
                B = len(text_tokens)
                dim = text_tokens[0].size(-1)
                max_len = max(t.size(0) for t in text_tokens)
                padded = torch.zeros(B, max_len, dim,
                                     device=ecg_tokens.device, dtype=ecg_tokens.dtype)
                for i, t in enumerate(text_tokens):
                    padded[i, :t.size(0)] = t.to(ecg_tokens.device)
                text_tokens = padded
        return self.adapters(ecg_tokens, image_tokens, text_tokens)

    def forward_cgl_fusion(self, ecg_tokens, image_tokens):
        if self._use_alternating_fusion:
            fused = self._run_alternating_fusion(ecg_tokens, image_tokens)
        else:
            fused = self.fusion_si(
                primary=ecg_tokens,
                cross_kv=image_tokens,
            )

        B = ecg_tokens.size(0) // 3
        cls_pos = fused[:B, 0, :]
        cls_neg1 = fused[B:2*B, 0, :]
        cls_neg2 = fused[2*B:, 0, :]

        fused_out = fused[:B]
        return cls_pos, cls_neg1, cls_neg2, fused_out

    def forward(self, ecg_signal=None, ecg_image=None,
                text_input_ids=None, text_attention_mask=None,
                batch=None, labels=None, tgl_text_ids=None,
                ecg_signal_aug=None, ecg_image_aug=None):
        if batch is not None:
            device = self.device
            ecg_signal = batch['ecg_signal'].to(device, non_blocking=True)
            ecg_image = batch['ecg_image'].to(device, non_blocking=True)
            text_ids = batch['text_input_ids'].to(device, non_blocking=True)
            text_mask = batch['text_attention_mask'].to(device, non_blocking=True)
            dec_ids = batch.get('tgl_text_ids')
            if dec_ids is not None:
                dec_ids = dec_ids.to(device, non_blocking=True)
            labels = batch.get('labels')
            if labels is not None:

                labels = labels.to(device, non_blocking=True)
            aug_sig = batch.get('ecg_signal_aug')
            aug_img = batch.get('ecg_image_aug')
            if aug_sig is not None:
                aug_sig = aug_sig.to(device, non_blocking=True)
            if aug_img is not None:
                aug_img = aug_img.to(device, non_blocking=True)
            if not self.has_cgl:
                return self._forward_sgl_only(
                    ecg_signal, ecg_image, text_ids, text_mask,
                    ecg_signal_aug=aug_sig, ecg_image_aug=aug_img,
                )
            return self._forward_full(ecg_signal, ecg_image, text_ids, text_mask,
                                      labels=labels, tgl_text_ids=dec_ids,
                                      ecg_signal_aug=aug_sig, ecg_image_aug=aug_img)

        if not self.has_cgl:
            return self._forward_sgl_only(
                ecg_signal, ecg_image, text_input_ids, text_attention_mask,
                ecg_signal_aug=ecg_signal_aug, ecg_image_aug=ecg_image_aug,
            )
        return self._forward_full(
            ecg_signal, ecg_image, text_input_ids, text_attention_mask,
            labels=labels, tgl_text_ids=tgl_text_ids,
            ecg_signal_aug=ecg_signal_aug, ecg_image_aug=ecg_image_aug,
        )

    def _forward_sgl_only(self, ecg_signal, ecg_image, text_input_ids, text_attention_mask=None,
                             ecg_signal_aug=None, ecg_image_aug=None):

        ecg_tokens = self.ecg_encoder.encode_tokens(ecg_signal)                   # [B, 481, 192]
        image_tokens = self.image_encoder.encode_tokens(ecg_image)                 # [B, N, D]
        text_tokens = self.text_encoder.encode_tokens(text_input_ids, text_attention_mask)  # [B, L, 768]
        if text_attention_mask is None:
            text_mask = torch.ones(text_tokens.size(0), text_tokens.size(1),
                                   device=text_tokens.device, dtype=torch.bool)
        else:
            text_mask = text_attention_mask.bool()

        ecg_embed, image_embed, text_embed = self.adapters(
            ecg_tokens, image_tokens, text_tokens, text_mask,
        )

        has_aug = (ecg_signal_aug is not None or ecg_image_aug is not None)
        if has_aug:
            ecg_embed_aug, img_embed_aug = self._encode_aug_embeds(
                ecg_signal_aug, ecg_image_aug)
            loss, loss_dict = self.sgl_loss.forward_masked(
                ecg_embed, image_embed, text_embed, ecg_embed_aug, img_embed_aug)
        else:
            loss, loss_dict = self._compute_sgl_loss(ecg_embed, image_embed, text_embed)
        self._nan_accum = {'sgl': torch.isfinite(loss)}

        return loss, loss_dict, self._make_embedding_dict(
            ecg_embed, image_embed, text_embed,
        )

    def _forward_full(self, ecg_signal, ecg_image, text_input_ids, text_attention_mask=None,
                      labels=None, tgl_text_ids=None,
                      ecg_signal_aug=None, ecg_image_aug=None):

        ecg_tokens, image_tokens, text_tokens, text_mask = self._encode_all(
            ecg_signal, ecg_image, text_input_ids, text_attention_mask
        )

        (ecg_embed, image_embed, text_embed), \
            (ecg_proj, img_proj, txt_proj) = self.adapters.forward_combined(
            ecg_tokens, image_tokens, text_tokens, text_mask,
        )

        if txt_proj is not None and text_mask is not None:
            txt_proj = txt_proj * text_mask.unsqueeze(-1).float()

        has_aug = (ecg_signal_aug is not None or ecg_image_aug is not None)
        if has_aug:
            ecg_embed_aug, img_embed_aug = self._encode_aug_embeds(
                ecg_signal_aug, ecg_image_aug)
            loss_sgl, sgl_dict = self.sgl_loss.forward_masked(
                ecg_embed, image_embed, text_embed, ecg_embed_aug, img_embed_aug)
            sim_et, sim_ei, sim_it = self.sgl_loss.compute_similarity(
                ecg_embed, image_embed, text_embed
            )
        elif dist.is_initialized() and dist.get_world_size() > 1:

            sim_et, sim_ei, sim_it = self.sgl_loss.compute_similarity(
                ecg_embed, image_embed, text_embed
            )
            loss_sgl, sgl_dict = self._compute_global_sgl_loss(
                ecg_embed, image_embed, text_embed
            )
        else:
            loss_sgl, sgl_dict, (sim_et, sim_ei, sim_it) = self.sgl_loss(
                ecg_embed, image_embed, text_embed
            )

        sig_all, img_all, _ = self._build_cgl_triplet(ecg_proj, img_proj, sim_ei)
        cls_si_pos, cls_si_neg1, cls_si_neg2, fused_si_pos = self.forward_cgl_fusion(
            sig_all, img_all
        )
        loss_matching, acc_matching = self.cgl_loss_module.compute_stage_loss(
            1, cls_si_pos, cls_si_neg1, cls_si_neg2
        )

        if labels is not None and self.label_names is not None:
            prompt_features = self.encode_prompt_bank(
                self.label_names, fused_si_pos.device,
            )  # [C, 768]
            mlc_logits = self.query_guided_tracing_head(prompt_features, fused_si_pos)  # [B, C]
            if self._mlc_label_weights.numel() != mlc_logits.shape[-1]:
                self._prepare_mlc_label_weights(
                    self.label_names, mlc_logits.device,
                )
            label_weights = self._mlc_label_weights.to(mlc_logits.device)

            if self.asl_loss is not None:
                loss_mlc = self.asl_loss(
                    mlc_logits.float(), labels.float(), label_weights,
                )
            else:
                elementwise = F.binary_cross_entropy_with_logits(
                    mlc_logits.float(), labels.float(), reduction="none",
                )
                loss_mlc = (
                    elementwise * label_weights.unsqueeze(0)
                ).sum() / (labels.shape[0] * label_weights.sum().clamp_min(1e-8))

            pred = (mlc_logits > 0.5).float()
            pos = labels.float()
            tp = (pred * pos).sum()
            mlc_precision = tp / (pred.sum() + 1e-8)
            mlc_recall = tp / (pos.sum() + 1e-8)
        elif labels is not None:

            logger.warning(
                "[MultiLabel] labels provided but label_names is None -- "
                "multi-label BCE loss is SKIPPED. Inject model.label_names "
                "(via dataset annotation_csv) to enable it."
            )
            mlc_logits = None
            loss_mlc = torch.zeros((), device=fused_si_pos.device,
                                   dtype=fused_si_pos.dtype)
            mlc_precision = mlc_recall = None
        else:
            mlc_logits = None
            loss_mlc = torch.zeros((), device=fused_si_pos.device,
                                   dtype=fused_si_pos.dtype)
            mlc_precision = mlc_recall = None

        self._last_mlc_logits = (
            mlc_logits.detach() if mlc_logits is not None else None
        )

        (L_mlm, L_ar, L_lm, mlm_correct, mlm_total, ar_tokens,
         sep_correct, sep_total, rep_correct, rep_total,
         expl_correct, expl_total) = self._maybe_tgl_loss(
            text_input_ids, tgl_text_ids, fused_si_pos)

        total_loss = self.sgl_weight * loss_sgl + self.cgl_weight * (
            self.w_matching * loss_matching + self.w_multilabel * loss_mlc)
        if L_lm is not None:
            total_loss = total_loss + self.lm_weight * L_lm

        lm_losses = [L_mlm, L_ar] if L_lm is not None else None
        lm_stats = [mlm_correct, mlm_total] if mlm_correct is not None else None
        sep_stats = [sep_correct, sep_total] if sep_correct is not None else None

        region_stats = ([rep_correct, rep_total, expl_correct, expl_total]
                        if L_lm is not None else None)
        loss_dict = self._build_loss_dict(
            sgl_dict, total_loss,
            loss_matching, acc_matching,
            lm_losses=lm_losses, lm_stats=lm_stats, ar_tokens=ar_tokens,
            sep_stats=sep_stats, region_stats=region_stats,
        )
        if mlc_logits is not None:
            loss_dict['loss_mlc'] = loss_mlc.detach()
            loss_dict['mlc_precision'] = mlc_precision.detach()
            loss_dict['mlc_recall'] = mlc_recall.detach()

        emb_dict = self._make_embedding_dict(
            ecg_embed, image_embed, text_embed,
        )

        self._nan_accum = {
            'sgl': torch.isfinite(loss_sgl),
            'matching': torch.isfinite(loss_matching),
            'mlc': torch.isfinite(loss_mlc),
            'mlm': torch.isfinite(L_mlm) if L_mlm is not None else None,
            'ar': torch.isfinite(L_ar) if L_ar is not None else None,
        }
        return total_loss, loss_dict, emb_dict

    def _maybe_tgl_loss(self, text_input_ids, tgl_text_ids, fused_si_pos):
        if self.has_tgl:
            if self.loss_mode_ar:
                if tgl_text_ids is None:
                    raise RuntimeError(
                        "[TGL AR] tgl_text_ids missing from batch — "
                        "autoregressive training requires a separate marked target stream."
                    )
                input_ids = tgl_text_ids
            else:
                input_ids = text_input_ids
            (L_mlm, L_ar, mlm_correct, mlm_total, ar_tokens,
             sep_correct, sep_total, rep_correct, rep_total,
             expl_correct, expl_total) = self._compute_tgl_loss(
                input_ids, fused_si_pos)
            if self.loss_mode_ar:
                ar_mean = L_ar

                L_lm = self.tgl_loss.ar_loss_weight * ar_mean
            else:
                ar_mean = L_ar / max(ar_tokens, 1)
                L_lm = self.tgl_loss.mlm_loss_weight * L_mlm \
                       + self.tgl_loss.ar_loss_weight * ar_mean

            return (L_mlm, ar_mean, L_lm, mlm_correct, mlm_total,
                    ar_tokens, sep_correct, sep_total,
                    rep_correct, rep_total, expl_correct, expl_total)
        return (None, None, None, None, None, None, None, None,
                None, None, None, None)

    def _compute_global_sgl_loss(self, ecg_embed, image_embed, text_embed):
        """Gather embeddings across ranks and compute distributed semantic alignment."""

        if torch.is_grad_enabled():

            # (forward) + ONE SUM all_reduce (backward) on a stacked
            # [B,3,D] tensor. Same data volume, 2 fewer NCCL launches;
            # rank/order, modality correspondence, SUM semantics and
            # local-batch slice are preserved (mathematically identical).
            embeds = torch.stack([ecg_embed, image_embed, text_embed], dim=1)  # [B, 3, D]
            embeds_global = gather_features(embeds)  # [total_B, 3, D]
            ecg_global = embeds_global[:, 0].contiguous()   # [total_B, D]
            image_global = embeds_global[:, 1].contiguous()
            text_global = embeds_global[:, 2].contiguous()
        else:
            ecg_global   = gather_features_nograd(ecg_embed)
            image_global = gather_features_nograd(image_embed)
            text_global  = gather_features_nograd(text_embed)

        self.sgl_loss._clamp_temp()
        logit_scale = self.sgl_loss.logit_scale.exp()
        total_B = ecg_global.size(0)
        labels = torch.arange(total_B, device=ecg_global.device, dtype=torch.long)

        # ECG-Text
        sim_et = logit_scale * ecg_global @ text_global.T
        loss_et = (F.cross_entropy(sim_et, labels)
                   + F.cross_entropy(sim_et.T, labels)) / 2

        # Image-Text
        sim_it = logit_scale * image_global @ text_global.T
        loss_it = (F.cross_entropy(sim_it, labels)
                   + F.cross_entropy(sim_it.T, labels)) / 2

        # ECG-Image
        sim_ei = logit_scale * ecg_global @ image_global.T
        loss_ei = (F.cross_entropy(sim_ei, labels)
                   + F.cross_entropy(sim_ei.T, labels)) / 2

        w_et, w_it, w_ei = self.sgl_loss.loss_weights
        total_loss = w_et * loss_et + w_it * loss_it + w_ei * loss_ei

        loss_dict = {
            'loss_sgl': total_loss.detach(),
            'loss_ecg_text': loss_et.detach(),
            'loss_image_text': loss_it.detach(),
            'loss_ecg_image': loss_ei.detach(),
            'temperature': (1.0 / logit_scale).detach(),
        }
        return total_loss, loss_dict

    def _compute_sgl_loss(self, ecg_embed, image_embed, text_embed):
        should_gather = (dist.is_initialized() and dist.get_world_size() > 1)
        if should_gather:
            return self._compute_global_sgl_loss(ecg_embed, image_embed, text_embed)
        else:
            loss, loss_dict, _ = self.sgl_loss(ecg_embed, image_embed, text_embed)
            return loss, loss_dict

    def get_similarity_matrices(self, ecg_signal, ecg_image, text_input_ids, text_attention_mask=None):
        with torch.no_grad():
            ecg_tokens = self.ecg_encoder.encode_tokens(ecg_signal)
            image_tokens = self.image_encoder.encode_tokens(ecg_image)
            text_tokens = self.text_encoder.encode_tokens(text_input_ids, text_attention_mask)

            tmask = (text_attention_mask.bool()
                     if text_attention_mask is not None else None)
            ecg_embed, image_embed, text_embed = self.adapters(
                ecg_tokens, image_tokens, text_tokens, tmask,
            )

            logit_scale = self.sgl_loss.logit_scale.exp()

            sim_ecg_text = logit_scale * ecg_embed @ text_embed.T
            sim_image_text = logit_scale * image_embed @ text_embed.T
            sim_ecg_image = logit_scale * ecg_embed @ image_embed.T

        return sim_ecg_text, sim_image_text, sim_ecg_image

    @torch.no_grad()
    def extract_fusion_si_cls(self, sig_tok, img_tok):
        """Return the unprojected fused-memory CLS feature for downstream linear probing."""
        if not self.has_cgl:
            raise RuntimeError("[LaUER] extract_fusion_si_cls requires CGL mode (mode='sgl+cgl').")

        if self._use_alternating_fusion:
            fused = self._run_alternating_fusion(sig_tok, img_tok)
            cls_feat = fused[:, 0, :]
        else:
            fused = self.fusion_si(primary=sig_tok, cross_kv=img_tok)
            cls_feat = fused[:, 0, :]

        return cls_feat

    @torch.no_grad()
    def extract_tgl_cross_kv(self, ecg_signal, ecg_image):
        """Return projected fused ECG memory [B, N, H] for text generation."""
        if not self.has_cgl:
            raise RuntimeError(
                "[LaUER] extract_tgl_cross_kv requires CGL mode."
            )
        if not self.has_tgl or self.tgl is None:
            raise RuntimeError(
                "[LaUER] extract_tgl_cross_kv requires an enabled Decoder."
            )
        if self.ecg_encoder is None or self.image_encoder is None:
            raise RuntimeError(
                "[LaUER] extract_tgl_cross_kv requires raw signal and image encoders."
            )

        device = self.device
        ecg_tokens = self.ecg_encoder.encode_tokens(ecg_signal.to(device))
        image_tokens = self.image_encoder.encode_tokens(ecg_image.to(device))
        ecg_proj, img_proj, _ = self.adapters.project_tokens(
            ecg_tokens, image_tokens, None,
        )

        if self._use_alternating_fusion:
            fused_si = self._run_alternating_fusion(ecg_proj, img_proj)
        else:
            fused_si = self.fusion_si(primary=ecg_proj, cross_kv=img_proj)

        return self.tgl_proj(fused_si)

    @torch.no_grad()
    def generate_report_ar(self, cross_kv, max_len=None, temperature=1.0):
        """Autoregressively generate report and explanation; terminate at the second separator."""
        if not self.has_tgl or self.tgl is None:
            raise RuntimeError(
                "[LaUER] generate_report_ar requires an enabled Decoder."
            )
        tok = self.text_encoder.tokenizer
        rep_marker = getattr(self.config.data, 'tgl_report_marker', 'report:')
        expl_marker = getattr(self.config.data, 'tgl_explanation_marker', 'explanation:')
        rep_ids = tok.encode(rep_marker, add_special_tokens=False)
        expl_ids = tok.encode(expl_marker, add_special_tokens=False)
        prefix = torch.tensor(
            [[self.cls_token_id] + rep_ids], device=cross_kv.device, dtype=torch.long,
        )
        if max_len is None:
            max_len = getattr(self.config.tgl, 'max_generation_len', 512)
        return self.tgl.generate_ar(
            cross_kv, prefix, rep_ids, expl_ids,
            max_len=max_len,
            sep_token_id=self.sep_token_id,
            cls_token_id=self.cls_token_id,
            mask_token_id=self.mask_token_id,
            pad_token_id=self.pad_token_id,
            temperature=temperature,
        )

    @torch.no_grad()
    def generate_text(self, cross_kv, max_len=None, temperature=1.0):
        """Generate report and explanation tokens using the configured text-decoder objective."""
        if not self.has_tgl or self.tgl is None:
            raise RuntimeError(
                "[LaUER] generate_text requires an enabled Decoder."
            )
        if max_len is None:
            max_len = getattr(self.config.tgl, 'max_generation_len', 512)
        if not self.loss_mode_ar:

            return self.tgl.generate_hybrid(
                cross_kv,
                mask_token_id=self.mask_token_id,
                sep_token_id=self.sep_token_id,
                cls_token_id=self.cls_token_id,
                pad_token_id=self.pad_token_id,
                max_report_len=128,
                max_llm_len=max_len,
                min_report_len=5,
                temperature=temperature,
            )
        return self.generate_report_ar(
            cross_kv, max_len=max_len, temperature=temperature,
        )

    def decode_ids(self, generated_ids, skip_special=True):
        """Decode token IDs while preserving the report/explanation boundary."""
        ids_list = generated_ids[0].cpu().tolist()
        if skip_special:

            special_tokens = {
                self.pad_token_id, self.cls_token_id, self.sep_token_id,
                self.mask_token_id,
            }
            ids_list = [t for t in ids_list if t not in special_tokens]
        if self.text_encoder is not None and hasattr(self.text_encoder, 'tokenizer'):
            return self.text_encoder.tokenizer.decode(ids_list, skip_special_tokens=True)
        else:
            return str(ids_list)
