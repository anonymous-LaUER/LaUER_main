import torch
import models.query_guided_tracing_head as mlc_module
import models.tgl as tgl_module
import models.fusion_encoder as fusion_module
from visualization.grad_cam.engine import Engine as BaseEngine
from visualization.grad_cam.cam import text_score
from .attention import capture_attention, attention_cam

class Engine(BaseEngine):
    """Capture ALBEF-style attention x positive-gradient CAMs from a frozen model.

    attention_layer: int = Python layer index used for every target (historical behaviour,
        taken modulo the target depth); None = pick per target (see AUTO_* below).

    Why the auto default is not uniformly the final layer: on this architecture the last
    cross-attention layer of the native multilabel head is saturated. Measured over 3 MEETI
    studies x 3 ground-truth labels, the peak-normalized CAM there has ~32 effective cells
    out of the 480 patch bins with the top 8 holding ~65% of the mass, versus ~219 effective
    cells and ~23% for the head's first layer. ALBEF itself visualises block 8 of 12, i.e.
    deliberately not the final block. Pass --attention-layer -1 to restore the old default.
    """

    attention_layer = None
    AUTO_ENCODER_LAYER = -1       # final ViT block (all ViT blocks measured non-degenerate)
    AUTO_MULTILABEL_LAYER = 0     # first multilabel-head layer; the last one collapses
    AUTO_DECODER_LAYER = -1       # final tgl layer
    AUTO_FUSION_LAYER = -1        # final CGL fusion layer (downstream-probe path)

    def _layer_index(self, depth, automatic):
        layer = automatic if self.attention_layer is None else self.attention_layer
        if not -depth <= layer < depth:
            raise ValueError(f'Attention layer {layer} out of range for depth {depth}')
        return layer % depth

    def encoder_cam(self, data, modality, text, head=None, index=None, amp=None):
        m = self.model
        encoder = m.ecg_encoder if modality == 'signal' else m.image_encoder
        adapter = m.adapters.ecg_adapter if modality == 'signal' else m.adapters.image_adapter
        core = encoder.encoder
        layer_index = self._layer_index(core.depth, self.AUTO_ENCODER_LAYER)
        target = getattr(core, f'block{layer_index}').attn.fn.attend
        with torch.enable_grad(), self.context(amp), capture_attention(target) as captured:
            tokens = encoder.encode_tokens(data[modality])
            if head is None:
                with torch.no_grad():
                    enc = m.text_encoder.tokenizer(text, return_tensors='pt', truncation=True,
                                                   max_length=m.config.text_encoder.max_length)
                    enc = {k: v.to(self.device) for k, v in enc.items()}
                    tt = m.text_encoder.encode_tokens(enc['input_ids'], enc['attention_mask'])
                    te = m.adapters.text_adapter.forward_sgl(tt, enc['attention_mask'])
                score = (adapter.forward_sgl(tokens) * te).sum()
            else:
                with torch.autocast(device_type=str(self.device).split(':')[0], enabled=False):
                    score = head(tokens[:, 0].float())[0, index]
            maps, info = attention_cam(score, captured, [0])
        return maps, dict(info, score=float(score.detach()),
                          layer=f'{modality}.block{layer_index}.self_attention',
                          resolved_layer_index=layer_index,
                          score_type='cosine_similarity' if head is None else 'downstream_logit')

    def classification_cam(self, data, index, head=None, amp=None):
        m = self.model
        if head is None:
            depth = len(m.query_guided_tracing_head.layers)
            layer_index = self._layer_index(depth, self.AUTO_MULTILABEL_LAYER)
            target = m.query_guided_tracing_head.layers[layer_index].cross_attention
            owner, rows = mlc_module, [index]
            layer_name = f'query_guided_tracing_head.layers[{layer_index}].cross_attention'
        else:
            stack = m.fusion_si_layers if m._use_alternating_fusion else m.fusion_si.layers
            layer_index = self._layer_index(len(stack), self.AUTO_FUSION_LAYER)
            target, owner, rows = stack[layer_index].self_attn, fusion_module, [0]
            layer_name = f'fusion.layers[{layer_index}].self_attention'
        with torch.enable_grad(), self.context(amp), capture_attention(target, owner) as captured:
            fused = self.fuse(data)
            if head is None:
                bank = m.encode_prompt_bank(m.label_names, self.device)
                score = m.query_guided_tracing_head(bank, fused)[0, index]
            else:
                with torch.autocast(device_type=str(self.device).split(':')[0], enabled=False):
                    score = head(fused[:, 0].float())[0, index]
            maps, info = attention_cam(score, captured, rows)
        return maps, dict(info, score=float(score.detach()), probability=float(score.detach().sigmoid()),
                          score_type='class_logit', layer=layer_name,
                          resolved_layer_index=layer_index)

    def generation_cam(self, data, ids, positions):
        m = self.model
        layer_index = self._layer_index(len(m.tgl.layers), self.AUTO_DECODER_LAYER)
        target = m.tgl.layers[layer_index].cross_attn
        rows = [p - 1 for p in positions]
        with torch.enable_grad(), self.context(), capture_attention(target, tgl_module) as captured:
            fused = self.fuse(data)
            logits = m.tgl(ids[:, :-1], torch.zeros(1, dtype=torch.long, device=self.device),
                               m.tgl_proj(fused), pad_token_id=m.pad_token_id)
            score = text_score(logits, ids, positions)
            maps, info = attention_cam(score, captured, rows)
        return maps, dict(info, score=float(score.detach()),
                          score_type='mean_generated_token_log_probability',
                          layer=f'tgl.layers[{layer_index}].cross_attention',
                          resolved_layer_index=layer_index, token_positions=positions)
