from contextlib import nullcontext
import hashlib

import torch

from .cam import Capture, token_cam, generation_regions, text_score

class Engine:
    def __init__(self, model, device, amp='none'):
        self.model, self.device, self.amp = model, device, amp

    def context(self, amp=None):
        amp = amp or self.amp
        return (torch.autocast('cuda', dtype={'bf16': torch.bfloat16, 'fp16': torch.float16}[amp])
                if str(self.device).startswith('cuda') and amp != 'none' else nullcontext())

    def fuse(self, data):
        m = self.model
        s = m.ecg_encoder.encode_tokens(data['signal'])
        i = m.image_encoder.encode_tokens(data['image'])
        s, i, _ = m.adapters.project_tokens(s, i, None)
        return (m._run_alternating_fusion(s, i) if m._use_alternating_fusion else
                m.fusion_si(primary=s, cross_kv=i))

    def encoder_cam(self, data, modality, text, head=None, index=None, amp=None):
        m = self.model
        encoder = m.ecg_encoder if modality == 'signal' else m.image_encoder
        adapter = m.adapters.ecg_adapter if modality == 'signal' else m.adapters.image_adapter
        core = encoder.encoder
        target = getattr(core, f'block{core.depth - 1}').attn.norm
        capture = Capture(target)
        try:
            with torch.enable_grad(), self.context(amp):
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
            maps, diagnostics = token_cam(score, capture.value)
            return maps, dict(diagnostics, score=float(score.detach()),
                **({'probability': float(score.detach().sigmoid())} if head is not None else {}),
                layer=f'{modality}_encoder.encoder.block{core.depth - 1}.attn.norm',
                score_type='cosine_similarity' if head is None else 'downstream_logit')
        finally:
            capture.close()

    def classification_cam(self, data, index, head=None, amp=None):
        m = self.model
        layer = (m.fusion_si_layers[-1] if m._use_alternating_fusion else m.fusion_si.layers[-1])
        # CLS-only probes require capture before the final token mixing operation.
        target = layer.norm1 if head is not None else layer
        capture = Capture(target)
        try:
            with torch.enable_grad(), self.context(amp):
                fused = self.fuse(data)
                if head is None:
                    bank = m.encode_prompt_bank(m.label_names, self.device)
                    score = m.query_guided_tracing_head(bank, fused)[0, index]
                else:
                    with torch.autocast(device_type=str(self.device).split(':')[0], enabled=False):
                        score = head(fused[:, 0].float())[0, index]
            maps, diagnostics = token_cam(score, capture.value)
            return maps, dict(diagnostics, score=float(score.detach()),
                probability=float(score.detach().sigmoid()), score_type='class_logit',
                layer='fusion.final.norm1' if head is not None else 'fusion.final.output')
        finally:
            capture.close()

    def generate(self, data, seed, max_len, temperature, sample_key):
        m = self.model
        if m.config.tgl.effective_loss_mode != 'ar':
            raise ValueError('Generation CAM currently requires AR; hybrid is not relabeled as AR')
        actual_seed = (seed + int(hashlib.sha256(sample_key.encode()).hexdigest()[:8], 16)) % (2**31)
        devices = [torch.cuda.current_device()] if str(self.device).startswith('cuda') else []
        with torch.random.fork_rng(devices=devices), torch.no_grad(), self.context():
            torch.manual_seed(actual_seed)
            fused = self.fuse(data)
            full, report, explanation = m.generate_text(m.tgl_proj(fused), max_len=max_len,
                                                        temperature=temperature)
        tok = m.text_encoder.tokenizer
        rep = tok.encode(m.config.data.tgl_report_marker, add_special_tokens=False)
        expl = tok.encode(m.config.data.tgl_explanation_marker, add_special_tokens=False)
        regions, complete = generation_regions(full[0].tolist(), m.sep_token_id, rep, expl)
        return full, regions, dict(seed=actual_seed, max_len=max_len, temperature=temperature,
            complete=complete, token_ids=full[0].tolist(),
            report=tok.decode(report[0], skip_special_tokens=True),
            explanation=tok.decode(explanation[0], skip_special_tokens=True), regions=regions)

    def generation_cam(self, data, ids, positions):
        m = self.model
        layer = m.fusion_si_layers[-1] if m._use_alternating_fusion else m.fusion_si.layers[-1]
        capture = Capture(layer)
        try:
            with torch.enable_grad(), self.context():
                fused = self.fuse(data)
                # sep=0 is full causal; position t-1 predicts token t.
                logits = m.tgl(ids[:, :-1], torch.zeros(1, dtype=torch.long, device=self.device),
                                   m.tgl_proj(fused), pad_token_id=m.pad_token_id)
                score = text_score(logits, ids, positions)
            maps, diagnostics = token_cam(score, capture.value)
            return maps, dict(diagnostics, score=float(score.detach()),
                score_type='mean_generated_token_log_probability',
                layer='fusion.final.output', token_positions=positions)
        finally:
            capture.close()
