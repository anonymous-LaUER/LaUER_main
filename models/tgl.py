"""Text decoder for token-granularity learning.

The paper configuration uses causal autoregressive generation with tied input
and output embeddings initialized from MedCPT-Article. The text stream is
[CLS] report: R [SEP] explanation: E [SEP]. The second separator ends generation."""
import torch
import torch.nn as nn

from models.attention import mha_fp32

def _valid_token_ids(token_ids, vocab_size):
    return sorted({
        int(token_id)
        for token_id in token_ids
        if token_id is not None and 0 <= int(token_id) < vocab_size
    })

def _mask_invalid_token_logits(logits, token_ids, vocab_size):
    invalid_ids = _valid_token_ids(token_ids, vocab_size)
    if invalid_ids:
        logits[..., invalid_ids] = float('-inf')
    return logits

# Hybrid Attention Mask

def build_hybrid_mask(seq_len, sep_positions, device):
    """Build the attention mask; a zero report boundary gives a fully causal mask."""
    L = seq_len
    row = torch.arange(L, device=device).view(1, L, 1)   # [1, L, 1]
    col = torch.arange(L, device=device).view(1, 1, L)   # [1, 1, L]
    sep = sep_positions.view(-1, 1, 1)                    # [B, 1, 1]

    is_cls = (row == 0) & (col <= sep)                    # [B, L, L]

    is_sep = (row == sep) & (col <= sep)                  # [B, L, L]

    is_report = (row >= 1) & (row < sep) & (col <= sep)   # [B, L, L]

    is_llm_causal = (row > sep) & (col <= row)            # [B, L, L]

    is_llm_see_report = (row > sep) & (col <= sep)        # [B, L, L]

    mask = is_cls | is_sep | is_report | is_llm_causal | is_llm_see_report  # [B, L, L]

    attn_mask = torch.zeros(mask.shape, dtype=torch.float, device=device).masked_fill(~mask, float('-inf'))
    return attn_mask  # [B, L, L]

# Decoder Layer

class DecoderLayer(nn.Module):

    def __init__(self, hidden_size=768, num_heads=12, ffn_size=3072, dropout=0.1):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_heads = num_heads

        self.self_attn = nn.MultiheadAttention(
            hidden_size, num_heads, dropout=dropout, batch_first=True,
        )
        self.norm1 = nn.LayerNorm(hidden_size)

        # Cross-Attention (Q=tgl_hidden, K/V=fused_si)
        self.cross_attn = nn.MultiheadAttention(
            hidden_size, num_heads, dropout=dropout, batch_first=True,
        )
        self.norm2 = nn.LayerNorm(hidden_size)

        # Feed-Forward
        self.ffn = nn.Sequential(
            nn.Linear(hidden_size, ffn_size),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(ffn_size, hidden_size),
            nn.Dropout(dropout),
        )
        self.norm3 = nn.LayerNorm(hidden_size)

    def forward(self, x, cross_kv, attn_mask, key_padding_mask=None):
        # 1. Self-Attention with Hybrid Mask

        if attn_mask is not None and attn_mask.dim() == 3 and attn_mask.size(0) == x.size(0):

            attn_mask = attn_mask.repeat_interleave(self.num_heads, dim=0)

        residual = x
        x_norm = self.norm1(x)
        attn_out, _ = mha_fp32(
            self.self_attn, x_norm, x_norm, x_norm,
            attn_mask=attn_mask,
            key_padding_mask=key_padding_mask,
        )
        x = residual + attn_out

        residual = x
        x_norm = self.norm2(x)
        attn_out, _ = mha_fp32(
            self.cross_attn, x_norm, cross_kv, cross_kv,
        )
        x = residual + attn_out

        # 3. Feed-Forward
        residual = x
        x_norm = self.norm3(x)
        ffn_out = self.ffn(x_norm)
        x = residual + ffn_out

        return x

# TGL text decoder

class AutoregressiveDecoder(nn.Module):
    """Memory-conditioned text decoder with a tied language-model head."""

    def __init__(self, config):
        super().__init__()
        self.hidden_size = config.hidden_size          # 768
        self.vocab_size = config.vocab_size            # 30522
        self.num_layers = config.num_layers            # 4
        self.num_heads = config.num_heads              # 12
        self.max_position = 512                        # BERT max_position_embeddings

        # Token Embedding
        self.embedding = nn.Embedding(self.vocab_size, self.hidden_size)

        self.position_embeddings = nn.Embedding(self.max_position, self.hidden_size)

        # Decoder Layers
        self.layers = nn.ModuleList([
            DecoderLayer(
                hidden_size=self.hidden_size,
                num_heads=self.num_heads,
                ffn_size=config.ffn_size,
                dropout=config.dropout,
            ) for _ in range(self.num_layers)
        ])

        # LM Head (weight-tied with embedding)
        self.lm_head = nn.Linear(self.hidden_size, self.vocab_size, bias=False)
        self.lm_head.weight = self.embedding.weight  # weight tying

        self.final_norm = nn.LayerNorm(self.hidden_size)

    def init_embedding_from_medcpt(self, medcpt_path):
        from transformers import AutoModel
        import os
        from configs.config import get_project_root

        if not os.path.isabs(medcpt_path):
            medcpt_path = os.path.join(get_project_root(), medcpt_path)

        print(f"[AutoregressiveDecoder] Loading MedCPT embeddings from: {medcpt_path}")
        medcpt = AutoModel.from_pretrained(medcpt_path, local_files_only=True)
        src_weight = medcpt.embeddings.word_embeddings.weight.data

        if src_weight.shape != self.embedding.weight.shape:
            raise ValueError(
                f"[AutoregressiveDecoder] Embedding shape mismatch: "
                f"MedCPT={src_weight.shape}, TGL={self.embedding.weight.shape}"
            )

        self.embedding.weight.data.copy_(src_weight)

        pos_weight = medcpt.embeddings.position_embeddings.weight.data
        if pos_weight.shape[0] >= self.max_position:
            self.position_embeddings.weight.data.copy_(pos_weight[:self.max_position])
        else:
            self.position_embeddings.weight.data[:pos_weight.shape[0]].copy_(pos_weight)
        print(f"[AutoregressiveDecoder] Position embeddings loaded: "
              f"{pos_weight.shape[0]} → {self.max_position} positions")

        # lm_head.weight is tied to embedding.weight, auto-synced
        print(f"[AutoregressiveDecoder] Embedding initialized: {self.vocab_size} × {self.hidden_size}")

    # ---- Forward ----

    def forward(self, input_ids, sep_positions, cross_kv, pad_token_id=0):
        B, L = input_ids.shape

        # Token Embedding + Position Embedding
        positions = torch.arange(L, device=input_ids.device).unsqueeze(0)  # [1, L]
        x = self.embedding(input_ids) + self.position_embeddings(positions)  # [B, L, 768]

        # Hybrid Attention Mask
        attn_mask = build_hybrid_mask(L, sep_positions, x.device)  # [B, L, L]

        if attn_mask.dim() == 3 and attn_mask.size(0) == x.size(0):
            attn_mask = attn_mask.repeat_interleave(self.num_heads, dim=0)

        key_padding_mask = (input_ids == pad_token_id)  # [B, L], True = ignore

        # Decoder Layers
        for layer in self.layers:
            x = layer(x, cross_kv, attn_mask, key_padding_mask=key_padding_mask)

        # LM Head
        x = self.final_norm(x)
        logits = self.lm_head(x)  # [B, L, vocab_size]

        return logits

    # ---- Hybrid Generation: MLM (report) + AR (LLM) ----

    @torch.no_grad()
    def predict_report_variable_length(self, cross_kv, max_report_len,
                                       mask_token_id, sep_token_id,
                                       cls_token_id, pad_token_id,
                                       min_report_len=1, temperature=1.0):
        if temperature <= 0.0:
            raise ValueError("temperature must be positive")
        if cross_kv.dim() != 3:
            raise ValueError("cross_kv must have shape [B, N, hidden]")
        if max_report_len < 1:
            raise ValueError("max_report_len must be positive")
        if max_report_len + 2 > self.max_position:
            raise ValueError(
                f"max_report_len={max_report_len} exceeds Decoder capacity "
                f"({self.max_position - 2})"
            )
        if min_report_len < 0 or min_report_len > max_report_len:
            raise ValueError(
                f"min_report_len must satisfy 0 <= min_report_len <= max_report_len, "
                f"got {min_report_len}"
            )

        device = cross_kv.device
        batch_size = cross_kv.size(0)
        M = max_report_len

        ids = torch.full(
            (batch_size, M + 2), pad_token_id, dtype=torch.long, device=device,
        )
        ids[:, 0] = cls_token_id
        ids[:, 1:1 + M] = mask_token_id
        sep_positions = torch.full(
            (batch_size,), M + 1, device=device, dtype=torch.long,
        )
        ids.scatter_(1, sep_positions.unsqueeze(1), sep_token_id)

        logits = self.forward(
            ids, sep_positions, cross_kv, pad_token_id=pad_token_id,
        )[:, 1:1 + M] / temperature
        logits = logits.clone()

        _mask_invalid_token_logits(
            logits,
            [cls_token_id, mask_token_id, pad_token_id],
            self.vocab_size,
        )
        log_probs = torch.log_softmax(logits, dim=-1)
        selected = log_probs.argmax(dim=-1)  # [B, M]

        is_sep = selected == sep_token_id                # [B, M]
        if min_report_len > 0:
            is_sep_effective = is_sep.clone()
            is_sep_effective[:, :min_report_len] = False
        else:
            is_sep_effective = is_sep
        terminated = is_sep_effective.any(dim=1)         # [B]
        first_sep = is_sep_effective.long().argmax(dim=1)

        ends = torch.where(
            terminated, first_sep,
            torch.full_like(first_sep, M),
        )

        pos = torch.arange(M, device=device).unsqueeze(0)  # [1, M]
        valid_mask = pos < ends.unsqueeze(1)              # [B, M]
        report_ids = torch.where(
            valid_mask, selected,
            torch.full_like(selected, pad_token_id),
        )

        token_log_probs = log_probs.gather(-1, selected.unsqueeze(-1)).squeeze(-1)
        token_log_probs = torch.where(
            valid_mask, token_log_probs, torch.zeros_like(token_log_probs),
        )

        safe_ends = ends.clamp(min=1).to(token_log_probs.dtype)
        mean_log_probability = token_log_probs.sum(dim=1) / safe_ends
        return report_ids, valid_mask, mean_log_probability

    @torch.no_grad()
    def generate_llm_from_report(self, cross_kv, report_ids, max_llm_len,
                                 sep_token_id, cls_token_id, mask_token_id,
                                 pad_token_id, temperature=1.0):
        """Generate the AR continuation once for one selected report."""
        if temperature <= 0.0:
            raise ValueError("temperature must be positive")
        if cross_kv.size(0) != 1 or report_ids.dim() != 2 or report_ids.size(0) != 1:
            raise ValueError("generate_llm_from_report currently requires batch size 1")
        if report_ids.size(1) < 1:
            raise ValueError("report_ids must contain at least one token")
        if report_ids.size(1) + 2 > self.max_position:
            raise ValueError(
                f"Report exceeds Decoder capacity ({self.max_position - 2})"
            )

        device = cross_kv.device
        prefix = torch.cat([
            torch.tensor([[cls_token_id]], device=device, dtype=torch.long),
            report_ids.to(device=device, dtype=torch.long),
            torch.tensor([[sep_token_id]], device=device, dtype=torch.long),
        ], dim=1)
        prefix_sep_pos = torch.tensor([report_ids.size(1) + 1], device=device)
        effective_max_llm_len = min(
            int(max_llm_len), self.max_position - prefix.size(1),
        )
        invalid_ar_tokens = _valid_token_ids(
            [cls_token_id, mask_token_id, pad_token_id], self.vocab_size,
        )
        generated = []
        for _ in range(max(0, effective_max_llm_len)):
            cur_len = prefix.size(1)
            attn_mask = build_hybrid_mask(cur_len, prefix_sep_pos, device)
            positions = torch.arange(cur_len, device=device).unsqueeze(0)
            x = self.embedding(prefix) + self.position_embeddings(positions)
            key_pad = (prefix == pad_token_id)
            for layer in self.layers:
                x = layer(x, cross_kv, attn_mask, key_padding_mask=key_pad)
            logits = self.lm_head(self.final_norm(x[:, -1:, :])).squeeze(1) / temperature
            _mask_invalid_token_logits(logits, invalid_ar_tokens, self.vocab_size)
            top_k = min(50, self.vocab_size)
            top_k_logits, top_k_indices = torch.topk(logits, top_k, dim=-1)
            probs = torch.softmax(top_k_logits, dim=-1)
            next_token = top_k_indices[0, torch.multinomial(probs, 1)[0]]
            prefix = torch.cat([prefix, next_token.view(1, 1)], dim=1)
            generated.append(next_token.item())
            if next_token.item() == sep_token_id:
                break

        if generated:
            return torch.tensor([generated], device=device, dtype=torch.long)
        return torch.empty((1, 0), device=device, dtype=torch.long)

    @torch.no_grad()
    def generate_hybrid(self, cross_kv, mask_token_id, sep_token_id,
                        cls_token_id, pad_token_id,
                        max_report_len=128, max_llm_len=256,
                        min_report_len=5, temperature=1.0):
        device = cross_kv.device
        B = cross_kv.size(0)

        if B != 1:
            raise ValueError(f"generate_hybrid currently requires batch size 1, got {B}")
        if max_report_len < 1:
            raise ValueError("max_report_len must be positive")
        if max_report_len + 2 > self.max_position:
            raise ValueError(
                f"max_report_len={max_report_len} exceeds tgl capacity "
                f"({self.max_position - 2} report tokens)"
            )
        if not (0 <= min_report_len <= max_report_len):
            raise ValueError(
                f"min_report_len must satisfy 0 <= min_report_len <= max_report_len, "
                f"got {min_report_len}"
            )
        if temperature <= 0.0:
            raise ValueError("temperature must be positive")

        invalid_ar_tokens = _valid_token_ids(
            [cls_token_id, mask_token_id, pad_token_id],
            self.vocab_size,
        )

        report_ids, valid_mask, _ = self.predict_report_variable_length(
            cross_kv, max_report_len,
            mask_token_id=mask_token_id,
            sep_token_id=sep_token_id,
            cls_token_id=cls_token_id,
            pad_token_id=pad_token_id,
            min_report_len=min_report_len,
            temperature=temperature,
        )

        report_clean = report_ids[0][valid_mask[0]]  # [R]

        # Step 2: AR Generation for LLM

        prefix = torch.tensor(
            [[cls_token_id] + report_clean.tolist() + [sep_token_id]],
            device=device, dtype=torch.long
        )

        prefix_sep_pos = torch.tensor([len(report_clean) + 1], device=device)

        llm_ids_list = []

        max_llm_steps = min(
            int(max_llm_len), self.max_position - prefix.size(1),
        )
        for step in range(max(0, max_llm_steps)):
            cur_len = prefix.size(1)

            attn_mask = build_hybrid_mask(cur_len, prefix_sep_pos, device)  # [1, L, L]

            positions = torch.arange(cur_len, device=device).unsqueeze(0)
            x = self.embedding(prefix) + self.position_embeddings(positions)
            key_pad = (prefix == pad_token_id)
            for layer in self.layers:
                x = layer(x, cross_kv, attn_mask, key_padding_mask=key_pad)

            logits = self.lm_head(self.final_norm(x[:, -1:, :])) / temperature
            logits = logits.squeeze(1)  # [1, V]

            _mask_invalid_token_logits(
                logits, invalid_ar_tokens, self.vocab_size,
            )

            top_k = min(50, self.vocab_size)
            top_k_logits, top_k_indices = torch.topk(logits, top_k, dim=-1)
            probs = torch.softmax(top_k_logits, dim=-1)
            next_token = top_k_indices[0, torch.multinomial(probs, 1)[0]]

            prefix = torch.cat([prefix, next_token.view(1, 1)], dim=1)
            llm_ids_list.append(next_token.item())

            if next_token.item() == sep_token_id:
                break

        llm_ids = torch.tensor(
            [llm_ids_list], device=device, dtype=torch.long,
        ) if llm_ids_list else torch.tensor([[]], device=device, dtype=torch.long)

        full_ids = prefix

        return full_ids, report_clean.unsqueeze(0) if report_clean.dim() == 1 else report_clean, llm_ids

    @torch.no_grad()
    def generate_ar(self, cross_kv, prefix_ids, report_marker_ids, explanation_marker_ids,
                    max_len, sep_token_id, cls_token_id, mask_token_id, pad_token_id,
                    temperature=1.0):
        if temperature <= 0.0:
            raise ValueError("temperature must be positive")
        if cross_kv.dim() != 3:
            raise ValueError("cross_kv must have shape [1, N, hidden]")
        if cross_kv.size(0) != 1:
            raise ValueError("generate_ar currently requires batch size 1")

        device = cross_kv.device
        prefix = torch.as_tensor(prefix_ids, dtype=torch.long, device=device).view(1, -1)
        if prefix.size(1) < 1:
            raise ValueError("prefix_ids must be non-empty ([CLS] + report marker)")
        rep_m = [int(t) for t in report_marker_ids]
        expl_m = [int(t) for t in explanation_marker_ids]
        max_steps = min(int(max_len), self.max_position - prefix.size(1))
        if max_steps < 1:
            raise ValueError(
                f"max_len={max_len} leaves no room to generate (prefix len "
                f"{prefix.size(1)} vs capacity {self.max_position})"
            )
        invalid_ar = _valid_token_ids(
            [cls_token_id, mask_token_id, pad_token_id], self.vocab_size,
        )

        sep_count = 0
        for _ in range(max_steps):
            cur_len = prefix.size(1)

            attn_mask = build_hybrid_mask(
                cur_len, torch.zeros(1, device=device, dtype=torch.long), device,
            )
            positions = torch.arange(cur_len, device=device).unsqueeze(0)
            x = self.embedding(prefix) + self.position_embeddings(positions)
            key_pad = (prefix == pad_token_id)
            for layer in self.layers:
                x = layer(x, cross_kv, attn_mask, key_padding_mask=key_pad)
            logits = self.lm_head(self.final_norm(x[:, -1:, :])).squeeze(1) / temperature
            _mask_invalid_token_logits(logits, invalid_ar, self.vocab_size)
            top_k = min(50, self.vocab_size)
            top_k_logits, top_k_indices = torch.topk(logits, top_k, dim=-1)
            probs = torch.softmax(top_k_logits, dim=-1)
            nxt = top_k_indices[0, torch.multinomial(probs, 1)[0]]
            prefix = torch.cat([prefix, nxt.view(1, 1)], dim=1)
            if nxt.item() == sep_token_id:
                sep_count += 1
                if sep_count >= 2:
                    break

        full_ids = prefix
        toks = full_ids[0].tolist()
        sep_pos = [i for i, t in enumerate(toks) if t == sep_token_id]
        b1 = sep_pos[0] if len(sep_pos) >= 1 else len(toks)
        b2 = sep_pos[1] if len(sep_pos) >= 2 else len(toks)

        rep_start = 1 + len(rep_m)
        rep_end = max(rep_start, b1)
        report_ids = torch.tensor(
            [toks[rep_start:rep_end]], device=device, dtype=torch.long,
        )

        expl_start = b1 + 1
        if toks[expl_start:expl_start + len(expl_m)] == expl_m:
            expl_start += len(expl_m)
        expl_end = b2
        explanation_ids = torch.tensor(
            [toks[expl_start:expl_end]], device=device, dtype=torch.long,
        )
        return full_ids, report_ids, explanation_ids
