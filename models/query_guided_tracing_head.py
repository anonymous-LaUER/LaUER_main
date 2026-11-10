"""Query-guided tracing head for concept-granularity learning.

Frozen disease-query features are projected to fusion width. Each tracing
block applies query self-attention, cross-attention to fused ECG tokens, and
a feed-forward network. A shared affine classifier emits one logit per query."""

import torch.nn as nn

from models.attention import mha_fp32

class TracingBlock(nn.Module):
    """Pre-normalized query self-attention, memory cross-attention, and feed-forward update."""

    def __init__(self, hidden_size, num_heads, intermediate_size, dropout,
                 query_self_attn=True):
        super().__init__()
        self.query_self_attn = bool(query_self_attn)
        self.query_norm = nn.LayerNorm(hidden_size)
        self.memory_norm = nn.LayerNorm(hidden_size)
        self.cross_attention = nn.MultiheadAttention(
            hidden_size, num_heads, dropout=dropout, batch_first=True,
        )
        self.ffn_norm = nn.LayerNorm(hidden_size)
        self.ffn = nn.Sequential(
            nn.Linear(hidden_size, intermediate_size),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(intermediate_size, hidden_size),
            nn.Dropout(dropout),
        )
        if self.query_self_attn:
            self.self_attention = nn.MultiheadAttention(
                hidden_size, num_heads, dropout=dropout, batch_first=True,
            )
            self.cross_norm = nn.LayerNorm(hidden_size)

    def forward(self, queries, memory):
        if self.query_self_attn:

            q_norm = self.query_norm(queries)
            q2, _ = mha_fp32(self.self_attention, q_norm, q_norm, q_norm)
            queries = queries + q2
            # 2. Cross-attention (queries → fused_si)
            normalized_memory = self.memory_norm(memory)
            attention, _ = mha_fp32(
                self.cross_attention, self.cross_norm(queries),
                normalized_memory, normalized_memory,
            )
        else:

            normalized_memory = self.memory_norm(memory)
            attention, _ = mha_fp32(
                self.cross_attention, self.query_norm(queries),
                normalized_memory, normalized_memory,
            )
        queries = queries + attention
        # 3. Feed-Forward
        return queries + self.ffn(self.ffn_norm(queries))

class QueryGuidedTracingHead(nn.Module):
    """Read disease evidence from fused ECG memory using a shared query-conditioned classifier."""

    def __init__(self, text_dim=768, hidden_size=768, num_layers=2,
                 num_heads=12, intermediate_size=3072, dropout=0.1,
                 query_proj_hidden=768, query_self_attn=True, query_proj=True):
        super().__init__()
        if query_proj:

            self.query_proj = nn.Sequential(
                nn.Linear(text_dim, query_proj_hidden),
                nn.LayerNorm(query_proj_hidden),
            )
            nn.init.normal_(self.query_proj[0].weight, std=0.02)
            nn.init.zeros_(self.query_proj[0].bias)
        else:
            if text_dim != query_proj_hidden:
                raise ValueError(
                    f"query_proj=False requires text_dim ({text_dim}) == "
                    f"query_proj_hidden ({query_proj_hidden})"
                )
            self.query_proj = nn.Identity()
        if query_proj_hidden != hidden_size:
            raise ValueError(
                f"query_proj_hidden ({query_proj_hidden}) must match fused hidden_size ({hidden_size})"
            )
        self.hidden_size = int(hidden_size)
        self.layers = nn.ModuleList([
            TracingBlock(hidden_size, num_heads, intermediate_size, dropout,
                            query_self_attn=query_self_attn)
            for _ in range(num_layers)
        ])
        self.final_norm = nn.LayerNorm(hidden_size)
        self.classifier = nn.Linear(hidden_size, 1)

        nn.init.normal_(self.classifier.weight, std=1e-3)
        nn.init.zeros_(self.classifier.bias)

    def forward(self, prompt_features, fused_memory):
        classes = prompt_features.shape[0]
        batch = fused_memory.shape[0]
        queries = self.query_proj(prompt_features)          # [C, H]
        queries = queries.unsqueeze(0).expand(batch, -1, -1)  # [B, C, H]
        for layer in self.layers:
            queries = layer(queries, fused_memory)
        evidence = self.final_norm(queries)                 # [B, C, H]
        logits = self.classifier(evidence).squeeze(-1)      # [B, C]
        return logits
