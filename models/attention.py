"""FP32 attention arithmetic for numerical stability under mixed precision."""

import os
import torch
import torch.nn.functional as F

def _sdpa_enabled() -> bool:
    return os.environ.get('LAUER_MHA_SDPA', '0') in ('1', 'true', 'True')

def _build_additive_mask(attn_mask, key_padding_mask, batch, query_length,
                         key_length, heads, device):
    mask = None

    if attn_mask is not None:
        m = attn_mask
        if m.dim() == 2:
            m = m[None, None]
        elif m.dim() == 3 and m.shape[0] == batch * heads:
            m = m.view(batch, heads, query_length, key_length)
        elif m.dim() == 3 and m.shape[0] == batch:
            m = m[:, None]
        else:
            raise ValueError("Unsupported attention mask shape: {}".format(tuple(m.shape)))
        if m.dtype == torch.bool:
            zeros = torch.zeros(m.shape, device=device, dtype=torch.float32)
            m = zeros.masked_fill(m, float('-inf'))
        else:
            m = m.float()
        mask = m

    if key_padding_mask is not None:
        p = key_padding_mask[:, None, None, :]           # [B, 1, 1, Lk]
        if p.dtype == torch.bool:
            zeros = torch.zeros(p.shape, device=device, dtype=torch.float32)
            p = zeros.masked_fill(p, float('-inf'))
        else:
            p = p.float()
        mask = p if mask is None else mask + p

    return None if mask is None else mask.to(torch.float32).contiguous()

def _mha_fp32_legacy(module, query, key, value, key_padding_mask=None, attn_mask=None):
    batch, query_length, hidden = query.shape
    key_length = key.shape[1]
    heads = module.num_heads
    head_dim = hidden // heads

    weight = module.in_proj_weight
    bias = module.in_proj_bias
    q_weight, k_weight, v_weight = weight.chunk(3, dim=0)
    if bias is None:
        q_bias = k_bias = v_bias = None
    else:
        q_bias, k_bias, v_bias = bias.chunk(3, dim=0)
    q = F.linear(query, q_weight, q_bias)
    k = F.linear(key, k_weight, k_bias)
    v = F.linear(value, v_weight, v_bias)
    q = q.view(batch, query_length, heads, head_dim).transpose(1, 2)
    k = k.view(batch, key_length, heads, head_dim).transpose(1, 2)
    v = v.view(batch, key_length, heads, head_dim).transpose(1, 2)

    scores = torch.matmul(q.float(), k.float().transpose(-2, -1))
    scores.mul_(head_dim ** -0.5)
    if attn_mask is not None:
        mask = attn_mask
        if mask.dim() == 2:
            mask = mask[None, None]
        elif mask.dim() == 3 and mask.shape[0] == batch * heads:
            mask = mask.view(batch, heads, query_length, key_length)
        elif mask.dim() == 3 and mask.shape[0] == batch:
            mask = mask[:, None]
        else:
            raise ValueError("Unsupported attention mask shape: {}".format(tuple(mask.shape)))
        if mask.dtype == torch.bool:
            scores.masked_fill_(mask, float("-inf"))
        else:
            scores.add_(mask.float())
    if key_padding_mask is not None:
        padding = key_padding_mask[:, None, None, :]
        if padding.dtype == torch.bool:
            scores.masked_fill_(padding, float("-inf"))
        else:
            scores.add_(padding.float())
    probabilities = torch.softmax(scores, dim=-1)
    if module.dropout and module.training:
        probabilities = F.dropout(probabilities, p=module.dropout)
    attended = torch.matmul(probabilities, v.float())
    attended = attended.transpose(1, 2).reshape(batch, query_length, hidden)
    attended = attended.to(query.dtype)
    return F.linear(attended, module.out_proj.weight, module.out_proj.bias), probabilities

def _mha_sdpa(module, query, key, value, key_padding_mask=None, attn_mask=None):
    batch, query_length, hidden = query.shape
    key_length = key.shape[1]
    heads = module.num_heads
    head_dim = hidden // heads

    weight = module.in_proj_weight
    bias = module.in_proj_bias
    q_weight, k_weight, v_weight = weight.chunk(3, dim=0)
    if bias is None:
        q_bias = k_bias = v_bias = None
    else:
        q_bias, k_bias, v_bias = bias.chunk(3, dim=0)
    q = F.linear(query, q_weight, q_bias)
    k = F.linear(key, k_weight, k_bias)
    v = F.linear(value, v_weight, v_bias)
    q = q.view(batch, query_length, heads, head_dim).transpose(1, 2)
    k = k.view(batch, key_length, heads, head_dim).transpose(1, 2)
    v = v.view(batch, key_length, heads, head_dim).transpose(1, 2)

    mask = _build_additive_mask(attn_mask, key_padding_mask, batch,
                                query_length, key_length, heads, q.device)
    dropout_p = module.dropout if module.training else 0.0

    attended = F.scaled_dot_product_attention(
        q, k, v,
        attn_mask=mask,
        dropout_p=dropout_p,
    )
    attended = attended.transpose(1, 2).reshape(batch, query_length, hidden)
    attended = attended.to(query.dtype)
    return F.linear(attended, module.out_proj.weight, module.out_proj.bias), None

def mha_fp32(module, query, key, value, key_padding_mask=None, attn_mask=None):
    if _sdpa_enabled():
        return _mha_sdpa(module, query, key, value, key_padding_mask, attn_mask)
    return _mha_fp32_legacy(module, query, key, value, key_padding_mask, attn_mask)
