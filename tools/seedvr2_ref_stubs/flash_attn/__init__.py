"""Stand-in for flash_attn on machines without CUDA: flash_attn_varlen_func
as one scaled_dot_product_attention per packed sequence (the same math, in
the inputs' dtype upcast to float32)."""
import torch
import torch.nn.functional as F


def flash_attn_varlen_func(q, k, v, cu_seqlens_q, cu_seqlens_k, max_seqlen_q=None,
                           max_seqlen_k=None, dropout_p=0.0, softmax_scale=None,
                           causal=False, deterministic=False, **kwargs):
    out = torch.empty_like(q)
    cq = cu_seqlens_q.tolist()
    ck = cu_seqlens_k.tolist()
    for i in range(len(cq) - 1):
        qs = q[cq[i]:cq[i + 1]].transpose(0, 1).float()   # h l d
        ks = k[ck[i]:ck[i + 1]].transpose(0, 1).float()
        vs = v[ck[i]:ck[i + 1]].transpose(0, 1).float()
        o = F.scaled_dot_product_attention(qs, ks, vs, scale=softmax_scale, is_causal=causal)
        out[cq[i]:cq[i + 1]] = o.transpose(0, 1).to(q.dtype)
    return out
