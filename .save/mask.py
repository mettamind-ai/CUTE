#!/usr/bin/env python3

'''####################################################################
from flash_attn import flash_attn_func, flash_attn_varlen_func
USAGE https://github.com/Dao-AILab/flash-attention/blob/main/flash_attn/flash_attn_interface.py
flash_attn_varlen_func(q, k, v,
    cu_seqlens_q, cu_seqlens_k,
    max_seqlen_q, max_seqlen_k,
    dropout_p=0.0, softmax_scale=None,
    causal=False, window_size=(-1, -1), # -1 means infinite context window
)
If causal=True, the causal mask is:
seqlen_q=3 & seqlen_k=4 | seqlen_q=3 & seqlen_k=2
       1 1 0 0                   0 0
       1 1 1 0                   1 0
       1 1 1 1                   1 1

If window_size != (-1, -1), implements sliding window local attention.
Query at position i will only attend to keys between
[ i + seqlen_k - seqlen_q - window_size[0], 
  i + seqlen_k - seqlen_q + window_size[1] ] inclusive.

q: (total_q, nheads,   headdim), where total_q = total number of query tokens in the batch.
k: (total_k, nheads_k, headdim), where total_k = total number of key tokens in the batch.
v: (total_k, nheads_k, headdim), where total_k = total number of key tokens in the batch.

cu_seqlens_q: (bs + 1,), int32. The cumulative sequence lengths
cu_seqlens_k: (bs + 1,), int32. ... used to index into kv.
max_seqlen_q: int. Maximum query sequence length in the batch.
max_seqlen_k: int. Maximum key sequence length in the batch.

def flash_attn_func(
    q: (batch_size, seqlen, nheads,   headdim)
    k: (batch_size, seqlen, nheads_k, headdim)
    v: (batch_size, seqlen, nheads_k, headdim)

!!! IMPORTANT !!!

Với flash_attn_varlen_func thì phải unpack batch thành 1 chuỗi dài

'''####################################################################

import torch
from torch import Tensor

# Giả sử đây là batch input (bs=3, seq_len=6)
input_sequences = torch.tensor([
    [2, 4, 7, 0, 5, 6],
    [3, 0, 2, 0, 6, 0],
    [9, 1, 0, 2, 3, 0]
])

mask = (input_sequences == 0)
mask[:, -1] = True

print(input_sequences)
print(mask)

cu_seqlens = torch.cat([
    torch.zeros(1, dtype=torch.int32, device=input_sequences.device), 
    torch.where(mask.flatten())[0].to(torch.int32)
])
print(cu_seqlens)

max_seqlen = int(torch.max(torch.diff(cu_seqlens)))
print(max_seqlen)