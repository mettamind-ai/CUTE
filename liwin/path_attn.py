#!/usr/bin/env python3
# https://github.com/fla-org/flash-linear-attention/blob/main/fla/layers/path_attn.py
# -*- coding: utf-8 -*-
# Copyright (c) 2023-2025, Songlin Yang, Yu Zhang
from __future__ import annotations

from typing import TYPE_CHECKING, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint
from einops import rearrange
from transformers.utils import logging

from fla.layers.utils import pad_input, unpad_input
from fla.modules import RMSNorm, ShortConvolution
from fla.modules.l2norm import l2_norm
from path_attn.parallel import parallel_path_attention

if TYPE_CHECKING:
    from fla.models.utils import Cache

logger = logging.get_logger(__name__)


class PaTHAttention(nn.Module):
    def __init__(
        self,
        hidden_size: int = 2048,
        num_heads: int = 32,
        num_kv_heads: Optional[int] = None,
        use_forget_gate: bool = False,
        use_qk_norm: bool = False,
        layer_idx: int = None,
        use_low_rank_w: bool = True,
        use_w_shortconv: bool = True,
        conv_size: int = 3,
        conv_bias: bool = False,
    ):
        super().__init__()

        self.hidden_size = hidden_size
        self.num_heads = num_heads
        if num_kv_heads is None:
            self.num_kv_heads = self.num_heads
        else:
            self.num_kv_heads = num_kv_heads
        self.head_dim = self.hidden_size // self.num_heads
        self.kv_dim = self.num_kv_heads * self.head_dim

        self.layer_idx = layer_idx

        self.q_proj = nn.Linear(self.hidden_size, self.hidden_size, bias=False)
        self.k_proj = nn.Linear(self.hidden_size, self.kv_dim, bias=False)
        self.v_proj = nn.Linear(self.hidden_size, self.kv_dim, bias=False)

        # We use low-rank parameterization for the w_proj to reduce parameters in MHA settings.
        if use_low_rank_w:
            self.w_proj = nn.Sequential(
                nn.Linear(self.hidden_size, 32, bias=False),
                nn.Linear(32, self.kv_dim, bias=False)
            )
        # In MQA/GQA settings, key/value heads are shared, so we use a standard linear projection
        # which doesn't introduce too many parameters
        else: self.w_proj = nn.Linear(self.hidden_size, self.kv_dim, bias=False)

        # TODO: per head norm?
        if use_qk_norm:
            self.maybe_q_norm = RMSNorm(self.hidden_size)
            self.maybe_k_norm = RMSNorm(self.kv_dim)
        else:
            self.maybe_q_norm = nn.Identity()
            self.maybe_k_norm = nn.Identity()

        self.use_w_shortconv = use_w_shortconv
        if use_w_shortconv:
            self.w_conv1d = ShortConvolution(hidden_size=self.kv_dim, kernel_size=conv_size, bias=conv_bias, activation='silu')

        self.use_forget_gate = use_forget_gate
        if use_forget_gate:
            self.g_proj = nn.Linear(self.hidden_size, self.num_heads, bias=True)

        self.beta_proj = nn.Linear(self.hidden_size, self.num_kv_heads, bias=True)
        self.o_proj = nn.Linear(self.hidden_size, self.hidden_size, bias=False)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Cache] = None,
        output_attentions: bool = False,
        use_cache: bool = False,
        **kwargs,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
        if use_cache:
            assert past_key_values is not None, "past_key_values must be provided when use_cache is True"
        if attention_mask is not None:
            assert len(attention_mask.shape) == 2, (
                "Expected attention_mask as a 0-1 matrix with shape [batch_size, seq_len] "
                "for padding purposes (0 indicating padding). "
                "Arbitrary attention masks of shape [batch_size, seq_len, seq_len] are not allowed."
            )
        batch_size, q_len, _ = hidden_states.size()
        q = self.q_proj(hidden_states)
        k = self.k_proj(hidden_states)
        v = self.v_proj(hidden_states)
        w = self.w_proj(hidden_states)
        beta = self.beta_proj(hidden_states).sigmoid() * 2  # allowing negative eigenvalues
        g = F.logsigmoid(self.g_proj(hidden_states).float()) if self.use_forget_gate else None
        q, k = self.maybe_q_norm(q), self.maybe_k_norm(k)
        cu_seqlens = kwargs.get('cu_seqlens', None)

        # Training
        assert use_cache is False, "use_cache should be False in training"
        if self.use_w_shortconv:
            w, _ = self.w_conv1d(w, cache=None, output_final_state=False, cu_seqlens=cu_seqlens)
        q = rearrange(q, '... (h d) -> ... h d', d=self.head_dim)
        k = rearrange(k, '... (h d) -> ... h d', d=self.head_dim)
        v = rearrange(v, '... (h d) -> ... h d', d=self.head_dim)
        w = rearrange(w, '... (h d) -> ... h d', d=self.head_dim)
        w = l2_norm(w)
        o, _ = parallel_path_attention(q=q, k=k, v=v, w=w, beta=beta, g=g, cu_seqlens=cu_seqlens)

        o = rearrange(o, '... h d -> ... (h d)')
        o = self.o_proj(o)
        return o, None, past_key_values
