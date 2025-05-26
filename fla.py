# https://github.com/fla-org/flash-linear-attention/blob/main/fla/modules/convolution.py
from causal_conv1d import causal_conv1d_fn, causal_conv1d_update

@checkpoint
def proj_then_conv1d(
    x: torch.Tensor,
    proj_weight: torch.Tensor,
    conv1d_weight: torch.Tensor,
    conv1d_bias: Optional[torch.Tensor] = None,
    cache: Optional[torch.Tensor] = None
) -> torch.Tensor:
    # We do matmul and transpose BLH -> HBL at the same time
    x = rearrange(proj_weight @ rearrange(x, "b t d -> d (b t)"), "d (b t) -> b d t", t=x.shape[-2])

    if causal_conv1d_fn is None:
        raise ImportError("`causal_conv1d_fn` is not available. Please install `causal-conv1d` first.")
    if cache is None:
        x = causal_conv1d_fn(
            x=x,
            weight=rearrange(conv1d_weight, "d 1 w -> d w"),
            bias=conv1d_bias,
            activation="silu",
        ).transpose(1, 2)
    else:
        assert x.shape[-1] == 1, "Only support decoding with 1 token at a time for now"
        x = x.squeeze(-1)
        x = causal_conv1d_update(
            x=x,
            weight=rearrange(conv1d_weight, "d 1 w -> d w"),
            bias=conv1d_bias,
            cache=cache,
            activation="silu",
        )
    return x


@triton.jit
def causal_conv1d_varlen_states_fwd_kernel(
    x,
    cache,
    offsets,
    D,
    W,
    BD: tl.constexpr,
    BW: tl.constexpr
):
    i_d, i_w, i_n = tl.program_id(0), tl.program_id(1), tl.program_id(2)
    eos = tl.load(offsets + i_n + 1)
    bos = tl.maximum(tl.load(offsets + i_n), eos - W)
    o_t = eos - (i_w + 1) * BW + tl.arange(0, BW)
    o_d = i_d * BD + tl.arange(0, BD)
    o_w = W - (i_w + 1) * BW + tl.arange(0, BW)

    b_x = tl.load(x + o_t * D + o_d[:, None], mask=(o_t >= bos) & (o_d[:, None] < D), other=0)
    tl.store(cache + i_n * D*W + o_d[:, None] * W + o_w, b_x, mask=(o_d[:, None] < D) & (o_w >= 0))


@input_guard
def causal_conv1d_varlen_states_fwd(
    x: torch.Tensor,
    cache: torch.Tensor,
    cu_seqlens: torch.Tensor,
    state_len: int
) -> torch.Tensor:
    N, D, W = len(cu_seqlens) - 1, x.shape[-1], state_len
    cache = torch.empty(N, D, W, dtype=x.dtype, device=x.device) if cache is None else cache
    BD = min(triton.next_power_of_2(D), 256)
    BW = min(triton.next_power_of_2(state_len), 16)
    grid = (triton.cdiv(D, BD), triton.cdiv(W, BW), N)
    with torch.cuda.device(x.device.index):
        causal_conv1d_varlen_states_fwd_kernel[grid](
            x=x,
            cache=cache,
            offsets=cu_seqlens,
            D=D,
            W=W,
            BW=BW,
            BD=BD
        )
    return cache

class ShortConvolution(nn.Conv1d):
    """
    Simple wrapper around `nn.Conv1d` that accepts dimension last.
    """

    def __init__(
        self,
        hidden_size: int,
        kernel_size: int,
        bias: bool = False,
        activation: Optional[str] = 'silu',
        use_fast_conv1d: Optional[bool] = True,
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None,
    ):
        super().__init__(
            in_channels=hidden_size,
            out_channels=hidden_size,
            kernel_size=kernel_size,
            groups=hidden_size,
            bias=bias,
            padding=kernel_size - 1,
            device=device,
            dtype=dtype,
        )

        self.hidden_size = hidden_size
        self.activation = None
        if activation is not None:
            assert activation in ['silu', 'swish'], f"Activation `{activation}` not supported yet."
            self.activation = activation

        if causal_conv1d_fn is None:
            if use_fast_conv1d:
                raise RuntimeError(
                    "Please either install `causal-conv1d>=1.4.0` to enable fast causal short convolution CUDA kernel "
                    "or set `use_fast_conv1d` to False"
                )
            else:
                warnings.warn(
                    "The naive Pytorch verison is very slow in practice, "
                    "please run `pip install causal-conv1d>=1.4.0` to install fast causal short convolution CUDA kernel",
                    category=ImportWarning
                )
        self.use_fast_conv1d = use_fast_conv1d

    def extra_repr(self):
        s = ('{in_channels}, {out_channels}, kernel_size={kernel_size}'
             ', stride={stride}')
        if self.padding != (0,) * len(self.padding):
            s += ', padding={padding}'
        if self.dilation != (1,) * len(self.dilation):
            s += ', dilation={dilation}'
        if self.output_padding != (0,) * len(self.output_padding):
            s += ', output_padding={output_padding}'
        if self.groups != 1:
            s += ', groups={groups}'
        if self.bias is None:
            s += ', bias=False'
        if self.padding_mode != 'zeros':
            s += ', padding_mode={padding_mode}'
        if self.activation is not None:
            s += ', activation={activation}'
        if not self.use_fast_conv1d:
            s += ', use_fast_conv1d={use_fast_conv1d}'
        return s.format(**self.__dict__)

    def forward(
        self,
        x: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        cache: Optional[torch.Tensor] = None,
        output_final_state: bool = False,
        cu_seqlens: Optional[torch.LongTensor] = None,
        **kwargs,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            x (`torch.Tensor`):
                Tensor of shape `[B, T, D]`.
                If `seq_idx` is provided, `B` must be 1.
            mask (`Optional[torch.Tensor]`):
                Attention mask dealing with padded positions.
            cache (`Optional[torch.Tensor]`):
                Previous cache tensor of shape `[N, D, W]`, where `W` is the kernel size.
                If provided, the cache is updated **inplace**.
            output_final_state (Optional[bool]):
                Whether to output the final state of shape `[N, D, W]`. Default: `False`.
            cu_seqlens (Optional[torch.LongTensor]):
                Cumulative sequence lengths for each batch. Used for varlen. Default: `None`.
                Shape: [B+1]

        Returns:
            Tensor of shape `[B, T, D]`.
        """

        B, T, D, W = *x.shape, self.kernel_size[0]
        N = B if cu_seqlens is None else len(cu_seqlens) - 1
        if mask is not None:
            if cu_seqlens is not None:
                raise ValueError("`mask` and `cu_seqlens` cannot be provided at the same time")
            x = x.mul_(mask.unsqueeze(-1))
        if output_final_state and cache is None:
            cache = x.new_zeros(N, D, W)
        # during the decoding phase, we assume the batch is composed of sequences of length 1
        if cache is not None and B * T == N:
            return self.step(x, cache, cu_seqlens)

        if cache is not None:
            if cu_seqlens is not None:
                cache = causal_conv1d_varlen_states_fwd(x, cache, cu_seqlens, W)
            else:
                cache[:, :, -min(W, T):].copy_(rearrange(x[..., -min(W, T):, :], 'n w d -> n d w'))

        x = rearrange(x, 'b t d -> b d t')
        if self.use_fast_conv1d:
            # Sequence index for each token. Used for varlen.
            # Suppose a batch consists of two sequences with lengths 3 and 4,
            # seq_idx=[0, 0, 0, 1, 1, 1, 1] for this batch.
            # NOTE: No need to provide this arg if `cu_seqlens` is passed.
            # This arg is just for BC, and will be removed in the future.
            # [B, T]
            seq_idx = kwargs.get('seq_idx', None)
            if cu_seqlens is not None and seq_idx is None:
                seq_idx = prepare_sequence_ids(cu_seqlens).to(torch.int32).unsqueeze(0)
            x = causal_conv1d_fn(
                x=x,
                weight=rearrange(self.weight, "d 1 w -> d w"),
                bias=self.bias,
                activation=self.activation,
                seq_idx=seq_idx,
            )
        else:
            if cu_seqlens is not None:
                raise ValueError("`cu_seqlens` is not supported for the naive Pytorch version")
            x = self._conv_forward(x, self.weight, self.bias)[..., :x.shape[-1]]
            if self.activation is not None:
                x = ACT2FN[self.activation](x)
        return rearrange(x, "b d t -> b t d"), cache

    def step(
        self,
        x: torch.Tensor,
        cache: torch.Tensor,
        cu_seqlens: Optional[torch.LongTensor] = None
    ):
        shape = x.shape
        x = x.squeeze(0) if cu_seqlens is not None else x.squeeze(1)
        if self.use_fast_conv1d:
            x = causal_conv1d_update(
                x=x,
                conv_state=cache,
                weight=rearrange(self.weight, "d 1 w -> d w"),
                bias=self.bias,
                activation=self.activation,
            )
        else:
            dtype = x.dtype
            # we follow the fast mode that updates the cache in-place
            cache.copy_(cache.roll(shifts=-1, dims=-1))
            cache[:, :, -1] = x
            x = torch.sum(cache * rearrange(self.weight, "d 1 w -> d w"), dim=-1)
            if self.bias is not None:
                x = x + self.bias
            if self.activation is not None:
                x = ACT2FN[self.activation](x).to(dtype=dtype)
        return x.view(shape), cache

    @property
    def state_size(self) -> int:
        return self.hidden_size * self.kernel_size
