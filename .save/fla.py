import functools
import torch, triton
import triton.language as tl
from torch import nn

from typing import Any, Callable, Dict, Literal, Optional, Tuple
from causal_conv1d import causal_conv1d_fn, causal_conv1d_update
from einops import rearrange


# https://github.com/fla-org/flash-linear-attention/blob/main/fla/utils.py
def input_guard(
    fn: Callable[..., torch.Tensor]
) -> Callable[..., torch.Tensor]:
    """
    A decorator to make sure all input tensors are contiguous and set the device based on input tensors.
    """

    @functools.wraps(fn)
    def wrapper(*args, **kwargs):
        contiguous_args = (i if not isinstance(i, torch.Tensor) else i.contiguous() for i in args)
        contiguous_kwargs = {k: (v if not isinstance(v, torch.Tensor) else v.contiguous()) for k, v in kwargs.items()}

        tensor = None
        for arg in args:
            if isinstance(arg, torch.Tensor):
                tensor = arg
                break
        if tensor is None:
            for value in kwargs.values():
                if isinstance(value, torch.Tensor):
                    tensor = value
                    break

        if tensor is not None:
            ctx = custom_device_ctx(tensor.device.index)
        else:
            ctx = contextlib.nullcontext()

        with ctx:
            return fn(*contiguous_args, **contiguous_kwargs)

    return wrapper
contiguous = input_guard


def tensor_cache(
    fn: Callable[..., torch.Tensor]
) -> Callable[..., torch.Tensor]:
    """
    A decorator that caches the most recent result of a function with tensor inputs.

    This decorator will store the output of the decorated function for the most recent set of input tensors.
    If the function is called again with the same input tensors, it will return the cached result.

    Args:
        fn (Callable[..., torch.Tensor]):
            The function to be decorated. It should take tensor inputs and return tensor outputs.

    Returns:
        Callable[..., torch.Tensor]:
            A wrapped version of the input function with single-entry caching.
    """
    last_args: Optional[Tuple] = None
    last_kwargs: Optional[Dict] = None
    last_result: Any = None

    @functools.wraps(fn)
    def wrapper(*args: Any, **kwargs: Any) -> Any:
        nonlocal last_args, last_kwargs, last_result

        if last_args is not None and last_kwargs is not None:
            if len(args) == len(last_args) and len(kwargs) == len(last_kwargs):
                if all(a is b for a, b in zip(args, last_args)) and \
                        all(k in last_kwargs and v is last_kwargs[k] for k, v in kwargs.items()):
                    return last_result

        result = fn(*args, **kwargs)
        last_args, last_kwargs, last_result = args, kwargs, result
        return result

    return wrapper


# https://github.com/fla-org/flash-linear-attention/blob/main/fla/ops/utils/index.py
@tensor_cache
def prepare_lens(cu_seqlens: torch.LongTensor) -> torch.LongTensor:
    return cu_seqlens[1:] - cu_seqlens[:-1]

@tensor_cache
def prepare_position_ids(cu_seqlens: torch.LongTensor) -> torch.LongTensor:
    return torch.cat([
        torch.arange(n, dtype=cu_seqlens.dtype, device=cu_seqlens.device)
        for n in prepare_lens(cu_seqlens).unbind()
    ])

@tensor_cache
def prepare_sequence_ids(cu_seqlens: torch.LongTensor) -> torch.LongTensor:
    return prepare_position_ids(cu_seqlens).eq(0).cumsum(0) - 1


# https://github.com/fla-org/flash-linear-attention/blob/main/fla/modules/convolution.py
class ShortConvolution(nn.Conv1d):
    """ Simple wrapper around `nn.Conv1d` that accepts dimension last.
    """
    def __init__(
        self,
        hidden_size: int,
        kernel_size: int,
        bias: bool = False,
        activation: Optional[str] = 'silu',
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


    def forward(
        self,
        x: torch.Tensor,
        cu_seqlens: Optional[torch.LongTensor],
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            x (`torch.Tensor`):
                Tensor of shape `[T, D]`.
            cu_seqlens (Optional[torch.LongTensor]):
                Cumulative sequence lengths for each batch. Used for varlen.
        Returns:
            Tensor of shape `[T, D]`.
        """

        # print(x.shape, self.kernel_size[0], "<<<<<")
        T, D, W = *x.shape, self.kernel_size[0]
        N = len(cu_seqlens) - 1

        # Sequence index for each token. Used for varlen.
        # Suppose a batch consists of two sequences with lengths 3 and 4,
        # seq_idx=[0, 0, 0, 1, 1, 1, 1] for this batch.
        seq_idx = prepare_sequence_ids(cu_seqlens).to(torch.int32).unsqueeze(0)
        x = rearrange(x, 't d -> 1 d t')
        x = causal_conv1d_fn(
            x=x,
            weight=rearrange(self.weight, "d 1 w -> d w"),
            bias=self.bias,
            activation=self.activation,
            seq_idx=seq_idx,
        )
        return rearrange(x, "1 d t -> t d")
