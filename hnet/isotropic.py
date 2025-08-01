import re, copy, optree, torch
from torch import nn, Tensor

from typing import Optional
from functools import partial
from dataclasses import dataclass, field

from flash.ops.layer_norm import RMSNorm
from flash.mamba2 import Mamba2
from flash.modules.mha import CausalMHA
from flash.modules.mlp import SwiGLU
from utils import get_seq_idx, get_stage_cfg, HNetConfig

class Mamba2Wrapper(Mamba2):
    ''' Mamba2 wrapper class that has the same inference interface as the CausalMHA class. '''
    def step(self, hidden_states, inference_params):
        # Don't use _get_states_from_cache because we want to assert that they exist
        conv_state, ssm_state = inference_params.key_value_memory_dict[self.layer_idx] # init class accepts layer_idx
        result, conv_state, ssm_state = super().step(hidden_states, conv_state, ssm_state)

        # Update the state cache in-place
        inference_params.key_value_memory_dict[self.layer_idx][0].copy_(conv_state)
        inference_params.key_value_memory_dict[self.layer_idx][1].copy_(ssm_state)
        return result

class Block(nn.Module):
    def __init__(self, d_model, mixer_cls=None, mlp_cls=None, norm_cls=None, residual_in_fp32=True):
        super().__init__()
        self.residual_in_fp32 = residual_in_fp32
        self.norm1 = norm_cls(d_model)
        self.mixer = mixer_cls(d_model)
        if mlp_cls is not nn.Identity:
                self.norm2 = norm_cls(d_model)
                self.mlp = mlp_cls(d_model)
        else:   self.mlp = None
        assert RMSNorm is not None, "Triton is not installed"
        assert isinstance(self.norm1, RMSNorm), "Only RMSNorm is supported"

    def forward(self, hidden_states: Tensor, residual: Optional[Tensor]=None, inference_params=None, mixer_kwargs=None):
        hidden_states, residual = self.norm1(hidden_states, residual, prenorm=True, residual_in_fp32=self.residual_in_fp32)
        if mixer_kwargs is None: mixer_kwargs = {}
        hidden_states = self.mixer(hidden_states, inference_params=inference_params, **mixer_kwargs)
        if self.mlp is not None:
            hidden_states, residual = self.norm2(hidden_states, residual, prenorm=True, residual_in_fp32=self.residual_in_fp32)
            hidden_states = self.mlp(hidden_states)
        return hidden_states, residual

    def allocate_inference_cache(self, batch_size, max_seqlen, dtype=None, **kwargs):
        return self.mixer.allocate_inference_cache(batch_size, max_seqlen, dtype=dtype, **kwargs)

    def step(self, hidden_states, inference_params, residual=None):
        hidden_states, residual = self.norm1(hidden_states, residual, prenorm=True, residual_in_fp32=self.residual_in_fp32)
        hidden_states = self.mixer.step(hidden_states, inference_params)
        if self.mlp is not None:
            hidden_states, residual = self.norm2(hidden_states, residual, prenorm=True, residual_in_fp32=self.residual_in_fp32)
            hidden_states = self.mlp(hidden_states)
        return hidden_states, residual

def create_block(arch, d_model, d_intermediate=None, ssm_cfg=dict(), attn_cfg=dict(), \
        norm_epsilon=1e-5, layer_idx=None, residual_in_fp32=True, device=None, dtype=None):
    # Mixer
    if   arch in "t|T": mixer_cls = partial(CausalMHA,    **attn_cfg, layer_idx=layer_idx, device=device, dtype=dtype)
    elif arch in "m|M": mixer_cls = partial(Mamba2Wrapper, **ssm_cfg, layer_idx=layer_idx, device=device, dtype=dtype)
    # MLP
    if   arch in "T|M": mlp_cls = partial(SwiGLU, d_intermediate=d_intermediate, device=device, dtype=dtype)
    elif arch in "t|m": mlp_cls = nn.Identity

    norm_cls = partial(RMSNorm, eps=norm_epsilon, device=device, dtype=dtype)
    return Block(d_model, mixer_cls, mlp_cls, residual_in_fp32=residual_in_fp32, norm_cls=norm_cls)

@dataclass
class IsotropicInferenceParams:
    ''' Inference parameters that are passed to the main model to calculate and store the context during inference. '''
    max_seqlen: int
    max_batch_size: int
    seqlen_offset: int = 0
    batch_size_offset: int = 0
    key_value_memory_dict: dict = field(default_factory=dict)
    lengths_per_sample: Optional[torch.Tensor] = None

    def reset(self, max_seqlen, max_batch_size):
        self.max_seqlen     = max_seqlen
        self.max_batch_size = max_batch_size
        self.seqlen_offset  = 0

        if self.lengths_per_sample is not None:
           self.lengths_per_sample.zero_()

        optree.tree_map(  # áp dụng lambda func lên lá của cây key_value_memory_dict
            lambda x: x.zero_() if isinstance(x, torch.Tensor) else x,
            self.key_value_memory_dict,
        )


class Isotropic(nn.Module):
    def __init__(self, config: HNetConfig, pos_idx: int, stage_idx: int, device=None, dtype=None):
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__()

        self.stage_idx = stage_idx
        self.d_model   = config.d_model[self.stage_idx]
        self.ssm_cfg   = get_stage_cfg(config.ssm_cfg, stage_idx)
        self.attn_cfg  = get_stage_cfg(config.attn_cfg, stage_idx)

        arch_layout = config.arch_layout
        for _ in range(stage_idx): arch_layout = arch_layout[1]
        arch_layout = arch_layout[pos_idx]
        layout_parse = re.findall(r"([mMtT])(\d+)", arch_layout)

        layers = []
        layer_idx = 0
        self.arch_full = []

        # self.height counts the number of things that get added to the residual stream
        self.height = 0

        for arch, n_layer in layout_parse:
            assert arch in ("m", "M", "t", "T")
            assert n_layer.isdigit()
            layers += [
                create_block(
                    arch, self.d_model,
                    d_intermediate=config.d_intermediate[self.stage_idx],
                    ssm_cfg=self.ssm_cfg,
                    attn_cfg=self.attn_cfg,
                    layer_idx=(layer_idx + i),
                    **factory_kwargs,
                )
                for i in range(int(n_layer))
            ]

            if arch.islower(): self.height += int(n_layer)
            else:              self.height += int(n_layer)*2

            self.arch_full.extend([arch for _ in range(int(n_layer))])
            layer_idx += int(n_layer)

        self.layers = nn.ModuleList(layers)
        self.rmsnorm = RMSNorm(self.d_model, eps=1e-5, **factory_kwargs)


    def allocate_inference_cache(self, batch_size, max_seqlen, dtype=None):
        """
        Allocate the inference cache for the Isotropic module.

        Arguments:
            batch_size: int. The number of sequences in the batch.
            max_seqlen: int. The maximum sequence length in the batch, not used for this module.
            dtype: torch.dtype. The dtype of the inference cache.

        The inference cache contains a list of inference caches, one for each block.
        """
        key_value_memory_dict = {}
        for i, layer in enumerate(self.layers):
            key_value_memory_dict[i] = layer.allocate_inference_cache(batch_size, max_seqlen, dtype=dtype)

        return IsotropicInferenceParams(
            key_value_memory_dict=key_value_memory_dict,
            max_seqlen=max_seqlen,
            max_batch_size=batch_size,
        )

    def forward(self, hidden_states, cu_seqlens=None, max_seqlen=None, mask=None, inference_params=None, **mixer_kwargs):
        assert (mask is not None) or (cu_seqlens is not None and max_seqlen is not None), \
            "Either mask or cu_seqlens and max_seqlen must be provided"

        attn_mixer_kwargs = copy.deepcopy(mixer_kwargs)
        ssm_mixer_kwargs = copy.deepcopy(mixer_kwargs)

        if mask is not None:
            packed = False
            hidden_states.dim() == 3, "Hidden states must be (B, L, D) in unpacked mode"
        else:
            attn_mixer_kwargs.update({"cu_seqlens": cu_seqlens.int(), "max_seqlen": max_seqlen})
            ssm_mixer_kwargs.update({"seq_idx": get_seq_idx(cu_seqlens, device=hidden_states.device)})
            packed = True

        residual = None
        for layer, arch in zip(self.layers, self.arch_full):
            if arch in ("m", "M"):
                layer_mixer_kwargs = ssm_mixer_kwargs
                if hidden_states.dim() == 2:
                    hidden_states = hidden_states.unsqueeze(0)
                    residual = None if residual is None else residual.unsqueeze(0)

            elif arch in ("t", "T"):
                layer_mixer_kwargs = attn_mixer_kwargs
                if hidden_states.dim() == 3 and packed:
                    hidden_states = hidden_states.squeeze(0)
                    residual = None if residual is None else residual.squeeze(0)
            else:
                # Currently supporting only Mamba2 and MHA
                raise NotImplementedError

            hidden_states, residual = layer(
                hidden_states,
                residual,
                inference_params=inference_params,
                mixer_kwargs=layer_mixer_kwargs,
            )

        # Setting prenorm=False ignores the residual
        hidden_states = self.rmsnorm(hidden_states, residual=residual, prenorm=False, residual_in_fp32=True)
        if hidden_states.dim() == 3 and packed: hidden_states = hidden_states.squeeze(0)

        if inference_params is not None:
            # here we also explicitly assume the mask is all True
            assert mask.shape[0] == 1, "seqlen_offset handling assumes batch size 1"
            inference_params.seqlen_offset += hidden_states.shape[1]

        return hidden_states


    def step(self, hidden_states, inference_params):
        """ Assumes hidden_states is (B, 1, D). Steps each of the layers in order, and then steps the main model.
        """
        residual = None
        for layer in self.layers:
            hidden_states, residual = layer.step(hidden_states, inference_params, residual=residual)

        hidden_states = self.rmsnorm(hidden_states, residual=residual, prenorm=False, residual_in_fp32=True)
        inference_params.seqlen_offset += 1

        return hidden_states
