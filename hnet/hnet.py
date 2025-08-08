from dataclasses import dataclass, field
from typing import List, Union, Optional

import torch, torch.nn as nn
from utils import AttnConfig, SSMConfig, HNetConfig
from dc import RoutingModule, ChunkLayer, DeChunkLayer, RoutingModuleState, DeChunkState
from isotropic import Isotropic, IsotropicInferenceParams


def apply_optimization_params(param: torch.Tensor, **kwargs) -> None:
    """ Annotates a parameter with optimization parameters.
    Specifically, updates the parameter's `_optim` attribute with the given kwargs.
    """
    if hasattr(param, "_optim"): param._optim.update(kwargs)
    else:                        param._optim = kwargs


class STE(torch.autograd.Function):
    '''Straight-Through Estimator
H-Net cần phải đưa ra một quyết định "cứng" (hard decision): tại một vị trí, có gộp (chunk) hay không? 
Đây là một lựa chọn có/không, và phép toán để đưa ra lựa chọn này (ví dụ như lấy giá trị lớn nhất, hoặc làm tròn)
thường không có đạo hàm. Nếu không có đạo hàm, gradient không thể lan truyền ngược qua bước này, 
và mô hình không thể học được.

STE giải quyết vấn đề này như sau:
1. Tính toán xuôi (Forward Pass): Sử dụng kết quả của quyết định "cứng". Ví dụ, nếu xác suất là 0.8, 
nó sẽ làm tròn thành 1.0.
2. Tính toán ngược (Backward Pass): "Giả vờ" rằng không có bước quyết định "cứng" nào cả. 
Nó sao chép gradient một cách trực tiếp đi qua, như thể bước đó chỉ là một hàm đồng nhất (identity function).
    '''
    @staticmethod
    def forward(ctx, x):
        return torch.ones_like(x)

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output

def ste_func(x):
    return STE.apply(x)


@dataclass
class HNetState:
    encoder_state: Optional[IsotropicInferenceParams] = None
    routing_module_state: Optional[RoutingModuleState] = None
    main_network_state: Optional[Union["HNetState", IsotropicInferenceParams]] = None
    dechunk_state: Optional[DeChunkState] = None
    decoder_state: Optional[IsotropicInferenceParams] = None


class HNet(nn.Module):
    def __init__(self, config: HNetConfig, stage_idx: int, device=None, dtype=None) -> None:
        super().__init__()
        factory_kwargs = {"device": device, "dtype": dtype}

        self.stage_idx = stage_idx
        self.d_model = config.d_model[stage_idx]

        arch_layout = config.arch_layout
        for _ in range(stage_idx): arch_layout = arch_layout[1]

        assert isinstance(arch_layout, list), f"Wrong arch_layout: {arch_layout}"
        if len(arch_layout) == 3:
            sub_model_names = ["encoder", "main_network", "decoder"]
            self.is_innermost = False
        elif len(arch_layout) == 1:
            sub_model_names = ["main_network"]
            self.is_innermost = True
        else:
            raise NotImplementedError

        for _name, _layout in zip(sub_model_names, arch_layout):
            if self.is_innermost or _name in ("encoder", "decoder"):
                SubModel = Isotropic        # chuỗi các lớp xử lý tuần tự, không phân cấp
                _stage_idx = stage_idx      # << giữ nguyên stage_idx
                _pos_idx = None
                if _name == "encoder": _pos_idx = 0

                # if innermost, then len(layer_layout) == 1
                elif self.is_innermost: _pos_idx = 0
                elif _name == "decoder": _pos_idx = 2
                _pos_idx_dict = {"pos_idx": _pos_idx}
            else:
                SubModel = HNet             # tiếp tục cấu trúc phân cấp
                _stage_idx = stage_idx + 1  # << Tăng stage_idx
                _pos_idx_dict = {}

            _sub_model = SubModel(
                config=config,
                stage_idx=_stage_idx,
                **_pos_idx_dict,
                **factory_kwargs,
            )
            self.add_module(_name, _sub_model)

        if not self.is_innermost:  # chưa phải stage cuối trong hierachy
            self.routing_module = RoutingModule(self.d_model, **factory_kwargs)
            self.chunk_layer    = ChunkLayer()
            self.dechunk_layer  = DeChunkLayer(self.d_model)

            # do the residual in fp32
            self.residual_proj = nn.Linear(
                self.d_model, self.d_model, device=device, dtype=torch.float32
            )
            nn.init.zeros_(self.residual_proj.weight)
            self.residual_proj.weight._no_reinit = True
            self.residual_func = lambda out, residual, p: out * ste_func(p) + residual

        if stage_idx > 0 and self.d_model - config.d_model[stage_idx - 1] > 0:
                self.pad_dimension = nn.Parameter(
                    torch.zeros(self.d_model - config.d_model[stage_idx - 1], **factory_kwargs)
                )
        else:   self.pad_dimension = None


    def allocate_inference_cache(self, batch_size, max_seqlen, dtype=None):
        """
        Allocate the inference cache for the HNet.

        Arguments:
            batch_size: int. The number of sequences in the batch.
            max_seqlen: int. The maximum sequence length in the batch.
            dtype: torch.dtype. The dtype of the inference cache.

        The structure of the inference cache is as follows:
            - [encoder state]
            - [routing module state]
            - [main network state]
            - [dechunk state]
            - [decoder state]
        It is thus a list of length 5.
        """
        if self.is_innermost:
            return HNetState(
                main_network_state=self.main_network.allocate_inference_cache(
                    batch_size, max_seqlen, dtype=dtype
                )
            )
        else:
            device = self.residual_proj.weight.device
            return HNetState(
                encoder_state=self.encoder.allocate_inference_cache(
                    batch_size, max_seqlen, dtype=dtype
                ),
                routing_module_state=self.routing_module.allocate_inference_cache(
                    batch_size, max_seqlen, device, dtype=dtype
                ),
                main_network_state=self.main_network.allocate_inference_cache(
                    batch_size, max_seqlen, dtype=dtype
                ),
                dechunk_state=self.dechunk_layer.allocate_inference_cache(
                    batch_size, max_seqlen, device, dtype=dtype
                ),
                decoder_state=self.decoder.allocate_inference_cache(
                    batch_size, max_seqlen, dtype=dtype
                ),
            )

    def forward(
        self,
        hidden_states,
        cu_seqlens=None,
        max_seqlen=None,
        mask=None,
        inference_params=None,
        **mixer_kwargs,
    ):
        assert mask is not None or (
            cu_seqlens is not None and max_seqlen is not None
        ), "Either mask or cu_seqlens and max_seqlen must be provided"

        if inference_params is None:
            inference_params = HNetState(main_network_state=None)
        else:
            assert (
                mask is not None
            ), "Mask must be provided if inference_params is provided"

        D = hidden_states.shape[-1]
        EARLY_DIMS = hidden_states.shape[:-1]

        if self.pad_dimension is not None:
            hidden_states = torch.cat(
                (hidden_states, self.pad_dimension.expand(EARLY_DIMS + (-1,))), dim=-1
            )

        if self.is_innermost:
            hidden_states = self.main_network(
                hidden_states,
                cu_seqlens=cu_seqlens,
                max_seqlen=max_seqlen,
                mask=mask,
                inference_params=inference_params.main_network_state,
                **mixer_kwargs,
            )
            hidden_states = hidden_states[..., :D]
            return hidden_states, []

        hidden_states = self.encoder(
            hidden_states,
            cu_seqlens=cu_seqlens,
            max_seqlen=max_seqlen,
            mask=mask,
            inference_params=inference_params.encoder_state,
            **mixer_kwargs,
        )

        hidden_states_for_residual = hidden_states.to(
            dtype=self.residual_proj.weight.dtype
        )
        residual = self.residual_proj(hidden_states_for_residual)

        bpred_output = self.routing_module(
            hidden_states,
            cu_seqlens=cu_seqlens,
            mask=mask,
            inference_params=inference_params.routing_module_state,
        )
        hidden_states, next_cu_seqlens, next_max_seqlen, next_mask = self.chunk_layer(
            hidden_states, bpred_output.boundary_mask, cu_seqlens, mask=mask
        )

        hidden_states, prev_boundary_predictions = self.main_network(
            hidden_states,
            cu_seqlens=next_cu_seqlens,
            max_seqlen=next_max_seqlen,
            mask=next_mask,
            inference_params=inference_params.main_network_state,
            **mixer_kwargs,
        )

        hidden_states = self.dechunk_layer(
            hidden_states,
            bpred_output.boundary_mask,
            bpred_output.boundary_prob,
            next_cu_seqlens,
            mask=mask,
            inference_params=inference_params.dechunk_state,
        )

        hidden_states = self.residual_func(
            hidden_states.to(dtype=residual.dtype), residual, bpred_output.selected_probs
        ).to(hidden_states.dtype)

        hidden_states = self.decoder(
            hidden_states,
            cu_seqlens=cu_seqlens,
            max_seqlen=max_seqlen,
            mask=mask,
            inference_params=inference_params.decoder_state,
            **mixer_kwargs,
        )

        hidden_states = hidden_states[..., :D]
        return hidden_states, [bpred_output, *prev_boundary_predictions]


    def step(self, hidden_states, inference_params):
        D = hidden_states.shape[-1]

        if self.pad_dimension is not None:
            # Tạo padding tensor có kích thước [batch_size, sequence_length, pad_dim]
            padding = self.pad_dimension.expand(hidden_states.shape[:-1] + (-1,))
            hidden_states = torch.cat([hidden_states, padding], dim=-1)

        if self.is_innermost:
            hidden_states = self.main_network.step(hidden_states, inference_params.main_network_state)
            hidden_states = hidden_states[..., :D]
            return hidden_states, []

        hidden_states = self.encoder.step(hidden_states, inference_params.encoder_state)
        hidden_states_for_residual = hidden_states.to(dtype=self.residual_proj.weight.dtype)
        residual = self.residual_proj(hidden_states_for_residual)

        bpred_output = self.routing_module.step(hidden_states, inference_params.routing_module_state)
        hidden_states_inner = self.chunk_layer.step(hidden_states, bpred_output.boundary_mask)

        prev_boundary_predictions = []
        if hidden_states_inner.shape[0] > 0:
            hidden_states_inner, prev_boundary_predictions = \
                self.main_network.step(hidden_states_inner, inference_params.main_network_state)

        hidden_states = self.dechunk_layer.step(
            hidden_states_inner,
            bpred_output.boundary_mask,
            bpred_output.boundary_prob,
            inference_params.dechunk_state,
        )

        hidden_states = self.residual_func(
            hidden_states.to(dtype=residual.dtype), residual, bpred_output.selected_probs
        ).to(hidden_states.dtype)

        hidden_states = self.decoder.step(hidden_states, inference_params.decoder_state)
        hidden_states = hidden_states[..., :D]

        return hidden_states, [bpred_output, *prev_boundary_predictions]


    def _init_weights(self, initializer_range: float = 0.02, parent_residuals: int = 0) -> None:
        n_residuals = parent_residuals
        if self.is_innermost:
            n_residuals += self.main_network.height
            for name, m in self.main_network.named_modules():
                if isinstance(m, nn.Linear) and not getattr(m.weight, "_no_reinit", False):
                    if "out_proj" in name or "fc2" in name:
                        nn.init.normal_(m.weight, mean=0.0, std=initializer_range / (n_residuals ** 0.5))
                    else:
                        nn.init.normal_(m.weight, mean=0.0, std=initializer_range)

        else:
            n_residuals += self.encoder.height + self.decoder.height
            for name, m in self.encoder.named_modules():
                if isinstance(m, nn.Linear) and not getattr(m.weight, "_no_reinit", False):
                    if "out_proj" in name or "fc2" in name:
                        nn.init.normal_(m.weight, mean=0.0, std=initializer_range / (n_residuals ** 0.5))
                    else:
                        nn.init.normal_(m.weight, mean=0.0, std=initializer_range)
            for name, m in self.decoder.named_modules():
                if isinstance(m, nn.Linear) and not getattr(m.weight, "_no_reinit", False):
                    if "out_proj" in name or "fc2" in name:
                        nn.init.normal_(m.weight, mean=0.0, std=initializer_range / (n_residuals ** 0.5))
                    else:
                        nn.init.normal_(m.weight, mean=0.0, std=initializer_range)
                    
            self.main_network._init_weights(initializer_range, n_residuals)
    

    def _apply_lr_multiplier(self, lr_multiplier: list[float]) -> None:
        """ Applies the learning rate multipliers to the parameters of the model.
        """
        # a little stupid: we apply lr_multiplier to all parameters, and then for the main stage (which may have another hierarchy), we just apply it again there.
        for param in self.parameters():
            apply_optimization_params(param, lr_multiplier=lr_multiplier[self.stage_idx])
        
        if not self.is_innermost:
            self.main_network._apply_lr_multiplier(lr_multiplier)

class HNetForCausalLM(nn.Module):
    def __init__(self, config: HNetConfig, device=None, dtype=None) -> None:
        self.config = config
        super().__init__()

        vocab_size, d_embed = self.config.vocab_size, self.config.d_model[0]
        self.embeddings = nn.Embedding(vocab_size, d_embed, device=device, dtype=dtype)
        self.backbone = HNet(config=config, stage_idx=0, device=device, dtype=dtype)
        self.lm_head = nn.Linear(d_embed, vocab_size, bias=False, device=device, dtype=dtype)
        if self.config.tie_embeddings: self.lm_head.weight = self.embeddings.weight

    def allocate_inference_cache(self, batch_size, max_seqlen, dtype=None, **kwargs):
        return self.backbone.allocate_inference_cache(batch_size, max_seqlen, dtype=dtype, **kwargs)

    def forward(self, input_ids, mask=None, inference_params=None, num_last_tokens=0, **mixer_kwargs):
        """ num_last_tokens: if > 0, only return the logits for the last n tokens """
        x = self.embeddings(input_ids)
        B, L, D = x.shape
        cu_seqlens = max_seqlen = None

        if mask is None:
            # Nếu người dùng gọi hàm forward mà không cung cấp một mask (tức là mask is None), thì chúng ta sẽ mặc định rằng
            # dữ liệu đầu vào đã được chuẩn bị để chạy ở chế độ Nén (Packed Mode). Do đó, code sẽ tiến hành nối các chuỗi lại
            # (flatten) và tự tạo ra cu_seqlens để các lớp bên trong (như Mamba) có thể xử lý hiệu quả.
            assert inference_params is None, "Inference params are not supported in packed mode"
            x = x.flatten(0, 1)  # Merge batch và sequence dimensions
            cu_seqlens = torch.arange(B + 1, device=x.device) * L
            max_seqlen = torch.tensor(L, dtype=torch.int, device=x.device)

        x, bpred_output = self.backbone(x, cu_seqlens, max_seqlen, mask, inference_params, **mixer_kwargs)
        x = x.view(B, L, D)
        if num_last_tokens > 0: x = x[:, -num_last_tokens:]
        logits = self.lm_head(x)
        return (logits, bpred_output, inference_params)

    def step(self, input_ids, inference_params):
        bs = input_ids.shape[0]
        assert bs == 1, "HNetForCausalLM step currently only supports batch size 1"
        x = self.embeddings(input_ids)
        x, bpred_output = self.backbone.step(x, inference_params)
        logits = self.lm_head(x)
        return (logits, bpred_output, inference_params)
