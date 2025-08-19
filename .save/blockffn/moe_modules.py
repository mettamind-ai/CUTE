import torch
import bmtrain as bmt
from typing import Optional
import torch.nn.functional as F
from .activation_function import get_activation_fn
from .activation_context import ActivationContext


class MoELinearActiveGate(bmt.DistributedModule):
    def __init__(self, dim_model, num_expert, activate_fn: str = "relu", activate_kwargs: dict = {},
                 dtype=torch.bfloat16, init_mean=0.0, init_std=0.02):
        super().__init__()
        self.w_gate = bmt.DistributedParameter(
            torch.empty((num_expert, dim_model), dtype=dtype),
            init_method=bmt.ParameterInitializer(torch.nn.init.normal_, mean=init_mean, std=init_std),
        )
        self.act = get_activation_fn(activate_fn)

    def forward(self, x):
        """
        Args: x (:obj:`torch.Tensor` of shape ``(batch * seq_len, dim_in)``)
        Return: gating_scores (:obj`torch.Tensor` of shape ``(batch * seq_len, expert_num)``)
        """
        raw_score = F.linear(x, self.w_gate)
        return raw_score, self.act(raw_score)


class MoEUpDownExperts(bmt.DistributedModule):
    def __init__(self, dim_model: int, dim_expert: int, num_expert: int, expert_gated: bool,
                 activate_fn: str = "silu", activate_kwargs: dict = {}, use_linear: bool = False,
                 dropout_p: Optional[float] = None, dtype: torch.dtype = torch.bfloat16,
                 init_mean: float = 0.0, init_std: float = 0.02):
        super().__init__()

        # MoE DenseGatedACT
        if use_linear:
            if expert_gated:
                self.moe_w_gate = bmt.DistributedParameter(
                    torch.empty((num_expert * dim_expert, dim_model), dtype=dtype),
                    init_method=bmt.ParameterInitializer(torch.nn.init.normal_, mean=init_mean, std=init_std),
                )
            self.moe_w_in = bmt.DistributedParameter(
                torch.empty((num_expert * dim_expert, dim_model), dtype=dtype),
                init_method=bmt.ParameterInitializer(torch.nn.init.normal_, mean=init_mean, std=init_std),
            )
            self.moe_w_out = bmt.DistributedParameter(
                torch.empty((dim_model, num_expert * dim_expert), dtype=dtype),
                init_method=bmt.ParameterInitializer(torch.nn.init.normal_, mean=init_mean, std=init_std),
            )
        else:
            if expert_gated:
                self.moe_w_gate = bmt.DistributedParameter(
                    torch.empty((num_expert, dim_expert, dim_model), dtype=dtype),
                    init_method=bmt.ParameterInitializer(torch.nn.init.normal_, mean=init_mean, std=init_std),
                )
            self.moe_w_in = bmt.DistributedParameter(
                torch.empty((num_expert, dim_expert, dim_model), dtype=dtype),
                init_method=bmt.ParameterInitializer(torch.nn.init.normal_, mean=init_mean, std=init_std),
            )
            self.moe_w_out = bmt.DistributedParameter(
                torch.empty((num_expert, dim_model, dim_expert), dtype=dtype),
                init_method=bmt.ParameterInitializer(torch.nn.init.normal_, mean=init_mean, std=init_std),
            )
        self.act = get_activation_fn(activate_fn)
        if dropout_p is not None:
            self.dropout = torch.nn.Dropout(dropout_p)
        else:
            self.dropout = None
        self.num_expert, self.dim_expert, self.dim_model, self.use_linear, self.expert_gated = num_expert, dim_expert, dim_model, use_linear, expert_gated

        if not self.use_linear:
            raise NotImplementedError("block_linear if more recommended for efficiency")

    def forward(self, x, router_score, router_up_proj=None):
        seq_len = router_score.shape[0]
        if self.expert_gated:
            if self.use_linear:
                x_score = F.linear(x, self.moe_w_gate)
                x_in = F.linear(x, self.moe_w_in)
            else:
                x_score = torch.matmul(x, self.moe_w_gate.transpose(1, 2))
                x_in = torch.matmul(x, self.moe_w_in.transpose(1, 2))
            x_in = self.act(x_score) * x_in
        else:
            if self.use_linear:
                x_in = F.linear(x, self.moe_w_in)
            else:
                x_in = torch.matmul(x, self.moe_w_in.transpose(1, 2))
            x_in = self.act(x_in)
        if self.dropout is not None:
            x_in = self.dropout(x_in)
        ActivationContext.stat_moe_intermediate_activation(x_in)
        if self.use_linear:
            if router_up_proj is not None:
                assert router_up_proj.shape[0] == seq_len
                scored_x_in = x_in.view(seq_len, self.num_expert, self.dim_expert) * router_up_proj
            else:
                scored_x_in = x_in.view(seq_len, self.num_expert, self.dim_expert) * router_score.unsqueeze(-1)
            x_out = F.linear(scored_x_in.view(seq_len, self.num_expert * self.dim_expert), self.moe_w_out)
        else:
            if router_up_proj is not None:
                assert router_up_proj.shape[0] == seq_len
                scored_x_in = x_in * router_up_proj.transpose(0, 1)
            else:
                scored_x_in = x_in * router_score.T.unsqueeze(-1)
            # [num_expert-e, seq_len-s, dim_expert-d] @ [num_expert-e, dim_model-m, dim_expert-d]
            x_out = torch.einsum("esd,emd->sm", scored_x_in, self.moe_w_out)
        return x_out