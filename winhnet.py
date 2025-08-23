import math, torch, torch.nn.functional as F
from torch import nn
from torch.utils.checkpoint import checkpoint
from einops import rearrange
from wingpt import norm, init_linear, GptBlock, get_cu_max_seqlens_from
from flash.ops.layernorm_gated import RMSNorm as RMSNormGated
from flash.mamba.ssd_combined import mamba_split_conv1d_scan_combined

class Mamba2Block(nn.Module):
    def __init__(self, dim=512, d_state=128, d_conv=4, headdim=64):
        super().__init__()
        self.dim = dim
        self.d_state = d_state
        self.d_conv  = d_conv
        self.headdim = headdim
        
        self.d_inner = 2*dim # expand=2
        assert self.d_inner % self.headdim == 0
        self.nheads = self.d_inner // self.headdim
        
        # Order: [z, x, B, C, dt]
        self.d_in_proj = 2*self.d_inner + 2*self.d_state + self.nheads
        self.fuse_proj = nn.Linear(dim, 8*dim, bias=False)
        self.out_proj  = nn.Linear(self.d_inner, dim, bias=False)
        self.down_proj = nn.Linear(4*dim, dim)

        conv_dim = self.d_inner + 2*self.d_state
        self.conv1d = nn.Conv1d(
            in_channels=conv_dim, out_channels=conv_dim,
            bias=True, kernel_size=d_conv, groups=conv_dim, padding=d_conv - 1,
        )
        self.norm = RMSNormGated(self.d_ssm, eps=1e-5, norm_before_gate=False, group_size=self.d_ssm // ngroups)

        # dt bias initialization
        dt_min, dt_max = 0.001, 0.1
        dt = torch.exp(torch.rand(self.nheads) * (math.log(dt_max) - math.log(dt_min)) + math.log(dt_min))
        inv_dt = dt + torch.log(-torch.expm1(-dt))
        self.dt_bias = nn.Parameter(inv_dt)
        
        # A parameter (negative exponential)
        A = torch.empty(self.nheads).uniform_(1, 16)
        self.A_log = nn.Parameter(torch.log(A))
        # D skip parameter
        self.D = nn.Parameter(torch.ones(self.nheads))        

    def forward(self, u, cu_seqlens):
        """
        u: (total_tokens, dim) - packed varlen sequences
        cu_seqlens: (batch_size + 1,) - cumulative sequence lengths
        """
        total_tokens, dim = u.shape
        fuse = self.fuse_proj(u)

        def prepare():
            up  = fuse[..., : 4*self.dim]
            act = F.relu(up).square()

            zxbcdt = fuse[..., -self.d_in_proj : ]  # (total_tokens, d_in_proj)
            A = -torch.exp(self.A_log.float())      # (nheads) or (d_inner, d_state)

            out = mamba_split_conv1d_scan_combined(
                zxbcdt,
                rearrange(self.conv1d.weight, "d 1 w -> d w"), self.conv1d.bias,
                self.dt_bias,
                A,
                D=self.D, chunk_size=self.chunk_size,
                seq_idx=None, activation="silu",
                rmsnorm_weight=None, rmsnorm_eps=None, norm_before_gate=False,
                outproj_weight=self.out_proj.weight, outproj_bias=self.out_proj.bias,
                headdim=self.headdim, ngroups=self.ngroups,
            )
            u_out = u + out
            return u_out, act
        
        u_out, act = checkpoint(prepare, use_reentrant=False)
        ffn = self.down_proj(act)
        return u_out + ffn


class Routing(nn.Module):
    def __init__(self, dim, device=None, dtype=None):
        super().__init__()

        self.dim = dim
        factory_kwargs = {"device": device, "dtype": dtype}

        self.q_proj_layer = nn.Linear(dim, dim, bias=False, **factory_kwargs)
        self.k_proj_layer = nn.Linear(dim, dim, bias=False, **factory_kwargs)

        with torch.no_grad():
            self.q_proj_layer.weight.copy_(torch.eye(dim))
            self.k_proj_layer.weight.copy_(torch.eye(dim))
            self.q_proj_layer.weight._no_reinit = True
            self.k_proj_layer.weight._no_reinit = True


    def forward(self, hidden_states, cu_seqlens, inference_params=None):
        # We are in packed mode, so hidden_states is (T, D).
        q = F.normalize(self.q_proj_layer(hidden_states[:,  :-1]), dim=-1)  # shape [t, d]
        k = F.normalize(self.k_proj_layer(hidden_states[:, 1:  ]), dim=-1)  # shape [t, d]

        cos_sim = (q * k).sum(dim=-1)      # shape [b, l], value in [-1, 1]
        boundary_prob = (1 - cos_sim) / 2  # convert [-1, 1] to [0, 1]
        boundary_prob = torch.clamp(boundary_prob, min=0.0, max=1.0)

        # Force boundary probability of the first element to 1.0
        PAD_PROB = 1.0
        boundary_prob = F.pad(boundary_prob, (1, 0), "constant", PAD_PROB)
        boundary_prob[cu_seqlens[:-1]] = PAD_PROB

        _probs = torch.stack(((1 - boundary_prob), boundary_prob), dim=-1)
        selected_idx  = torch.argmax(_probs, dim=-1)
        boundary_mask = selected_idx == 1   # (shape hidden_states.shape[:-1])

        selected_probs = boundary_prob.gather(dim=-1, index=selected_idx.unsqueeze(-1))
                                            # (shape hidden_states.shape[:-1], 1)
        return  boundary_prob,              # (shape hidden_states.shape[:-1], 2)
                boundary_mask,              # (shape hidden_states.shape[:-1])
                selected_probs              # (shape hidden_states.shape[:-1], 1)


class Chunk(nn.Module):
    def forward(self, hidden_states, boundary_mask, cu_seqlens):
        next_hidden_states = hidden_states[boundary_mask]
        next_cu_seqlens = boundary_mask.cumsum(dim=0)[cu_seqlens[1:] - 1]
        next_cu_seqlens = F.pad(next_cu_seqlens, (1, 0)) # thêm 0 vào đầu
        next_max_seqlen = int((next_cu_seqlens[1:] - next_cu_seqlens[:-1]).max())
        return next_hidden_states, next_cu_seqlens, next_max_seqlen


def get_seq_idx(cu_seqlens, device=None):
    seq_idx = torch.zeros(cu_seqlens[-1], dtype=torch.long, device=device)
    seq_idx[cu_seqlens[:-1]] = 1
    seq_idx = (torch.cumsum(seq_idx, dim=0) - 1).unsqueeze(0).int()
    return seq_idx

class DeChunk(nn.Module):
    def __init__(self, dim, block_size=256, headdim=32):
        super().__init__()
        self.dim = dim
        # For Mamba2 kernell
        self.block_size = block_size
        self.headdim = headdim
        assert dim % self.headdim == 0
        self.nheads = dim // self.headdim

    def forward(self, hidden_states, boundary_mask, boundary_prob, cu_seqlens):
        p = torch.clamp(boundary_prob[..., -1].float(), min=1e-4, max=1-(1e-4))
        p = p[boundary_mask]
        seq_idx = get_seq_idx(cu_seqlens, device=hidden_states.device)

        # Reuse Mamba2 kernel for EMA Deaggregator.
        dt = torch.log(1 / (1 - p)).to(torch.bfloat16)
        x = (hidden_states / dt[..., None]).to(torch.bfloat16)

        A = -torch.ones((self.nheads,), device=hidden_states.device, dtype=torch.float32)
        b = p.to(torch.bfloat16)
        c = torch.ones_like(b)

        out = mamba_chunk_scan_combined(
            rearrange(x, "t (h d) -> t h d", d=self.headdim),
            repeat(dt, "t -> t h", h=self.nheads),
            A,
            rearrange(b, "t -> t 1 1"),
            rearrange(c, "t -> t 1 1"),
            chunk_size=self.block_size,
            seq_idx=seq_idx,
        )
        out = rearrange(out, "t h d -> t (h d)")
        plug_back_idx = boundary_mask.cumsum(dim=0) - 1
        index = plug_back_idx.unsqueeze(-1).expand(-1, self.dim)
        out = torch.gather(out, dim=0, index=index)
        return out.to(hidden_states.dtype)
