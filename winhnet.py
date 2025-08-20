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
    def forward(self, hidden_states, boundary_mask, cu_seqlens=None):
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
        dt = torch.log(1 / (1 - p))#.to(torch.bfloat16)
        x = (hidden_states / dt[..., None])#.to(torch.bfloat16)

        A = -torch.ones((self.nheads,), device=hidden_states.device, dtype=torch.float32)
        b = p#.to(torch.bfloat16)
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
