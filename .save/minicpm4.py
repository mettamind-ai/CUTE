# https://huggingface.co/openbmb/MiniCPM4-0.5B/blob/main/modeling_minicpm.py

class MiniCPMRMSNorm(nn.Module):
    def __init__(self, hidden_size, eps=1e-6):
        """
        MiniCPMRMSNorm is equivalent to T5LayerNorm
        """
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.variance_epsilon = eps

    def forward(self, hidden_states):
        return rms_layernorm(hidden_states, self.weight, self.variance_epsilon)


ALL_LAYERNORM_LAYERS.append(MiniCPMRMSNorm)


class MiniCPMRotaryEmbedding(nn.Module):
    def __init__(self, dim, max_position_embeddings=2048, base=10000, device=None):
        super().__init__()

        self.dim = dim
        self.max_position_embeddings = max_position_embeddings
        self.base = base
        inv_freq = 1.0 / (self.base ** (torch.arange(0, self.dim, 2).float().to(device) / self.dim))
        self.register_buffer('inv_freq', inv_freq, persistent=False)

        # Build here to make `torch.jit.trace` work.
        self._set_cos_sin_cache(
            # seq_len=max_position_embeddings, device=self.inv_freq.device, dtype=torch.get_default_dtype()
            seq_len=max_position_embeddings, device=self.inv_freq.device, dtype=torch.float32
        )

    def _set_cos_sin_cache(self, seq_len, device, dtype):
        self.max_seq_len_cached = seq_len
        t = torch.arange(self.max_seq_len_cached, device=device, dtype=self.inv_freq.dtype)
        freqs = torch.outer(t, self.inv_freq)
        # Different from paper, but it uses a different permutation in order to obtain the same calculation
        emb = torch.cat((freqs, freqs), dim=-1)

        self.register_buffer('cos_cached', emb.cos().to(dtype), persistent=False)
        self.register_buffer('sin_cached', emb.sin().to(dtype), persistent=False)

    def forward(self, x, seq_len=None):
        # x: [bs, num_attention_heads, seq_len, head_size]
        if seq_len > self.max_seq_len_cached:
            self._set_cos_sin_cache(seq_len=seq_len, device=x.device, dtype=x.dtype)

        return (
            self.cos_cached[:seq_len].to(dtype=x.dtype),
            self.sin_cached[:seq_len].to(dtype=x.dtype),
        )


class MiniCPMLongRoPE(MiniCPMRotaryEmbedding):
    """MiniCPMRotaryEmbedding extended with Dynamic NTK scaling. Credits to the Reddit users /u/bloc97 and /u/emozilla"""

    def __init__(self, dim, max_position_embeddings=2048, base=10000, device=None, short_factor=None, long_factor=None, original_max_position_embeddings=None):
        self.short_factor = short_factor
        self.long_factor = long_factor
        self.original_max_position_embeddings = original_max_position_embeddings
        scale = (max_position_embeddings / self.original_max_position_embeddings)
        self.scaling_factor = math.sqrt(1 + math.log(scale) / math.log(self.original_max_position_embeddings))
        super().__init__(dim, max_position_embeddings, base, device)

    def _set_cos_sin_cache(self, seq_len, device, dtype):
        self.max_seq_len_cached = seq_len
        t = torch.arange(self.max_seq_len_cached, device=device, dtype=self.inv_freq.dtype)
        if seq_len > self.original_max_position_embeddings:
            ext_factors = torch.tensor(self.long_factor, dtype=torch.float32, device=device)
        else:
            ext_factors = torch.tensor(self.short_factor, dtype=torch.float32, device=device)

        freqs = torch.mul(
            torch.outer(t, 1.0 / ext_factors).to(device=device),
            self.inv_freq.to(device=device).to(dtype)
        )
        # Different from paper, but it uses a different permutation in order to obtain the same calculation
        emb = torch.cat((freqs, freqs), dim=-1)
        self.register_buffer('cos_cached', emb.cos().to(dtype) * self.scaling_factor, persistent=False)
        self.register_buffer('sin_cached', emb.sin().to(dtype) * self.scaling_factor, persistent=False)


class MiniCPMLinearScalingRotaryEmbedding(MiniCPMRotaryEmbedding):
    """MiniCPMRotaryEmbedding extended with linear scaling. Credits to the Reddit user /u/kaiokendev"""

    def __init__(self, dim, max_position_embeddings=2048, base=10000, device=None, scaling_factor=1.0):
        self.scaling_factor = scaling_factor
        super().__init__(dim, max_position_embeddings, base, device)

    def _set_cos_sin_cache(self, seq_len, device, dtype):
        self.max_seq_len_cached = seq_len
        t = torch.arange(self.max_seq_len_cached, device=device, dtype=self.inv_freq.dtype)
        t = t / self.scaling_factor

        freqs = torch.outer(t, self.inv_freq)
        # Different from paper, but it uses a different permutation in order to obtain the same calculation
        emb = torch.cat((freqs, freqs), dim=-1)
        self.register_buffer('cos_cached', emb.cos().to(dtype), persistent=False)
        self.register_buffer('sin_cached', emb.sin().to(dtype), persistent=False)


class MiniCPMDynamicNTKScalingRotaryEmbedding(MiniCPMRotaryEmbedding):
    """MiniCPMRotaryEmbedding extended with Dynamic NTK scaling. Credits to the Reddit users /u/bloc97 and /u/emozilla"""

    def __init__(self, dim, max_position_embeddings=2048, base=10000, device=None, scaling_factor=1.0):
        self.scaling_factor = scaling_factor
        super().__init__(dim, max_position_embeddings, base, device)

    def _set_cos_sin_cache(self, seq_len, device, dtype):
        self.max_seq_len_cached = seq_len

        if seq_len > self.max_position_embeddings:
            base = self.base * (
                (self.scaling_factor * seq_len / self.max_position_embeddings) - (self.scaling_factor - 1)
            ) ** (self.dim / (self.dim - 2))
            inv_freq = 1.0 / (base ** (torch.arange(0, self.dim, 2).float().to(device) / self.dim))
            self.register_buffer('inv_freq', inv_freq, persistent=False)

        t = torch.arange(self.max_seq_len_cached, device=device, dtype=self.inv_freq.dtype)

        freqs = torch.outer(t, self.inv_freq)
        # Different from paper, but it uses a different permutation in order to obtain the same calculation
        emb = torch.cat((freqs, freqs), dim=-1)

        self.register_buffer('cos_cached', emb.cos().to(dtype), persistent=False)
        self.register_buffer('sin_cached', emb.sin().to(dtype), persistent=False)


def rotate_half(x):
    """Rotates half the hidden dims of the input."""
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2:]
    return torch.cat((-x2, x1), dim=-1)


    def _init_rope(self):
        if self.config.rope_scaling is None:
            self.rotary_emb = MiniCPMRotaryEmbedding(
                self.head_dim,
                max_position_embeddings=self.max_position_embeddings,
                base=self.rope_theta,
            )
        else:
            scaling_type = self.config.rope_scaling['rope_type']
            scaling_factor = self.config.rope_scaling.get('factor', None)
            if scaling_type == 'linear':
                self.rotary_emb = MiniCPMLinearScalingRotaryEmbedding(
                    self.head_dim,
                    max_position_embeddings=self.max_position_embeddings,
                    scaling_factor=scaling_factor,
                    base=self.rope_theta,
                )
            elif scaling_type == 'dynamic':
                self.rotary_emb = MiniCPMDynamicNTKScalingRotaryEmbedding(
                    self.head_dim,
                    max_position_embeddings=self.max_position_embeddings,
                    scaling_factor=scaling_factor,
                    base=self.rope_theta,
                )
            elif scaling_type == 'longrope':
                self.rotary_emb = MiniCPMLongRoPE(
                    self.head_dim,
                    max_position_embeddings=self.max_position_embeddings,
                    short_factor=self.config.rope_scaling['short_factor'],
                    long_factor=self.config.rope_scaling['long_factor'],
                    base=self.rope_theta,
                    original_max_position_embeddings=self.config.rope_scaling['original_max_position_embeddings']
                )
            else:
                raise ValueError(f'Unknown RoPE scaling type {scaling_type}')

    def _shape(self, tensor: torch.Tensor, seq_len: int, bsz: int):
        return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()


