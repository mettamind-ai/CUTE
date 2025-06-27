class Gemma3nTextScaledWordEmbedding(nn.Embedding):
    """
    This module overrides nn.Embeddings' forward by multiplying with embeddings scale.
    """

    def __init__(self, num_embeddings: int, embedding_dim: int, padding_idx: int, embed_scale: float = 1.0):
        super().__init__(num_embeddings, embedding_dim, padding_idx)
        self.register_buffer("embed_scale", torch.tensor(embed_scale), persistent=False)

    def forward(self, input_ids: torch.Tensor):
        return super().forward(input_ids) * self.embed_scale.to(self.weight.dtype)


class Gemma3nTextLaurelBlock(nn.Module):
    """Learned Augmented Residual Layer"""

    def __init__(self, config: Gemma3nTextConfig):
        super().__init__()
        self.config = config

        self.linear_left  = nn.Linear(self.config.hidden_size, self.config.laurel_rank, bias=False)
        self.linear_right = nn.Linear(self.config.laurel_rank, self.config.hidden_size, bias=False)
        self.post_laurel_norm = Gemma3nRMSNorm(self.config.hidden_size, eps=self.config.rms_norm_eps)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        laurel_hidden_states = self.linear_left (hidden_states)
        laurel_hidden_states = self.linear_right(laurel_hidden_states)
        normed_laurel_hidden_states = self.post_laurel_norm(laurel_hidden_states)
        return hidden_states + normed_laurel_hidden_states


class Gemma3nTextMLP(nn.Module):
    def __init__(self, config: Gemma3nTextConfig, layer_idx: int = 0):
        super().__init__()
        self.config = config
        self.hidden_size = config.hidden_size
        self.intermediate_size = config.intermediate_size[layer_idx]
        self.gate_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
        self.up_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
        self.down_proj = nn.Linear(self.intermediate_size, self.hidden_size, bias=False)
        self.act_fn = ACT2FN[config.hidden_activation]
        self.activation_sparsity = config.activation_sparsity_pattern[layer_idx]

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        gate_proj = self.gate_proj(hidden_states)
        if self.activation_sparsity > 0.0:
            gate_proj = self._gaussian_topk(gate_proj)
        activations = self.act_fn(gate_proj)
        up_proj = self.up_proj(hidden_states)
        down_proj = self.down_proj(activations * up_proj)
        return down_proj

    def _gaussian_topk(self, inputs: torch.Tensor) -> torch.Tensor:
        target_sparsity_tensor = torch.tensor(self.activation_sparsity, dtype=torch.float32, device=inputs.device)
        # normal_dist and std_multiplier are adapted from jax.scipy.stats.norm.ppf().
        #
        # References:
        #   *   https://docs.jax.dev/en/latest/_autosummary/jax.scipy.stats.norm.ppf.html
        #   *   https://pytorch.org/docs/stable/distributions.html#torch.distributions.normal.Normal
        #   *   https://pytorch.org/docs/stable/distributions.html#torch.distributions.transformed_distribution.TransformedDistribution.icdf
        normal_dist = torch.distributions.normal.Normal(0, 1)
        std_multiplier: torch.Tensor = normal_dist.icdf(target_sparsity_tensor)
        std_multiplier = std_multiplier.type(inputs.dtype)
        inputs_mean = torch.mean(inputs, dim=-1, keepdim=True)
        inputs_std = torch.std(inputs, dim=-1, keepdim=True, unbiased=False)
        cutoff_x = inputs_mean + inputs_std * std_multiplier
        return nn.functional.relu(inputs - cutoff_x)


class Gemma3nTextAltUp(nn.Module):
    """Alternating Updates (AltUp)

    The AltUp module wraps transformer layers. The `predict` step modifies the
    input to the transformer layer, and the `correct` step propagates the output
    of the transformer layer to the sparsely updated dimensions.

    See more in the research paper:

    https://proceedings.neurips.cc/paper_files/paper/2023/file/f2059277ac6ce66e7e5543001afa8bb5-Paper-Conference.pdf
    """

    def __init__(self, config: Gemma3nTextConfig):
        super().__init__()
        self.config = config
        self.correct_output_scale = nn.Parameter(torch.zeros(self.config.hidden_size))
        self.correction_coefs = nn.Linear(self.config.altup_num_inputs, self.config.altup_num_inputs, bias=False)
        self.prediction_coefs = nn.Linear(self.config.altup_num_inputs, self.config.altup_num_inputs**2, bias=False)
        self.modality_router = nn.Linear(self.config.hidden_size, self.config.altup_num_inputs, bias=False)
        self.router_norm = Gemma3nRMSNorm(self.config.hidden_size, eps=self.config.rms_norm_eps)
        self.register_buffer("router_input_scale", torch.tensor(self.config.hidden_size**-1.0), persistent=False)

    def compute_router_modalities(self, x: torch.Tensor) -> torch.Tensor:
        router_inputs = self.router_norm(x) * self.router_input_scale
        routed = self.modality_router(router_inputs)
        return torch.tanh(routed.float()).type_as(x)

    def predict(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """Predicts the output of a layer using a trainable map.

        Args:
            hidden_states: A 4D tensor of shape `[num_altup_inputs, batch_size, num_tokens, hidden_size]` derived by
                stacking the input embeddings and preprocessing the last `num_altup_inputs - 1` matrices.

        Returns:
            A 4D tensor of shape `[num_altup_inputs, batch_size, num_tokens, hidden_size]` containing the predictions.
        """
        modalities = self.compute_router_modalities(hidden_states[self.config.altup_active_idx])

        if self.training and self.config.altup_coef_clip is not None:
            self.prediction_coefs.weight.data.clamp_(-self.config.altup_coef_clip, self.config.altup_coef_clip)

        # Project and then transpose all 2D matrices contained so that mulmat gives the correct result
        all_coefs: torch.Tensor = (
            self.prediction_coefs(modalities)
            .reshape(*modalities.shape[:-1], self.config.altup_num_inputs, self.config.altup_num_inputs)
            .permute(0, 1, 3, 2)
        )

        # permute hidden_states to [batch_size, num_tokens, hidden_size, altup_num_inputs]
        predictions = torch.matmul(hidden_states.permute(1, 2, 3, 0), all_coefs)
        predictions = predictions.permute(3, 0, 1, 2)  # undo the permute
        predictions += hidden_states  # add the original input
        return predictions.contiguous().type_as(hidden_states)

    def correct(self, predictions: torch.Tensor, activated: torch.Tensor) -> torch.Tensor:
        """Corrects the predictions relative to the

        Args:
            predictions: A 4D tensor of shape `[num_altup_inputs, batch_size, num_tokens, hidden_size]` derived by
                stacking the input embeddings and preprocessing the last `num_altup_inputs - 1` matrices.
            activated: A 3D tensor of shape `[batch_size, num_tokens, hidden_size]` containing the activated inputs.

        Returns:
            A 4D tensor of shape `[num_altup_inputs, batch_size, num_tokens, hidden_size]` correcting the original
                predictions relative to the activated input embeddings.
        """
        modalities = self.compute_router_modalities(activated)
        innovation = activated - predictions[self.config.altup_active_idx]  # (batch, num_tokens, hidden_size)
        innovation = innovation.repeat(self.config.altup_num_inputs, 1, 1, 1)  # Repeat on dim0 to match predictions

        if self.config.altup_coef_clip is not None:
            self.correction_coefs.weight.data.clamp_(-self.config.altup_coef_clip, self.config.altup_coef_clip)

        # all_coefs adapted from jax.numpy.einsum("...p,pi->...i", ...)
        # Permute to (altup_num_inputs, batch_size, num_tokens) as the last dim is a scalar applied to each altup input
        # and expand on dim1 for broadcastability
        all_coefs: torch.Tensor = self.correction_coefs(modalities) + 1.0
        all_coefs = all_coefs.permute(2, 0, 1).unsqueeze(-1)

        corrected = torch.mul(innovation, all_coefs)
        corrected += predictions  # add the original input
        return corrected.contiguous().type_as(activated)

    def scale_corrected_output(self, corrected: torch.Tensor) -> torch.Tensor:
        """Scales the provided 3D tensor of shape [batch_size, num_tokens, hidden_size]."""
        return (corrected.type_as(self.correct_output_scale) * self.correct_output_scale).type_as(corrected)


class Gemma3nTextRotaryEmbedding(nn.Module):
    def __init__(self, config: Gemma3nTextConfig, device=None):
        super().__init__()
        # BC: "rope_type" was originally "type"
        if hasattr(config, "rope_scaling") and config.rope_scaling is not None:
            self.rope_type = config.rope_scaling.get("rope_type", config.rope_scaling.get("type"))
        else:
            self.rope_type = "default"
        self.max_seq_len_cached = config.max_position_embeddings
        self.original_max_seq_len = config.max_position_embeddings

        self.config = config
        self.rope_init_fn = ROPE_INIT_FUNCTIONS[self.rope_type]

        inv_freq, self.attention_scaling = self.rope_init_fn(self.config, device)
        self.register_buffer("inv_freq", inv_freq, persistent=False)
        self.original_inv_freq = self.inv_freq

    @torch.no_grad()
    @dynamic_rope_update  # power user: used with advanced RoPE types (e.g. dynamic rope)
    def forward(self, x, position_ids):
        inv_freq_expanded = self.inv_freq[None, :, None].float().expand(position_ids.shape[0], -1, 1).to(x.device)
        position_ids_expanded = position_ids[:, None, :].float()

        device_type = x.device.type if isinstance(x.device.type, str) and x.device.type != "mps" else "cpu"
        with torch.autocast(device_type=device_type, enabled=False):  # Force float32
            freqs = (inv_freq_expanded.float() @ position_ids_expanded.float()).transpose(1, 2)
            emb = torch.cat((freqs, freqs), dim=-1)
            cos = emb.cos() * self.attention_scaling
            sin = emb.sin() * self.attention_scaling

        return cos.to(dtype=x.dtype), sin.to(dtype=x.dtype)


def rotate_half(x):
    """Rotates half the hidden dims of the input."""
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


def repeat_kv(hidden_states: torch.Tensor, n_rep: int) -> torch.Tensor:
    """
    This is the equivalent of torch.repeat_interleave(x, dim=1, repeats=n_rep). The hidden states go from (batch,
    num_key_value_heads, seqlen, head_dim) to (batch, num_attention_heads, seqlen, head_dim)
    """
    batch, num_key_value_heads, slen, head_dim = hidden_states.shape
    if n_rep == 1:
        return hidden_states
    hidden_states = hidden_states[:, :, None, :, :].expand(batch, num_key_value_heads, n_rep, slen, head_dim)
    return hidden_states.reshape(batch, num_key_value_heads * n_rep, slen, head_dim)


class Gemma3nTextAttention(nn.Module):
    """Multi-headed attention from 'Attention Is All You Need' paper"""

    def __init__(self, config: Gemma3nTextConfig, layer_idx: int):
        super().__init__()
        self.is_sliding = config.layer_types[layer_idx] == "sliding_attention"
        self.config = config
        self.layer_idx = layer_idx
        self.head_dim = getattr(config, "head_dim", config.hidden_size // config.num_attention_heads)
        self.num_key_value_groups = config.num_attention_heads // config.num_key_value_heads
        self.attention_dropout = self.config.attention_dropout
        self.is_causal = True

        self.q_proj = nn.Linear(config.hidden_size, config.num_attention_heads * self.head_dim, bias=config.attention_bias)
        self.k_proj = nn.Linear(config.hidden_size, config.num_key_value_heads * self.head_dim, bias=config.attention_bias)
        self.v_proj = nn.Linear(config.hidden_size, config.num_key_value_heads * self.head_dim, bias=config.attention_bias)
        self.o_proj = nn.Linear(config.num_attention_heads * self.head_dim, config.hidden_size, bias=config.attention_bias)

        self.sliding_window = config.sliding_window if self.is_sliding else None

        self.q_norm = Gemma3nRMSNorm(dim=config.head_dim, eps=config.rms_norm_eps)
        self.k_norm = Gemma3nRMSNorm(dim=config.head_dim, eps=config.rms_norm_eps)
        self.v_norm = Gemma3nRMSNorm(dim=config.head_dim, eps=config.rms_norm_eps, with_scale=False)

        first_kv_shared_layer_idx = self.config.num_hidden_layers - self.config.num_kv_shared_layers
        self.is_kv_shared_layer   = layer_idx >= first_kv_shared_layer_idx
        # Find the index of the last sliding or full layer before sharing starts (or None if no sharing)
        layer_type = config.layer_types[layer_idx]
        self.kv_shared_layer_index = (
            first_kv_shared_layer_idx - 1 - config.layer_types[first_kv_shared_layer_idx - 1 :: -1].index(layer_type)
            if self.is_kv_shared_layer
            else None
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        position_embeddings: torch.Tensor,
        attention_mask: Optional[torch.Tensor],
        past_key_value: Optional[Cache] = None,
        cache_position: Optional[torch.LongTensor] = None,
        **kwargs: Unpack[FlashAttentionKwargs],
    ) -> tuple[torch.Tensor, Optional[torch.Tensor], Optional[tuple[torch.Tensor]]]:
        input_shape = hidden_states.shape[:-1]
        hidden_shape = (*input_shape, -1, self.config.head_dim)

        cos, sin = position_embeddings

        query_states = self.q_proj(hidden_states).view(hidden_shape)
        query_states = self.q_norm(query_states)
        query_states = apply_rotary_pos_emb(query_states, cos, sin, unsqueeze_dim=2)
        query_states = query_states.transpose(1, 2)

        if self.is_kv_shared_layer and self.kv_shared_layer_index is not None and past_key_value is not None:
            # HybridCache has complex slicing when layer_type == "sliding_attention" that impact Shared KV Cache.
            if isinstance(past_key_value, HybridCache) and self.is_sliding:
                max_length = past_key_value.sliding_window
                if cache_position.shape[0] > max_length:
                    # If in the prefill phase for a "sliding_attention" layer and the prefill is larger than the cache,
                    # slice into the entire cache.
                    indices = slice(0, max_length)
                else:
                    # If prefill fits or generating for a "sliding_attention" layer, clamp to max_cache_len - 1
                    indices = cache_position.clamp(min=0, max=max_length - 1)
            else:
                indices = cache_position

            key_states = past_key_value.key_cache[self.kv_shared_layer_index][:, :, indices]
            value_states = past_key_value.value_cache[self.kv_shared_layer_index][:, :, indices]
        else:
            key_states = self.k_proj(hidden_states).view(hidden_shape)
            key_states = self.k_norm(key_states)
            key_states = apply_rotary_pos_emb(key_states, cos, sin, unsqueeze_dim=2)
            key_states = key_states.transpose(1, 2)

            value_states = self.v_proj(hidden_states).view(hidden_shape)
            value_states = self.v_norm(value_states)
            value_states = value_states.transpose(1, 2)

        if past_key_value is not None:
            # sin and cos are specific to RoPE models; cache_position needed for the static cache
            cache_kwargs = {
                "sin": sin,
                "cos": cos,
                "cache_position": cache_position,
                "sliding_window": self.sliding_window,
            }
            key_states, value_states = past_key_value.update(key_states, value_states, self.layer_idx, cache_kwargs)

        attention_interface: Callable = eager_attention_forward
        if self.config._attn_implementation != "eager":
            attention_interface = ALL_ATTENTION_FUNCTIONS[self.config._attn_implementation]

        attn_output, attn_weights = attention_interface(
            self,
            query_states,
            key_states,
            value_states,
            attention_mask,
            dropout=self.attention_dropout if self.training else 0.0,
            scaling=1.0,
            sliding_window=self.sliding_window,
            **kwargs,
        )

        attn_output = attn_output.reshape(*input_shape, -1).contiguous()
        attn_output = self.o_proj(attn_output)
        return attn_output, attn_weights


class Gemma3nTextDecoderLayer(GradientCheckpointingLayer):
    def __init__(self, config: Gemma3nTextConfig, layer_idx: int):
        super().__init__()
        self.config = config
        self.hidden_size = config.hidden_size
        self.layer_idx = layer_idx
        self.attention_type = config.layer_types[layer_idx]
        self.self_attn = Gemma3nTextAttention(config, layer_idx)
        self.mlp = Gemma3nTextMLP(config, layer_idx=layer_idx)
        self.input_layernorm = Gemma3nRMSNorm(self.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = Gemma3nRMSNorm(self.hidden_size, eps=config.rms_norm_eps)
        self.pre_feedforward_layernorm = Gemma3nRMSNorm(self.hidden_size, eps=config.rms_norm_eps)
        self.post_feedforward_layernorm = Gemma3nRMSNorm(self.hidden_size, eps=config.rms_norm_eps)

        self.hidden_size_per_layer_input = config.hidden_size_per_layer_input
        self.act_fn = ACT2FN[config.hidden_activation]

        self.altup = Gemma3nTextAltUp(config)
        self.laurel = Gemma3nTextLaurelBlock(config)
        self.per_layer_input_gate = nn.Linear(self.hidden_size, self.hidden_size_per_layer_input, bias=False)
        self.per_layer_projection = nn.Linear(self.hidden_size_per_layer_input, self.hidden_size, bias=False)
        self.post_per_layer_input_norm = Gemma3nRMSNorm(self.hidden_size, eps=config.rms_norm_eps)

    @deprecate_kwarg("last_cache_position", version="4.53.0")
    def forward(
        self,
        hidden_states: torch.Tensor,
        position_embeddings_global: torch.Tensor,
        position_embeddings_local: torch.Tensor,
        per_layer_input: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Cache] = None,
        output_attentions: Optional[bool] = False,
        use_cache: Optional[bool] = False,
        cache_position: Optional[torch.LongTensor] = None,
        **kwargs, ) -> tuple[torch.Tensor, Optional[tuple[torch.FloatTensor, torch.FloatTensor]]]:

        predictions = self.altup.predict(hidden_states)
        active_prediction = predictions[self.config.altup_active_idx]

        active_prediction_normed = self.input_layernorm(active_prediction)
        laurel_output = self.laurel(active_prediction_normed)

        # apply global RoPE to non-sliding layer only
        position_embeddings = position_embeddings_local if self.self_attn.is_sliding \
                         else position_embeddings_global

        attn, self_attn_weights = self.self_attn(
            hidden_states=active_prediction_normed, position_embeddings=position_embeddings,
            attention_mask=attention_mask, position_ids=position_ids,
            past_key_value=past_key_value, output_attentions=output_attentions,
            use_cache=use_cache, cache_position=cache_position, **kwargs,
        )
        attn = self.post_attention_layernorm(attn)

        attn_gated = active_prediction + attn
        attn_laurel = (attn_gated + laurel_output) / math.sqrt(2)

        attn_norm = self.pre_feedforward_layernorm(attn_laurel)
        attn_ffw  = self.mlp(attn_norm)
        attn_ffw_norm = self.post_feedforward_layernorm(attn_ffw)
        attn_ffw_laurel_gated = attn_laurel + attn_ffw_norm
        corrected_predictions = self.altup.correct(predictions, attn_ffw_laurel_gated)

        first_prediction = corrected_predictions[self.config.altup_active_idx]
        first_prediction_clone = first_prediction.clone()
        if self.config.altup_correct_scale:
            first_prediction = self.altup.scale_corrected_output(first_prediction_clone)

        # per_layer_input_gate adapted from jax.numpy.einsum("btd,dp->btp", ...)
        first_prediction = self.per_layer_input_gate(first_prediction)
        first_prediction = self.act_fn(first_prediction)
        first_prediction = torch.multiply(first_prediction, per_layer_input)

        # per_layer_projection adapted from jax.numpy.einsum("btp,pd->btd", ...)
        first_prediction = self.per_layer_projection(first_prediction)
        first_prediction = self.post_per_layer_input_norm(first_prediction)
        corrected_predictions[1:] += first_prediction

        return outputs + (self_attn_weights,) else outputs


@auto_docstring(custom_intro="The base Gemma 3n language model without a language modeling head.")
class Gemma3nTextModel(Gemma3nPreTrainedModel):
    config_class = Gemma3nTextConfig

    def __init__(self, config: Gemma3nTextConfig):
        super().__init__(config)
        self.padding_idx = config.pad_token_id
        self.vocab_size  = config.vocab_size

        # Gemma3n downcasts the below to bfloat16, causing sqrt(3072)=55.4256 to become 55.5. See https://github.com/huggingface/transformers/pull/29402
        self.embed_tokens = Gemma3nTextScaledWordEmbedding(
            config.vocab_size, config.hidden_size, self.padding_idx, embed_scale=self.config.hidden_size**0.5
        )
        self.layers = nn.ModuleList(
            [Gemma3nTextDecoderLayer(config, layer_idx) for layer_idx in range(config.num_hidden_layers)]
        )

        self.norm = Gemma3nRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.rotary_emb = Gemma3nTextRotaryEmbedding(config=config)
        self.gradient_checkpointing = False

        # TODO (raushan): Fix this after RoPE refactor. For now we hack it by
        # reassigning thetas when we want to create a local RoPE layer. Config
        # defaults should hold values for global RoPE.
        config = copy.deepcopy(config)
        config.rope_theta = config.rope_local_base_freq
        config.rope_scaling = {"rope_type": "default"}
        self.rotary_emb_local = Gemma3nTextRotaryEmbedding(config=config)

        self.hidden_size = config.hidden_size
        self.hidden_size_per_layer_input = config.hidden_size_per_layer_input

        self.embed_tokens_per_layer = Gemma3nTextScaledWordEmbedding(
            config.vocab_size_per_layer_input,
            config.num_hidden_layers * config.hidden_size_per_layer_input,
            self.padding_idx,
            embed_scale=config.hidden_size_per_layer_input**0.5,
        )

        self.per_layer_model_projection = nn.Linear(
            self.hidden_size,
            config.num_hidden_layers * config.hidden_size_per_layer_input,
            bias=False,
        )

        self.per_layer_projection_norm = Gemma3nRMSNorm(config.hidden_size_per_layer_input, eps=config.rms_norm_eps)

        self.altup_projections = nn.ModuleList(
            [nn.Linear(self.hidden_size, self.hidden_size, bias=False) for _ in range(1, self.config.altup_num_inputs)]
        )

        self.altup_unembed_projections = nn.ModuleList(
            [nn.Linear(self.hidden_size, self.hidden_size, bias=False) for _ in range(1, self.config.altup_num_inputs)]
        )

        self.register_buffer("per_layer_projection_scale", torch.tensor(self.hidden_size**-0.5), persistent=False)
        self.register_buffer("per_layer_input_scale", torch.rsqrt(torch.tensor(2.0)), persistent=False)

        # Initialize weights and apply final processing
        self.post_init()

    def get_input_embeddings(self):
        return self.embed_tokens

    def set_input_embeddings(self, value):
        self.embed_tokens = value

    def get_per_layer_inputs(self, input_ids: torch.LongTensor) -> torch.Tensor:
        return self.embed_tokens_per_layer(input_ids).
            reshape(*input_ids.shape, self.config.num_hidden_layers, self.hidden_size_per_layer_input)

    def project_per_layer_inputs(self, inputs_embeds: torch.Tensor, per_layer_inputs: Optional[torch.Tensor] = None,):
        per_layer_projection  = self.per_layer_model_projection(inputs_embeds)
        per_layer_projection *= self.per_layer_projection_scale.type(inputs_embeds.dtype)
        per_layer_projection  = per_layer_projection.reshape(
            *inputs_embeds.shape[:-1],
            self.config.num_hidden_layers,
            self.hidden_size_per_layer_input,
        )
        per_layer_projection = self.per_layer_projection_norm(per_layer_projection)

        if per_layer_inputs is None: return per_layer_projection

        if per_layer_projection.shape != per_layer_inputs.shape:
            # per-layer inputs are sometimes padded with zeros, slice the relevant embeddings.
            per_layer_inputs = per_layer_inputs[..., : self.config.num_hidden_layers, :]

        return (per_layer_projection + per_layer_inputs) * self.per_layer_input_scale.type(inputs_embeds.dtype)


    @can_return_tuple
    @auto_docstring
    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        per_layer_inputs: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Cache] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        cache_position: Optional[torch.LongTensor] = None,
        **flash_attn_kwargs: Unpack[FlashAttentionKwargs],
    ) -> BaseModelOutputWithPast:
        r"""
        per_layer_inputs (torch.Tensor, *optional*, defaults to None):
            Pre-computed per-layer embeddings. If None, they are derived from input_ids if provided.
        """
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        use_cache = use_cache if use_cache is not None else self.config.use_cache

        if (input_ids is None) ^ (inputs_embeds is not None):
            raise ValueError("You must specify exactly one of input_ids or inputs_embeds")

        if self.gradient_checkpointing and self.training and use_cache:
            logger.warning_once(
                "`use_cache=True` is incompatible with gradient checkpointing. Setting `use_cache=False`."
            )
            use_cache = False

        if input_ids is not None:
            inputs_embeds = self.embed_tokens(input_ids)
            per_layer_inputs = self.get_per_layer_inputs(input_ids)

        per_layer_inputs = self.project_per_layer_inputs(inputs_embeds, per_layer_inputs)

        if use_cache and past_key_values is None and not self.training:
            past_key_values = DynamicCache()

        if cache_position is None:
            past_seen_tokens = past_key_values.get_seq_length() if past_key_values is not None else 0
            cache_position = torch.arange(
                past_seen_tokens,
                past_seen_tokens + inputs_embeds.shape[1],
                device=inputs_embeds.device,
            )

        if position_ids is None:
            position_ids = cache_position.unsqueeze(0)

        # It may already have been prepared by e.g. `generate`
        if not isinstance(causal_mask_mapping := attention_mask, dict):
            # Prepare mask arguments
            mask_kwargs = {
                "config": self.config,
                "input_embeds": inputs_embeds,
                "attention_mask": attention_mask,
                "cache_position": cache_position,
                "past_key_values": past_key_values,
            }
            # Create the masks
            causal_mask_mapping = {
                "full_attention": create_causal_mask(**mask_kwargs),
                "sliding_attention": create_sliding_window_causal_mask(**mask_kwargs),
            }

        # embed positions
        hidden_states_0 = inputs_embeds

        # Initialize RoPE embeddings
        position_embeddings_global = self.rotary_emb(hidden_states_0, position_ids)
        position_embeddings_local = self.rotary_emb_local(hidden_states_0, position_ids)

        # Expand hidden_states to support per-layer inputs
        target_magnitude: torch.Tensor = torch.mean(hidden_states_0**2, dim=-1, keepdim=True) ** 0.5
        epsilon_tensor = torch.tensor(torch.finfo().min)

        temp_hidden_states = [hidden_states_0]
        for i in range(1, self.config.altup_num_inputs):
            # altup_proj adapted from jax.numpy.einsum("btp,pd->btd", ...)
            altup_proj: torch.Tensor = self.altup_projections[i - 1](hidden_states_0)
            current_hidden_state = altup_proj.type(hidden_states_0.dtype)
            new_magnitude = torch.mean(current_hidden_state**2, dim=-1, keepdim=True) ** 0.5
            current_hidden_state = current_hidden_state * (
                target_magnitude / torch.maximum(new_magnitude, epsilon_tensor)
            )
            temp_hidden_states.append(current_hidden_state)

        hidden_states = torch.stack(temp_hidden_states, dim=0)  # [num_altup_inputs, batch, seq_len, hidden_size]

        # decoder layers
        all_hidden_states = () if output_hidden_states else None
        all_self_attns = () if output_attentions else None

        for decoder_layer in self.layers[: self.config.num_hidden_layers]:
            if output_hidden_states:
                all_hidden_states += (hidden_states,)

            causal_mask = causal_mask_mapping[decoder_layer.attention_type]
            per_layer_input = per_layer_inputs[:, :, decoder_layer.layer_idx, :]

            layer_outputs = decoder_layer(
                hidden_states,
                position_embeddings_global=position_embeddings_global,
                position_embeddings_local=position_embeddings_local,
                per_layer_input=per_layer_input,
                attention_mask=causal_mask,
                position_ids=position_ids,
                past_key_value=past_key_values,
                output_attentions=output_attentions,
                use_cache=use_cache,
                cache_position=cache_position,
                **flash_attn_kwargs,
            )

            hidden_states = layer_outputs[0]

            if output_attentions:
                all_self_attns += (layer_outputs[1],)

        # add hidden states from the last decoder layer (but before reprojecting to stay consistent with layer output)
        if output_hidden_states:
            all_hidden_states += (hidden_states,)

        # Per-layer inputs to single output
        target_magnitude = torch.mean(hidden_states[0] ** 2, dim=-1, keepdim=True) ** 0.5
        temp_hidden_states = [hidden_states[0]]
        for i in range(1, self.config.altup_num_inputs):
            # altup_unembed_projections adapted from jax.numpy.einsum("btp,pd->btd", ...)
            altup_unemb_proj: torch.Tensor = self.altup_unembed_projections[i - 1](hidden_states[i])
            current_hidden_state = altup_unemb_proj.type(hidden_states_0.dtype)
            new_magnitude = torch.mean(current_hidden_state**2, dim=-1, keepdim=True) ** 0.5
            current_hidden_state = current_hidden_state * (
                target_magnitude / torch.maximum(new_magnitude, epsilon_tensor)
            )
            temp_hidden_states.append(current_hidden_state)

        hidden_states = torch.stack(temp_hidden_states)
        hidden_states = torch.mean(hidden_states, dim=0)
        hidden_states = self.norm(hidden_states)

        return BaseModelOutputWithPast(
            last_hidden_state=hidden_states,
            past_key_values=past_key_values,
            hidden_states=all_hidden_states,
            attentions=all_self_attns,
        )
