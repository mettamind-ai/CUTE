''' https://huggingface.co/deepseek-ai/DeepSeek-V3/blob/main/config.json
{   https://huggingface.co/deepseek-ai/DeepSeek-V3/blob/main/README_WEIGHTS.md
  "model_type": "deepseek_v3",
  "architectures": [ "DeepseekV3ForCausalLM" ],
  "hidden_act": "silu",
  "hidden_size": 7168,
  "intermediate_size": 18432,
  "first_k_dense_replace": 3,
  "initializer_range": 0.02,

  "ep_size": 1,
  "moe_intermediate_size": 2048,
  "moe_layer_freq": 1,
  "n_group": 8,
  "n_routed_experts": 256,
  "n_shared_experts": 1,
  "norm_topk_prob": true,
  "num_experts_per_tok": 8,
  "routed_scaling_factor": 2.5,
  "scoring_func": "sigmoid",
  "topk_group": 4,
  "topk_method": "noaux_tc",

  "vocab_size": 129280
  "num_hidden_layers": 61,
  "max_position_embeddings": 163840,
  "num_nextn_predict_layers": 1,    // number of Multi-Token Prediction (MTP) Modules

  "kv_lora_rank": 512,
  "q_lora_rank": 1536,
  "num_attention_heads": 128,
  "num_key_value_heads": 128,
  "qk_nope_head_dim": 128,
  "qk_rope_head_dim": 64,
  "v_head_dim": 128,

  "quantization_config": {
    "activation_scheme": "dynamic",
    "fmt": "e4m3",
    "quant_method": "fp8",
    "weight_block_size": [128, 128]
  },
  "rms_norm_eps": 1e-06,
  "rope_scaling": {
    "beta_fast": 32, "beta_slow": 1, "factor": 40,
    "mscale": 1.0, "mscale_all_dim": 1.0,
    "original_max_position_embeddings": 4096,
    "type": "yarn"
  },
  "rope_theta": 10000,
  "tie_word_embeddings": false,
  "torch_dtype": "bfloat16",
}
'''

