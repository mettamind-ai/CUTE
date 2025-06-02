Các kernels mạnh đang có trong tay:

- INT8 Mixed Matmul
- bf16 Flash Attention vẫn nhanh nhất trong tổng thể các trường hợp
  - một vài trường hợp sageattn thắng ở fwd

LoRA, freezed, inference thì:
- Model weights ở int8 row scale => Cần Int8 Tensor class
- Activation + LoRA weights ở bf16


| Định dạng       | Exponent (mũ) | Mantissa (phần định trị) | Phạm vi động (Dynamic Range) | Độ chính xác (Precision) | Phù hợp với vai trò trong huấn luyện |
| --------------- | ------------- | ------------------------ | ---------------------------- | ------------------------ | ------------------------------------ |
| `float8_e5m2`   | 5 bits        | 2 bits                   | Cao                          | Thấp                     | Gradient (truyền ngược)              |
| `float8_e4m3fn` | 4 bits        | 3 bits                   | Trung bình                   | Cao                      | Trọng số và kích hoạt (truyền xuôi)  |


https://huggingface.co/allenai/OLMoE-1B-7B-0125

https://stanford-cs336.github.io/spring2025-lectures/?trace=var%2Ftraces%2Flecture_10.json&step=383

|<img src="https://pbs.twimg.com/media/GsWiTOXaMAAsjhn?format=jpg">|<img src="https://pbs.twimg.com/media/GsWin8PawAA8s9A?format=jpg">|
|-|-|
|![](https://pbs.twimg.com/media/GsWjPvna4AAMwgA?format=jpg)|![](https://pbs.twimg.com/media/GsWk1s2bsAAbdGp?format)|
|![]()|![]()|

![](https://github.com/Dao-AILab/grouped-latent-attention/raw/main/assets/gta.png)
**Grouped-Tied Attention (GTA)**, which ties the key and value representations into a single shared state, leverages grouping to share the tied KV heads among a small set of query heads combined with partial RoPE; it roughly reduces KV cache size and improves the arithmetic intensity by up to a factor of two relative to its GQA counterpart with the same number of groups, while preserving quality and parallelism benefits. The following is an overview of GTA, where a single projection produces a *tied KV*. The full *tied KV* dimension is for the value. For the keys, half of the dimension is from *tied KV* (yellow) designated for no positional encoding, and the other half (red) comes from a separate single-head projection where RoPE is applied; this separate half is broadcast to all heads in the group and concatenated with the *tied KV* half. 

```py https://github.com/Dao-AILab/grouped-latent-attention/blob/main/modeling_llama_GTA.py
key_rope = self.W_rope_k(hidden_states)
key_rope = key_rope.view(bsz, q_len, 1 , self.rope_dim) 
key_rope = repeat(key_rope, 'b l 1 d -> b l h d', h=self.num_heads) 

query_rope, key_rope = apply_rotary_pos_emb(query_rope, key_rope, cos, sin,unsqueeze_dim=2)          
query_states = torch.cat([query, query_rope], dim=-1) 

kv_states_tied, value_states = \
  torch.split(kv_states, [kv_states.size(-1) - self.rope_dim, self.rope_dim], dim=-1)  

kv_states_tied = repeat_kv(kv_states_tied, self.num_key_value_groups, dim=2) 
value_states = repeat_kv(value_states, self.num_key_value_groups, dim=2) 

key_states = torch.cat([kv_states_tied, key_rope], dim=-1) 
value_states = torch.cat([kv_states_tied, value_states], dim=-1) 
```
