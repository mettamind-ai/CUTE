# Flash Attn in Triton
- [`./save/attn.py`](https://github.com/bryanzhang/triton_fusedattention/blob/main/fused-attention.py)
- https://www.youtube.com/watch?v=zEuwuCTEf_0
- https://www.youtube.com/watch?v=4jQTb6sRGLg
- https://www.youtube.com/watch?v=zy8ChVd_oTM

# mô phỏng long short layers
- Train 2k ctxlen trước, sau đó freeze 2/3 layers rồi train tiếp với 4k ctxlen

---

- optim scheduler: scaling laws for wd & bs in llm training
  https://x.com/dmsobol/status/1925273068840390801
  https://x.com/dmsobol/status/1895179989664047442
  - `wd = 0.1` is suboptimal, should scales linearly with bs
  - `EMA` (Exponential Moving Average) 

- command+a https://alphaxiv.org/abs/2504.00698
  - n x { `3 swa` (RoPE) + `1 full attn` (NoPE) }
  - NoPE giúp tổng quát hoá tốt hơn
  - fp8 then bf16 to stable training

- gemma3 & https://ai.google.dev/gemma/docs/gemma-3n
  
- weighted loss https://x.com/kalomaze/status/1880923963880300941
