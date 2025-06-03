- [x] [SageAttention3 int8, fp8](https://arxiv.org/abs/2505.11594) fwd/bwd can be a good choice for LoRA.
  Thử nghiệm [sage.py](/speed/sage.py) cho thấy ở 4k ctx sage's fwd x1.5 flash_attn, 8k và 16k thì x2 flash_attn
  Với LoRA thì chỉ cần tính input gradient nên code sẽ tinh giản hơn ...

- [ ] Trước mắt chỉ cần áp dụng phép nhân ma trận 8bit vào [lora.py](lora.py) là cũng đã save vram và speedup kha khá ...

- [ ] https://github.com/mobiusml/gemlite tìm hiểu quant matmul kernel, có cái nào dùng được cho finetune?

- [ ] save/quant params + inference
  - https://github.com/pytorch-labs/gpt-fast
  - https://pytorch.org/blog/accelerating-generative-ai-2
  - https://github.com/turboderp-org/exllamav3/blob/master/doc/exl3.md

- Áp dụng NAS trong việc tìm 1 cấu hình finetune linh hoạt / phù hợp với computing budget ...
  - https://github.com/IST-DASLab/DarwinLM Evolutionary Structured Pruning for Language 
  - kết hợp freeze / full / lora ...
