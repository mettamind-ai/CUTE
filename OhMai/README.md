# OhMai - Biến UnSloth thành Sloth
Điểm mạnh của Unsloth là 1 nhân [fast lora](https://github.com/unslothai/unsloth/blob/main/unsloth/kernels/fast_lora.py) giúp LoRA finetune nhanh và tiết kiệm vram hơn bình thường. Sau đó bổ xung thêm vài fused kernels và một vài mẹo tăng tốc và chữa lỗi cho các models mới ra.

Điểm yếu của Unsloth là chưa hỗ trợ sample packing một kỹ thuật tối quan trọng trong pretrain và finetune. Chưa hỗ trợ những kỹ thuật mới mẻ như INT8 hay Muon ... Về bản chất Unsloth vẫn lười ...

Nếu biến nhân fast lora xài được INT8, sử dụng Muon optimizer, hỗ trợ sample packing và các kỹ thuật chưa từng có được tối ưu cho gamming GPUs như tự động giảm kích thước embeddings và lm_head (thường chiếm 1 phần rất lớn weights của model 1-3b) để giúp finetuning hiệu quả hơn? Và trở thành một finetuning framework linh hoạt (a.k.a FlexTune) để với mỗi một model module's weight có thể tuỳ chọn freeze (ko tune), full finetune hoặc LoRA ... và tự động search xem cấu hình nào là phù hợp nhất với finetuning computing buget?

---

- [x] [SageAttention3 int8, fp8](https://arxiv.org/abs/2505.11594) fwd/bwd can be a good choice for LoRA.
  Thử nghiệm [sage.py](/speed/sage.py) cho thấy ở 4k ctx sage's fwd x1.5 flash_attn, 8k và 16k thì x2 flash_attn
  Với LoRA thì chỉ cần tính input gradient nên code sẽ tinh giản hơn ...

- [x] OhMaiEmbedding cần giảm tối thiểu IO
  - [x] chỉ update những tokens ko có trong batch và 
  - [x] chỉ load những tokens không có sẵn trong vram

- [x] OhMaiHead
  - [x] Fused linear + chunked cross entropy loss
  - [ ] Sử dụng freq based sample softmax
  ```py playground/683d5a3ef829ed36a76977b1
# L2 norm từ pretrained
pretrained_norm = model.lm_head.weight.norm(dim=1)  # ← đây là L2

# Kết hợp với dataset freq
combined_score = alpha * dataset_freq + (1-alpha) * pretrained_norm
'''
- Nếu finetune dataset khác biệt nhiều với pretrained data thì chọn α cao (0.7-0.8) để ưu tiên frequency từ dataset. 
- Nếu dataset tương tự hoặc chỉ là mở rộng thì chọn α thấp (0.3-0.5) để giữ lại kiến thức pretrained. 
- α = 0.5 là điểm xuất phát an toàn để thử nghiệm.
- α = 0.65 để Finetune thêm tiếng Việt cho Qwen vì Qwen đã support tiếng Việt nhưng chưa mạnh
  Cần boost Vietnamese tokens, Nhưng vẫn giữ multilingual capability.
'''
def sample_negative_tokens(batch_tokens, sample_probs, k=24000):
    # Exclude batch tokens từ sampling
    mask = torch.ones_like(sample_probs)
    mask[batch_tokens] = 0
    masked_probs = sample_probs * mask
    masked_probs = masked_probs / masked_probs.sum()
    # Sample k negative tokens
    neg_tokens = torch.multinomial(masked_probs, k, replacement=False)
    return neg_tokens

# => 40960 head size,  ~8k active tokens, 1:4 positive/negative sampling
# => 51200 head size, ~10k active tokens, 1:5 positive/negative sampling
  ```

- [ ] Trước mắt chỉ cần áp dụng phép nhân ma trận 8bit vào [lora.py](lora.py) là cũng đã save vram và speedup kha khá ...

- [ ] Áp dụng NAS trong việc tìm 1 cấu hình finetune linh hoạt / phù hợp với computing budget ...
  - https://github.com/IST-DASLab/DarwinLM Evolutionary Structured Pruning for Language 
  
- [ ] Tìm hiểu các kỹ thuật PEFT khác nhau
  - DORA của Nvidia (có trong torchtune)
  - ROSA https://github.com/IST-DASLab/RoSA
