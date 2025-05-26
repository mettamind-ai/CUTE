# OhMai - Biến UnSloth thành Sloth
Điểm mạnh của Unsloth là 1 nhân [fast lora](https://github.com/unslothai/unsloth/blob/main/unsloth/kernels/fast_lora.py) giúp LoRA finetune nhanh và tiết kiệm vram hơn bình thường. Sau đó bổ xung thêm vài fused kernels và một vài mẹo tăng tốc và chữa lỗi cho các models mới ra.

Điểm yếu của Unsloth là chưa hỗ trợ sample packing một kỹ thuật tối quan trọng trong pretrain và finetune. Chưa hỗ trợ những kỹ thuật mới mẻ như INT8 hay Muon ... Về bản chất Unsloth vẫn lười ...

Nếu biến nhân fast lora xài được INT8, sử dụng Muon optimizer, hỗ trợ sample packing và các kỹ thuật chưa từng có được tối ưu cho gamming GPUs như tự động giảm kích thước embeddings và lm_head (thường chiếm 1 phần rất lớn weights của model 1-3b) để giúp finetuning hiệu quả hơn? Và trở thành một finetuning framework linh hoạt (a.k.a FlexTune) để với mỗi một model module's weight có thể tuỳ chọn freeze (ko tune), full finetune hoặc LoRA ... và tự động search xem cấu hình nào là phù hợp nhất với finetuning computing buget?

- [SageAttention3 int8, fp8](https://arxiv.org/abs/2505.11594) fwd/bwd can be a good choice for LoRA.
  Thử nghiệm [sage.py](sage.py) cho thấy ở 4k ctx sage's fwd x1.5 flash_attn, 8k và 16k thì x2 flash_attn
  Với LoRA thì chỉ cần tính input gradient nên code sẽ tinh giản hơn ...

- [LigerKernel Embedding](../liger_kernel.py) can be a good start to build flexible embedding and lm_head based on up-comming batch(es) data. Có thể học cách họ tối ưu IO trong GPU ... Giả sử mỗi batch cần 1s để train thì chỉ cần look ahead 10 batches là đủ ...

- [ ] Trước mắt chỉ cần áp dụng phép nhân ma trận 8bit vào [lora.py](lora.py) là cũng đã speedup rất nhiều ...
