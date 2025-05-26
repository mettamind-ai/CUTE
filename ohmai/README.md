# OhMai - Biến UnSloth thành Sloth
Điểm mạnh của Unsloth là 1 nhân [fast lora](https://github.com/unslothai/unsloth/blob/main/unsloth/kernels/fast_lora.py) giúp LoRA finetune nhanh và tiết kiệm vram hơn bình thường. Sau đó bổ xung thêm vài fused kernels và một vài mẹo tăng tốc và chữa lỗi cho các models mới ra.

Điểm yếu của Unsloth là chưa hỗ trợ sample packing một kỹ thuật tối quan trọng trong pretrain và finetune. Chưa hỗ trợ những kỹ thuật mới mẻ như INT8 hay Muon ... Về bản chất Unsloth vẫn lười ...

Nếu biến nhân fast lora xài được INT8, sử dụng Muon optimizer, hỗ trợ sample packing và các kỹ thuật chưa từng có được tối ưu cho gamming GPUs như tự động giảm kích thước embeddings và lm_head (thường chiếm 1 phần rất lớn weights của model 1-3b) để giúp finetuning hiệu quả hơn? Và trở thành một finetuning framework linh hoạt (a.k.a FlexTune) để với mỗi một model module's weight có thể tuỳ chọn freeze (ko tune), full finetune hoặc LoRA ... và tự động search xem cấu hình nào là phù hợp nhất với finetuning computing buget?

Học hỏi kernels cho LoRA từ Axolotl để build INT8 Mixed LoRA

```
wget https://raw.githubusercontent.com/axolotl-ai-cloud/axolotl/refs/heads/main/src/axolotl/kernels/swiglu.py
wget https://raw.githubusercontent.com/axolotl-ai-cloud/axolotl/refs/heads/main/src/axolotl/kernels/lora.py
```