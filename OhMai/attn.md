# Flash Infer (serving)
- https://github.com/flashinfer-ai/flashinfer
- https://flashinfer.ai/2024/12/16/flashinfer-v02-release.html#jit-compilation-for-attention-customization

# Flash Attn (in Triton)
- [`./save/attn.py`](https://github.com/bryanzhang/triton_fusedattention/blob/main/fused-attention.py)
- https://www.youtube.com/watch?v=zEuwuCTEf_0
- https://www.youtube.com/watch?v=4jQTb6sRGLg
- https://www.youtube.com/watch?v=zy8ChVd_oTM

Tốc độ các bản triton thua flash_attn'2 với 16 bit, hoặc flash_attn'3 với 8 bit (có thể gặp compile error với 5090)
Trừ khi ép xuống 4 bit https://github.com/IST-DASLab/Quartet

## INT8 SageBwd
https://www.alphaxiv.org/abs/2505.11594
Trong forward, SageBwd áp dụng `per-block quantization` cho Q, K, V và `per-token quantization` cho P. Đối với backward pass, phương pháp này quantize 4 trong 5 phép nhân ma trận xuống INT8, nhưng `giữ phép toán dOV^T ở FP16` để duy trì độ chính xác. Lý do là gradient của attention map rất nhạy cảm với quantization errors và có thể tích lũy lỗi qua các sequence dài. SageBwd nhanh hơn FlashAttention `1.67x` @ RTX4090, với `end-to-end speedup khoảng 1.15` lần cho Llama models => `Bwd` có vấn đề !!! SageBwd có tốc độ hội tụ chậm hơn BF16 trong pretraining vì **quantization errors tích lũy** ảnh hưởng đến chất lượng gradient.
