
**Có ba thứ cần tìm hiểu về blackwell gamming gpus (rtx 50xx)**

1. có thể train fp8 weights được không? => fp4 kernel (ĐƯỢC!) 

2. flash_attn'3 fp8 có hỗ trợ 5090? (Cài từ source đc?)
  - FA3 hỗ trợ FP16 / BF16 fwd & bwd, FP8 fwd
  - Có kernels nào nhanh hơn flash_attn không?
    - https://www.alphaxiv.org/overview/2505.11594 INT8 SageBwd tốt cho finetune, pretrain yếu

3. Kỹ thuật nào hiệu quả nhất (tốc độ cao + chính xác) fp4/fp8/int8/int4/mixed matmul?

**TODOs**
- [ ] Round & smooth thuộc 3. và có thể áp dụng ngược lại cho INT8 Mixed
  - Áp dụng HT trong fwd và SR trong bwd (xem ROUNDING & SMOOTHING)

- [ ] Activations đang chiếm nhiều vram nhất => nên quant (giảm 1/2)

- Tile scale or row-wise / col-wise or tensor scale sẽ phụ thuộc vào round & smooth methods

ROUNDING & SMOOTHING
--------------------
|Method | effN (fwd) | effD (bwd) | Misalignment |
|-------|------------|------------|--------------|
|QuEST  |  0.65 ✅  |   0.18 ❌  |  1.3×10⁻²    |
|SR     |  0.44 ❌  |   0.85 ✅  |  0           |

- QuEST Fwd
  - `Hadamard Transform` (HT) biến dist gần gaussian để smooth outlier
  - `MSE-optimal fitting` tìm tensor scaling factor tối ưu để min L2 error
  - `RMS norm` chuẩn hoá về `N(0, 1)` trước khi quant

- Quartet Bwd 
  - `final_gradient = stochastic_round(computed_gradient)`

=> Giống https://alphaxiv.org/overview/2502.20586#key-innovation-mxfp4-with-random-hadamard-transform-and-stochastic-rounding

## SageBwd
<img src="https://paper-assets.alphaxiv.org/figures/2505.11594/x10.png" width="60%">

|Activation|Storage|Computation|Lý do|
|-|-|-|-|
|Q, K, V (input)|	FP16	|INT8	|Input precision, quantize khi compute|
|S (scores)     |	FP16	|FP16	|Intermediate softmax cần precision   |
|P (attention)  |	FP16	|INT8	|Quantize cho PV multiplication       |
|dOV^T	        | FP16	|🔴 FP16	|Critical cho gradient accuracy!  |
|dS, dQ, dK	    | FP16	|INT8	|Gradients cần FP16 storage           |

The accuracy loss in `dS` will continuously accumulate errors into `dQ` (Q's grad) and `dK` during the recurrent process along the sequence length in FlashAttention’s backward pass, meaning longer sequences lead to greater error accumulation. Therefore, we maintain `dOV^T` in FP16 while accelerating the other four matrix multiplications using INT8 per-block quantization.

|![alt text](.save/sagebwd-00.png)|![](.save/sagebwd-01.png)|
|-|-|

- Áp dụng cả per token scaling (PV) và block scaling (V)
- 