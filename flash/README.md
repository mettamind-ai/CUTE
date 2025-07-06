
## Có ba thứ cần tìm hiểu về gamming gpus

1. có thể train fp8 weights được không? => 
   Khó, 8/4-bit chỉ nên dùng cho mixed precision. weight / main activations vẫn nên giữ ở 16/32 bits

2. Full (fwd+bwd) attention kernels nào phù hợp?
- Có kernels nào nhanh hơn flash_attn không? => Có nhưng chỉ cho finetune
  - https://www.alphaxiv.org/overview/2505.11594 INT8 SageBwd tốt cho finetune, pretrain yếu
  - => customized flash attn để hỗ trợ masking tốt hơn và sparse vẫn là lựa chọn hàng đầu.

3. Kỹ thuật nào hiệu quả nhất (tốc độ cao + chính xác) fp4/fp8/int8/int4/mixed matmul?

**TODO**
- [x] ~~Áp dụng HT trong fwd và SR trong bwd trong INT8 Mixed~~
  int8 mm row scale hiện đã đủ tốt và nhanh, áp dụng thêm HT sẽ làm giảm tốc

- [x] Activations đang chiếm nhiều vram nhất ~~=> nên quant~~ (phức tạp hoá code)

- [x] Dùng block quant để ~~tăng độ chính xác và~~ tái sử dụng và có thể lưu activation ở 8/4 bit
  - DeepSeek quant cho fp8 https://github.com/pytorch/ao/tree/main/torchao/prototype/blockwise_fp8
    - Activations are quantized in blocks of size 128x1 using the FP8 format
    - Weights are quantized in blocks of size 128x128 using the FP8 format
    - **Tốc độ đang chậm**, có lẽ chỉ hợp để giảm comm cost khi train đa nodes

|    m |     k |     n | block_size | dtype         | fp16_latency (ms) | blockwise_latency (ms) | blockwise_speedup |
|-----:|------:|------:|-----------:|:--------------|------------------:|-----------------------:|------------------:|
|   64 |  8192 | 57344 |        128 | float8_e4m3fn |           471.44  |                245.2   |          1.92268  |
|  256 | 28672 |  8192 |        128 | float8_e4m3fn |           260.224 |                215.392 |          1.20814  |
|  512 |  8192 |  8192 |        128 | float8_e5m2   |           121.184 |                114.592 |          1.05753  |
| 1024 |  8192 | 10240 |        128 | float8_e5m2   |           289.408 |                283.424 |          1.02111  |
| 2048 |  8192 | 10240 |        128 | float8_e5m2   |           575.36  |                582.016 |          0.988564 |
| 4096 |  8192 | 10240 |        128 | float8_e5m2   |          1170.62  |               1188.45  |          0.985002 |
| 8192 |  8192 | 57344 |        128 | float8_e4m3fn |         13004     |              13990.6   |          0.929482 |


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

## SageBwd
|Activation|Storage|Computation|Lý do|
|-|-|-|-|
|Q, K, V (input)|	FP16	|INT8	|Input precision, quantize khi compute|
|S (scores)     |	FP16	|FP16	|Intermediate softmax cần precision   |
|P (attention)  |	FP16	|INT8	|Quantize cho PV multiplication       |
|dOV^T	        | FP16	|🔴 FP16	|Critical cho gradient accuracy!  |
|dS, dQ, dK	    | FP16	|INT8	|Gradients cần FP16 storage           |

|||
|-|-|
|The accuracy loss in `dS` will continuously accumulate errors into `dQ` (Q's grad) and `dK` during the recurrent process along the sequence length in FlashAttention’s backward pass, meaning longer sequences lead to greater error accumulation. Therefore, we maintain `dOV^T` in FP16 while accelerating the other four matrix multiplications using INT8 per-block quantization.|![](https://paper-assets.alphaxiv.org/figures/2505.11594/x10.png)|

## infllmv2_cuda (biến flash_attn thành sparse)
|![](https://pbs.twimg.com/media/GsxEgeOa0AAYvL3?format=png)|![](https://pbs.twimg.com/media/GsxFmaTa0AAZ70F?format=jpg)|
|-|-|
|![](https://pbs.twimg.com/media/GsxHgUnbAAAnPV3?format=jpg)|![](https://pbs.twimg.com/media/GsxInb6aQAAHXtB?format=jpg)|
|![](https://pbs.twimg.com/media/GsxTQU-aQAAySEY?format=png)|![](https://pbs.twimg.com/media/GsxUc82a4AAk4_g?format=jpg)|
|![](https://pbs.twimg.com/media/GsxWtlDa0AA35ER?format=png)|![](https://pbs.twimg.com/media/GsxZV1pbwAE_yBT?format=jpg)|
