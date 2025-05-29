FP4 TRAINING
------------

# MX Formats
- https://www.nvidia.com/en-us/on-demand/session/gtc25-s72778
- https://www.opencompute.org/documents/ocp-microscaling-formats-mx-v1-0-spec-final-pdf

MX formats have `multiple scaling factors per tensor`:
- Element dtype and encoding
- Scale dtype and encoding
- Scaling block size


# Training LLMs with MXFP4
- https://github.com/amazon-science/mxfp4-llm
![](https://arxiv.org/html/2502.20586v2/x1.png)
> Figure 1:Our method uses stochastic rounding (SR) to compute unbiased gradients and the random Hadamard transform to bound the variance of SR. This enables us to perform more accurate model updates with MXFP4 in the backward pass, enabling a speedup of > 1.3× over FP8 and > 1.7× over BF16.

# Quartet
- https://github.com/IST-DASLab/QuEST
- https://github.com/IST-DASLab/Quartet
- https://www.alphaxiv.org/abs/2505.14669
<table><tr><td width="40%"><img src="https://pbs.twimg.com/media/GsB8pYda8AAosOH?format=jpg"></td>
<td width="60%"><img src="https://pbs.twimg.com/media/GsB-vd6bwAAKKeK?format=jpg"></td></tr></table>

> Rất nhiều models như Llama, qwen, gemma có tỉ lệ params (N) và data (D) rơi vào vùng tối ưu của fp4

**Quartet** leverages:
- Fwd: `QuEST` (Hadamard + RMSE-based clipping) để minimize MSE
- Bwd: `Stochastic Rounding` (SR) để maintain unbiased gradient estimation
  - `final_gradient = stochastic_round(computed_gradient)`
<img width="70%" src="https://pbs.twimg.com/media/Gr4-Y4MXoAA0m8y?format=jpg">

**QuEST fwd**:
- `Hadamard Transform` (HT) biến dist gần gaussian để smooth outlier
- `MSE-optimal fitting` tìm α* tensor scaling factor tối ưu để min L2 error (tương thích bwd)
- `RMS norm` chuẩn hoá về N(0, 1) trước khi quant
```py Forward
xh = HT(x)        # Hadamard transform
wh = HT(w)        # Same for weights
x̂h = proj_α*(xh)  # MSE-optimal quantization
ŵh = proj_α*(wh)  # Same for weights
y = x̂h @ ŵhᵀ      # Matrix multiplication
```

**2 tham số của scaling laws**
- hiệu quả tham số `effN`: liên quan trực tiếp đến `lỗi nén thuận` của mỗi phương pháp huấn luyện
- hiệu quả dữ liệu `effD`: liên quan đến `độ lệch trong bộ ước lượng gradient`, đo bằng chỉ số lệch hướng
- `MXFP4` tối đa hoá cả `effN` và `effD`; đạt độ chính xác vượt trội trên các compute budgets và model sizes
- Treat FP8 as the lossless baseline for the purposes of comparison.


|![]()|![]()|
|-|-|