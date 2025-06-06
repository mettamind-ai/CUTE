## QuEST - Quantization-Aware (QAT) Stable Training of LLMs with 1-Bit Weights and Activations
- https://github.com/IST-DASLab/QuEST
- https://www.alphaxiv.org/abs/2502.05003
- https://arxiv.org/html/2502.05003v1
- https://www.youtube.com/watch?v=e3ClKTyVrkI

- `Hadamard Transform` (HT) biến dist gần gaussian để smooth outlier
- `MSE-optimal fitting` tìm α* tensor scaling factor tối ưu để min L2 error (tương thích bwd)
- `RMS norm` chuẩn hoá về N(0, 1) trước khi quant

**Trust Gradient Estimator (bwd)**:
- `grad error = ||∇wL - ∇ŵL||²`

- Phân chia weights:
  - `S_small`: Weights có quantization error ≤ T (trust threshold)
  - `S_large`: Weights có quantization error > T

- Trust mask `∂/∂x ≈ I_{|x̂-x|≤T} ⊙ ∂/∂x̂`
  - ✅ "Tin tưởng" gradient từ weights có error nhỏ
  - ❌ "Không tin" gradient từ weights có error lớn (outliers)
```
Forward:
xh = HT(x)        # Hadamard transform
wh = HT(w)        # Same for weights
x̂h = proj_α*(xh)  # MSE-optimal quantization
ŵh = proj_α*(wh)  # Same for weights
y = x̂h @ ŵhᵀ      # Matrix multiplication

Backward:
∂L/∂x = IHT(M_α*(xh; x̂h) ⊙ ∂L/∂x̂h)    # IHT là Inversed Hadamard Transform
∂L/∂w = IHT(M_α*(wh; ŵh) ⊙ ∂L/∂ŵh)    # Trust-masked gradient
```
- `M_α(xh; x̂h)`: Trust mask - binary/float mask [0,1]
- `∂L/∂x̂h`: Gradient tensor - same shape as mask
- `⊙`: Hadamard product hay Element-wise multiplication (phép nhân từng phần tử). 

|![](https://pbs.twimg.com/media/GsCE20NbwAADNj3?format=jpg)|![](https://pbs.twimg.com/media/GsCFPonasAAY7Yw?format=jpg&name=medium)|
|-|-|
|![](https://pbs.twimg.com/media/GsCFxSlbAAAfU3X?format=jpg)|![](https://pbs.twimg.com/media/GsCJYc9awAAb29B?format=jpg)|
|![](https://pbs.twimg.com/media/GsCKHQ9bMAACaMy?format=jpg)|![](https://pbs.twimg.com/media/GsCLczNbYAAFdPO?format=jpg)|
|![](https://pbs.twimg.com/media/GsCOkKhawAA9DVE?format=jpg)STE = Straight Through Estimation|![](https://pbs.twimg.com/media/GsCPbL2acAAZL1o?format=jpg)|
|![](https://pbs.twimg.com/media/GsCQLMka4AAau2D?format=png)|![](https://pbs.twimg.com/media/GsCQVkWa8AAQ306?format=jpg)|
|![](https://pbs.twimg.com/media/GsCQe3tbIAAN7Ev?format=png)|![](https://pbs.twimg.com/media/GsCQ72TbkAAWQyp?format=jpg)|
|![](https://pbs.twimg.com/media/GsCRS3bbQAA_3zm?format=jpg)|![](https://pbs.twimg.com/media/GsCWEm7WAAAtxhb?format=jpg)|

<img width="70%" src="https://arxiv.org/html/2502.05003v1/x1.png">

> **Figure 1**: The scaling law induced by QuEST when training Llama-family models from 30 to 800M parameters on C4, with quantized weights and activations from 1 to 4 bits, in the 100 tokens/parameter regime (higher compression uses proportionally more data at fixed memory). QuEST allows for stable training at 1-bit weights and activations (W1A1), and the QuEST W4A4 model is Pareto-dominant relative to BF16, with lower loss at lower size.
