QUARTET FP4 TRAINING
--------------------
- https://github.com/IST-DASLab/Quartet
- https://arxiv.org/html/2505.14669v1
- https://www.alphaxiv.org/abs/2505.14669
- https://x.com/DAlistarh/status/1927046856179081281
- https://x.com/DAlistarh/status/1927046864219550073

<table><tr><td width="40%"><img src="https://pbs.twimg.com/media/GsB8pYda8AAosOH?format=jpg"></td>
<td width="60%"><img src="https://pbs.twimg.com/media/GsB-vd6bwAAKKeK?format=jpg"></td></tr></table>

> Rất nhiều models như Llama, qwen, gemma có tỉ lệ params (N) và data (D) rơi vào vùng tối ưu của fp4

Quartet leverages:

- `QuEST-based quantization-aware` training (QAT) for **minimal forward-pass error**

- Unbiased `stochastic rounding` for **stable backward-pass propagation** 

<img width="70%" src="https://pbs.twimg.com/media/Gr4-Y4MXoAA0m8y?format=jpg">

## QuEST - Quantization-Aware (QAT) Stable Training of LLMs with 1-Bit Weights and Activations
<img width="70%" src="https://arxiv.org/html/2502.05003v1/x1.png">

> **Figure 1**: The scaling law induced by QuEST when training Llama-family models from 30 to 800M parameters on C4, with quantized weights and activations from 1 to 4 bits, in the 100 tokens/parameter regime (higher compression uses proportionally more data at fixed memory). QuEST allows for stable training at 1-bit weights and activations (W1A1), and the QuEST W4A4 model is Pareto-dominant relative to BF16, with lower loss at lower size.

- https://github.com/IST-DASLab/QuEST
- https://www.alphaxiv.org/abs/2502.05003
- https://arxiv.org/html/2502.05003v1
- https://www.youtube.com/watch?v=e3ClKTyVrkI

|![](https://pbs.twimg.com/media/GsCE20NbwAADNj3?format=jpg)|![](https://pbs.twimg.com/media/GsCFPonasAAY7Yw?format=jpg&name=medium)|
|-|-|
|![](https://pbs.twimg.com/media/GsCFxSlbAAAfU3X?format=jpg)|![](https://pbs.twimg.com/media/GsCJYc9awAAb29B?format=jpg)|
|![](https://pbs.twimg.com/media/GsCKHQ9bMAACaMy?format=jpg)|![](https://pbs.twimg.com/media/GsCLczNbYAAFdPO?format=jpg)|
|![](https://pbs.twimg.com/media/GsCOkKhawAA9DVE?format=jpg)STE = Straight Through Estimation|![](https://pbs.twimg.com/media/GsCPbL2acAAZL1o?format=jpg)|
|![](https://pbs.twimg.com/media/GsCQLMka4AAau2D?format=png)|![](https://pbs.twimg.com/media/GsCQVkWa8AAQ306?format=jpg)|
|![](https://pbs.twimg.com/media/GsCQe3tbIAAN7Ev?format=png)|![](https://pbs.twimg.com/media/GsCQ72TbkAAWQyp?format=jpg)|
|![]()|![]()|
|![]()|![]()|
|![]()|![]()|
|![]()|![]()|
|![]()|![]()|
