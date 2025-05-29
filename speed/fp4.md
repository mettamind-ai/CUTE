QUARTET FP4 TRAINING
--------------------
- https://github.com/IST-DASLab/QuEST
- https://www.alphaxiv.org/abs/2502.05003
- https://github.com/IST-DASLab/Quartet
- https://www.alphaxiv.org/abs/2505.14669

<table><tr><td width="40%"><img src="https://pbs.twimg.com/media/GsB8pYda8AAosOH?format=jpg"></td>
<td width="60%"><img src="https://pbs.twimg.com/media/GsB-vd6bwAAKKeK?format=jpg"></td></tr></table>

> Rất nhiều models như Llama, qwen, gemma có tỉ lệ params (N) và data (D) rơi vào vùng tối ưu của fp4

Quartet leverages:

- `QuEST-based quantization-aware` training (`QAT` or `Mixed` or `MXFP4`) for **minimal forward-pass error**
- Unbiased `stochastic rounding` for **stable backward-pass propagation** 

<img width="70%" src="https://pbs.twimg.com/media/Gr4-Y4MXoAA0m8y?format=jpg">

## QuEST: "Minimize the difference between true gradient and quantized gradient"
- `Hadamard Transform` (HT) biến dist gần gaussian để smooth outlier
- `MSE-optimal fitting` tìm α* tensor scaling factor tối ưu để min L2 error
- `RMS norm` chuẩn hoá về N(0, 1) trước khi quant

## Quartet

|![]()|![]()|
|-|-|
|![]()|![]()|
|![]()|![]()|
|![]()|![]()|

**2 tham số của scaling laws**
- hiệu quả tham số `N^eff`: liên quan trực tiếp đến `lỗi nén thuận` của mỗi phương pháp huấn luyện
- hiệu quả dữ liệu `D^eff`: liên quan đến `độ lệch trong bộ ước lượng gradient`, đo bằng chỉ số lệch hướng
- Quartet (MXFP4) tối đa hoá cả N^eff và D^eff; đạt độ chính xác vượt trội trên 
  nhiều ngân sách tính toán và kích thước mô hình
- We will treat FP8 as the lossless baseline for the purposes of comparison.
