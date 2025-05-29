
**Có ba thứ cần tìm hiểu về blackwell gamming gpus (rtx 50xx)**

1. có thể train fp8 weights được không? => fp4 kernel (ĐƯỢC!) 

2. flash_attn'3 fp8 có hỗ trợ 5090? (lý thuyết được, thực tế đang bị lỗi?)
   Nếu không thì có kernels nào nhanh hơn flash_attn không?

3. Kỹ thuật nào hiệu quả nhất (tốc độ cao + chính xác) fp4/fp8/int8/int4/mixed matmul?

- [ ] Round & smooth thuộc 3. và có thể áp dụng ngược lại cho INT8 Mixed trên 4090
- [ ] Activations đang chiếm nhiều vram nhất => nên quant (giảm 1/2)
- Tile scale sẽ đều và tốt hơn row-wise / col-wise? Nếu đã smooth thì tile còn tác dụng?
- 4090 có fp8 => fp8 weight?
- int8 weights https://github.com/pytorch/ao/pull/644 @ RTX 3090?

ROUNDING & SMOOTHING
--------------------
- QuEST
  - `Hadamard Transform` (HT) biến dist gần gaussian để smooth outlier
  - `MSE-optimal fitting` tìm tensor scaling factor tối ưu để min L2 error
  - `RMS norm` chuẩn hoá về N(0, 1) trước khi quant
