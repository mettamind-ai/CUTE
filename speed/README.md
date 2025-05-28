**Có 4 thứ cần tìm hiểu về blackwell gamming gpus (rtx 50xx)**
a) có thể train fp8 weights được không? => fp4 kernel (ĐƯỢC!) 
b) flash_attn'3 fp8 có hỗ trợ 5090? (lý thuyết được, thực tế đang bị lỗi?)
   Nếu không thì có kernels nào nhanh hơn flash_attn không? => Quartet FP4
c) Kỹ thuật nào hiệu quả nhất (tốc độ cao + chính xác) fp4/fp8/int8/int4/mixed matmul?

- [ ] Round & smooth thuộc c) và có thể áp dụng ngược lại cho INT8 Mixed trên 4090
- [ ] Activations đang chiếm nhiều vram nhất mà nên được quant (giảm 1/2)
- [ ] Tile scale sẽ đều và tốt hơn row-wise / col-wise?
- [ ] 4090 có fp8 => fp8 weight + INT4 kernels?
- int8 weights https://github.com/pytorch/ao/pull/644 @ RTX 3090?

ROUNDING & SMOOTHING
--------------------
...
