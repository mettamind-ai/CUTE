**Có 3 thứ cần tìm hiểu về  blackwell gamming gpus (rtx 50xx)**
a) có thể train fp8 weights được không?
b) có làm attention kernels nhanh hơn flash_attn'3 fp8 không?
c) kỹ thuật nào hiệu quả nhất (tốc độ cao + chính xác) Linear matmul?
d) flash_attn đã quá tối ưu và tinh vi (khi hỗ trợ varlen) liệu các attn kernels mới có *ăn* được không?

- [ ] Round & smooth thuộc c) và có thể áp dụng ngược lại cho INT8 Mixed trên 4090
- [ ] Activations đang chiếm nhiều vram nhất mà nên được quant (giảm 1/2)
- [ ] Block scale sẽ đều và tốt hơn row-wise / col-wise?
- [ ] 4090 cũng hỗ trợ fp8, tìm hiểu xem có tận dụng được gì cho 8bit mixed / traing không?
  - fp8 weights
  - int8 weights https://github.com/pytorch/ao/pull/644

ROUNDING & SMOOTHING
--------------------
...

