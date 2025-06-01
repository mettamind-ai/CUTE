Các kernels mạnh đang có trong tay:

- INT8 Mixed Matmul
- bf16 Flash Attention vẫn nhanh nhất trong tổng thể các trường hợp
  - một vài trường hợp sageattn thắng ở fwd

LoRA, freezed, inference thì:
- Model weights ở int8 row scale => Cần Int8 Tensor class
- Activation + LoRA weights ở bf16


| Định dạng       | Exponent (mũ) | Mantissa (phần định trị) | Phạm vi động (Dynamic Range) | Độ chính xác (Precision) | Phù hợp với vai trò trong huấn luyện |
| --------------- | ------------- | ------------------------ | ---------------------------- | ------------------------ | ------------------------------------ |
| `float8_e5m2`   | 5 bits        | 2 bits                   | Cao                          | Thấp                     | Gradient (truyền ngược)              |
| `float8_e4m3fn` | 4 bits        | 3 bits                   | Trung bình                   | Cao                      | Trọng số và kích hoạt (truyền xuôi)  |


https://huggingface.co/allenai/OLMoE-1B-7B-0125

https://stanford-cs336.github.io/spring2025-lectures/?trace=var%2Ftraces%2Flecture_10.json&step=383

|<img src="https://pbs.twimg.com/media/GsWiTOXaMAAsjhn?format=jpg">|<img src="https://pbs.twimg.com/media/GsWin8PawAA8s9A?format=jpg">|
|-|-|
