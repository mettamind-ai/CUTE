RWKV7: state update là tuyến tính theo thời gian, dạng RNN tuyến tính có gating.
- `state = state * w + sa * b + k * v`
- `y = state · r` (trong kernel code của `r` là `q`), r là gating trong RNN
- `sa = Σ a[j] * state[j]` cơ chế “state attention” vector 64 chiều (HEAD_SIZE)
- HEAD_SIZE mặc định cố định = 64
- heads = dim / 64

## r: receptance “độ tiếp nhận” (khả năng nhận/tiếp thu)
- x là input của layer
- xx = x_{t-1} - x_t
- xr = x_t + xx * x_r
-  r = xr × W_r

## state = state * w + sa * b + k * v
- w: hệ số giảm dần (decay) cho state cũ.
  state * w nghĩa là giữ lại state quá khứ nhưng giảm bớt theo w.

- k và v: giống “key” và “value”.
  k * v là đưa thông tin mới vào state.

- b: hệ số trộn state-attention.
  sa * b là phần điều chỉnh state dựa trên sa.
