## `LiWin` `Li`near Attention + `Win`dow Attention (samba linh hoạt)
__Kết hợp best SWA (local) với Linear Attention (global)__
![](liwin-00-crunch.png)

### [Taipan: Mamba + Selective Attention Layers (SALs)](https://arxiv.org/html/2410.18572v1)
![](https://arxiv.org/html/2410.18572v1/x2.png)
![](https://arxiv.org/html/2410.18572v1/x3.png)

### Based = Li + Win mỏng (Linear Attn + SWA mỏng)
- https://www.alphaxiv.org/abs/2402.18668v2
- https://arxiv.org/html/2402.18668v2
![](https://arxiv.org/html/2402.18668v2/x1.png)

### Cách phối layers
- `6k ctxlen` LiWin: (n // 2) * (`rwkv7` + `swa_2k` + `rwkv7` + `swa_4k`)

- tăng dần embedding size của các khối
    - 768 + 896 + 1024 + 1152 + 1280 + 1408 ( + 1536 + 1664 + 1792 )
    - 24 layers, 4 layers 1 size => avg 1088

## Các biến thể của Attn
- `Glo` thiên về tóm tắt / toàn cục, kiểu như `Li`
- `Sel` chọn những tokens quan trọng để attn
- `Win` cục bộ theo cửa sổ, điển hình là `SWA` vô cùng đơn giản và hiệu quả nhất
- `Loc` local_attention mở rộng và look backward xa hơn `SWA`

## Win should use Trainable Sparse Attention
- `MoSA` = `Sel`; có thể kết hợp với local_attention nên hoàn toàn có thể thay thế `SWA`
- `NSA` = `Glo` + `Sel` + `Win`; có thể thay thế mọi loại block (cả Li và Win)
    - Vì Li đã có global, có thể chỉ dùng `Win` và `Sel` của NSA?
- `Taipan` = `Sel` + `Win`; => Có thể viết lại `Taipan Sel` (SALs) để dùng với `Win`
- `Based` = Li`Glo` + `Win` mỏng; `LiGlo` = Taylor Exponential Linear Attention nhanh nhẹ

# Learn at Test Time
- Titans: Learning to Memorize at Test Time https://arxiv.org/abs/2501.00663
- RNNs with Expressive Hidden States https://arxiv.org/abs/2407.04620
- Gated Delta Networks: Improving Mamba2 with Delta Rule https://arxiv.org/abs/2412.06464
