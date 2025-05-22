# WinGPT 6k, 8k vocab

![](/.save/wingpt.jpg)

Model kết hợp nhiều input và ouput:
- input có vẻ thắng thế
- loss không nói được nhiều
- cần eval để biết kết quả cuối

---

| Config    | Loss  | VRAM   | toks | t/s    | t/step |
|-----------|-------|--------|------|--------|--------|
|333m 12*3k cp0| 2.57 | 22.9G | 74m | 60_310 | 36k    |
|666m 16*4k cp1| 2.63 | 16.8G | 74m | 32_855 | 64k    |
|666m 24*4k cp1| 2.61 | 23.1G | 74m | 33_969 | 96k    |
|999m 16*4k cp1| 2.56 | 21.8G | 74m | 25_413 | 64k    |
```bash
./pretrain.py --S --steps 2000          --bs  12    # 36k
./pretrain.py --M --steps 2250 --actcp 2 --bs  8    # 32k OOM
./pretrain.py --M --steps 1125 --actcp 1 --bs 16    # 64k
./pretrain.py --M --steps  750 --actcp 1 --bs 24    # 96k
./pretrain.py --L --steps 1125 --actcp 1 --bs 16    # 64k
```

## 600m; 64k vs 96kt/step; TextBooks.en+Wiki.en.vi
| Config       | Loss  | VRAM   | toks | toks/s   | mins | speedup |
|--------------|-------|--------|------|----------|------|---------|
|[96k=24x4k](https://wandb.ai/tiendung/2/runs/508suqgu)  | 2.62  | 23.9G  | 98m  | 35_308   | 46.2 | 1.00    |
|[64k=16x4k](https://wandb.ai/tiendung/2/runs/kdc48s5k)  | 2.52  | 17.1G  | 98m  | 36_840   | 44.3 | 1.04    |
<!-- |[96k=32*3k](https://wandb.ai/tiendung/2/runs/j1y7xilx)  |  -->
**=> 64k @ 16bs lợi nhất! `tốc độ nhanh nhất` + `loss giảm sâu nhất`**


---

# WinGPT 32k vocab
## 700m 64k vs 32kt/step; Muon; TinyStories
| Config       | Loss  | VRAM   | toks | toks/s   | mins | speedup |
|--------------|-------|--------|------|----------|------|---------|
|[32k_bf16](https://wandb.ai/tiendung/2/runs/n3jxz5cc)|**1.07**|  21.3G  | 98m | 22_625 | 72 |  1.00  |
|[32k_int8](https://wandb.ai/tiendung/2/runs/jgnbig7f)|  1.10  |**14.5G**| 98m | 34_420 | 48 | *1.52* |
|[64k_int8](https://wandb.ai/tiendung/2/runs/rbumxf9z)| *1.09* |  24.0G  | 98m | 37_338 | 44 |**1.65**|
|[96k_bf16](https://wandb.ai/tiendung/2/runs/cya6x9w7)|**1.06**|  22.0G  | 98m | xx_xxx |102 |  x.xx  |

- `64kt/step + int8abit` vừa 24G VRAM => fit 4090
- `32kt/step + int8abit` vừa 16G VRAM => fit 4070, 5070ti

<img src="/.save/win_700m_64kt_step-crunch.png" width="80%">
<img src="/.save/tinymonster-01-crunch.png" width="80%">
<br /><br />

<img src="/.save/tinymonster-00-crunch.png" width="80%">
> 32kt/step + int8abit vừa 16G VRAM => fit 4070, 5070ti

## 760m 32kt/step int8's `none` vs `abit` vs `half` vs `full` vs `hack` rounding
| Config       | Loss  | toks | toks/s   | mins | speedup |
|--------------|-------|------|----------|------|---------|
|[bf16](https://wandb.ai/tiendung/2/runs/86sadb32)        |**1.26**| 33m | 22_580 | 24 |  1.00  |
|[int8_none_rd](https://wandb.ai/tiendung/2/runs/09dw41us)|~~3.33~~| 33m | 36_421 | 15 |~~1.61~~|
|[int8_abit_rd](https://wandb.ai/tiendung/2/runs/r65t6xuu)|  1.30  | 33m | 34_283 | 16 |**1.52**|
|[int8_half_rd](https://wandb.ai/tiendung/2/runs/sdt3820n)|  1.31  | 33m | 32_386 | 17 |  1.43  |
|[int8_full_rd](https://wandb.ai/tiendung/2/runs/3cpryttj)|**1.26**| 33m | 29_313 | 19 | *1.30* |
|[int8_hack_rd](https://wandb.ai/tiendung/2/runs/cdd4yfd8)| *1.27* | 33m | 28_720 | 19 |  1.27  |

![](/.save/int8+muon+various_rd-crunch.png)

**NOTE**: trường hợp `int8` loss bị chênh so với `bf16`
- pretrain => `abit`
- midtrain => `hack` hoặc `full`
- finetune => `hack` hoặc `bf16`
