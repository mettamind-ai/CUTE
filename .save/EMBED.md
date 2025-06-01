# 32k full vocab vs 3.2k ohmai

PHÂN CHIA PARAMS VÀO DTYPES:
* 130 INT8 Mixed Weights 88.7% 542,703,616
* 3 BF16/ FP32 Weights 11.3% 69,222,504
INT8: ['*attn.kv_proj', '*attn.o_proj', '*attn.q_proj', '*mlp.fc1_proj', '*mlp.fc2_proj']

PHÂN CHIA PARAMS VÀO OPTIMIZERS:
* Adam: 11.3% 69,222,504
* Muon: 88.7% 542,703,616
 TOTAL: 100.0% 611,926,120
Adam: ['embeddings.active_weight', 'lm_head', 'scalars']
Muon: ['*attn.kv_proj', '*attn.o_proj', '*attn.q_proj', '*mlp.fc1_proj', '*mlp.fc2_proj']
>>> torch.compile(lossf) <<<

CHUẨN BỊ HUẤN LUYỆN:
* GPU(s) 1
* compile? True
* future? 0.0
* simple_loss_fn
* 32k seq/step

wandb: Currently logged in as: tiendung to https://api.wandb.ai. Use `wandb login --relogin` to force relogin
wandb: Tracking run with wandb version 0.19.11
wandb: Run data is saved locally in /tmp/wandb/run-20250601_132301-l6dxsiug
wandb: Run `wandb offline` to turn off syncing.
wandb: Syncing run woven-disco-1409
wandb: ⭐️ View project at https://wandb.ai/tiendung/2
wandb: 🚀 View run at https://wandb.ai/tiendung/2/runs/l6dxsiug
  0%|              | 1/1000 [01:38<27:16:46, 98.30s/it, loss=10.4, lr=0.0006]>>> First Step Took 753 Seconds <<<
100%|██████████████| 1000/1000 [18:13<00:00,  1.00it/s, loss=1.62, lr=0.0002]wandb:                                                                       
wandb:
wandb: Run history:
wandb:                  adam_lr ▁▅███████████████████████████████▇▆▅▄▃▃▂
wandb:                     loss █▅▄▄▃▃▃▂▂▃▃▂▂▂▂▂▂▂▂▂▂▂▁▁▂▁▁▁▂▁▂▂▂▂▁▂▁▂▂▁
wandb:     max_memory_allocated ▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁
wandb:                  muon_lr ▁▇▇████████████████████████████████▆▄▃▃▁
wandb: num_tokens_seen_millions ▁▁▂▂▂▃▃▃▃▃▃▃▃▃▄▄▄▄▄▄▄▅▅▅▅▆▆▆▆▆▆▆▇▇▇▇▇███
wandb:        tokens_per_second ▁███████████████████████████████████████
wandb:
wandb: Run summary:
wandb:                  adam_lr 2e-05
wandb:                     loss 1.61773
wandb:     max_memory_allocated 12976196608
wandb:                  muon_lr 0.0002
wandb: num_tokens_seen_millions 32.768
wandb:        tokens_per_second 32790.9786
wandb:
wandb: 🚀 View run woven-disco-1409 at: https://wandb.ai/tiendung/2/runs/l6dxsiug


PHÂN CHIA PARAMS VÀO DTYPES:
* 130 INT8 Mixed Weights 71.8% 542,703,616
* 3 BF16/ FP32 Weights 28.2% 212,992,104
INT8: ['*attn.kv_proj', '*attn.o_proj', '*attn.q_proj', '*mlp.fc1_proj', '*mlp.fc2_proj']

PHÂN CHIA PARAMS VÀO OPTIMIZERS:
* Adam: 28.2% 212,992,104
* Muon: 71.8% 542,703,616
 TOTAL: 100.0% 755,695,720
Adam: ['embeddings', 'lm_head', 'scalars']
Muon: ['*attn.kv_proj', '*attn.o_proj', '*attn.q_proj', '*mlp.fc1_proj', '*mlp.fc2_proj']
>>> torch.compile(lossf) <<<

CHUẨN BỊ HUẤN LUYỆN:
* GPU(s) 1
* compile? True
* future? 0.0
* simple_loss_fn
* 32k seq/step

wandb: Currently logged in as: tiendung to https://api.wandb.ai. Use `wandb login --relogin` to force relogin
wandb: Tracking run with wandb version 0.19.11
wandb: Run data is saved locally in /tmp/wandb/run-20250601_125336-o9h0bk56
wandb: Run `wandb offline` to turn off syncing.
wandb: Syncing run ethereal-moon-1407
wandb: ⭐️ View project at https://wandb.ai/tiendung/2
wandb: 🚀 View run at https://wandb.ai/tiendung/2/runs/o9h0bk56
  0%|              | 1/1000 [01:31<25:23:13, 91.49s/it, loss=10.4, lr=0.0006]>>> First Step Took 689 Seconds <<<
100%|██████████████| 1000/1000 [17:57<00:00,  1.01it/s, loss=1.62, lr=0.0002]wandb:                                                                       
wandb:
wandb: Run history:
wandb:                  adam_lr ▅▇████████████████████████████▇▇▅▄▄▃▃▃▃▁
wandb:                     loss █▇▆▆▆▅▅▄▄▃▄▃▂▃▃▂▂▂▃▃▂▂▂▃▂▂▂▁▂▁▂▃▁▃▂▁▂▁▂▁
wandb:     max_memory_allocated ▁▂▂▃▃▃▃▇▇▇▇▇▇███████████████████████████
wandb:                  muon_lr ▂███████████████████████████████▇▅▅▄▄▃▃▁
wandb: num_tokens_seen_millions ▁▁▁▁▂▂▂▂▂▂▂▃▃▃▃▃▄▄▄▄▅▅▆▆▆▆▆▇▇▇▇▇▇▇██████
wandb:        tokens_per_second █▇▄▃▂▂▂▂▂▂▂▂▂▁▂▁▂▁▂▁▂▂▂▁▁▁▂▁▁▁▁▁▁▂▂▁▂▂▂▁
wandb:
wandb: Run summary:
wandb:                  adam_lr 2e-05
wandb:                     loss 1.61744
wandb:     max_memory_allocated 17913653248
wandb:                  muon_lr 0.0002
wandb: num_tokens_seen_millions 32.768
wandb:        tokens_per_second 33156.33899
wandb:
wandb: 🚀 View run ethereal-moon-1407 at: https://wandb.ai/tiendung/2/runs/o9h0bk56

--

### all ve, max te (full vocab active embbeddings)
```
PHÂN CHIA PARAMS VÀO OPTIMIZERS:
* Adam: 39.8% 372,736,156
* Muon: 60.2% 564,854,784
 TOTAL: 100.0% 937,590,940
```
### all ve, max te + ohmai (1/3 vocab active embeddings)
```
PHÂN CHIA PARAMS VÀO OPTIMIZERS:
* Adam: 18.3% 126,517,404
* Muon: 81.7% 564,854,784
 TOTAL: 100.0% 691,372,188
```

![](/.save/embs-00-crunch.png)

KẾT LUẬN:
- Bổ xung value embeddings ở mọi layer có lợi nhất về loss, vram và speed
- OhMaiEmbedding giúp giảm đáng kể lượng params phải load trong vram

---


# 3 ve, 1 te
PHÂN CHIA PARAMS VÀO OPTIMIZERS:
* Adam: 5.2% 31,129,756
* Muon: 94.8% 564,854,784
 TOTAL: 100.0% 595,984,540
Adam: ['embeddings', 'lm_head', 'scalars']
Muon: ['*attn.kv_proj', '*attn.o_proj', '*attn.q_proj', '*mlp.fc', '*mlp.proj']
>>> torch.compile(lossf) <<<

100%|██████████████| 1000/1000 [16:26<00:00,  1.08it/s, loss=1.22, lr=0.0002]wandb:                                                                       
wandb:
wandb: Run history:
wandb:                  adam_lr ▁▅███████████████████████████████████▇▇▅
wandb:                     loss █▆▄▃▂▂▂▂▂▂▁▂▂▁▁▂▂▂▂▁▂▁▁▂▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁
wandb:     max_memory_allocated ▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁
wandb:                  muon_lr ▂▃▇████████████████████████████████▅▄▃▃▁
wandb: num_tokens_seen_millions ▁▁▁▁▂▂▂▂▃▃▃▃▃▃▄▄▄▄▄▄▅▅▅▅▅▆▆▆▆▆▆▆▆▆▇▇▇███
wandb:        tokens_per_second ▁███████████████████████████████████████
wandb:
wandb: Run summary:
wandb:                  adam_lr 2e-05
wandb:                     loss 1.22074
wandb:     max_memory_allocated 11777459200
wandb:                  muon_lr 0.0002
wandb: num_tokens_seen_millions 32.768
wandb:        tokens_per_second 35395.05433
wandb:
wandb: 🚀 View run pleasant-waterfall-1290 at: https://wandb.ai/tiendung/2/runs/p405cfj8

# max ve, 1 te
PHÂN CHIA PARAMS VÀO OPTIMIZERS:
* Adam: 15.9% 106,496,156
* Muon: 84.1% 564,854,784
 TOTAL: 100.0% 671,350,940
Adam: ['embeddings', 'lm_head', 'scalars']
Muon: ['*attn.kv_proj', '*attn.o_proj', '*attn.q_proj', '*mlp.fc', '*mlp.proj']
>>> torch.compile(lossf) <<<

  0%|              | 1/1000 [01:07<18:35:50, 67.02s/it, loss=8.76, lr=0.0006]>>> First Step Took 392 Seconds <<<
100%|██████████████| 1000/1000 [16:38<00:00,  1.07it/s, loss=1.01, lr=0.0002]wandb:                                                                       
wandb:
wandb: Run history:
wandb:                  adam_lr ▁▇█████████████████████████████████▆▅▅▄▃
wandb:                     loss █▇▆▄▃▃▃▃▄▂▃▂▂▃▃▃▃▂▂▂▂▂▂▂▂▂▁▂▃▁▁▁▁▁▁▁▁▁▁▂
wandb:     max_memory_allocated ▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁
wandb:                  muon_lr ▄▇█████████████████████████████████▇▇▆▃▁
wandb: num_tokens_seen_millions ▁▁▁▁▂▂▂▂▂▃▃▃▃▃▄▄▄▄▄▄▅▅▅▅▅▅▅▆▆▆▆▆▇▇▇▇▇███
wandb:        tokens_per_second ▅▆▆▅▄▁▅▄▄▅▄▄▄▄▅▆▆█▆▃▄▄▆▇▇▄▇▇█▆▆▇▆▇▄▆▆▇▅▄
wandb:
wandb: Run summary:
wandb:                  adam_lr 2e-05
wandb:                     loss 1.0135
wandb:     max_memory_allocated 14118120960
wandb:                  muon_lr 0.0002
wandb: num_tokens_seen_millions 32.768
wandb:        tokens_per_second 35045.88511
wandb:
wandb: 🚀 View run lemon-hill-1291 at: https://wandb.ai/tiendung/2/runs/26k0m18u

# 3 ve, max te
PHÂN CHIA PARAMS VÀO OPTIMIZERS:
* Adam: 34.5% 297,369,756
* Muon: 65.5% 564,854,784
 TOTAL: 100.0% 862,224,540
Adam: ['embeddings', 'lm_head', 'scalars']
Muon: ['*attn.kv_proj', '*attn.o_proj', '*attn.q_proj', '*mlp.fc', '*mlp.proj']
>>> torch.compile(lossf) <<<

  0%|              | 1/1000 [01:05<18:14:08, 65.71s/it, loss=8.76, lr=0.0006]>>> First Step Took 322 Seconds <<<
100%|██████████████| 1000/1000 [17:42<00:00,  1.00it/s, loss=1.13, lr=0.0002]wandb:                                                                       
wandb:
wandb: Run history:
wandb:                  adam_lr ▁▇█████████████████████████████████▇▅▄▃▁
wandb:                     loss █▄▄▄▃▃▃▂▂▂▁▂▂▂▂▂▂▂▂▂▂▂▂▂▂▂▂▂▂▂▂▁▂▁▂▂▁▁▁▁
wandb:     max_memory_allocated ▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁
wandb:                  muon_lr ▃██████████████████████████████████▆▅▅▄▁
wandb: num_tokens_seen_millions ▁▁▁▁▁▂▂▂▃▃▃▃▃▃▄▄▄▄▅▅▅▅▅▅▆▆▆▆▆▇▇▇▇▇▇▇████
wandb:        tokens_per_second █▆▃▂▄▂▃▃▂▄▁▁▃▃▄▃▃▄▄▄▂▄▃▃▃▃▃▅▄▅▄▄▅▄▃▄▄▃▂▄
wandb:
wandb: Run summary:
wandb:                  adam_lr 2e-05
wandb:                     loss 1.133
wandb:     max_memory_allocated 19550088704
wandb:                  muon_lr 0.0002
wandb: num_tokens_seen_millions 32.768
wandb:        tokens_per_second 32841.43138
wandb:
wandb: 🚀 View run smooth-glitter-1292 at: https://wandb.ai/tiendung/2/runs/okzak59z

# all ve, max te

PHÂN CHIA PARAMS VÀO OPTIMIZERS:
* Adam: 39.8% 372,736,156
* Muon: 60.2% 564,854,784
 TOTAL: 100.0% 937,590,940
Adam: ['embeddings', 'lm_head', 'scalars']
Muon: ['*attn.kv_proj', '*attn.o_proj', '*attn.q_proj', '*mlp.fc', '*mlp.proj']
>>> torch.compile(lossf) <<<

CHUẨN BỊ HUẤN LUYỆN:
* GPU(s) 1
* compile? True
* future? 0.0
* simple_loss_fn
* 32k seq/step

  0%|              | 1/1000 [01:08<19:08:41, 68.99s/it, loss=8.76, lr=0.0006]>>> First Step Took 363 Seconds <<<
100%|██████████████| 1000/1000 [19:38<00:00,  1.00s/it, loss=1.52, lr=0.0002]wandb:                                                                       
wandb:
wandb: Run history:
wandb:                  adam_lr ▁▃▇████████████████████████████████▆▃▃▂▁
wandb:                     loss █▅▃▃▂▂▂▂▂▂▁▂▁▁▂▂▂▁▁▂▁▂▂▂▁▁▂▁▁▁▂▁▂▂▁▁▁▁▁▁
wandb:     max_memory_allocated ▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁
wandb:                  muon_lr ▁▇█████████████████████████████████▇▆▆▃▁
wandb: num_tokens_seen_millions ▁▁▁▁▁▂▂▂▂▂▃▃▃▃▃▄▄▄▄▄▄▄▅▅▅▅▅▅▅▅▆▆▆▆▆▇▇███
wandb:        tokens_per_second ▅▆▅▅▁▇▃▅▄▁▃▅▄▄▇▅▇█▃▅▆▆▆▇█▅▆▇▄▇▅▆█▅█▇▅▃▆█
wandb:
wandb: Run summary:
wandb:                  adam_lr 2e-05
wandb:                     loss 1.51639
wandb:     max_memory_allocated 21613424640
wandb:                  muon_lr 0.0002
wandb: num_tokens_seen_millions 32.768
wandb:        tokens_per_second 32570.88819
wandb:
wandb: 🚀 View run trim-glitter-1293 at: https://wandb.ai/tiendung/2/runs/gng9w91a

# all ve, max te + ohmai

PHÂN CHIA PARAMS VÀO DTYPES:
* 78 INT8 Mixed Weights 68.9% 476,250,112
* 55 BF16/ FP32 Weights 31.1% 215,122,076
INT8: ['*attn.o_proj', '*mlp.fc', '*mlp.proj']

PHÂN CHIA PARAMS VÀO OPTIMIZERS:
* Adam: 18.3% 126,517,404
* Muon: 81.7% 564,854,784
 TOTAL: 100.0% 691,372,188
Adam: ['embeddings.active_weight', 'lm_head', 'scalars']
Muon: ['*attn.kv_proj', '*attn.o_proj', '*attn.q_proj', '*mlp.fc', '*mlp.proj']
>>> torch.compile(lossf) <<<

CHUẨN BỊ HUẤN LUYỆN:
* GPU(s) 1
* compile? True
* future? 0.0
* simple_loss_fn
* 32k seq/step

  0%|              | 1/1000 [01:25<23:40:51, 85.34s/it, loss=8.76, lr=0.0006]>>> First Step Took 168 Seconds <<<
100%|██████████████| 1000/1000 [20:14<00:00,  1.07s/it, loss=1.52, lr=0.0002]wandb:                                                                       
wandb:
wandb: Run history:
wandb:                  adam_lr ▃▄▇█████████████████████████████████▇▅▃▁
wandb:                     loss ██▆▅▄▃▃▃▄▃▂▂▂▄▃▃▁▃▂▂▃▃▃▂▃▂▃▂▃▂▂▃▂▃▁▂▂▂▂▃
wandb:     max_memory_allocated ▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁
wandb:                  muon_lr ▁▃▄▅▇█████████████████████████████▆▅▄▄▃▁
wandb: num_tokens_seen_millions ▁▁▁▁▂▂▂▂▂▂▃▃▃▃▃▃▄▄▄▄▅▅▅▅▅▅▆▆▆▆▆▆▇▇▇▇████
wandb:        tokens_per_second █▇█▃▅▃█▆▄▇▅▆▇▃▄▆▄▆▄▇▆▂▁▃▅▄▅▁▅▄▆▆▅▄▆█▅█▅▅
wandb:
wandb: Run summary:
wandb:                  adam_lr 2e-05
wandb:                     loss 1.522
wandb:     max_memory_allocated 23952928768
wandb:                  muon_lr 0.0002
wandb: num_tokens_seen_millions 32.768
wandb:        tokens_per_second 30564.18333
wandb:
wandb: 🚀 View run lively-armadillo-1296 at: https://wandb.ai/tiendung/2/runs/ulhv1mjl
