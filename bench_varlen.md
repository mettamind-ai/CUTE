# RWKV7 Varlen Kernel Benchmark

Benchmark so sánh WinRWKV (original) vs WinRWKV (varlen) trên cùng số tokens.

**Hardware**: RTX 3050 Ti  
**Model**: dim=256, layers=6, vocab=4k

## End-to-End Training Benchmark

### Context Length 4096

| AvgSeqLen | NumSeqs | Orig ms | Varlen ms | Speedup |
|-----------|---------|---------|-----------|---------|
| 128       | 31      | 195.3   | 100.1     | 1.95x   |
| 256       | 17      | 186.5   | 101.7     | 1.84x   |
| 512       | 9       | 188.0   | 105.3     | 1.78x   |
| 1024      | 4       | 185.5   | 114.7     | 1.62x   |
| full      | 1       | 193.4   | 188.9     | 1.02x   |

### Context Length 8192

| AvgSeqLen | NumSeqs | Orig ms | Varlen ms | Speedup |
|-----------|---------|---------|-----------|---------|
| 128       | 60      | 389.4   | 193.3     | 2.01x   |
| 256       | 31      | 409.1   | 196.3     | 2.08x   |
| 512       | 17      | 431.5   | 220.2     | 1.96x   |
| 1024      | 9       | 435.1   | 257.8     | 1.69x   |
| full      | 1       | 434.8   | 400.3     | 1.09x   |

## Kết luận

- Varlen nhanh hơn **1.62x - 2.08x** với packed sequences
- Single sequence (full): ~1.02-1.09x (baseline, gần như tương đương)
- Speedup đến từ việc kernel xử lý nhiều sequences ngắn hiệu quả hơn

## Files

- `wkv7_varlen.cu` - Varlen kernel implementation
- `wkv7.cu` - Original kernel (không sửa đổi)
- `winrwkv_varlen.py` - WinRWKV model với varlen kernel
- `test_wkv7_varlen.py` - Test suite
- `bench_varlen.py` - Benchmark script
