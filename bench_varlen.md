# RWKV7 Varlen Kernel Benchmark

Benchmark so sánh original kernel (với padding) vs varlen kernel (packed sequences).

**Hardware**: RTX 3050 Ti  
**Config**: H=4 heads, C=64 (HEAD_SIZE), warmup=3, iterations=10

## Context Length 4096

| Num Seqs | Orig ms | Varlen ms | Speedup | Tokens | Padded | Waste% |
|----------|---------|-----------|---------|--------|--------|--------|
| 5        | 9.25    | 9.02      | 1.03x   | 4096   | 10800  | 62.1%  |
| 10       | 4.65    | 4.05      | 1.15x   | 4096   | 14880  | 72.5%  |
| 20       | 2.68    | 2.19      | 1.22x   | 4096   | 12160  | 66.3%  |
| 30       | 4.14    | 2.47      | 1.68x   | 4096   | 18240  | 77.5%  |
| 40       | 5.05    | 2.28      | 2.21x   | 4096   | 23040  | 82.2%  |
| 50       | 3.78    | 1.56      | 2.42x   | 4096   | 16000  | 74.4%  |

## Context Length 8192

| Num Seqs | Orig ms | Varlen ms | Speedup | Tokens | Padded | Waste% |
|----------|---------|-----------|---------|--------|--------|--------|
| 5        | 12.26   | 11.54     | 1.06x   | 8192   | 21600  | 62.1%  |
| 10       | 9.48    | 8.13      | 1.17x   | 8192   | 29760  | 72.5%  |
| 20       | 5.70    | 4.25      | 1.34x   | 8192   | 24640  | 66.8%  |
| 30       | 8.69    | 4.65      | 1.87x   | 8192   | 36960  | 77.8%  |
| 40       | 10.27   | 4.28      | 2.40x   | 8192   | 46080  | 82.2%  |
| 50       | 7.62    | 3.12      | 2.44x   | 8192   | 32000  | 74.4%  |

## Kết luận

- **Varlen nhanh hơn 1.03x - 2.44x** tùy thuộc vào số sequences
- Speedup tăng khi số sequences tăng (nhiều padding waste hơn)
- Với 40-50 sequences, varlen nhanh hơn **~2.4x**
- Waste% = (Padded - Tokens) / Padded, cho thấy lượng compute lãng phí khi dùng padding

## Files

- `wkv7_varlen.cu` - Varlen kernel implementation
- `wkv7.cu` - Original kernel (không sửa đổi)
- `test_wkv7_varlen.py` - Test suite
- `bench_varlen.py` - Benchmark script
