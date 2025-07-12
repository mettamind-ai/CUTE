# Dynamic Chunking End2End HNet
https://arxiv.org/html/2507.07955v1

File `end2end_hnet.py` implement một end-to-end tokenizer-free language model theo tinh thần của H-Net paper:

- No explicit tokenization: Làm việc trực tiếp với ký tự (character-level)
- Hierarchical processing: Có nhiều cấp độ abstraction nhưng implement khác paper
- Dynamic representation: Sử dụng signal processing + attention thay vì fixed chunking

2. So sánh với H-Net paper

| Aspect            | H-Net paper                  | Implementation này                  | Đánh giá            |
|-------------------|------------------------------|-------------------------------------|---------------------|
| Input level       | Byte/character               | Character                           | ✅ Đúng tinh thần   |
| Chunking strategy | Học dynamic chunk boundaries | Fixed 8 tokens                      | ❌ Khác             |
| Compression       | Data-dependent pooling       | Signal encoding + softmax weighting | ⚠️ Khác             |
| Hierarchy         | Multiple stages              | Single stage (encoder→decoder)      | ❌ Chỉ 1 cấp        |
| End-to-end        | True end-to-end              | True end-to-end                     | ✅ Đúng tinh thần   |

## PatternEncoder (dòng 50-132)

Thực hiện 3-stage processing:

1. Signal Encoding (dòng 98-114):
- Dùng sine waves với frequencies từ prime numbers
- Concept tương tự positional encoding nhưng khác implement
- Vấn đề: Không rõ ràng về meaning của prime-based frequencies

2. Tokenization (dòng 116-124):
- Linear projection → softmax → weighted embedding
- Giống paper: Soft combination thay vì hard selection
- Khác paper: Fixed 8 tokens thay vì dynamic chunk sizes

3. Token Refinement (dòng 126-132):
- Multi-head self-attention giữa 8 tokens
- Tốt: Cho phép tokens interact với nhau

## PatternDecoder (dòng 137-188)

- Decode từ 8 tokens → character logits
- Có attention refinement giống paper
- Thiếu: `Multi-stage` hierarchy như trong paper

4. Vấn đề và hạn chế

So với paper:

- `Không có dynamic chunking`: Sử dụng fixed 8 tokens
- Thiếu hierarchy: `Chỉ có 1 cấp` thay vì multiple stages
- `Không có learned boundaries`: Không học chunk boundaries như paper

Technical issues:

- Prime tensor usage: Sử dụng prime numbers cho frequencies là heuristic
- Fixed m_tokens: Không adaptive với content length
- Character-level: Có thể inefficient cho sequences dài


5. Khuyến nghị cải tiến theo paper

- Dynamic chunking: Implement learned chunk boundaries
- Multi-stage hierarchy: Thêm multiple levels của processing
- Better compression: Dùng learned pooling thay vì signal encoding
- Byte-level: Consider byte thay vì character
- Scaling: Test với larger models và datasets

6. Kết luận

Implementation này capture được tinh thần của H-Net (end-to-end tokenizer-free) nhưng **implementation khác significantly về technical details**. Đây là một **proof-of-concept hướng tới end-to-end LM**, nhưng `cần nhiều cải tiến để match performance` của paper đề xuất.