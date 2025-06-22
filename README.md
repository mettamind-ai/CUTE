## 🌸`CUTE`🌸 Center for Upgrading Training Efficient
- `ONE_` gamming GPUs can train ~1b models
- `TWO_` gamming GPUs can train ~2b models
- `FOUR` gamming GPUs can train ~4b models
```
                             BF16        INT8
3090      350W  24G     71 TFLOPS    284 TOPS
4090      450W  24G    165 TFLOPS    660 TOPS
5090      575W  32G    210 TFLOPS    838 TOPS
```
| Cấu hình        | Giá     | TFLOPs | TFLOPs/$ | DLPerf | DLPerf/$ |
| --------------- | ------- | ------ | -------- |------- | ---------|
|`032G`_2x5070ti  |$0.32/hr | 088    |**275.00**| 132    |**412.50**|
|`064G`_2x5090    |$0.97/hr | 216    |  222.68  | 283    | *291.75* |
| --------------- | ------- | ------ | -------- |------- | -------- |
|`096G`_4x4090    |$1.28/hr | 324    | *253.12* | 310    |  242.18  |
|`128G`_4x5090    |$1.96/hr | 432    |  220.41  | 524    |  267.34  |

- [x] **Muon**          2.0x
- [x] **int8**          1.5x
- [x] **Dense Arch**    1.5x @ 6k ctxlen (chưa đo lường)
- [x] **OhMai**         1.3x
- [ ] **Flexible MoE**  1.5x
- [ ] **Super Token**   2.0x (better & denser representations in the hidden space)

🌸__!!! TARGET x10 SPEEPUP !!!__🌸

## [Kết quả thử nghiệm](/.save/EXPER.md)
- Muon is super good! vram = 1/4 + loss giảm sâu hơn adam
- int8 hữu dụng trong cả speedup và giảm vram
- int8 cần kết hợp stochastic rounding (rd) để đường loss bám sát bf16
- `muon + torch.optim.AdamW(fused=True) + int8rd` chạy rất tốt
- `value embeddings` + `multi exits` + `future prediction` should be good nhưng chưa thể hiện trên loss


## DATA
- Chỉ nên làm bilingual LM (Anh-Việt, Trung-Việt), và cạnh tranh theo chiều sâu ở từng domain hẹp
- `Best data` = `LLM mạnh nhất` + `sức người` **để đạt độ đậm đặc value**
- [Dùng GPU xử lý data](https://github.com/ServiceNow/Fast-LLM/blob/main/fast_llm/csrc/data.cpp)

## Tiny Monster Models
- `SyMaTo` (`Sy`llable + `Ma`rk + `To`ne) để tiền xử lý
- `4k BPE vocab` + Stochastok (random phân giã) + 2,3-gram embeddings (random tổng hợp)
- Bài toán bộ gõ thông minh:
  - `auto/smart-edit`
  - `auto/smart-complete` + 
  - `sửa lỗi chính tả` + 
  - `convert gõ không dấu => có dấu` (tự động điền Mark + Tone)
- TTS cần 1 bộ tokenization khác thiên về phát âm
- VLM đọc screenshots

# 🌸LINH HOẠT🌸 Dense + MoE + Reused Block + Precision + Size + Flex Text Token/Super Token + Multi-Modals + Đa Mục Tiêu Học?
Một sự linh hoạt toàn diện trong cách xây dựng model, và tìm kiếm hiệu quả thực sự trong các cách kết hợp
linh hoạt đó? `Linh hoạt không khó, linh hoạt mang lại hiệu quả mới khó!`

- LOGITS DISTILL: Có thể **kết hợp logits distill + pre-train** để:
  - giảm dataset phải chuẩn bị cho nó học?
  - học cách phân bổ dữ liệu nhanh hơn?
  - nhìn data dist dưới góc nhìn của logits (2D: seq x vocab)
  - tknz là cách cân bằng giữa `hidden dim` vs `seq_len` vs `vocab_size`
    hdim cố định, vocab tăng giúp giảm seq len nhưng làm tăng vocab size nhanh chóng
  - Logits distill: chỉ có top-5 tokens là quan trọng ... => tính thưa rất cao!
    - BiLD loss với chỉ top-8 logits https://www.alphaxiv.org/abs/2406.13555

- LEARNING OBJECTIVES
  - [x] MTP rất hiệu quả với 1 future head
  - [ ] LongCE => WEIGHTED LOSS một cách thông minh (dùng chính model để đo độ quan trọng của token)
    - Cần 1 phương pháp load training data vào context nhanh để tìm ra những tokens khó dự đoán.
      Chỉ cần nắm tương quan tokens nào khó dự đoán, tokens nào dễ dự đoán ko cần ra xác xuất chính xác.
    - => INT8 Linear + INT8 SageFwd + Early Exit to maximum speedup! 
  - [ ] Hạn chế tác hại của Causual Attn => GLM and other learning objectives?
  - [ ] Dùng final NTP loss của mỗi token làm weighted cho early exit prediction (EE) và next of next token prediction (MTP)
    Lý do: token nào mà final dễ đoán thì dồn sức cho EE; token nào khó đoán thì dồn sức cho MTP

- LINH TOK (flexible tokenization & token usage)
  - Token được TỰ DO LỰA CHỌN:
    - cách nó attn (chính là query trong self-attn)
    - cách nó chọn số computing / hidden dim để biểu diễn chính nó (MoE)
      https://newsletter.maartengrootendorst.com/p/a-visual-guide-to-mixture-of-experts

  - Input là 2-gram nhưng output là gram (NTP) + gram (MTP với 1 prediction head) để giữ head bé
    - trong lúc tknz có tỉ lệ nhỏ 1-2% tự động phân mảnh token hoặc gộp 2 token liền nhau (dùng model để chọn)
      nhằm huấn luyện model thích ứng với nhiều cách tknz khác nhau và build embeddings của n-grams (n > 2)

  - Tận dụng loss/logits/gradient per token từ step trước để quyết định phân tách / gộp chính xác hơn.
    **Cách nhóm tokens thế nào sẽ do model tính điểm 1 lần trên based tokens trước rồi mới quyết**

  - Bắt đầu bằng bất kỳ khởi tạo vocab nào và sắp xếp lại thứ tự quan trọng của các tokens trong vocab dựa vào grad score / logits score / loss score ..., từ đó bình chọn lại, sẽ có những unit tokens luôn được giữ nhưng chỉ là số lượng nhỏ (2k/total 8k chẳng hạn)

  - **Nếu liên tục promote những tokens mới thì model tự nhiên sẽ tạo ra SUPER TOKENS của riêng nó.**

---

- [x] chậm, chưa gain? <= ~~Canon https://github.com/fla-org/flash-linear-attention/blob/main/fla/modules/convolution.py~~
- [x] ~~stablemax~~ hoặc fp32 unembeddings + fp64 softmax, ortho optim đã có muon

- [ ] Thống kê 2-gram và map 2-gram vào token_id từ data/6400.bin
- [ ] Dùng final NTP loss của mỗi token làm weighted cho early exit prediction (EE) và next of next token prediction (MTP)
  Lý do: token nào mà final dễ đoán thì dồn sức cho EE; token nào khó đoán thì dồn sức cho MTP
- [ ] impl weighted LCE hiệu quả
- [ ] impl longce hiệu quả "only less than 10% tokens are highly influenced by long context"
- [ ] tính và giữ lại per token `logit_score` và `grad_score` sau mỗi lần fwd và bwd.
  => có thể pre-filter hoặc sample intelligently.

- [ ] diffusion LLM từ Bert https://huggingface.co/HPLT/hplt_bert_base_2_0_vie-Latn/blob/main/config.json
  - https://huggingface.co/HPLT/translate-vi-en-v2.0-hplt
  - https://huggingface.co/HPLT/translate-en-vi-v2.0-hplt

- [ ] flash-attn đang viết lại = cute dsl => đọc hiểu để impl
  - flashmask ![](https://pbs.twimg.com/media/GtqLLRsbAAEXNTq?format=jpg&name=4096x4096)
  - cách nhanh và rẻ nhất là masking ở **block level** với block là processing unit của FA
  - [ ] sửa FA để nó luôn attn vào first block of seq bất kể SWA là bao nhiêu
  - [ ] sửa FA để hỗ trợ prefix LLM (GLM paper, UL2 paper ...)

- [ ] grokking với spectral clipping https://leloykun.github.io/ponder/spectral-clipping

- [ ] fast inference
  - https://scalingintelligence.stanford.edu/blogs/tokasaurus
