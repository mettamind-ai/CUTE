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

- [x] **Muon** 2-3x
- [x] **int8** 1.5x
- [x] **Arch** 1.5x @ 6k ctxlen (chưa đo lường)
- **Token**    1.5x (và **representation** nói chung)

🌸__!!! TARGET 10x SPEEPUP !!!__🌸

## [Kết quả thử nghiệm](/.save/EXPER.md)
- Muon is super good! vram = 1/4 + loss giảm sâu hơn adam
- int8 hữu dụng trong cả speedup và giảm vram
- int8 cần kết hợp stochastic rounding (rd) để đường loss bám sát bf16
- `muon + torch.optim.AdamW(fused=True) + int8rd` chạy rất tốt
- `value embeddings` + `multi exits` + `future prediction` should be good nhưng chưa thể hiện trên loss

---

# 🌸 TODO 🌸
## DATA
- Chỉ nên làm bilingual LM (Anh-Việt, Trung-Việt), và cạnh tranh theo chiều sâu ở từng domain hẹp
- `Best data` = `LLM mạnh nhất` + `sức người` **để đạt độ đậm đặc value**

## PLANING
- Tìm hiểu `exits > 1` + `torch.compile` khiến vram bị đội lên
- Canon https://github.com/fla-org/flash-linear-attention/pull/388
- Gluon https://www.alphaxiv.org/abs/2505.13416
- Scion https://github.com/LIONS-EPFL/scion
- GLM / UL2 learning objs

## Super Token
1 visual token có rất nhiều thông tin (diễn đạt bằng nhiều text token)
Liệu có thể làm tương tự như visual encoder nhưng mà cho text? => **SUPER TEXT TOKEN**
1 cụm n text tokens giờ đc biểu diễn = 1 embedding vector thay vì n như trước =>
đó là biên giới hạn của ngôn ngữ con người.

Máy có thể khai thác cái này để thông minh hơn ví dụ: Con lai của hổ và sư tử miêu tả là 
`"1 con gì đó giống con hổ và con sư tử"` => từ mới `hổ sư` thay vì biểu diễn bằng 1 câu 
sẽ có 1 miền trong token embeddings biểu diễn cái này
miền đó nằm giữa trung điểm của vector hổ và vector sư tử

biển diễn cho input và ouput thì vẫn là tokens trong vocab (con chữ) 
để đảm bảo tính cross entropy loss như bình thường, 
nhưng khi vào model mình có thể **map con chữ thành concept vector** ...

## Build `SyMaTo` (`Sy`llable + `Ma`rk + `To`ne) Tiny Monster Models
- `6k vocab` = `3k symato` (Vietnam) + `3k BPE` (English)
- Bài toán bộ gõ thông minh:
  - `auto-complete` + 
  - `sửa lỗi chính tả` + 
  - `convert gõ không dấu => có dấu` (tự động điền Mark + Tone)
- TTS cần 1 bộ tokenization khác thiên về phát âm

## OhMaiMơ - Biến UnSloth thành Sloth
Điểm mạnh của Unsloth là 1 nhân [fast lora](https://github.com/unslothai/unsloth/blob/main/unsloth/kernels/fast_lora.py) giúp LoRA finetune nhanh và tiết kiệm vram hơn bình thường. Sau đó bổ xung thêm vài fused kernels và một vài mẹo tăng tốc và chữa lỗi cho các models mới ra.

Điểm yếu của Unsloth là chưa hỗ trợ sample packing một kỹ thuật tối quan trọng trong pretrain và finetune. Chưa hỗ trợ những kỹ thuật mới mẻ như INT8 hay Muon ... Về bản chất Unsloth vẫn lười ...

Nếu biến nhân fast lora xài được INT8, sử dụng Muon optimizer, hỗ trợ sample packing và các kỹ thuật chưa từng có được tối ưu cho gamming GPUs như tự động giảm kích thước embeddings và lm_head (thường chiếm 1 phần rất lớn weights của model 1-3b) để giúp finetuning hiệu quả hơn? Và trở thành một finetuning framework linh hoạt (a.k.a FlexTune) để với mỗi một model module's weight có thể tuỳ chọn freeze (ko tune), full finetune hoặc LoRA ... và tự động search xem cấu hình nào là phù hợp nhất với finetuning computing buget?

## [DONE](.save/DONE.md)

🌸__DOING__🌸
- [ ] `tinymonster01` VLM đọc `screenshots (Anh + Việt)`
- [ ] save params + inference
  - https://github.com/pytorch-labs/gpt-fast
  - https://pytorch.org/blog/accelerating-generative-ai-2
- [ ] tìm hiểu mọi thứ về embedding / representation
- [ ] Học cách thu nhỏ model và NAS
  - https://huggingface.co/nvidia/Llama-3_3-Nemotron-Super-49B-v1
  - NAS https://arxiv.org/abs/2411.19146
  - Llama-Nemotron https://arxiv.org/abs/2505.00949
