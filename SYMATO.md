# SYMATO - SYMbol + MArk + TOne

> Symato là viết tắt của Symbol + Mark + Tone. Google translate sang tiếng Việt là "Đồng Cảm"

## Tổng quan

SYMATO là phương pháp tokenization tối ưu cho tiếng Việt, tách âm tiết thành 3 thành phần:

1. **SYM** (Symbol): Âm tiết viết không dấu (2535 syms)
2. **MARK** (Nét phụ): Biến thể nguyên âm - ă, â, ê, ô, ơ, ư, đ
3. **TONE** (Thanh điệu): ngang, sắc, huyền, hỏi, ngã, nặng

## Vocab Structure (2816 tokens)

| Range | Count | Mô tả |
|-------|-------|-------|
| 0-255 | 256 | Bytes (fallback cho non-Vietnamese) |
| 256-273 | 18 | Marktones |
| 274-2808 | 2535 | Syms (âm tiết không dấu) |
| 2814 | 1 | `^^` viết hoa toàn bộ |
| 2815 | 1 | `^` viết hoa chữ cái đầu |

## Marktone Encoding (18 loại)

Format: `|` + mark + tone

| Mark | Ký tự | Code |
|------|-------|------|
| NONE | a, e, i, o, u, y | (không có) |
| HORN | ơ, ư | `w` |
| HAT | â, ê, ô | `z` |
| BREVE | ă | `w` (trên a) |
| BAR | đ | Encoded trong sym (`dd`) |

| Tone | Tên | Code |
|------|-----|------|
| NONE | ngang | (không có) |
| ACUTE | sắc | `s` |
| GRAVE | huyền | `f` |
| HOOK | hỏi | `r` |
| TILDE | ngã | `x` |
| DOT | nặng | `j` |

### Bảng Marktone đầy đủ

```
|     không dấu + ngang     (a, e, o, u)
|s    không dấu + sắc       (á, é, ó, ú)
|f    không dấu + huyền     (à, è, ò, ù)
|r    không dấu + hỏi       (ả, ẻ, ỏ, ủ)
|x    không dấu + ngã       (ã, ẽ, õ, ũ)
|j    không dấu + nặng      (ạ, ẹ, ọ, ụ)

|w    horn + ngang          (ơ, ư)
|ws   horn + sắc            (ớ, ứ)
|wf   horn + huyền          (ờ, ừ)
|wr   horn + hỏi            (ở, ử)
|wx   horn + ngã            (ỡ, ữ)
|wj   horn + nặng           (ợ, ự)

|z    hat + ngang           (â, ê, ô)
|zs   hat + sắc             (ấ, ế, ố)
|zf   hat + huyền           (ầ, ề, ồ)
|zr   hat + hỏi             (ẩ, ể, ổ)
|zx   hat + ngã             (ẫ, ễ, ỗ)
|zj   hat + nặng            (ậ, ệ, ộ)
```

## Xử lý chữ "đ"

Chữ `đ` được encode thành `dd` trong SYM. Cả `d` và `đ` đều có SYM riêng biệt:

| d (không gạch) | đ (có gạch) |
|----------------|-------------|
| `di` (di cư) | `ddi` (đi bộ) |
| `da` (da thịt) | `dda` (đa số) |
| `do` (do đó) | `ddo` (đo lường) |
| `dau` (dau dau) | `ddau` (đau đớn) |

**Lưu ý**: `đ` KHÔNG nằm trong MARKTONE. MARKTONE chỉ xử lý:
- Nét phụ trên **nguyên âm** (ă, â, ê, ô, ơ, ư)
- Thanh điệu (sắc, huyền, hỏi, ngã, nặng)

Ví dụ:
| UTF-8 | SYM | Marktone |
|-------|-----|----------|
| đi | `ddi` | `\|` |
| đường | `dduong` | `\|wf` |
| đất | `ddat` | `\|zs` |
| đỏ | `ddo` | `\|r` |

## Ví dụ Encoding

| Tiếng Việt | SYM | Marktone | Token IDs |
|------------|-----|----------|-----------|
| a | `a` | `\|` | [274, 256] |
| à | `a` | `\|f` | [274, 258] |
| ă | `a` | `\|w` | [274, 262] |
| â | `a` | `\|z` | [274, 268] |
| việt | `viet` | `\|zj` | [sym_id, 273] |
| người | `nguoi` | `\|wf` | [sym_id, 264] |
| đường | `dduong` | `\|wf` | [sym_id, 264] |

## Encoding Format

Text được encode với byte `\x10` (16) làm delimiter:

```
"\x10a|\x10"         → a (không dấu)
"\x10a|f\x10"        → à (huyền)
"\x10^a|s\x10"       → Á (viết hoa + sắc)
"\x10^^viet|zj\x10"  → VIỆT (viết hoa toàn bộ + hat + nặng)
```

## Thứ tự Token

Mặc định: **SYM trước, MARKTONE sau**
```
"việt" → [sym_id("viet"), marktone_id("|zj")]
```

Option `rev=True`: đảo ngược (marktone trước sym)

## Space Handling

- Space giữa 2 âm tiết được **loại bỏ khi encode** để tiết kiệm context
- Khi decode, tự động thêm space giữa 2 sym liên tiếp

```python
# "người việt" encode thành:
[sym_nguoi, marktone_wf, sym_viet, marktone_zj]  # không có space token

# Decode tự động thêm space:
"người việt"
```

## Chuẩn hóa về dạng không dấu

SYMATO **strip hết dấu** từ input → chỉ giữ SYM:

```
ngùoi  → sym: nguoi (strip: ù→u)
nguòi  → sym: nguoi (strip: ò→o)
nguoì  → sym: nguoi (strip: ì→i)
người  → sym: nguoi (strip: ư→u, ờ→o)
```

**Lưu ý**: Marktone được lấy từ input gốc, không tự động sửa lỗi.

## Ưu điểm

### 1. Vocab size nhỏ
- SYMATO: 2816 tokens
- PhoBert: ~64k tokens
- → Model nhẹ hơn 2x, loss tốt hơn 3.5x

### 2. Tập trung biểu diễn SYM
- 80% text tiếng Việt là âm tiết
- SYM mang phần lớn thông tin ngữ nghĩa
- Marktone chỉ là modifier

### 3. Robust với typo
- 15 biến thể lỗi của "người" đều map về sym `nguoi`
- Model học embedding của `nguoi` rất tốt (nhiều samples)
- Chịu lỗi cao với các lỗi gõ phổ biến

### 4. Dual-task capability
- Vừa sinh văn bản (generate sym sequence)
- Vừa thêm dấu tự động (predict marktone từ context)

## Hạn chế

- Chỉ tối ưu cho text tiếng Việt (80%+ âm tiết)
- Non-Vietnamese text dùng byte fallback (256 tokens) - kém hiệu quả hơn

## Task: Thêm dấu tự động

Input: sequence các SYM (không dấu) + context
Output: predict MARKTONE cho mỗi SYM

```
Input:  nguoi viet
Output: |wf   |zj
Result: người việt

Input:  nuoc ngoai
Output: |ws  |f
Result: nước ngoài  (không phải "người"!)
```

Model cần học ngữ cảnh để phân biệt các trường hợp đồng âm.

## Marktone thực tế chỉ có 18 loại, không phải đầy đủ

**Lưu ý quan trọng**: BREVE (ă) dùng chung code `w` với HORN (ơ,ư):
- `|w` trên nguyên âm `a` → `ă` (BREVE)
- `|w` trên nguyên âm `o,u` → `ơ,ư` (HORN)

Điều này hoạt động vì trong tiếng Việt:
- `ă` chỉ xuất hiện từ `a`
- `ơ,ư` chỉ xuất hiện từ `o,u`
- Không có sự nhập nhằng

## Symato vocab

**symato-4k** (~4096 tokens):
- 256 bytes (fallback)
- 18 marktones
- 2535 syms
- ~1287 BPE tokens (cho non-Vietnamese và mở rộng)

## Kết quả thực nghiệm

### So sánh với PhoBert tokenizer (cùng kiến trúc model)

| Tokenizer | Vocab size | Params | Embedding | Blocks | Loss (20 epochs) |
|-----------|------------|--------|-----------|--------|------------------|
| PhoBert | ~64k | 47.8M | 20.6M | 6.7M | 0.14 |
| Symato | 2816 | 23.4M | 1.4M | 20.5M | 0.04 |

**Kết quả**: Symato nhỏ hơn 2x params nhưng loss tốt hơn 3.5x!

### Model đã huấn luyện

- `symato-vlc-23m`: ctx_len=512, embedding=512, layers=6, loss=0.04
- `symato-vlc-7m`: ctx_len=512, embedding=320, layers=4, loss=0.08

### Ví dụ output (thêm dấu + sinh văn bản)

```
Input:  nghia vu nop thue
Output: nghĩa vụ nộp thuế của Kho bạc Nhà nước...
```

Model tự động:
1. Thêm dấu thanh cho input không dấu
2. Tiếp tục sinh văn bản dựa trên context

## Tại sao RWKV?

SYMATO được thiết kế để dùng với RWKV vì:
- **Huấn luyện song song** như Transformer
- **Inference như RNN** - chỉ cần hidden state, tiết kiệm memory
- **Context length vô hạn** (lý thuyết)
- **Tốc độ nhanh hơn**, tiết kiệm tài nguyên hơn Transformer
- Phù hợp với tài nguyên hạn chế (1 GPU) và dữ liệu tiếng Việt

## Code Reference

- `tools/symato/symato_2816.py`: Tokenizer implementation
- `tools/symato/symato_2944.py`: Phiên bản mở rộng
- `tools/symato/bogo/`: Mark và Tone processing (từ ibus-bogo)
  - `mark.py`: Xử lý nét phụ (BREVE, HAT, HORN, BAR)
  - `tone.py`: Xử lý thanh điệu (6 tones)
  - `core.py`: Telex → UTF-8 conversion
- `tools/symato/racoon/`: RWKV training code đơn giản hóa
- `tools/utf8_to_symato.py`: UTF-8 → SYMATO converter

## UTF-8 to SYMATO Converter

### Sử dụng

```bash
# Command line
python3 tools/utf8_to_symato.py "Việt Nam"
# Output: ^viet|zj ^nam|

# Chạy tests
python3 tools/utf8_to_symato.py --test

# Từ stdin
echo "Xin chào" | python3 tools/utf8_to_symato.py
```

### Logic chuyển đổi

Chỉ convert sang SYMATO nếu thỏa **cả 3 điều kiện**:

1. **SYM hợp lệ**: Từ nằm trong danh sách 2535 SYM tiếng Việt
2. **Pure VN chars**: Từ chỉ chứa ký tự tiếng Việt (a-z + dấu VN)
3. **Word boundary hợp lệ**: Đằng trước là:
   - Đầu text
   - Space
   - Dấu câu mở (trừ `.` `@` `/` `\` - thường trong URL/email/path)

### Ví dụ

```python
from utf8_to_symato import text_to_symato

# Tiếng Việt thuần túy
text_to_symato("Việt Nam")           # → "^viet|zj ^nam|"
text_to_symato("Nguyễn Văn An")      # → "^nguyen|zx ^van|w ^an|"

# Có dấu câu - convert bình thường
text_to_symato("(Việt)")             # → "(^viet|zj)"
text_to_symato('"Xin chào!"')        # → '"^xin| chao|f!"'
text_to_symato("Việt-Nam")           # → "^viet|zj-^nam|"

# URL/email - KHÔNG convert (có . @ / dính liền)
text_to_symato("email@gmail.com")    # → "email@gmail.com"
text_to_symato("https://vi.wikipedia.org")  # → giữ nguyên

# Tiếng nước ngoài - KHÔNG convert
text_to_symato("hello world")        # → "hello world" (không phải SYM)
text_to_symato("café")               # → "café" (ký tự lạ)
text_to_symato("naïve")              # → "naïve" (ký tự lạ)

# Hỗn hợp Anh-Việt
text_to_symato("I love Việt Nam")    # → "^i| love ^viet|zj ^nam|"
# "i" là SYM VN, "love" không phải
```

### Tại sao cần word boundary?

Tiếng Việt mỗi âm tiết cách nhau bằng space. Nếu không có space/dấu câu đằng trước thì không phải âm tiết riêng biệt:

```python
# "com" là SYM tiếng Việt (cơm, còm...)
text_to_symato("com")                # → "com|" ✓
text_to_symato("gmail.com")          # → "gmail.com" (không convert vì . dính)

# "vi" là SYM tiếng Việt
text_to_symato("vi")                 # → "vi|" ✓  
text_to_symato("vi.wikipedia.org")   # → giữ nguyên (trong URL)
```

### Các ký tự "dính" (không tạo word boundary)

| Ký tự | Lý do |
|-------|-------|
| `.` | Domain, file extension |
| `@` | Email |
| `/` | URL path |
| `\` | Windows path |

Các dấu câu khác như `(`, `)`, `"`, `[`, `]`, `!`, `?`, `-` đều tạo word boundary hợp lệ.
