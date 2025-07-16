# H-Net: Dynamic Chunking for End-to-End Hierarchical Sequence Modeling
(Phân đoạn động để mô hình hoá chuỗi theo phân cấp từ đầu đến cuối)
- https://arxiv.org/html/2507.07955v1
- https://goombalab.github.io/blog/2025/hnet-past
- https://goombalab.github.io/blog/2025/hnet-future
- https://goombalab.github.io/blog/2025/tradeoffs
- https://main-horse.github.io/posts/hnet-inf

<table><tr>
<td width="45%"><img src="https://github.com/goombalab/hnet/raw/main/assets/code.gif"></td>
<td width="55%"><img src="https://raw.githubusercontent.com/goombalab/hnet/refs/heads/main/assets/arch.png"></td>
</tr></table>

## Các khái niệm cốt lõi (từ bài báo)
1.  **Học End-to-End trực tiếp từ Byte:** H-Net loại bỏ hoàn toàn bước tiền xử lý token hóa với một bộ từ vựng cố định (như BPE). Thay vào đó, nó học trực tiếp từ các byte thô, cho phép mô hình tự xây dựng các biểu diễn của riêng mình.

2.  **Gộp chuỗi động (Dynamic Chunking - DC):** Đây là cơ chế trung tâm. Mô hình học cách phân đoạn một chuỗi thành các "đoạn" (chunk) có độ dài thay đổi dựa trên sự tương đồng về nội dung. Quá trình này phụ thuộc vào ngữ cảnh, linh hoạt hơn token cố định.

3.  **Xử lý phân cấp:** Kiến trúc có nhiều tầng. Tầng thấp nhất xử lý các byte thô để tạo ra các chunk. Các tầng cao hơn xử lý các biểu diễn của những chunk này, cho phép mô hình học các cấu trúc phức tạp và các phụ thuộc xa.

4.  **Hiệu suất và Độ bền vững:** Bài báo cho thấy H-Net vượt trội hơn các mô hình Transformer dựa trên token có cùng quy mô. Bản chất xử lý ở cấp độ byte giúp nó bền vững trước các lỗi chính tả, từ hiếm và dữ liệu đa ngôn ngữ.

## Phân tích triển khai chính (Kết nối giữa Code và Lý thuyết)
1. Cấu trúc phân cấp đệ quy (`hnet.py`)
Bản chất phân cấp của H-Net được triển khai một cách thanh lịch bằng cách sử dụng đệ quy.
- Trong quá trình khởi tạo `main_network`, nếu tầng hiện tại không phải là tầng cuối cùng (`is_innermost` là False), `self.main_network` sẽ trở thành một thực thể khác của `HNet` với `stage_idx` được tăng lên.
- Vòng đệ quy này dừng lại ở tầng trong cùng, nơi `main_network` là một mô hình `Isotropic` (một chuỗi các khối xử lý tuần tự). Thiết kế này phản ánh hoàn hảo cấu trúc phân cấp lý thuyết.

2. Cơ chế Gộp chuỗi động (`dynamic_chunking.py`)
Đây là phần hiện thực hóa ý tưởng cốt lõi của bài báo.
- `RoutingModule`: Dự đoán ranh giới của chunk. Nó tính toán độ tương đồng cosine (cosine similarity) giữa các trạng thái ẩn của hai token liền kề (`t` và `t+1`). Độ tương đồng thấp (khoảng cách lớn) cho thấy một ranh giới tự nhiên, dẫn đến `boundary_prob` cao.
- `ChunkLayer`: Một lớp đơn giản nhưng hiệu quả, sử dụng `boundary_mask` từ `RoutingModule` để lọc chuỗi, chỉ chuyển các trạng thái ẩn tại các vị trí ranh giới lên tầng phân cấp tiếp theo.
- `DeChunkLayer`: Phần phức tạp nhất, thực hiện việc "trải phẳng" chuỗi đã xử lý. Nó sử dụng một phép quét giống như EMA (Trung bình động hàm mũ) để lan truyền thông tin từ các biểu diễn của chunk trở lại chuỗi có độ dài ban đầu. **Một điểm thú vị là nó tái sử dụng một cách thông minh kernel `mamba_chunk_scan_combined` để thực hiện thao tác này một cách hiệu quả.**

3. Huấn luy���n một quyết định "Cứng" (`hnet.py`)
Việc quyết định một ranh giới chunk là một lựa chọn rời rạc, không khả vi. Vấn đề này được giải quyết bằng **Bộ ước tính truyền thẳng (Straight-Through Estimator - STE)**.
- Class `STE` được định nghĩa để hoạt động như một hàm đồng nhất (identity function) trong quá trình lan truyền ngược (`backward(ctx, grad_output): return grad_output`).
- Điều này _"đánh lừa"_ bộ tối ưu hóa bằng cách cho phép gradient đi qua điểm quyết định cứng như thể nó là một hàm liên tục, giúp `RoutingModule` có thể được huấn luyện end-to-end. Hàm `residual_func` đã áp dụng kỹ thuật này.

4. Kiến trúc linh hoạt, lai (Hybrid) (`config_hnet.py`, `block.py`)
Mô hình không bị trói buộc vào một loại khối xử lý duy nhất. `HNetConfig` cho phép định nghĩa một `arch_layout` để thiết kế các kiến trúc rất linh hoạt.

- `M`: Viết tắt của khối **Mamba**.
- `T`: Viết tắt của khối **Attention** (Transformer).
- **Số liền sau (ví dụ `6` trong `M6`)**: Là số lần lặp lại của khối đó. `M6` nghĩa là 6 khối Mamba nối tiếp nhau.
- **Cấu trúc lồng nhau `[..., [...], ...]`**: Thể hiện các tầng phân cấp. Phần tử ở giữa `[...]` là tầng xử lý sâu hơn.

**Ví dụ 1: `["M6", ["M12"], "M6"]`**
Đây là kiến trúc 2 tầng, chỉ dùng Mamba:
1.  **Tầng 0 (Encoder - `M6`)**: Dữ liệu đầu vào đi qua 6 khối Mamba.
2.  **Gộp chuỗi (Chunking)**: Dữ liệu được nén lại thành một chuỗi ngắn hơn.
3.  **Tầng 1 (Xử lý sâu - `M12`)**: Chuỗi ngắn này được xử lý bởi 12 khối Mamba.
4.  **Trải phẳng (De-chunking)**: Dữ liệu được giải nén về độ dài ban đầu.
5.  **Tầng 0 (Decoder - `M6`)**: Dữ liệu cuối cùng đi qua 6 khối Mamba nữa.

**Ví dụ 2: `["T4M2", ["M10T2"], "M2T4"]`**
Đây là một kiến trúc lai (hybrid) phức tạp hơn, kết hợp cả Attention và Mamba:
1.  **Tầng 0 (Encoder - `T4M2`)**: Dữ liệu đầu vào đi qua 4 khối Attention, rồi tiếp tục qua 2 khối Mamba.
2.  **Gộp chuỗi (Chunking)**.
3.  **Tầng 1 (Xử lý sâu - `M10T2`)**: Chuỗi ngắn được xử lý bởi 10 khối Mamba, rồi tiếp tục qua 2 khối Attention.
4.  **Trải phẳng (De-chunking)**.
5.  **Tầng 0 (Decoder - `M2T4`)**: Dữ liệu cuối cùng đi qua 2 khối Mamba, rồi kết thúc bằng 4 khối Attention.

Thiết kế này cho phép kết hợp sức mạnh của cả hai loại khối: Attention giỏi trong việc nắm bắt các mối quan hệ phức tạp, trong khi Mamba lại rất hiệu quả trong việc xử lý / nén dữ liệu thô.

---
## Hành trình: Từ Byte đến Byte

Để hiểu rõ cách H-Net hoạt động, hãy cùng theo dõi toàn bộ hành trình của dữ liệu, từ lúc là byte đầu vào cho đến khi dự đoán ra byte tiếp theo. Ta sẽ dùng kiến trúc `["M6", ["M12"], "M6"]` làm ví dụ.

### Hành trình đi vào (Encoding & Chunking)
1.  **Byte đầu vào** -> `nn.Embedding` -> Tạo ra vector biểu diễn ban đầu `h_0`.
2.  `h_0` đi qua **Encoder** của tầng ngoài (`M6`) -> Tạo ra `h_encoded`. Vector này chứa thông tin ngữ cảnh cục bộ.
3.  `h_encoded` được `RoutingModule` phân tích để tạo ra `boundary_mask` (mặt nạ ranh giới).
4.  `ChunkLayer` dùng `boundary_mask` để nén `h_encoded` thành một chuỗi ngắn hơn `h_chunked`.
5.  `h_chunked` được đưa vào tầng trong cùng (`M12`) để xử lý. Kết quả là `z`, một vector chứa thông tin ngữ cảnh ở mức độ rất cao và trừu tượng.

### Hành trình đi ngược ra (De-chunking & Decoding)
`z` không trực tiếp dự đoán byte. Nó bắt đầu một hành trình đi ngược ra để làm giàu thông tin cho các t��ng bên ngoài.
6.  **DeChunkLayer (Dùng EMA)**: Lớp này nhận `z` và `boundary_mask`. Nó dùng cơ chế EMA để "trải" thông tin trừu tượng trong `z` ra lại thành một chuỗi có độ dài đầy đủ, gọi là `h_dechunked`.
7.  **Kết nối phần còn lại (Residual Connection)**: Đây là bước cực kỳ quan trọng. Mô hình kết hợp `h_dechunked` (thông tin trừu tượng, mượt mà) với `h_encoded` (thông tin chi tiết, nguyên bản được giữ lại từ trước). Việc này cho phép mô hình có được cả hai:
  - cái nhìn tổng quan từ tầng sâu VÀ
  - chi tiết cụ thể từ tầng nông.
8.  **Decoder của tầng ngoài (`M6`)**: Vector kết hợp ở trên tiếp tục đi qua các khối xử lý cuối cùng của tầng ngoài (`M6`) để hòa trộn hai nguồn thông tin lại với nhau. Kết quả là `h_final`.
9.  **Chiếu ra Byte (LM Head)**: **Chỉ bây giờ**, `h_final` mới được đưa vào lớp `lm_head`. Lớp này chiếu `h_final` thành một vector `logits` có 256 chiều (tương ứng với 256 giá trị byte).
10. **Dự đoán**: Một hàm `softmax` được áp dụng lên `logits` để tạo ra phân phối xác suất, và byte có xác suất cao nhất được chọn làm dự đoán.

---
## Phân tích sâu: Cơ ch�� Gộp chuỗi động (`hnet/dynamic_chunking.py`)
File `dynamic_chunking.py` là trái tim của H-Net. Nó bao gồm 3 thành phần chính hoạt động như một dây chuyền:

1.  **`RoutingModule`**: **Người Ra Quyết Định** - Quyết định xem vị trí nào nên là ranh giới của một "chunk".
2.  **`ChunkLayer`**: **Người Thực Thi** - Dựa trên quyết định, nó thực sự lọc và tạo ra chuỗi ngắn hơn.
3.  **`DeChunkLayer`**: **Người Tái Tạo** - Sau khi chuỗi ngắn được xử lý, nó sẽ "trải" thông tin trở lại chuỗi dài ban đầu.

### 1. `RoutingModule` - Người Ra Quyết Định
**Mục đích:** Trả lời câu hỏi: "Tại mỗi vị trí trong chuỗi, xác suất để vị trí này là điểm kết thúc của một chunk là bao nhiêu?"

- **Ý tưởng cốt lõi:** Nếu hai token liền kề nhau có ý nghĩa rất khác nhau, thì khả năng cao giữa chúng là một ranh giới. Ngược lại, nếu chúng rất giống nhau (ví dụ: "c" và "h" trong "ch"), chúng nên thuộc cùng một chunk.
- **Triển khai trong code:**
    1.  Nó sử dụng hai lớp `Linear` (`q_proj_layer`, `k_proj_layer`) để chiếu (project) các vector `hidden_state` của mỗi token.
    2.  Sau đó, nó tính **độ tương đồng cosine (cosine similarity)** giữa mỗi token và token ngay sau nó.
    3.  **Bước quan trọng nhất:** Nó chuyển đổi độ tương đồng (từ -1 đến 1) thành xác suất ranh giới (từ 0 đến 1) bằng công thức `(1 - cos_sim) / 2`.
        - Nếu `cos_sim` = 1 (rất giống nhau) -> xác suất = 0 (không phải ranh giới).
        - Nếu `cos_sim` = -1 (rất khác nhau) -> xác suất = 1 (chắc chắn là ranh giới).
- **Điểm thông minh:**
    - Nó **ép xác suất của token đầu tiên luôn là 1.0**. Điều này đảm bảo rằng mọi chuỗi luôn bắt đầu bằng một chunk.
    - Kết quả cuối cùng là `boundary_mask` - một chuỗi boolean (`True`/`False`) cho biết vị trí nào được chọn làm ranh giới.

### 2. `ChunkLayer` - Người Thực Thi
**Mục đích:** Nhận vào `boundary_mask` và chuỗi đầy đủ, sau đó tạo ra một chuỗi mới chỉ chứa các token được đánh dấu là ranh giới.

- Lớp này cực kỳ đơn giản nhưng hiệu quả. Nó sử dụng **lập chỉ mục bằng mảng boolean (boolean array indexing)**.
- Dòng code cốt lõi là `hidden_states[boundary_mask]` sẽ tạo ra một tensor mới, chỉ chứa các token từ `hidden_states` mà tại đó `boundary_mask` có giá trị là `True`.
- Kết quả là một chuỗi ngắn hơn nhiều, sẵn sàng để được đưa vào tầng xử lý sâu hơn của H-Net.

### 3. `DeChunkLayer` - Người Tái Tạo
**Mục đích:** Sau khi tầng sâu hơn đã xử lý chuỗi ngắn, lớp này cần phải "phân phối" thông tin đó trở lại chuỗi có độ dài ban đầu.

- **Ý tưởng cốt lõi:** Mỗi token trong chuỗi gốc sẽ nhận thông tin từ chunk gần nhất mà nó thuộc về, được thực hiện bằng một phép toán tương tự như **Trung bình động hàm mũ (EMA)**.
- **Triển khai thông minh:**
    1.  Thay vì tự viết một kernel EMA, các tác giả đã **tái sử dụng kernel `mamba_chunk_scan_combined`** từ Mamba2, một kernel được tối ưu hóa ở mức độ rất cao.
    2.  Họ thiết lập các tham số đầu vào cho kernel Mamba (`A`, `dt`, `b`, `c`) theo một cách đặc biệt để nó thực hiện chính xác phép toán EMA, mô phỏng hành vi phân rã theo hàm mũ.
    3.  Sau khi kernel Mamba thực hiện phép quét, nó sử dụng `torch.gather` để "cắm" các giá trị đã được làm mịn này trở lại đúng vị trí của chúng trong chuỗi có độ dài ban đầu.

---
## Phân tích sâu: `DeChunkLayer` và vai trò của EMA

`DeChunkLayer` là một trong những thành phần thông minh và quan trọng nhất c��a H-Net. Nó giải quyết một bài toán khó: Làm thế nào để "trải" hoặc "lan tỏa" thông tin từ một vài vector biểu diễn chunk (đã được xử lý ở tầng sâu) ra tất cả các byte con mà nó đại diện một cách mượt mà?

### Vấn đề cần giải quyết
- **Đầu vào:** Một chuỗi ngắn gồm các vector `z` của chunk (ví dụ: 3 vector cho 3 chunk `[Học]`, `[chuỗi]`, `[phân cấp]`).
- **Đầu ra cần tạo:** Một chuỗi dài có cùng độ dài với chuỗi byte ban đầu (ví dụ: 18 vector cho 18 ký tự), trong đó mỗi byte đều có một vector biểu diễn mới, đã được làm giàu thông tin.

Một cách tiếp cận đơn giản là sao chép vector của chunk cho tất cả các byte con. Tuy nhiên, cách này tạo ra các "vách đá" thông tin đột ngột ở ranh giới chunk và làm mất thông tin về vị trí tương đối của byte bên trong chunk.

### Giải pháp thanh lịch với EMA (Trung bình động hàm mũ)
`DeChunkLayer` sử dụng một cơ chế tương tự EMA để giải quyết vấn đề này một cách hoàn hảo. Hãy tưởng tượng EMA như một cách "làm mịn" hoặc "lan tỏa" giá trị theo thời gian.

1.  **Khi gặp một ranh giới chunk (ví dụ, byte 'c' trong "chu���i"):**
    - Xác suất ranh giới `boundary_prob` (đóng vai trò là cổng `alpha` trong EMA) sẽ cao.
    - Công thức cập nhật sẽ ưu tiên mạnh mẽ cho giá trị mới (vector `z` của chunk `[chuỗi]`). Vector của byte 'c' sẽ gần như là vector của chunk này.

2.  **Khi ở bên trong một chunk (ví dụ, byte 'h' sau 'c'):**
    - `boundary_prob` (`alpha`) sẽ rất thấp.
    - Công thức sẽ ưu tiên mạnh mẽ cho giá trị cũ (vector của byte 'c' ngay trước đó). Vector của 'h' sẽ là một phiên bản "sao chép có giảm nhẹ" từ vector của 'c'. Quá trình này tiếp tục cho các byte tiếp theo, tạo ra một dòng chảy thông tin.

**Kết quả của việc dùng EMA:**
- **Chuyển tiếp mượt mà:** Thay vì một "vách đá", thông tin từ một chunk sẽ "phai" hoặc "lan tỏa" dần qua các byte bên trong nó.
- **Tạo nhận thức về vị trí:** Do hiệu ứng "phai" dần này, vector của byte ở đầu chunk sẽ hơi khác một chút so với vector của byte ở cuối chunk. Điều này giúp mô h��nh giữ lại được thông tin vị trí tương đối bên trong chunk.

---
# Đánh giá Tổng quan: Sức mạnh và Tiềm năng của H-Net

**Nói một cách ngắn gọn: H-Net không chỉ là một cải tiến nhỏ, mà là một sự thay đổi trong tư duy nền tảng về cách mô hình xử lý chuỗi thông tin. Sức mạnh của nó đến từ việc giải quyết một vấn đề gốc rễ, và tiềm năng của nó là vô cùng to lớn vì nó mở ra những hướng đi mới.**

## I. Sức mạnh Hiện tại (Những gì đã được chứng minh)

1.  **Giải quyết "Tội lỗi Nguyên thủy" của Tokenization:**
    *   Hầu hết các mô hình ngôn ngữ hiện đại (như GPT, Llama) đều bị phụ thuộc vào một bước tiền xử lý gọi là "tokenization" - chia câu thành các mảnh nhỏ dựa trên một bộ từ điển cố định. Điều này giống như việc bắt một đứa trẻ chỉ được đọc những từ có trong từ điển, nếu gặp từ mới, nó sẽ bối rối và phải bẻ từ đó ra thành các mảnh vô nghĩa.
    *   **Sức mạnh của H-Net:** Nó vứt bỏ bộ từ điển cố định đó. Thay vào đó, nó **tự học** cách nhóm các ký tự lại thành các đơn vị có ý nghĩa (từ, cụm từ) m���t cách linh động, tùy thuộc vào ngữ cảnh. Điều này giúp nó xử lý ngôn ngữ phức tạp, từ lóng, thuật ngữ chuyên ngành, và thậm chí cả các loại dữ liệu không phải văn bản (như DNA, code) một cách tự nhiên và hiệu quả hơn nhiều.

2.  **Hiệu quả tính toán vượt trội:**
    *   Các mô hình Transformer truyền thống có chi phí tính toán tăng theo cấp số nhân với độ dài chuỗi (`O(N^2)`), khiến chúng rất khó xử lý văn bản dài.
    *   **Sức mạnh của H-Net:** Bằng cách gộp các token cấp thấp thành các token cấp cao hơn, H-Net giảm đáng kể độ dài của chuỗi ở các tầng xử lý sâu hơn. Điều này giúp chi phí tính toán giảm xuống gần như tuyến tính (`O(N)`), cho phép nó xử lý các chuỗi dài hơn rất nhiều (ví dụ: cả một cuốn sách) với chi phí thấp hơn.

3.  **Mô hình hóa cấu trúc phân cấp của thế giới:**
    *   Thế giới của chúng ta vốn có cấu trúc phân cấp: Ký tự -> Từ -> Câu -> Đoạn văn -> Ý tưởng. Tương tự, trong hình ảnh: Pixel -> Đường nét -> Vật thể -> Khung cảnh.
    *   **Sức mạnh của H-Net:** Kiến trúc phân cấp của nó phản ánh chính xác cấu trúc này. Nó không chỉ xử lý một chuỗi phẳng các token, mà nó xây dựng một biểu diễn thông tin ngày càng trừu tượng và cô đọng hơn ở mỗi tầng. Đây là một cách tiếp cận tự nhiên và mạnh mẽ hơn nhiều.

## II. Tiềm năng Tương lai (Những gì nó hứa hẹn)

1.  **Kiến trúc nền tảng cho Đa phương thức (Multimodality):**
    *   Bài blog "THE FUTURE" đã chỉ ra rất rõ điều này. Cùng một kiến trúc H-Net có thể được áp dụng cho nhiều loại dữ liệu khác nhau mà không cần thay đổi nhiều.
    *   **Tiềm năng:**
        *   **Thị giác (Vision):** H-Net có thể học cách gộp các pixel thành các cạnh, các cạnh thành các bộ phận của vật thể, và các vật thể thành một khung cảnh hoàn chỉnh.
        *   **Âm thanh (Audio):** Nó có thể học cách gộp các mẫu âm thanh (samples) thành các âm vị (phonemes), các âm vị thành từ, và các từ thành câu nói.
    *   Điều này mở ra khả năng tạo ra một mô hình duy nhất có thể "nhìn", "nghe", và "đọc" một cách thống nhất.

2.  **Một bước tiến tới khả năng Suy luận (Reasoning):**
    *   Khả năng suy luận đòi hỏi việc hiểu các khái niệm ở nhiều mức độ trừu tượng khác nhau.
    *   **Tiềm năng:** Bằng cách tạo ra các biểu diễn phân cấp, H-Net có thể học được các khái niệm từ cấp thấp (một "cái chân") đến cấp cao (một "con chó đang chạy"). Việc có thể thao tác trên các khái niệm trừu tượng cấp cao này là một bước đệm quan trọng để xây dựng các "World Model" - các mô hình thực sự hiểu và suy luận về thế giới.

3.  **Trở thành "Mô hình Chuỗi Tổng quát" (General Sequence Model):**
    *   **Tiềm năng:** Vì không bị ràng buộc bởi tokenization, H-Net có tiềm năng trở thành một kiến trúc phổ quát cho **bất kỳ loại dữ liệu tuần tự nào**, từ mã nguồn lập trình, chuỗi gen, dữ liệu tài chính theo thời gian, cho đến nốt nhạc.

---
# Bối cảnh rộng hơn: Sự đối đầu giữa SSM và Transformer

Để thực sự hiểu tại sao H-Net lại là một hướng đi quan trọng, chúng ta cần đặt nó vào bối cảnh của cuộc "đối đầu" giữa hai trường phái kiến trúc: **SSM (mà Mamba là đại diện) và Transformer**. Bài blog "On the Tradeoffs of SSMs and Transformers" của Albert Gu đã đưa ra một góc nhìn triết lý rất sâu sắc về vấn đề này.

## 1. Hai triết lý xử lý thông tin hoàn toàn khác nhau
Sự khác biệt cốt lõi không nằm ở công thức, mà ở cách chúng "ghi nhớ" quá khứ:

*   **Transformers giống như một "Cơ sở dữ liệu" (Database):**
    * Nó lưu lại một bản sao (cache) của **mọi token** nó đã thấy trong KV cache.
    * **Điểm mạnh:** Khả năng truy hồi thông tin (recall) hoàn hảo và chính xác đến từng chi tiết.
    * **Điểm yếu:** Kích thước bộ nhớ tăng tuyến tính, và nó bị "trói buộc" vào các token được cung cấp.

*   **SSMs (như Mamba/H-Net) giống như một "Bộ não" (Brain):**
    * Nó liên tục **nén (compress)** toàn bộ lịch sử vào một trạng thái ẩn có kích thước **không đổi**.
    * **Điểm mạnh:** Xử lý online hiệu quả, có "bộ nhớ" dài vô tận (dù mờ ảo), và có thiên hướng tự học cách trừu tượng hóa thông tin.
    * **Điểm yếu:** Khả năng truy hồi thông tin chi tiết, chính xác sẽ kém hơn Transformer.

## 2. "Tội lỗi" của Tokenization và vai trò của H-Net
Đây là luận điểm mạnh mẽ nhất, giải thích tại sao các mô hình như H-Net lại cần thiết:

*   **Transformer phụ thuộc vào `Token "có ý nghĩa`":** Transformer hoạt động tốt nhất khi được cung cấp các token đã được tiền xử lý để có "ý nghĩa" ở một "mức độ trừu tượng phù hợp". Thiên hướng của Attention là muốn "chú ý" đến một vài token cụ thể. Nếu các token là vô nghĩa (ví dụ: từng ký tự một), Transformer sẽ bị "nhiễu" và hoạt động kém hiệu quả.
*   **SSMs tỏa sáng khi không có Tokenization:** Ngược lại, vì các mô hình này có thiên hướng "nén" thông tin, chúng rất phù hợp để xử lý dữ liệu thô, có độ phân giải cao (như ký tự, byte, DNA). **Cơ chế "Dynamic Chunking" của H-Net chính là hiện thực hóa của triết lý này**: nó tự động học cách nhóm các đơn vị vô nghĩa (byte) thành các khái niệm có ý nghĩa hơn (chunk), thay vì dựa vào một bộ token cố định.

## 3. Kết luận: Tại sao H-Net quan trọng?
H-Net không chỉ là một kiến trúc mới. Nó đại diện cho một triết lý khác biệt, giải quyết những điểm yếu cố hữu của Transformer:

1.  **Giải phóng khỏi Tokenization:** Bằng cách học trực tiếp từ byte và tự tạo ra các "chunk" có ý nghĩa, H-Net đi theo đúng tinh thần của deep learning là học end-to-end, mở ra tiềm năng cho dữ liệu đa ngôn ngữ, đa phương thức một cách tự nhiên.

2.  **Thiên hướng Nén và Trừu tượng hóa:** Việc ép mô hình phải nén thông tin (từ byte thành chunk, từ chunk thành biểu diễn cấp cao hơn) có thể chính là một "tính năng", buộc nó phải học các quy luật và cấu trúc cơ bản của dữ liệu, thay vì chỉ ghi nhớ bề mặt. Đây có thể là một bước tiến quan trọng hướng tới khả năng suy luận thực sự.
