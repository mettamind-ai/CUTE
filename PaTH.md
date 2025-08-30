# Positional Encodings and PaTH Attention
https://www.youtube.com/watch?v=l6_fdwRvMPk

## Động Lực Cho Mã Hóa Vị Trí
Khi transformers được phát minh, chúng chủ yếu được sử dụng để mô hình hóa hai chiều và cơ chế self-attention không có mask nhân quả, xử lý các token đầu vào như một tập không có thứ tự. Nếu không có positional embedding, cơ chế attention của transformer sẽ hoạt động giống như mô hình bag-of-words, không có cảm nhận về thứ tự từ. Đây là lý do tại sao chúng ta muốn có positional embedding ngay từ đầu.

Đây là một ví dụ trực quan: chúng ta có hai câu "con mèo ngồi trên tấm thảm" và "tấm thảm ngồi trên con mèo". Nếu không có positional embedding, mô hình không thể phân biệt giữa hai câu này khi không có mask nhân quả, và đây là điều kiện tiên quyết.

![](https://pbs.twimg.com/media/Gzg_VcUbkAELrAP?format=jpg&name=medium)

## Mã Hóa Vị Trí Tuyệt Đối và Tương Đối
Trong bài báo transformer gốc, họ đã thêm một mã hóa vị trí tuyệt đối rất đơn giản, sử dụng hàm sin và cos như positional encoding. Các hàm này có tính chất tuần hoàn nên có thể mã hóa một số thông tin vị trí. Sau đó, các nhà nghiên cứu nhận ra rằng mã hóa vị trí tuyệt đối không hoàn hảo, có nhiều hạn chế. Ví dụ, nội dung tương quan cao với vị trí tuyệt đối, không trực tiếp mô hình hóa vị trí tương đối giữa các từ. Vì vậy, sau này các nhà nghiên cứu tập trung vào mã hóa vị trí tương đối, và hôm nay chúng ta sẽ tập trung vào rotary position embedding (RoPE), một phương pháp phổ biến nhất.

## Tổng Quan Về Rotary Position Embedding
Đầu tiên, chúng ta có input embedding và RoPE chia các kênh đầu vào thành các cặp kênh khác nhau, mỗi cặp có hai chiều. Mỗi kênh có một góc tần số và sử dụng position ID tuyệt đối để tính toán phép quay dựa trên cả góc và position ID. Sau phép quay, chúng ta có các queries và keys đã được quay để tính toán attention logits. Các thao tác này được thực hiện độc lập cho từng cặp kênh, `không có tương tác giữa các cặp`. (mỗi cặp kênh tạo một siêu mặt phẳng và phép quay nằm trên mặt phẳng đó)

## Ma Trận Quay
Trước khi hiểu cơ chế RoPE, chúng ta cần ôn lại về ma trận quay. Ma trận quay hai chiều rất đơn giản, nếu bạn muốn quay một vector một góc θ (theta), bạn có thể sử dụng ma trận này. Ma trận quay có nhiều tính chất tốt:

1. Tính chất lũy thừa: khi tính tích lũy của ma trận quay, bạn không cần tính từng tích một mà có thể tính góc tích lũy và chỉ tính ma trận một lần.

2. Ma trận quay là phép biến đổi trực giao, nghịch đảo của nó bằng chuyển vị.

3. Có thể sử dụng góc âm để đảo ngược phép quay.

4. Tổ hợp nhiều ma trận quay rất đơn giản, chỉ cần cộng các góc và tính ma trận cuối cùng.

## RoPE là mã hoá vị trí tương đối
RoPE hoạt động trên các cặp chiều độc lập. Nếu input có chiều d, chúng ta có d/2 cặp kênh, mỗi cặp có góc quay khác nhau, có thể biểu diễn dưới dạng ma trận khối đường chéo. Các khối này độc lập với nhau và bảo toàn các tính chất của ma trận quay.

![](https://pbs.twimg.com/media/GzjucILbkAQZIiI?format=jpg&name=large)

Chúng ta có thể biểu diễn cơ chế RoPE dưới dạng ma trận, sử dụng position ID tuyệt đối làm số mũ cho ma trận quay. Mặc dù trông giống mã hóa tuyệt đối, nhưng thực chất nó mã hóa thông tin vị trí tương đối. Khi tính tích vô hướng giữa query và key, chúng ta có thể thấy rõ sự phụ thuộc vào hiệu vị trí tương đối giữa query và key. Mặc dù sử dụng position ID tuyệt đối, nhưng kết quả cuối cùng phụ thuộc vào sự khác biệt vị trí tương đối.

![](https://pbs.twimg.com/media/Gzjp9EnaoAA8JfK?format=jpg&name=large)

NOTE: lý do chỉ dùng d/2 cặp kênh là để giữ cấu trúc đường chéo khối và nhờ đó có được các thuộc tính tốt và nhất là tính chất không phụ thuộc vào vị trị tuyệt đối $(R^i)^\top R^j = R^{\,j-i}$

## Phân Tách Vai Trò Các Kênh Tần Số Trong RoPE

Một đặc điểm rất thú vị của RoPE là sau khi thực hiện phép quay và lấy tích vô hướng giữa query và key, kết quả thu được là một mã hóa vị trí tương đối. Điều này cho phép chúng ta chỉ cần thực hiện một phép biến đổi vị trí tuyệt đối đơn giản, sau đó dùng tích vô hướng để mã hóa thông tin vị trí tương đối một cách hiệu quả và rẻ về mặt tính toán. Tính chất này đã được ứng dụng rộng rãi trong nhiều mô hình mã nguồn mở như Llama, Qwen, v.v.

Như đã đề cập, `mỗi cặp kênh trong embedding sẽ có một góc quay (theta) khác nhau`. Dựa vào giá trị tần số của các kênh này, ta có thể chia chúng thành hai nhóm chính:

### 1. Kênh tần số cao (High Frequency Channels)
Các kênh này có giá trị theta lớn, tức là góc quay lớn, nên phép quay diễn ra rất nhanh. Một bài báo nổi bật về chủ đề này là "Round and Round We Go: What Makes Rotary Positional Encodings Useful?", trong đó phân tích sâu về chức năng của các kênh tần số khác nhau. Theo đó, các kênh tần số cao chủ yếu dùng để mã hóa các mẫu vị trí (positional patterns). Ví dụ, nếu muốn mô hình nhận biết các từ gần nhau nhất, nó sẽ dựa vào các kênh tần số cao này. Đặc biệt, trong các mô hình autoregressive, bản thân causal mask cũng đã cung cấp một phần thông tin vị trí, nhưng các phương pháp positional encoding khác như "Nope" không có cơ chế này để phát hiện các mẫu vị trí như RoPE.

### 2. Kênh tần số thấp (Low Frequency Channels)
Ngược lại, các kênh tần số thấp có góc quay nhỏ, phép quay diễn ra chậm. Các kênh này chủ yếu dùng để mã hóa ý nghĩa ngữ nghĩa (semantic meaning) vì phép quay chậm nên ảnh hưởng đến giá trị tích vô hướng là rất nhỏ. Do đó, giá trị attention chủ yếu dựa vào độ tương đồng giữa query và key, giúp mô hình học được thông tin ngữ nghĩa tốt hơn. Trong khi đó, các kênh tần số cao do quay nhanh nên giá trị tích vô hướng thay đổi mạnh, chủ yếu dùng để mã hóa thông tin vị trí.

Sự phân chia này giúp RoPE cân bằng hiệu quả giữa việc mã hóa thông tin vị trí và thông tin ngữ nghĩa.

## Hạn Chế Của RoPE và Giải Pháp

Mặc dù RoPE là một phương pháp mã hóa vị trí tương đối rất hiệu quả, nó vẫn tồn tại một số hạn chế. Một vấn đề lớn là khả năng mở rộng độ dài chuỗi (extrapolation). Ví dụ, `nếu mô hình được huấn luyện với chuỗi có độ dài tối đa 4K`, khi đánh giá với chuỗi dài hơn, perplexity sẽ tăng rất nhanh và có thể tăng vọt lên giá trị rất lớn. Đây là vấn đề phổ biến của RoPE: `nó không mở rộng tốt ra ngoài phạm vi độ dài đã được huấn luyện`.

Để khắc phục, các nhà nghiên cứu đã đề xuất nhiều giải pháp, trong đó có "positional interpolation". Phương pháp này thực hiện việc co giãn (scaling) góc quay để `đảm bảo` rằng tích giữa position ID tuyệt đối và góc quay vẫn nằm trong phạm vi giá trị đã xuất hiện trong quá trình huấn luyện, `tránh hiện tượng "out of distribution"`. Tuy nhiên, positional interpolation truyền thống lại quá đơn giản: nó co giãn đồng đều cho tất cả các kênh mà không phân biệt tần số. Điều này không tối ưu, vì như đã nói, các kênh tần số cao quay rất nhanh và đã trải qua nhiều chu kỳ trong quá trình huấn luyện nên không gặp vấn đề out of distribution, còn các kênh tần số thấp thì lại rất nhạy cảm với việc co giãn.

FAQ

**1. Tại sao perplexity của mô hình lại tăng vọt khi vượt quá context window 4K?**

Nguyên nhân phổ biến là với một số kênh quay chậm (tần số thấp), mô hình chưa từng thấy các góc quay lớn này trong quá trình huấn luyện. Khi extrapolate, các góc quay này vượt ra ngoài phạm vi đã học, dẫn đến mô hình không biết xử lý như thế nào, gây ra hiện tượng out of distribution.

**2. Có phải các chiều đầu của embedding chủ yếu mã hóa vị trí, còn các chiều sau mã hóa ngữ nghĩa không?**

Đúng vậy, các chiều đầu (tương ứng với kênh tần số cao) thực hiện phép quay nhanh, nên chủ yếu dùng để mã hóa thông tin vị trí tương đối. Các chiều sau (kênh tần số thấp) quay chậm, ảnh hưởng đến tích vô hướng nhỏ, nên chủ yếu dùng để mã hóa thông tin ngữ nghĩa. Nhờ vậy, RoPE cho phép mô hình đồng thời học được cả thông tin vị trí và ngữ nghĩa trong embedding.

## Phương Pháp Nội Suy Vị Trí: NTK, NTK-by-parts và YARN

Sau khi phân tích các kênh tần số trong RoPE, một vấn đề lớn là khi muốn mở rộng độ dài chuỗi (extrapolation), ta cần nội suy (interpolate) hoặc ngoại suy (extrapolate) góc quay cho các kênh. Tuy nhiên, phương pháp nội suy vị trí truyền thống lại áp dụng đồng đều cho tất cả các kênh, không xét đến độ nhạy khác nhau giữa các kênh tần số cao và thấp.

### NTK Interpolation

**NTK (Neural Tangent Kernel)** cho ta cách nhìn “mạng nơ-ron cực rộng + học bằng gradient” như **một máy lọc tín hiệu cố định**: mỗi “kiểu mẫu” trong dữ liệu (mode) được khuếch đại/bị làm yếu với cường độ riêng. Mode bị khuếch đại mạnh học nhanh; mode bị làm yếu học chậm.

Hãy tưởng tượng hàm mục tiêu bạn muốn học là một bản nhạc gồm bass (thấp tần) + mid + treble (cao tần).
Khi bạn dùng GD để huấn luyện một mạng rất rộng, hệ thống hành xử gần như một **bàn trộn tần số cố định**:

* Mỗi dải tần có **một cần gạt** (một mode riêng).
* Vị trí cố định của cần gạt chính là **riêng trị (eigenvalue)** của mode đó.
* Cần gạt cao (riêng trị lớn) ⇒ tín hiệu ở dải đó **lên rất nhanh** (học nhanh).
* Cần gạt thấp (riêng trị nhỏ) ⇒ tín hiệu **lên rất chậm** (học khó).

Khoa học phía sau “bàn trộn” ấy chính là NTK.

Khi mạng **rất rộng**, trong giai đoạn đầu (và thường suốt quá trình nếu đủ rộng), mô hình hầu như **không đổi “hình dạng”**, chỉ **tinh chỉnh tuyến tính** xung quanh điểm khởi tạo. Khi đó, học bằng GD ≈ **hồi quy kernel** với một kernel đặc trưng của chính mạng tại khởi tạo, gọi là **Neural Tangent Kernel** $K$.

* Toán tử “nhân kernel” này có các **hướng riêng** $\phi_k$ (những kiểu mẫu cơ bản), với **riêng trị** $\lambda_k$.
* Sai số trên mỗi hướng giảm theo mũ

Nếu đầu vào chỉ là **một con số** (vd: position ID 1,2,3,…), các **kiểu mẫu tự nhiên** của dữ liệu theo vị trí chính là **sóng sin/cos** ở các tần số khác nhau (Fourier).

* **Thấp tần**: biến thiên chậm theo vị trí (chu kỳ dài).
* **Cao tần**: biến thiên rất nhanh (chu kỳ ngắn).

Đa số kernel NTK “mặc định” là **trơn và làm mượt** ⇒ phổ Fourier của chúng **rụng nhanh ở tần số cao**. Nói bằng ngôn ngữ equalizer: **các cần gạt cao tần bị để rất thấp** ⇒ **$\lambda_{\text{cao tần}}$ nhỏ** ⇒ **học chậm**. Đây là hiện tượng **spectral bias** (thiên lệch về thấp tần).

**Tại sao “đầu vào 1 chiều” làm mọi thứ tệ hơn?**
Vì bạn chỉ có **một trục** để mô tả mọi biến thiên theo vị trí. Kernel trơn trên trục này càng giống **bộ lọc thông thấp**; mode cao tần bị bóp nghẹt mạnh hơn, nên riêng trị của chúng càng nhỏ → học càng chậm.

**RoPE/PE giúp gì trong bối cảnh NTK?**
**Sinusoidal PE / RoPE** biến một ID 1D thành **vector nhiều kênh** chứa sẵn các **thành phần sin–cos ở nhiều tần số**.

Dịch sang góc nhìn NTK:
* Bạn **bổ sung sẵn** các “cần gạt” cho những dải tần quan trọng ngay trong **đặc trưng đầu vào** thay vì bắt mạng tự “bịa ra”.
* Khi đó, các mode cao tần trong mục tiêu **trở nên tuyến tính sẵn** trong đặc trưng ⇒ **riêng trị hiệu dụng tăng**, học **nhanh** và **ổn định** hơn.

RoPE còn có thêm cấu trúc **phụ thuộc vị trí tương đối** (nhờ tính chất $(R^i)^\top R^j = R^{j-i}$), rất hợp với attention.

---

**“NTK-aware scaling” là gì khi kéo dài context?**

Khi bạn thay đổi thang tần số (ví dụ mở rộng context), nếu bạn **phóng to/thu nhỏ sai cách**, bạn sẽ **đẩy mô hình vào vùng tần số** mà nó **chưa từng được “nâng cần gạt” lúc train** → dễ **mất ổn định** hoặc **suy giảm chất lượng**.

**NTK-aware scaling** là cách **điều chỉnh công thức PE/RoPE** để **giữ phổ tần “trông giống” lúc huấn luyện**, tức giữ các $\lambda_k$ hiệu dụng của những mode quan trọng gần như cũ → mô hình suy luận tốt hơn ở context dài.

**Hạn chế của cách nhìn NTK**

* NTK mô tả tốt khi mạng **rất rộng** và học ở vùng **tuyến tính hóa quanh khởi tạo**.
* Với mạng vừa/nhỏ, fine-tune lớn, hoặc kỹ thuật học phức tạp, mô hình có thể **ra khỏi vùng NTK**, lúc đó phân tích kernel thuần túy kém chính xác.

* **Riêng trị (eigenvalue) là gì?** Hệ số khuếch đại của một kiểu mẫu riêng (mode).
* **Vì sao cao tần khó học?** Kernel trơn làm **suy hao** mạnh ở tần số cao ⇒ riêng trị nhỏ ⇒ cập nhật chậm.
* **Làm sao cải thiện?** Dùng **Fourier features / Sinusoidal PE / RoPE**, hoặc **NTK-aware scaling** khi mở context.
* **Có phải cứ thêm nhiều tần số là tốt?** Không luôn. Quá nhiều cao tần có thể gây nhiễu/overfit; cần cân bằng với nhiệm vụ và dữ liệu.

Ví dụ trực quan: Giả sử bạn muốn học $y(i)=\sin(10\,i)$ từ dữ liệu $i=1,2,\dots$.

* **Không PE**: đầu vào chỉ là $i$. Kernel trơn coi $y$ là **cao tần** ⇒ **lên rất chậm**.
* **Có Fourier/ RoPE**: đầu vào chứa $\sin(10\,i),\cos(10\,i)$ (và các tần số khác). Bây giờ **y** gần như **tuyến tính** theo đặc trưng ⇒ **học cực nhanh**.

---

NTK (Neural Tangent Kernel) là một lý thuyết cho thấy rằng các thành phần tần số cao trong embedding rất khó học khi đầu vào có chiều thấp (ví dụ, position ID chỉ là một số nguyên). Điều này đúng với positional encoding: position ID là một chiều, nhưng ta dùng vector nhiều chiều để mã hóa. Kết quả là, các kênh tần số cao (high frequency) rất nhạy cảm và khó học, dễ bị phá vỡ nếu ta thay đổi góc quay quá nhiều khi nội suy. Ngược lại, các kênh tần số thấp (low frequency) ít bị ảnh hưởng bởi việc thay đổi góc quay.

Vì vậy, NTK interpolation đề xuất rằng:
- **Kênh tần số cao**: Giữ nguyên góc quay khi nội suy, không thay đổi (vì rất nhạy cảm, dễ làm mất thông tin vị trí cục bộ).
- **Kênh tần số thấp**: Có thể nội suy góc quay thoải mái, vì ảnh hưởng đến ý nghĩa ngữ nghĩa là chủ yếu, và góc quay thay đổi ít.

NTK sử dụng một hàm mũ (exponential function) để tính hệ số co giãn (scaling factor) cho từng kênh dựa trên chỉ số kênh (channel ID). Hệ số này nhân với góc quay để ra góc quay mới khi mở rộng chuỗi.

**Trực giác chính:**  
- Kênh tần số cao mã hóa thứ tự cục bộ, cú pháp, thay đổi góc quay sẽ làm méo thông tin, giảm độ chính xác cho các ngữ cảnh ngắn.
- Kênh tần số thấp chủ yếu mã hóa ý nghĩa ngữ nghĩa, có thể nội suy mà không ảnh hưởng nhiều.

Vấn đề "perplexity tăng vọt" khi vượt quá context window là do các kênh tần số thấp chưa từng gặp các góc quay lớn trong quá trình huấn luyện, dẫn đến hiện tượng out-of-distribution. Do đó, ta cần nội suy để đảm bảo góc quay luôn nằm trong phạm vi đã học.

### NTK-by-parts

NTK-by-parts là một biến thể của NTK, thay vì dùng hàm mũ trơn, nó dùng hàm bậc thang (stairwise/step function):
- Một số kênh giữ nguyên (scaling = 0), một số kênh nội suy hoàn toàn (scaling = 1), các kênh ở giữa thì nội suy tuyến tính.
- Kênh tần số rất cao: chỉ ngoại suy, không nội suy.
- Kênh tần số rất thấp: chỉ nội suy, không ngoại suy.
- Kênh ở giữa: vừa nội suy vừa ngoại suy.

Cách này giúp kiểm soát tốt hơn việc thay đổi góc quay cho từng nhóm kênh, phù hợp với đặc tính của từng loại tần số.

### YARN

YARN là một mở rộng khác của RoPE, kết hợp NTK-by-parts với một hệ số nhiệt độ (temperature scaling) t, dựa trên tỉ lệ giữa độ dài chuỗi mới và chuỗi huấn luyện. Hệ số này điều chỉnh thêm góc quay khi mở rộng chuỗi. Tuy paper không giải thích sâu, nhưng blog của Jennings cung cấp nhiều trực giác về ý nghĩa của temperature scaling, liên quan đến entropy của attention logits.

**Tóm tắt trực giác:**  
- Hãy hình dung RoPE như một cây thước đo độ dài chuỗi, nhưng thước chỉ dài bằng context window huấn luyện.
- Khi chuỗi dài hơn, ta cần "kéo giãn" thước, nhưng không đều cho mọi kênh:  
    - Kênh tần số cao như độ phân giải cao, cần giữ nguyên để không mất chi tiết vị trí.
    - Kênh tần số thấp như độ phân giải thấp, có thể co giãn mà không ảnh hưởng nhiều.
- NTK-by-parts và YARN giúp mô hình mở rộng chuỗi tốt hơn mà không làm mất thông tin vị trí cục bộ quan trọng.

Kết quả thực nghiệm cho thấy YARN vượt trội hơn RoPE truyền thống khi mở rộng độ dài chuỗi, nhờ kiểm soát tốt hơn việc nội suy góc quay cho từng nhóm kênh tần số.

## Rotary Position Embedding trong các mô hình hiện đại
Các mô hình mã nguồn mở gần đây như DeepSeek, Llama 4 và GPT OSS đều sử dụng Rotary Position Embedding (RoPE) làm phương pháp mã hóa vị trí. Để mở rộng độ dài chuỗi, các mô hình này trước tiên điều chỉnh RoPE bằng cách thay đổi góc quay dựa trên NTK-aware scaling, sau đó huấn luyện trên các chuỗi dài hơn để phù hợp hơn với ngữ cảnh mở rộng. Kỹ thuật này đã trở thành tiêu chuẩn trong lĩnh vực này.

( Kết thúc RoPE / NTK / YARN )


## Giới hạn về khả năng biểu đạt của Transformer
Ngoài các thách thức về nội suy, transformer còn đối mặt với những giới hạn cơ bản về khả năng biểu đạt. Ta có thể phân tích điều này thông qua hai lớp độ phức tạp:

### Lớp độ phức tạp TC0
- Mạch logic có độ sâu hằng số và kích thước đa thức
- Cho phép số lượng đầu vào không giới hạn cho các cổng AND/OR/NOT
- Bao gồm các cổng ngưỡng (giống như bỏ phiếu đa số)
- Có thể thực hiện các phép đếm
- Bao gồm các phép toán cơ bản:
  - Cộng, so sánh
  - Theo dõi parity tiền tố  
  - Nhân/chia số nguyên

### Lớp độ phức tạp NC1  
- Mạch logic có độ sâu logarit (so với kích thước đầu vào)
- Số lượng cổng đa thức
- Số lượng đầu vào cho mỗi cổng AND/OR bị giới hạn
- Bao hàm các thuật toán `chia để trị`
- Xử lý các tác vụ phức tạp hơn:
  - Đánh giá công thức logic
  - Hợp thành hoán vị
  - `Theo dõi trạng thái`

Như đã chỉ ra trong bài báo "Illusion of State in State Space Models", cả transformer và state space model đều thuộc lớp TC0, điều này giới hạn khả năng xử lý các tác vụ suy luận phức tạp. Đây là lý do tại sao các kỹ thuật như chain-of-thought prompting được sử dụng để tăng cường năng lực của chúng.

![](https://pbs.twimg.com/media/GzkNdIjbsAAmqOW?format=jpg&name=large)

## Thông tin vị trí trong RoPE
Một câu hỏi quan trọng đặt ra: Vì RoPE chỉ áp dụng cho truy vấn (query) và khóa (key), liệu thông tin vị trí tuyệt đối có truyền qua giá trị (value) không?

Câu trả lời nằm ở các tính chất toán học của RoPE:
1. Dù sử dụng vị trí tuyệt đối, phép nhân vô hướng thực chất tạo ra mã hóa vị trí tương đối
2. Điều này xảy ra nhờ tính chất của ma trận quay
3. Value không nhận mã hóa vị trí trực tiếp
4. Điểm attention cuối cùng chỉ phụ thuộc vào vị trí tương đối
5. Do đó, không còn thông tin vị trí tuyệt đối trong value

## Bài toán hoán vị S5
Để minh họa khả năng của NC1, ta xét bài toán hoán vị 5 phần tử (S5):
- Hợp thành nhiều phép hoán đổi (swap)
- Mỗi hoán vị có thể phân rã thành các phép hoán đổi
- Cần theo dõi hiệu ứng tích lũy của các phép hoán đổi
- Bản chất là không giao hoán:
  - Swap(A,B) rồi Swap(B,C) ≠ Swap(B,C) rồi Swap(A,B)
  - Dẫn đến cấu hình cuối cùng khác nhau

## Vì sao RoPE thất bại với S5
RoPE không thể xử lý bài toán này vì:
1. Độc lập dữ liệu: Các phép quay chỉ phụ thuộc vào vị trí, không phụ thuộc vào nội dung
2. Tính giao hoán: Các ma trận quay dạng block-diagonal (khối 2D) là giao hoán
3. Những tính chất này mâu thuẫn trực tiếp với yêu cầu của S5:
   - Cần các phép toán phụ thuộc dữ liệu
   - Cần hợp thành không giao hoán

Điều này làm nổi bật những hạn chế cơ bản của RoPE trong việc mô hình hóa các `biến đổi phức tạp phụ thuộc trạng thái`. Chúng ta sẽ tìm hiểu lý do tại sao `RoPE không thể xử lý các tác vụ không giao hoán` và `cách sử dụng phép biến đổi tuyến tính khác` để mã hóa thao tác hoán đổi.

|![](https://pbs.twimg.com/media/GzkVRMebsAAFjaR?format=jpg&name=large)|![](https://pbs.twimg.com/media/GzkWHxgbwAAArCq?format=png&name=large)|
|-|-|

## Ma Trận Householder Cho Thao Tác Hoán Đổi
Chúng ta có thể sử dụng ma trận Householder để mã hóa thao tác hoán đổi:
- Ma trận Householder thực hiện phép phản xạ cơ bản
- Có thể hình dung như một tấm gương ánh xạ điểm này sang điểm khác
- Bằng cách chọn gương phù hợp, chúng ta có thể thay đổi vị trí của hai vector
- Đây là kết quả cổ điển trong đại số tuyến tính:
  - Lấy hiệu của hai vector để xây dựng phép biến đổi Householder
  - Áp dụng phép biến đổi tuyến tính này giữ nguyên các vector trực giao khác
- Tích lũy các ma trận Householder cho phép mô hình hóa thành phần hoán đổi

## Chain-of-Thought Có Giúp RoPE Xử Lý Hoán Đổi?
- Chain-of-thought có thể nâng cao khả năng biểu đạt của transformer
- Khi mô hình đủ lớn, có thể giải quyết các vấn đề trong lớp P
- Tuy nhiên tốc độ suy luận chậm hơn do cần chuỗi suy nghĩ dài
- Các giá trị riêng âm cũng có thể mã hóa hoán đổi (như trong Delta Product)

## Biến Đổi Householder Tổng Quát
- Phiên bản gốc: β=2 (phản xạ cơ bản)
- Khi β=0: phép biến đổi đồng nhất (không làm gì)
- Khi β=1: phép chiếu
- Khi β=2: phép phản xạ
- Đã được sử dụng trong các mô hình như Data Night và Data Product

## Ứng Dụng Trong PaTH
- Sử dụng tích lũy ma trận Householder làm mã hóa vị trí
- Giải quyết bài toán hoán đổi 5 phần tử (NC1 hoàn chỉnh)
- PaTH được chứng minh là NC1 hoàn chỉnh trong điều kiện nhẹ

## Đánh Giá Trên Tác Vụ Tổng Hợp
1. Tác vụ Multiquery:
   - Theo dõi 5 phần tử qua các hoán đổi
   - Hỏi vị trí phần tử sau mỗi hoán đổi
   - Không sử dụng chain-of-thought
   - RoPE không thể giải quyết dù train bao nhiêu epoch
   - PaTH giải quyết dễ dàng với ít epoch

2. Tác vụ Flip-Flop Language Modeling:
   - Ghi nhớ bit cuối cùng tại token đọc
   - PaTH giải quyết tốt với tỷ lệ lỗi thấp

3. Tác vụ A5 Word Movement:
   - Tương tự bài toán hoán đổi
   - PaTH chỉ cần số lớp logarit trong khi các phương pháp khác cần tuyến tính
   - A5 là nhóm thay phiên, tương tự bài toán hoán đổi

## Đánh Giá Trên Tác Vụ Tổng Hợp và Ứng Dụng Thực Tế
Đầu tiên chúng ta có một số benchmark tổng hợp, sau đó nếu muốn áp dụng PaTH vào thực tế, tôi đưa ra một số khuyến nghị:

### Phương Pháp Chưng Cất (Distillation)
- Hiện có nhiều mô hình được huấn luyện với RoPE
- Chúng ta có thể bắt đầu từ các checkpoint này và thay thế các lớp attention từ RoPE sang PaTH
- Quá trình gồm 2 giai đoạn:
  1. Giai đoạn 1: 
     - Lấy input/output chuẩn từ mô hình giáo viên (RoPE)
     - Thay thế toàn bộ RoPE bằng PaTH trong mô hình học sinh
     - Khởi tạo trọng số học sinh từ mô hình giáo viên
     - Tính toán loss L2 khoảng cách giữa output của học sinh và giáo viên
     - Tối thiểu hóa sai số bình phương trung bình
  2. Giai đoạn 2:
     - Tối thiểu hóa divergence KL của phân phối đầu ra
     - Đây chính là phương pháp knowledge distillation

### Giải Thích Toán Học
- PaTH cho phép mô hình hóa phản xạ, trong khi RoPE chỉ xoay
- Về mặt toán học: 2 phép quay có thể mô hình hóa 1 phép phản xạ (và ngược lại)
- Trong thực tế, chúng ta không cần tái tạo chính xác RoPE mà chỉ cần giảm thiểu loss một cách gián tiếp

### Kết Quả Thử Nghiệm
1. Thí nghiệm chưng cất:
   - Sử dụng mô hình CO 2.5 7B instruction làm giáo viên
   - Chỉ dùng 100 triệu token cho giai đoạn 1 và 3 tỷ token cho giai đoạn 2
   - So với pretrain từ đầu cần hàng nghìn tỷ token
   - Hiệu năng phục hồi gần như tương đương, thậm chí tốt hơn ở một số tác vụ

2. Tiếp tục huấn luyện (Continue Pre-training):
   - Sử dụng checkpoint trung gian từ Hugging Face
   - Dữ liệu chất lượng cao ở giai đoạn cuối
   - So sánh công bằng với cùng dữ liệu huấn luyện
   - PaTH vượt trội RoPE trên benchmark mã hóa và toán học (GSM8K, HumanEval, BBPP)
   - Đặc biệt hiệu quả với tác vụ theo dõi trạng thái trong lập trình

### Kiến Trúc PaTH Attention
- Sử dụng tích lũy ma trận Householder làm trọng số nhị phân
- Biến đổi Householder tổng quát với tham số β ∈ [0,2]
- Vector WT được tính từ phép chiếu tuyến tính và chuẩn hóa L2
- Đảm bảo tính ổn định bằng cách giới hạn giá trị riêng

### Tối Ưu Phần Cứng
1. Tính toán theo khối (Blockwise):
   - Chia sequence thành các khối có kích thước bằng nhau
   - Mỗi khóa được ánh xạ tới vị trí cuối cùng trong khối
   - Tái sử dụng tính toán thông qua lập trình hạn chót (deadline programming)

2. Thuật toán hiệu quả:
   - Sử dụng biến đổi UT (Upper Triangular) cổ điển
   - Tính tích lũy ma trận Householder hiệu quả
   - Độ phức tạp khối nhỏ (thường 64) nên nghịch đảo ma trận không thành vấn đề
   - Song song hóa nghịch đảo ma trận trên GPU

3. Xử lý truy vấn:
   - Áp dụng biến đổi Householder ngược chiều (từ phải sang trái)
   - Sử dụng phép thế thuận (forward substitution) cho ma trận tam giác dưới
   - Tối ưu hóa thông qua các phép nhân ma trận

## Tính Toán Tích Lũy Ma Trận Householder
Phép biến đổi được sử dụng để tính tích lũy ma trận Householder một cách hiệu quả. Chúng ta có thể biểu diễn nó dưới dạng phép nhân ma trận. Việc nghịch đảo ma trận không cần phải lo lắng vì kích thước khối thường nhỏ (khoảng 64). Ngay cả khi độ phức tạp là bậc 3, chúng ta chỉ thực hiện cục bộ trong khối đó. Với kích thước khối nhỏ, độ phức tạp này không phải vấn đề.

## Phương Pháp Tính Toán
1. **Thế thuận (Forward Substitution)**: Sử dụng cho ma trận tam giác dưới
2. **Biến đổi ma trận**: Để tính tích lũy cộng đồng
3. **Xử lý truy vấn và khóa**:
   - Với khóa: Biến đổi đến vị trí tương lai
   - Với truy vấn: Áp dụng biến đổi ngược chiều (từ phải sang trái)
   - Sử dụng mũi tên chỉ hướng biến đổi Householder

## Tối Ưu Phần Cứng
- **Phép nhân ma trận**: Hầu hết tính toán có thể biểu diễn bằng phép nhân ma trận, rất nhanh trên GPU
- **Nghịch đảo ma trận**: Thực hiện song song cho từng khối độc lập
- **Tích hợp Flash Attention**:
  - Tải khối truy vấn vào bộ nhớ chia sẻ
  - Quét từ phải sang trái (khác với RoPE)
  - Cập nhật thống kê trực tuyến tương tự Flash Attention
  - Khác biệt chính: Cần cập nhật truy vấn sau mỗi khối để tích hợp biến đổi Householder tích lũy

## Hiệu Suất Thực Tế
- Hiện chậm hơn Flash Attention khoảng 2 lần
- Có thể cải thiện bằng DSL tùy chỉnh
- Quan trọng là thuật toán phù hợp với phần cứng

## Suy Luận (Inference)
1. **Cơ chế**:
   - Mỗi bước thời gian nhận ma trận Householder mới
   - Áp dụng lên tất cả khóa lịch sử
   - Sử dụng thông tin tương lai để tinh chỉnh bộ đệm khóa
2. **Cập nhật Rank-1**:
   - Giữ phần identity của khóa cũ
   - Thêm cập nhật rank-1 cho thông tin mới
   - Cho phép thay đổi động bộ đệm khóa
3. **Tối ưu IO**:
   - Chỉ cần tải vector W đơn lẻ
   - Không cần lưu trữ lịch sử vector W
   - Chi phí chính: Ghi khóa đã cập nhật vào HBM
   - Có thể pipeline với tải khối tiếp theo

## Câu Hỏi Mở
- Có thể áp dụng cơ chế tương tự cho bộ đệm giá trị (value cache)?
- Cơ chế này mở ra khả năng nén bộ đệm khóa khi giá trị tiệm cận 0
- Mô hình có thể tự học hệ số β để điều chỉnh tốc độ suy giảm
## Tối Ưu KV Cache và Cập Nhật Ma Trận
- **Cập nhật khóa vào HBM**: Có thể thực hiện chồng chéo (pipeline) một phần
- **Quy trình xử lý**:
  1. Tải khối khóa vào bộ nhớ chia sẻ
  2. Thực hiện cập nhật rank-1
  3. Ghi lại kết quả
  4. Đồng thời tải khối khóa tiếp theo
- **Tối ưu hóa pipeline**: Có nhiều tiềm năng để cải thiện hiệu suất

## Thảo Luận Về KV Cache Động
- **Vấn đề hiện tại**: Các phương pháp tối ưu KV cache giả định cache cố định
- **Quan điểm mới**: Có thể thay đổi KV cache trong quá trình decoding
- **Mục tiêu chính**: Tập trung vào khả năng biểu diễn hơn là hiệu suất
- **Lợi ích tiềm năng**:
  - Giảm kích thước KV cache cần thiết
  - Giảm số bước decoding
  - Nén cache ở chiều thấp hơn
  - Tự động loại bỏ các khóa đã decay về 0

## Tính Chất Spectral và Kiểm Soát Decay
- **Phân tích spectral**:
  - Vector W được chuẩn hóa
  - Chuẩn spectral bị chặn trong khoảng [0,1]
  - Phép biến đổi Householder tổng quát không bảo toàn chuẩn
- **Cơ chế decay linh hoạt**:
  - Hệ số β có thể thay đổi động theo dữ liệu
  - Tương tự cơ chế gating trong mô hình attention
  - Cho phép kiểm soát khoảng cách attention (từ local đến long-range)
  - Có thể tự động tỉa (prune) các khóa không quan trọng

## So Sánh Với Các Phương Pháp Khác
- **RoPE vs Householder**:
  - RoPE sử dụng phép quay cố định
  - Householder cho phép reflection linh hoạt
  - Về lý thuyết có thể biểu diễn RoPE nhưng thực tế khác biệt
- **Lý do chọn Householder**:
  - Hiệu quả tính toán (O(n²) thay vì O(n³))
  - Cấu trúc ma trận đặc biệt
  - Hỗ trợ tính toán song song
  - Cho phép thao tác reflection đột ngột (khác với rotation liên tục)

## Kết Nối Với Forgetting Transformers
- **Mối quan hệ với ALiBi**:
  - Forgetting transformers là phiên bản phụ thuộc dữ liệu của ALiBi
  - Sử dụng tích lũy các giá trị scalar (forget gate)
  - Có thể kết hợp với cơ chế Householder
- **Biểu diễn toán học**:
  - Forget gate thường dùng hàm exponential
  - Có thể đưa vào trong biểu thức exponential như một bias term
  - Kết hợp encoding vị trí dạng multiplicative và additive

## Mã Hóa và Cơ Chế Attention
- **Vai trò mã hóa**: Có thể sử dụng như một thành phần bias cho đối tượng attention
- **Cơ chế mã hóa**:
  - RoPE và PaTH sử dụng phương pháp mã hóa multiplicative attention
  - Sử dụng trọng số nhị phân cho tính toán attention score
  - Kết hợp giữa mã hóa vị trí multiplicative và additive
- **Kết nối với NA attention**: Có mối liên hệ thú vị với cơ chế phát hiện NA attention

## So Sánh Các Mô Hình Attention
- **Mô hình tuyến tính**:
  - Phiên bản tương đương không dùng softmax
  - Ví dụ: Mamba 2 (khi bỏ hàm exponential)
- **Mô hình PaTH**:
  - Có thể xem như DeltaNet với softmax
  - Hoặc DeltaNet có gating kết hợp softmax
- **Ưu điểm bổ sung**:
  - Cơ chế mã hóa attention log mạnh mẽ hơn
  - Cho phép phân tách biểu diễn lý thuyết giữa NC1 và TC0

## Hạn Chế và Giải Pháp
- **Hạn chế về bộ nhớ**:
  - Mô hình phi tuyến có sức mạnh biểu diễn cao nhưng bộ nhớ hạn chế
  - Kích thước trạng thái ẩn cố định
- **Giải pháp softmax**:
  - Xem như phương pháp kernel ánh xạ sang không gian vô hạn chiều
  - Tương đương RNN với chiều ẩn vô hạn
  - Cải thiện đáng kể khả năng lưu trữ bộ nhớ

## Mở Rộng Lý Thuyết
- **Khai triển Taylor**:
  - Softmax vô hạn có thể biểu diễn qua khai triển Taylor
  - Các mô hình như Tensor Power Linear Attention sử dụng khai triển Taylor cụt
- **Kết nối kernel**:
  - Mối quan hệ giữa attention và linear attention qua hàm kernel
  - Hàm kích hoạt exponential giúp phân tách tốt hơn

## Động Lực Phát Triển PaTH
- **Kết hợp ưu điểm**:
  - Kết hợp softmax (bộ nhớ lớn) với delta product (sức mạnh biểu diễn)
  - Lấy cảm hứng từ Forgetting Transformers
- **Khác biệt với DeltaNet**:
  - Chỉ sử dụng một ma trận Householder
  - Tập trung vào khả năng theo dõi trạng thái

## Kết Quả Thực Nghiệm
- **Benchmark chung**:
  - PaTH và PaTH-Fox vượt trội RoPE và Fox
  - Hiệu quả trên các tác vụ common sense reasoning
- **Khả năng ngoại suy**:
  - RoPE: Hiệu suất giảm khi vượt quá độ dài huấn luyện 4K
  - Fox: Có thể ngoại suy nhưng hiệu suất dao động
  - PaTH: Cải thiện đáng kể khả năng ngoại suy
  - PaTH-Fox: Hiệu suất ổn định nhất, không bùng nổ ở đoạn giữa
- **Tác vụ theo dõi biến**:
  - PaTH và PaTH-Fox thể hiện ưu thế rõ rệt

## Kết Quả Đánh Giá Trên Benchmark
- Trong bài kiểm tra không cạnh tranh (non-contest benchmark), đặc biệt đáng chú ý là:
  - Tác vụ theo dõi biến (variable tracking task) trong bộ thước đo (ruler)
  - Đây là tác vụ chuyên biệt cho theo dõi trạng thái
  - Kết quả cho thấy PaTH và PaTH-Fox có lợi thế đáng kể so với Fox và RoPE trong hạng mục phụ này
