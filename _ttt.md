Đầu tiên hãy đọc qua bài viết sau để dẫn nhập vào TTT (test time training)

# Lịch sử ngắn về chú ý tuyến tính: Từ bắt chước, đổi mới đến phản hồi ngược
- Tô Kiếm Lâm | 20-06-2025
- https://kexue.fm/archives/11033
- playground/68575d9dd633325a8ba59b60

Trong cộng đồng tiếng Trung, trang web này có lẽ là một trong những nơi sớm quan tâm đến Attention tuyến tính. Khi viết bài blog đầu tiên về chủ đề này vào năm 2020 "Khám phá Attention tuyến tính: Attention có nhất thiết phải có Softmax không?", mọi người chủ yếu vẫn đang thảo luận về Softmax Attention liên quan đến BERT. Nhìn lại, việc xem xét Attention tuyến tính trong thời đại BERT không phải là quá khôn ngoan, vì khi đó độ dài huấn luyện khá ngắn và mô hình chủ yếu vẫn là Encoder, sử dụng Attention tuyến tính về cơ bản không có lợi thế. Về điều này, tôi cũng đã viết bài "Transformer tuyến tính có lẽ không phải là mô hình bạn đang chờ đợi" để bày tỏ quan điểm này.

Cho đến khi ChatGPT ra đời, buộc mọi người phải làm mô hình sinh dạng Decoder-only, điều này rất phù hợp với dạng RNN của Attention tuyến tính. Đồng thời, việc theo đuổi độ dài huấn luyện dài hơn cũng khiến nút thắt cổ chai độ phức tạp bậc hai của Softmax Attention càng trở nên rõ ràng. Trong bối cảnh mới này, Attention tuyến tính ngày càng thể hiện khả năng cạnh tranh, thậm chí xuất hiện dấu hiệu "phản hồi ngược" lại Softmax Attention.

## Độ phức tạp bình phương

Đầu tiên, hãy giới thiệu một số ký hiệu:
```
qi,ki,vi,oi ∈ Rd×1
Q=[q1,q2,⋯,qn]⊤ ∈ Rn×d
K=[k1,k2,⋯,kn]⊤ ∈ Rn×d  
V=[v1,v2,⋯,vn]⊤ ∈ Rn×d
O=[o1,o2,⋯,on]⊤ ∈ Rn×d
```
Một mô hình Attention về bản chất là một ánh xạ `Q,K,V → O`. Bài viết này chủ yếu quan tâm đến trường hợp nhân quả (causal), nghĩa là `ot` chỉ phụ thuộc vào `Q[:t],K[:t],V[:t]`. Về nguyên tắc, `d` của `Q,K` có thể khác với `d` của `V,O`, như GAU và MLA, nhưng việc đơn giản hóa chúng thành cùng một giá trị không thay đổi bản chất vấn đề.

Softmax Attention tiêu chuẩn thường là cơ chế Attention được đề xuất trong "Attention is All You Need": **`O=softmax(QK⊤+logM)V`**. Ở đây đã bỏ qua hệ số tỷ lệ `1/√d` vì nó luôn có thể được hấp thụ vào `Q,K`. softmax thực hiện chuẩn hóa mũ theo chiều thứ hai, và `M ∈ Rn×n` là ma trận tam giác dưới (causal), được gọi là ma trận mặt nạ.

## Hình dạng ban đầu

Ý tưởng ban đầu của Attention tuyến tính chủ yếu là bắt chước và xấp xỉ Softmax Attention.
Phương án đơn giản nhất là loại bỏ trực tiếp hàm softmax: **`O=((QK⊤)⊙M)V`**

Tại sao dạng này lại là Attention "tuyến tính"? Để hiểu nhanh điều này, chúng ta có thể xem xét phiên bản không-Causal không có `⊙M`. Khi đó `O=(QK⊤)V=Q(K⊤V)` (6), lưu ý rằng độ phức tạp tính `K⊤V` là `O(nd²)`, kết quả là ma trận `d×d`, sau đó nhân với `Q` cũng có độ phức tạp `O(nd²)`, do đó độ phức tạp của nó phụ thuộc tuyến tính vào `n`.

Đối với phiên bản Causal (6), chúng ta có thể hiểu từ dạng phần tử:

`ot = ∑j=1^t vj(kj⊤qt) = ∑j=1^t (vjkj⊤)qt = (∑j=1^t vjkj⊤)qt` (7)

Nếu chúng ta ký hiệu phần trong ngoặc là `St`, thì có:

**`ot = St qt, St = St-1 + vt kt^⊤` (8)**

Từ đây có thể thấy, **Attention dạng Causal có thể được viết như một RNN tuyến tính với St là State**, do đó độ phức tạp mỗi bước là hằng số, tổng độ phức tạp tỷ lệ với độ dài chuỗi n. Lưu ý ở đây xuất hiện "RNN tuyến tính", đây là khái niệm tổng quát hơn, Attention tuyến tính thuộc về một loại RNN tuyến tính. RNN tuyến tính cũng đã phát triển riêng một thời gian, như LRU, SSM đã được giới thiệu trước đây, nhưng **các kiến trúc tuyến tính có khả năng cạnh tranh gần đây đều có dạng Attention tuyến tính**.

Attention tuyến tính thời kỳ đầu còn có những đặc điểm bắt chước Softmax Attention rất rõ ràng, ví dụ thêm mẫu số vào công thức (6) để chuẩn hóa, và để chuẩn hóa thì `kj⊤ qt` phải không âm, do đó lại thêm hàm kích hoạt không âm cho `Q,K`. Một loạt công trình với đại diện là Performer, RFA càng lấy việc xấp xỉ `exp(QK⊤)` làm điểm xuất phát để xây dựng mô hình.

Tuy nhiên, các nghiên cứu sau này như "The Devil in Linear Transformer" phát hiện rằng chuẩn hóa theo chiều độ dài chuỗi không thể tránh hoàn toàn sự không ổn định số học, thà chuẩn hóa như sau còn hơn: **`O = RMSNorm(((QK⊤)⊙M)V)` (9)**

Và vì không cần chuẩn hóa, việc thêm hàm kích hoạt không âm cho `Q,K` để đảm bảo `kj⊤ qt` không âm trở nên không cần thiết. Vậy việc thêm hàm kích hoạt (không nhất thiết phải không âm) cho `Q,K` còn có ý nghĩa không? Quan điểm của tôi là, **thêm hàm kích hoạt là quyền tự do của mọi người**, không loại trừ việc thêm một hàm kích hoạt nào đó có thể điều chỉnh ra kết quả tốt hơn, nhưng thêm hàm kích hoạt không thay đổi dạng của Attention tuyến tính, do đó không ảnh hưởng đến mô tả của chúng ta. Ngoài ra, các kết quả hiện có cho thấy, **thực ra không thêm cũng đã đủ tốt**.

không âm trở nên không cần thiết. Vậy việc thêm hàm kích hoạt (không nhất thiết phải không âm) cho Q,K còn có ý nghĩa không? Quan điểm của tôi là, thêm hàm kích hoạt là quyền tự do của mọi người, không loại trừ việc thêm một hàm kích hoạt nào đó có thể điều chỉnh ra kết quả tốt hơn, nhưng thêm hàm kích hoạt không thay đổi dạng của Attention tuyến tính, do đó không ảnh hưởng đến mô tả của chúng ta. Ngoài ra, các kết quả hiện có cho thấy, thực ra không thêm cũng đã đủ tốt.

## Cổng quên đa dạng

Từ công thức (8) có thể thấy, Attention tuyến tính hiện tại về bản chất chỉ là `cumsum`, tức là cộng dồn tất cả thông tin lịch sử với trọng số bằng nhau. Không khó để tưởng tượng khi số lượng token được cộng dồn đủ lớn, tỷ lệ thông tin của mỗi token sẽ trở nên cực nhỏ, do đó chỉ dựa vào ma trận `St` kích thước cố định thậm chí không thể tái tạo chính xác bất kỳ token nào, ví dụ trực quan là **ký ức của mỗi token trở nên mờ nhạt**.

Để giảm thiểu vấn đề này, RetNet đã giới thiệu hiệu ứng quên vào Attention tuyến tính:

`ot = St qt, St = γSt - 1 + vt kt⊤` (10)

Trong đó hệ số suy giảm `γ∈(0,1)`, trong RetNet được đặt là hằng số, cũng có thể đặt là tham số có thể huấn luyện, hoặc thay γ thành ma trận đường chéo, v.v... **Attention tuyến tính được sử dụng trong MiniMax-01 cũng thuộc loại này**. Lưu ý, hệ số suy giảm đã có trước RetNet, nhưng chúng chủ yếu xuất hiện dưới dạng RNN tuyến tính, như LRU, SSM. RetNet có lẽ là lần đầu tiên kết hợp nó với Attention tuyến tính. Sau khi thêm hệ số suy giảm, mô hình sẽ có xu hướng quên đi thông tin lịch sử xa xưa hơn, từ đó ít nhất đảm bảo độ phân giải của các token gần đây, nói cách khác là thể hiện "nguyên tắc gần gũi (Recency Bias)" phù hợp với đặc tính của mô hình ngôn ngữ, do đó thường hoạt động tốt hơn.

Ngoài ra, một chi tiết đáng chú ý là RetNet còn thêm RoPE vào `Q,K`, điều này **tương đương với việc mở rộng hệ số suy giảm thành số phức `γe^(iθ)`**, từ góc nhìn LRU là xem xét giá trị riêng phức. Mặc dù việc thêm mã hóa vị trí cho RNN có vẻ hơi mâu thuẫn, nhưng một số thí nghiệm như TransXSSM gần đây cho thấy việc thêm RoPE vào Attention tuyến tính cũng có tác dụng tích cực nhất định. Tất nhiên, điều này có thể phụ thuộc vào biến thể mô hình cụ thể và thiết lập thí nghiệm.

Một mở rộng đơn giản của công thức (10) là thay γ bằng hàm của vị trí t là γt, điều này đã được thể hiện trong SSM. Sau đó, các công trình như DFW, Mamba, Mamba2 đã mở rộng nó thành phụ thuộc vào đầu vào, hình thành một loạt công trình về "data-dependent decay". Điều này thực sự rất giống với "cổng quên (forget gate)" của các RNN phi tuyến truyền thống như GRU, LSTM, chỉ khác là để duy trì tính tuyến tính của mô hình, đã loại bỏ sự phụ thuộc của cổng quên vào State (như St).

**Tại sao chúng ta ưa thích RNN tuyến tính?** Bởi vì RNN tuyến tính về cơ bản đều có thể tìm được cách nào đó để **huấn luyện song song**, điều này khiến nó cạnh tranh hơn so với Softmax Attention - cả về hiệu quả huấn luyện và hiệu quả suy luận đều không thua kém. Trong đó, "giải pháp chung" để song song hóa là chuyển đổi thành bài toán **Prefix Sum** rồi dùng **Associative Scan**, ý tưởng chính chúng tôi cũng đã giới thiệu đơn giản trong phần "Song song hóa" của bài "Công trình mới của Google cố gắng "hồi sinh" RNN: RNN có thể huy hoàng trở lại không?".

Tuy nhiên, **"giải pháp chung" không hiệu quả trên GPU**. VÌ GPU HIỆU QUẢ NHẤT VỚI PHÉP NHÂN MA TRẬN, do đó tìm được thuật toán song song sử dụng nhiều phép nhân ma trận là lý tưởng nhất. Thậm chí không cần song song, chỉ cần tìm được định dạng đệ quy Chunk by Chunk sử dụng đầy đủ phép nhân ma trận, cũng có thể cải thiện đáng kể hiệu quả huấn luyện. Điều này ngược lại đặt ra yêu cầu cho mô hình, như chỉ có cổng quên dạng tích ngoài mới có thể thực hiện mục đích này. Ví dụ phản biện điển hình là Mamba, nó là cổng quên phi tích ngoài, không thể phát huy đầy đủ hiệu năng GPU, do đó mới có các biến thể sau này như Mamba2 và GLA.

## Huấn luyện lúc kiểm tra

Đến nay, Attention tuyến tính từ việc bắt chước đơn giản Softmax Attention ban đầu, đến việc giới thiệu hệ số suy giảm tĩnh và thậm chí "data-dependent decay", đã hình thành đặc trưng riêng và phát huy giá trị trong nhiều tác vụ. Tuy nhiên, những tiến bộ này phần lớn được `thiết kế thủ công dựa trên kinh nghiệm`, chúng ta không khỏi tự hỏi: **Có nguyên tắc cấp cao hơn để chỉ đạo thiết kế Attention tuyến tính hoặc thậm chí là mô hình chuỗi tổng quát (Token-Mixer) không?**

Đối với câu hỏi này, **TTT (Test Time Training)** đã đưa ra câu trả lời của riêng mình. Nó **coi việc xây dựng mô hình chuỗi như một vấn đề "học trực tuyến (Online Learning)"** và đề xuất cách sử dụng bộ tối ưu (optimizer) để xây dựng RNN (không nhất thiết phải tuyến tính). Cụ thể, nó coi `K,V` như các cặp ngữ liệu `(k1,v1),(k2,v2),...,(kt,vt)`, dựa trên những ngữ liệu này để huấn luyện được một mô hình `v=f(St;k)`, cuối cùng xuất ra `ot=f(St;qt)`, trong đó `St` là tham số mô hình, còn cấu trúc mô hình phần lớn là tùy ý.

**Điều này liên quan gì đến RNN?** Rất đơn giản, các bộ tối ưu như SGD, Adam về bản chất là một RNN về tham số mô hình! Thực ra quan điểm này không mới, từ năm 2017 khi Meta Learning thịnh hành đã có nhà nghiên cứu đề xuất và sử dụng điều này, chỉ là lúc đó ý tưởng là cố gắng dùng RNN (LSTM) để mô phỏng một bộ tối ưu tốt hơn, chi tiết có thể tham khảo "Optimization as a Model for Few-Shot Learning".

Đúng là "phong thủy luân lưu chuyển", sau nhiều năm **TTT ngược lại đề xuất xây dựng RNN thông qua bộ tối ưu**. Quy trình như sau: Đầu tiên, tham số mô hình hiện tại là `St-1`, bộ tối ưu (SGD) nhận dữ liệu mới `(kt,vt)`, dựa vào dữ liệu này cập nhật tham số mô hình thành `St`, cuối cùng trả về kết quả dự đoán của `qt` là `f(St-1;qt)`, và cứ thế tiếp tục. Vì vậy, RNN được TTT thực hiện có thể được viết thống nhất là:

**`ot = f(St;qt), St = St-1 - ηt∇S_(t-1).L(f(S_(t-1);kt),vt)` (11)**

Trong đó `L(f(St-1;kt),vt)` là hàm mất mát của dữ liệu hiện tại `(kt,vt)` dưới tham số hiện tại `St-1`, `ηt` là tham số tốc độ học, tham khảo "data-dependent decay" ở phần trước, nó cũng có thể làm thành data-dependent. Dạng này có thể bao phủ rất nhiều mô hình RNN, ví dụ công thức (8) và (10) đều là trường hợp đặc biệt của nó.

TTT gốc tập trung khám phá RNN phi tuyến với mini-batch, sau đó Titans thêm động lượng vào SGD của TTT, tiếp theo **"Test-Time Training Done Right"** khám phá cách dùng TTT với large-batch, còn khám phá tổ hợp "TTT + Muon". Lưu ý, **TTT chỉ sử dụng bộ tối ưu để xây dựng RNN**, các tham số ngoài RNN như tham số có thể huấn luyện của Q,K,V vẫn được huấn luyện bằng bộ tối ưu tổng thể sau khi xây dựng toàn bộ mô hình.

### Một câu hỏi đáng suy nghĩ hơn là: **Tại sao TTT có thể trở thành "nguyên tắc chỉ đạo" để xây dựng RNN?** 

Mục tiêu cốt lõi của RNN là nén hiệu quả dữ liệu lịch sử vào một State kích thước cố định, trong khi `tham số mô hình đúng là có kích thước cố định`, **huấn luyện mô hình ở mức độ nào đó tương đương với nén dữ liệu huấn luyện vào trọng số mô hình**. TTT chính là tận dụng sự phù hợp cao độ của nó với mục tiêu RNN. Nói một cách trực tiếp, nếu coi RNN như một tác vụ nén, TTT coi mô hình `f` là "bộ giải nén", trọng số của nó là "file nén", thuật toán nén là SGD, tỷ lệ nén là loss `L`.

Như vậy, chúng ta không cần bận tâm xây dựng định dạng đệ quy, mà chuyển sang **xây dựng mô hình f và loss L**. Một RNN mạnh hay không, đáng tin cậy hay không, chúng ta **chỉ cần xem f và L tương ứng** là có thể nắm rõ.

Ngoài ra, **TTT dùng Online Learning để xây dựng RNN, có nghĩa RNN thu được chắc chắn rất phù hợp với tác vụ ICL (In Context Learning)**, đây cũng là ưu thế của TTT như một "nguyên tắc chỉ đạo". Trước đây "Why Can GPT Learn In-Context? Language Models Implicitly Perform Gradient Descent as Meta-Optimizers" thậm chí còn làm ngược lại, loại bỏ Softmax khỏi Softmax Attention thành Attention tuyến tính để giải thích khả năng ICL của nó. Với góc nhìn hiện tại, nó chính là xây dựng TTT tương ứng.

## Loại bỏ cũ và đón mới

Ví dụ, Attention tuyến tính ban đầu tương ứng với hàm mất mát là `-v⊤(Sk)`, nhìn vào đây đã thấy là một mục tiêu không đáng tin cậy lắm, vì nó không có cận dưới, điều này có thể dẫn đến S tiến tới vô cùng. Ngược lại, RetNet thêm điều chuẩn L2 vào hàm mất mát, tránh được rủi ro này, từ góc độ tối ưu cũng giảm thiểu rủi ro overfitting, từ đó có được một RNN tốt hơn.

Tuy nhiên, dùng tích vô hướng làm hàm mất mát tuy đơn giản và có lý nhất định, nhưng nó không trực tiếp khuyến khích Sk=v, do đó không phải là một mất mát hồi quy lý tưởng. Hàm mục tiêu tốt hơn nên là mất mát bình phương, tức là 1/2‖Sk-v‖², thay nó vào công thức TTT (11) ta được:

`ot = f(St;qt), St = St-1 - ηt(St-1 kt - vt)kt⊤` (12)

Đây chính là DeltaNet, tên này xuất phát từ "Parallelizing Linear Transformers with the Delta Rule over Sequence Length", sớm hơn thì được đề xuất bởi "Linear Transformers Are Secretly Fast Weight Programmers".

Lưu ý rằng `ηt(St-1 kt -vt)kt⊤ = (St-1(√ηt kt) - (√ηt vt))(√ηt kt)⊤`, điều này có nghĩa `ηt` luôn có thể được hấp thụ vào định nghĩa của `kt,vt`, do đó phân tích tiếp theo chúng ta chỉ xét trường hợp ηt=1:
```
St = St-1-(St-1kt-vt)kt⊤
   = St-1-(St-1kt)kt⊤+vtkt⊤
   = St-1(I-ktkt⊤)+vtkt⊤ (13)
```
Nếu cần thiết, chúng ta thay kt,vt bằng √ηt kt, √ηt vt, là có thể khôi phục lại ηt. So sánh với dạng ban đầu của Attention tuyến tính (8), sự khác biệt của DeltaNet là trước khi cộng vtkt⊤ có thêm trừ đi (St-1kt)kt⊤, trong đó St-1kt có thể hiểu là kết quả dự đoán của đầu vào mới kt dưới mô hình cũ St-1.

Suy nghĩ trực quan, "trừ trước cộng sau" là trước tiên loại bỏ nhận thức cũ của mô hình về kt, sau đó dựa vào (kt,vt) bổ sung nhận thức mới, đạt được hiệu ứng "loại bỏ cũ đón mới". Quy tắc này gọi là "Delta Rule", chính là nguồn gốc của "Delta" trong DeltaNet. Delta Rule không mới, nó còn gọi là Least Mean Square, Widrow-Hoff Algorithm, v.v., đã có từ thập niên 60 thế kỷ trước. Thực tế, trong lĩnh vực này rất ít thứ hoàn toàn mới, nhiều thay đổi đều có thể truy nguồn về một công trình "thời cổ đại" nào đó. Nỗ lực hiện tại chủ yếu tập trung vào khai thác phần có thể Scalable trong đó.

Cũng cần chỉ ra rằng, theo thứ tự thời gian, DeltaNet có trước, TTT có sau. Việc hiểu RNN từ góc độ Online Learning thực ra đã được thể hiện rải rác trong một số công trình trước TTT, nhưng **TTT đã đề xuất một cách có hệ thống "nguyên tắc chỉ đạo" này và sử dụng nó để xây dựng mô hình RNN mới**, do đó chúng tôi đặt TTT lên trước để cách giới thiệu tổng thể trôi chảy tự nhiên hơn.

Một số độc giả có thể thắc mắc: DeltaNet vẫn còn là RNN tuyến tính không? Câu trả lời là khẳng định. RNN tuyến tính mà chúng ta nói đến là công thức đệ quy phụ thuộc vào biến State theo quan hệ tuyến tính, nhưng sự phụ thuộc vào đầu vào hoặc q,k,v có thể là phi tuyến (tất nhiên hiệu quả song song của các dạng phụ thuộc khác nhau sẽ khác nhau). Từ công thức (13) có thể thấy, vế phải luôn chỉ xuất hiện St-1 bậc nhất, do đó nó thỏa mãn định nghĩa tuyến tính.

## Nghịch đảo đến trợ giúp

Như đã nói ở trên, thuật toán song song lý tưởng nhất cho RNN tuyến tính (tức là hiệu quả nhất trên GPU) là dạng sử dụng triệt để phép nhân ma trận. Để đạt được mục tiêu này, trước tiên chúng ta cần viết lại DeltaNet dưới một dạng thuận tiện hơn.

Chúng ta bắt đầu bằng cách viết lại công thức DeltaNet như sau: St bằng St-1 cộng với (vt trừ St-1kt) nhân với kt chuyển vị. Nếu đặt ut bằng vt trừ St-1kt, thì St sẽ bằng St-1 cộng ut nhân kt chuyển vị. Điều này có nghĩa là DeltaNet thực chất chỉ là thay thế V bằng U trong Attention tuyến tính ban đầu, trong đó U là ma trận chứa các vector u1, u2 cho đến un.

Bằng cách lặp lại công thức này t-1 lần, ta có thể biểu diễn St-1 dưới dạng tổng của các tích ngoài ujkj chuyển vị. Từ đó, ta có thể tính được ut bằng vt trừ đi tổng của các uj nhân với tích vô hướng giữa kj và kt. 

Khi viết dưới dạng ma trận, ta thu được một hệ phương trình tuyến tính: U bằng V trừ tích Hadamard của KK chuyển vị với (M-I) nhân với U. Nghiệm của hệ phương trình này có thể được biểu diễn trực tiếp thông qua ma trận nghịch đảo.

Cụ thể, U bằng ma trận nghịch đảo của (I cộng với tích Hadamard của KK chuyển vị và M-I) nhân với V. Để cho gọn, ta ký hiệu phần trong ngoặc là B.

Ở đây xuất hiện một vấn đề: chúng ta cần tính nghịch đảo của ma trận (I+B), một ma trận kích thước n nhân n. Độ phức tạp tiêu chuẩn của phép tính nghịch đảo này là O(n³), thậm chí còn cao hơn cả Softmax Attention!

Tuy nhiên, may mắn thay, chúng ta không cần phải tính toán tường minh ma trận nghịch đảo. Thay vào đó, chúng ta chỉ cần tìm U, điều này có thể thực hiện bằng cách giải hệ phương trình (I+B)U=V. Việc giải hệ phương trình này có độ phức tạp O(n²), thấp hơn đáng kể.

Hơn nữa, bằng cách tận dụng hai đặc điểm quan trọng - I+B là ma trận tam giác dưới và B có cấu trúc hạng thấp - chúng ta có thể giảm độ phức tạp xuống còn tuyến tính. Khi được viết dưới dạng phép nhân ma trận khối, thuật toán này có thể tận dụng tối đa sức mạnh tính toán của GPU.

Chi tiết cụ thể về cách thực hiện điều này khá phức tạp và đòi hỏi phải đọc kỹ bài báo gốc. Trong phạm vi bài viết này, chúng tôi tập trung vào việc giới thiệu rõ ràng các nguyên lý toán học cơ bản.

Sau DeltaNet, một phát triển tự nhiên tiếp theo là Gated DeltaNet (GDN), trong đó cổng quên được tích hợp vào DeltaNet. Đây là một thay đổi có thể dự đoán được, tiếp nối xu hướng phát triển của các mô hình Attention tuyến tính.


## Phản hồi ngược đang diễn ra

Đầu bài đã đề cập, Attention tuyến tính ngày nay không chỉ có thể cạnh tranh với Softmax Attention, thậm chí bắt đầu "phản hồi ngược" lại Softmax Attention. Điều này tưởng chừng không thể tin được, nhưng suy nghĩ kỹ thì không khó hiểu. Theo một góc nhìn nào đó, trong những năm gần đây, Softmax Attention liên tục bị "thoái hóa". Quá trình phát triển từ MHA (Multi-Head Attention) sang GQA (Grouped Query Attention) rồi đến MQA (Multi-Query Attention) thực chất đều là những phép "trừ" nhằm giảm kích thước KV Cache. Trong khi đó, Attention tuyến tính vốn không gặp phải vấn đề KV Cache, nên có thể tự do phát triển theo hướng ngày càng tốt hơn.

Để thấy rõ hơn điều này, chúng ta hãy viết tất cả các cơ chế Attention đã đề cập dưới dạng ma trận:

| | Công thức |
|---|---|
| Softmax Attention 			| (exp(QK⊤)⊙M)V |
| Attention tuyến tính ban đầu 	| ((QK⊤)⊙M)V |
| Sau khi thêm cổng quên 		| ((QK⊤)⊙Γ)V |
| DeltaNet 						| ((QK⊤)⊙M)(I+(KK⊤)⊙(M-I))^(-1)V |
| Gated DeltaNet 				| ((QK⊤)⊙Γ)(I+(KK⊤)⊙(Γ-I))^(-1)V |

Để thấy rõ điều này, chúng ta hãy so sánh các cơ chế Attention đã được đề cập dưới dạng công thức ma trận. Softmax Attention vẫn giữ nguyên dạng cổ điển với exp(QK⊤)⊙M nhân V. Attention tuyến tính ban đầu đơn giản loại bỏ hàm mũ, chỉ còn (QK⊤)⊙M nhân V. Khi thêm cổng quên, ta có (QK⊤)⊙Γ nhân V, trong đó Γ là ma trận chứa các tích lũy của hệ số suy giảm γ. DeltaNet phức tạp hơn với việc thêm ma trận nghịch đảo, còn Gated DeltaNet kết hợp cả cổng quên và nghịch đảo.

Ma trận Γ được định nghĩa với các phần tử Γi,j bằng tích của các γτ từ j+1 đến i khi i lớn hơn j, bằng 1 khi i bằng j, và bằng 0 khi i nhỏ hơn j. Đây là cách biểu diễn toán học của hiệu ứng quên theo thời gian.

Nhìn vào bảng so sánh này, ta thấy rõ Softmax Attention vẫn đang "dậm chân tại chỗ" ở dạng thức của Attention tuyến tính thời kỳ đầu. Tất nhiên, điều này cũng chứng minh sức mạnh vốn có của Softmax Attention - dù đơn giản nhưng vẫn cực kỳ hiệu quả.

Vậy làm thế nào để thực hiện "phản hồi ngược"? Câu trả lời nằm ở việc chuyển đổi Softmax Attention thành dạng Attention tuyến tính. May mắn thay, điều này không quá khó khăn. Từ rất sớm, trong bài viết "Con đường nâng cấp Transformer: 5, Attention tuyến tính như không gian vô hạn chiều", chúng tôi đã tổng kết ba phương pháp để chuyển đổi Softmax Attention thành Attention tuyến tính vô hạn chiều.

Cốt lõi của phương pháp này là tồn tại một ánh xạ φ (phi) có khả năng biến đổi Q và K từ không gian n×d sang không gian n×∞ (vô hạn chiều). Ánh xạ này thỏa mãn tính chất quan trọng: exp(QK⊤) bằng φ(Q) nhân φ(K) chuyển vị. Đây chính là "thủ thuật hạt nhân" (kernel trick) nổi tiếng trong học máy.

Với công cụ này trong tay, việc tiếp theo trở nên đơn giản. Chúng ta chỉ cần thay thế Q và K trong các công thức Attention tuyến tính bằng φ(Q) và φ(K), sau đó tìm cách khôi phục lại hàm mũ và chuẩn hóa nếu cần. Kết quả là ta thu được các biến thể mới của Softmax Attention với những đặc tính ưu việt được thừa hưởng từ Attention tuyến tính.

Ví dụ điển hình, khi áp dụng phương pháp này vào công thức có cổng quên, ta thu được exp(QK⊤+logΓ)V. Nếu γt là hằng số, đây chính là ALIBI (Attention with Linear Biases) được đề xuất trong "Train Short, Test Long". Nếu γt phụ thuộc vào đầu vào, ta có FoX (Forgetting Transformer) với cổng quên thích ứng.

Một kết quả thú vị hơn nữa là DeltaFormer từ bài báo "Understanding Transformer from the Perspective of Associative Memory". Như tên gọi, đây là phiên bản Softmax của DeltaNet, kết hợp những ưu điểm của cả hai thế giới.

Khi thay thế Q, K của DeltaNet bằng φ(Q), φ(K), chúng ta thu được một công thức phức tạp hơn nhưng cực kỳ mạnh mẽ. Công thức này bao gồm exp(QK⊤+logM) (ký hiệu là A) nhân với nghịch đảo của (I cộng exp(KK⊤+log(M-I))) (ký hiệu là B) rồi nhân với V. Nếu muốn chuẩn hóa, ta chỉ cần thay exp bằng softmax.

So với Softmax Attention gốc chỉ tính AV, DeltaFormer tính A(I+B)^(-1)V. Điều thú vị là khi khai triển (I+B)^(-1) thành chuỗi vô hạn (I-B+B²-B³+...), ta có thể hiểu DeltaFormer như sau: đầu tiên sử dụng K, K, V để tính nhiều lần Attention (V-BV+B²V-B³V+...), cộng dồn các kết quả lại thành V mới, sau đó mới tính Attention một lần nữa với Q, K và V mới này.

Đặc tính "multi-hop" này khiến DeltaFormer đặc biệt hiệu quả cho các tác vụ đòi hỏi suy luận nhiều bước, chẳng hạn như hiểu và sinh code. Hơn nữa, DeltaFormer còn rất phù hợp với kiến trúc MQA (Multi-Query Attention), bởi vì phần tính toán (I+B)^(-1)V chỉ liên quan đến K và V. Với MQA, K và V chỉ có single-head nên lượng tính toán giảm đáng kể so với MHA đầy đủ.

Tuy nhiên, theo quan điểm cá nhân của tôi, việc cộng dồn với hệ số cố định này có thể tuân theo nguyên lý "không có bữa trưa miễn phí". Thí nghiệm của tôi cho thấy loss của mô hình ngôn ngữ với DeltaFormer không thay đổi nhiều so với baseline. Điều này ám chỉ rằng nếu một số tác vụ được cải thiện đáng kể, chắc chắn sẽ có những tác vụ khác bị giảm hiệu suất để "bù lại".

## Kỹ thuật mã hóa cứng

Một công trình phản hồi ngược đáng chú ý khác là PaTH Attention, xuất phát từ bài báo "PaTH Attention: Position Encoding via Accumulating Householder Transformations". Công trình này tiếp cận từ góc độ mã hóa vị trí, đưa ý tưởng của DeltaNet ngược lại vào Softmax Attention.

Trong bài viết "Con đường nâng cấp Transformer: 6, Phân tích tính đầy đủ của mã hóa vị trí xoay", chúng tôi đã chỉ ra rằng với bất kỳ ma trận trực giao Ω nào, Rm=Ω^m đều là RoPE tổng quát. Ngoài ma trận xoay, còn có loại ma trận trực giao nào dễ xây dựng không? PaTH sử dụng ma trận Householder - một công cụ toán học cổ điển nhưng mạnh mẽ.

Cụ thể, nếu w là vector cột bất kỳ có độ dài bằng √2, thì I-ww⊤ là một ma trận trực giao. Điều này chúng tôi cũng đã chứng minh trong "Ma trận trực giao biến đổi từ vector đơn vị này sang vector đơn vị khác". Ý nghĩa hình học của phép biến đổi này là phản xạ gương qua một siêu phẳng.

Điều thú vị là dạng I-ww⊤ này giống hệt với thừa số mà St-1 được nhân trong DeltaNet (với w thay cho kt). Nhận ra điều này, PaTH quyết định "mượn" ý tưởng từ DeltaNet một cách triệt để. Họ từ bỏ ràng buộc về độ dài của w, từ bỏ dạng Ω^m truyền thống, và thay vào đó sử dụng một chuỗi các phép nhân I-ww⊤ để mã hóa thông tin vị trí.

Cụ thể, PaTH biến đổi tích vô hướng qi⊤kj thành qi⊤ nhân với một chuỗi các ma trận (I-wiw⊤i)(I-wi-1w⊤i-1)...(I-wj+1w⊤j+1), ký hiệu tích này là Ri,j, rồi nhân với kj. 

Viết dưới dạng đệ quy, ta có Ri,j bằng (I-wiw⊤i) nhân Ri-1,j, với điều kiện ban đầu Rj,j bằng I. So sánh với công thức DeltaNet, điều này tương đương với việc đặt tất cả vt bằng 0, nhưng ma trận trạng thái ban đầu S0 không còn là ma trận không.

Áp dụng kỹ thuật "nghịch đảo đến trợ giúp" tương tự như với DeltaNet, ta có thể biểu diễn Ri,j dưới dạng tường minh liên quan đến ma trận nghịch đảo. Lưu ý rằng ma trận cần nghịch đảo là tam giác dưới, và ma trận tam giác có tính chất đặc biệt: các phần tử đường chéo của ma trận nghịch đảo bằng nghịch đảo của các phần tử đường chéo tương ứng trong ma trận gốc.

Các phép biến đổi tiếp theo khá phức tạp về mặt ký hiệu, nhưng ý tưởng cốt lõi là tận dụng cấu trúc tam giác và tính chất low-rank để đơn giản hóa tính toán. Kết quả cuối cùng cho thấy PaTH thực hiện một phép biến đổi vị trí cực kỳ tinh vi dựa trên nội dung ngữ cảnh.

Toàn bộ ma trận attention (trước Softmax) trong PaTH có thể được viết dưới dạng:
A = (QK⊤)⊙M trừ đi một số hạng phức tạp liên quan đến ma trận W và nghịch đảo của nó.

Thật ấn tượng phải không? Nhưng đó vẫn chưa phải tất cả. Việc tính toán trực tiếp nghịch đảo có độ phức tạp O(n³), hoàn toàn không thể chấp nhận được. Do đó cần phải tìm cách tận dụng cấu trúc low-rank của WW⊤ để giảm độ phức tạp xuống O(n²). Sau đó còn phải tính đạo hàm cho lan truyền ngược, và cuối cùng là triển khai hiệu quả kiểu Flash Attention. 

Tất cả những chi tiết này đều cực kỳ phức tạp và "cứng", đòi hỏi người đọc phải tự nghiên cứu kỹ bài báo gốc. Tóm lại, toàn bộ quá trình đều rất khó nhằn về mặt kỹ thuật.

Từ góc độ mã hóa vị trí, PaTH là một dạng CoPE (Contextual Position Encoding) - mã hóa vị trí phụ thuộc ngữ cảnh. Vị trí trong PaTH không phải là các con số đơn giản 1, 2, 3,... mà là tín hiệu vị trí được tự động sinh ra dựa trên nội dung ngữ cảnh. Tương tự, FoX cũng có thể được xem như phiên bản ngữ cảnh của ALIBI. Thông tin vị trí phụ thuộc ngữ cảnh là đặc trưng chính của Attention tuyến tính hiện tại, và có thể là hướng chính để phản hồi ngược cải tiến Softmax Attention.

## Niềm vui đơn giản hóa #

Chúng ta hãy đào sâu hơn một chút vào PaTH. Điều này không chỉ giúp hiểu rõ hơn về PaTH mà còn giúp làm quen hơn với DeltaNet, vì hai mô hình này có mối liên hệ chặt chẽ. Phần này chúng ta sẽ xuất phát từ hai trường hợp đặc biệt của PaTH để hiểu rõ hơn mối quan hệ giữa PaTH và DeltaNet.

Trường hợp đặc biệt thứ nhất là khi W=K. Thay vào công thức (28), ta thu được một kết quả thú vị. Sau một số phép biến đổi đại số (sử dụng đẳng thức I-(I+A)^(-1)A=(I+A)^(-1)), ta thấy công thức cuối cùng chính xác là ma trận attention của DeltaNet!

Từ trường hợp đặc biệt này, ta thấy sự khác biệt giữa PaTH và DeltaFormer như sau: DeltaFormer dựa trên thủ thuật hạt nhân, thêm exp vào QK⊤ và KK⊤ của DeltaNet một cách riêng biệt. Trong khi đó, PaTH trực tiếp thêm exp vào toàn bộ ma trận attention của DeltaNet.

Trường hợp đặc biệt thứ hai là khi ta đưa lại ràng buộc ||w||=√2. Lúc này I-ww⊤ trở thành ma trận trực giao thực sự. Ta định nghĩa Ri là tích của các ma trận (I-wiw⊤i) từ i ngược về 1. Với một số phép biến đổi, ta có thể biểu diễn Ri dưới dạng liên quan đến ma trận nghịch đảo, tương tự như cách ta làm với DeltaNet.

Quan trọng hơn, trong trường hợp này ta có Ri,j=RiRj⊤. Đẳng thức này có ý nghĩa to lớn: chúng ta có thể triển khai PaTH theo cách tương tự RoPE, tức là chỉ cần nhân mỗi qi⊤, ki⊤ với Ri tương ứng, sau đó áp dụng triển khai Softmax Attention tiêu chuẩn.

Vậy phép nhân với Ri thực chất là gì? Qua một loạt phép biến đổi phức tạp (mà tôi sẽ không trình bày chi tiết ở đây), ta có thể chứng minh rằng kết quả cuối cùng có dạng:

Q trừ đi ((QW⊤)⊙M) nhân với (I+(WW⊤)⊙(M-I))^(-1) nhân W

Lại một lần nữa ta thấy sự xuất hiện của DeltaNet! Phần thứ hai chính là DeltaNet(Q,W,W). Vì vậy, trong trường hợp này, PaTH thực hiện hiệu ứng tương đương với:

SoftmaxAttention(Q-DeltaNet(Q,W,W), K-DeltaNet(K,W,W), V)

Nói cách khác, PaTH sử dụng DeltaNet để thêm mã hóa vị trí cho Q và K. Nhìn theo cách này, PaTH (với ràng buộc ||w||=√2) tương đương với một dạng kết hợp giữa các lớp Softmax Attention và DeltaNet.

Tất nhiên, chúng ta cũng có thể xem xét việc bỏ qua các suy luận phức tạp ở trên. Ngay cả khi ||w||≠√2, ta vẫn có thể triển khai theo công thức trên. Điều này tương tự như phương pháp Canon Layers, sử dụng tích chập để thêm thông tin vị trí cho Q, K. Chỉ khác là ở đây không phải tích chập ngắn mà là "tích chập dài" dưới dạng DeltaNet.

## Lối đi riêng

Cuối cùng, chúng ta xem xét một mô hình Attention tuyến tính gần đây cũng rất đáng chú ý - MesaNet (còn có một công trình đồng thời tương tự là Atlas). Góc nhìn Online Learning của TTT cho ta biết DeltaNet thực chất đang dùng SGD để tối ưu hàm mục tiêu 1/2||Sk-v||². Nếu quan sát kỹ, ta thấy Sk chỉ là hàm tuyến tính của k, vì vậy đây thực sự chỉ là bài toán hồi quy tuyến tính, mà hồi quy tuyến tính có nghiệm giải tích!

Nghiệm giải tích được biểu diễn như sau: St=GtHt^(-1), trong đó Gt là tổng tích lũy của vjkj⊤ và Ht là tổng tích lũy của kjkj⊤ từ j=1 đến t.

MesaNet chính là tận dụng nghiệm giải tích này để xây dựng mô hình chuỗi. Ý tưởng xuất phát từ "Uncovering mesa-optimization algorithms in Transformers", còn việc huấn luyện hiệu quả được thực hiện bởi "MesaNet: Sequence Modeling by Locally Optimal Test-Time Training". MesaNet thêm cổng quên vào Gt, Ht và thêm ma trận đường chéo Λt khi tính nghịch đảo để tránh trường hợp không khả nghịch.

Rõ ràng, độ phức tạp của Gt, Ht theo độ dài chuỗi là tuyến tính, do đó độ phức tạp tính ot cũng là tuyến tính. Vì vậy MesaNet vẫn thuộc phạm trù Attention tuyến tính. Do có nghiệm giải tích, về cơ bản có thể đảm bảo trong phần lớn trường hợp nó sẽ tốt hơn DeltaNet và thậm chí cả Gated DeltaNet. Từ góc độ xử lý tín hiệu, MesaNet và DeltaNet tương ứng với sự khác biệt giữa Recursive Least Square và Least Mean Square.

Nhìn qua toàn là ưu điểm, vậy tại sao tôi lại xếp nó vào mục "đi lối riêng"? Theo quan điểm cá nhân, MesaNet "thành cũng vì nghiệm giải tích, bại cũng vì nghiệm giải tích". Nghiệm giải tích khiến nó thường tốt hơn DeltaNet, nhưng cũng tạo cảm giác "đến đây là hết đường" - chỉ cần thay đổi một chút là hầu như không còn cơ hội tìm được nghiệm giải tích nữa. Nhìn lại toàn bộ lịch sử toán học, tất cả các nhánh phụ thuộc vào nghiệm giải tích ngày nay hầu như đều đã mai một, bởi nghiệm giải tích thực sự quá hiếm hoi và thiếu tính đại diện.

Từ góc độ triển khai, ma trận cần nghịch đảo Ht+Λt trong MesaNet không phải ma trận tam giác. Mặc dù (Ht+Λt)^(-1)qt vẫn có thể chuyển thành giải phương trình thay vì tính nghịch đảo tường minh, nhưng ma trận không phải tam giác khiến độ phức tạp giải tăng lên đáng kể. Làm thế nào để tính toán song song toàn bộ (Ht+Λt)^(-1)qt với chi phí thấp nhất sẽ là thách thức lâu dài của MesaNet. Hiện tại bài báo sử dụng "phương pháp gradient liên hợp" để tìm nghiệm xấp xỉ - có thể dùng được nhưng không hoàn hảo.

Hơn nữa, về mặt khả năng lý thuyết, MesaNet cũng không hẳn nghiêm ngặt vượt trội hơn DeltaNet. Lý do là quy tắc cập nhật Gt, Ht của MesaNet vẫn chỉ là dạng trung bình trượt đơn giản, phép nghịch đảo cũng không liên quan đến tương tác giữa các token. Do đó, giới hạn khả năng của nó có lẽ không bằng DeltaNet với Delta Rule. 

Hiểu một cách trực quan: MesaNet cố gắng ghi nhớ toàn bộ k,v - trong phần lớn trường hợp đây là điều tốt, nhưng đôi khi dẫn đến ký ức khá mờ nhạt. Trong khi đó, nguyên tắc của DeltaNet là "loại bỏ cũ đón mới" - nhờ có "loại bỏ cũ", nó có thể ghi nhớ lâu dài và chính xác một số nội dung nhất định.

Tóm lại, MesaNet là một mô hình đẹp đẽ và dễ chịu về mặt toán học, nhưng nghiệm giải tích cũng làm tăng độ phức tạp và hạn chế tính linh hoạt của nó, để lại không ít không gian cần khám phá. Nếu độc giả muốn tìm hiểu thêm về việc xây dựng mô hình chuỗi dựa trên hồi quy tuyến tính, có thể đọc thêm TTR - công trình này thảo luận chi tiết các mô hình chuỗi dưới nhiều hàm mục tiêu hồi quy tuyến tính khác nhau.

## Con đường phương hưng vị ngãi (方兴未艾) - đang phát triển và chưa dừng lại

Bài viết này đã lược thuật quá trình phát triển của Attention tuyến tính và giới thiệu nguyên lý toán học của một số mô hình. Attention tuyến tính khởi đầu từ việc bắt chước Softmax Attention, dần dần phát triển đặc trưng riêng, đến nay đã trở thành phương án mô hình hóa chuỗi cực kỳ cạnh tranh, thậm chí ngược lại còn cung cấp ý tưởng mới cho sự phát triển của Softmax Attention. Quá trình này đầy tính thú vị và mang lại nhiều cảm hứng.

Từ những mô hình đơn giản ban đầu, qua việc thêm cổng quên, đến các kỹ thuật nghịch đảo ma trận tinh vi, rồi đến việc phản hồi ngược cải tiến Softmax Attention - hành trình của Attention tuyến tính minh họa rõ ràng cách một ý tưởng có thể tiến hóa và trưởng thành trong nghiên cứu khoa học. Điều quan trọng là không ngừng đổi mới trong khi vẫn giữ được những nguyên lý cốt lõi.

Với sự phát triển mạnh mẽ của các mô hình ngôn ngữ lớn và nhu cầu xử lý chuỗi ngày càng dài, **Attention tuyến tính chắc chắn sẽ còn tiếp tục phát triển**. Có thể trong tương lai không xa, **ranh giới giữa Attention tuyến tính và Softmax Attention sẽ càng mờ nhạt**, và chúng ta sẽ chứng kiến **sự ra đời của những kiến trúc lai ghép mạnh mẽ hơn nữa**.

---

Vậy TTT là gì?
--------------

Test-Time Training (TTT) là một cách tiếp cận độc đáo để xây dựng mô hình xử lý chuỗi. Thay vì phải tự thiết kế các cơ chế phức tạp, TTT cho phép chúng ta sử dụng các thuật toán tối ưu quen thuộc như SGD để tự động xây dựng mô hình.

Cách TTT hoạt động khá đơn giản. Khi nhận được một chuỗi dữ liệu mới, nó coi các cặp (K, V) như một bộ dữ liệu huấn luyện nhỏ. Sau đó, nó dùng SGD để cập nhật tham số của mô hình dựa trên dữ liệu này. Cuối cùng, mô hình đã được cập nhật sẽ đưa ra dự đoán cho câu hỏi hiện tại.

**TTT nhìn nhận vấn đề từ góc độ nén dữ liệu**. Mô hình đóng vai trò như bộ giải nén, tham số của nó là file nén, còn SGD chính là thuật toán nén. Cách nhìn này giúp chúng ta hiểu rõ hơn **bản chất của việc xử lý chuỗi - lưu trữ thông tin quan trọng trong không gian hạn chế**.

TTT ĐẶC BIỆT PHÙ HỢP VỚI HỌC TRONG NGỮ CẢNH VÌ NÓ LIÊN TỤC HỌC TỪ DỮ LIỆU ĐANG XỬ LÝ. Các biến thể khác nhau của Attention tuyến tính như DeltaNet hay RetNet thực chất chỉ khác nhau ở hàm mất mát được sử dụng trong quá trình tối ưu.

[TTT: Test-Time Training - huấn luyện tại thời điểm kiểm tra; SGD: Stochastic Gradient Descent - giảm gradient ngẫu nhiên; RNN: Recurrent Neural Network - mạng nơ-ron hồi quy; ICL: In-Context Learning - học trong ngữ cảnh]

TTT đã có một số tiến bộ quan trọng theo từng giai đoạn phát triển:

**Giai đoạn đầu - TTT gốc:**
TTT ban đầu chỉ sử dụng SGD đơn giản với mini-batch nhỏ. Mặc dù đã chứng minh được ý tưởng cốt lõi, nhưng hiệu suất còn hạn chế và chưa thể cạnh tranh với các phương pháp truyền thống.

**Cải tiến với Titans:**
Titans đã thêm momentum vào SGD, giúp quá trình tối ưu ổn định hơn và hội tụ nhanh hơn. Đây là bước tiến quan trọng vì momentum giúp vượt qua các điểm cực tiểu cục bộ trong quá trình học.

**Đột phá với TTT Done Right:**
Đây là cải tiến lớn nhất, với hai điểm chính. Thứ nhất, họ khám phá cách sử dụng large-batch hiệu quả, giúp tận dụng tốt hơn sức mạnh tính toán song song của GPU. Thứ hai, họ kết hợp với Muon optimizer - một bộ tối ưu mới mạnh mẽ hơn SGD truyền thống.

**Kết quả thực tế:**
Các cải tiến này đã giúp TTT từ một ý tưởng thú vị trở thành phương pháp thực sự cạnh tranh. TTT Done Right đã cho thấy hiệu suất tương đương hoặc vượt trội so với Transformer truyền thống trên nhiều tác vụ, đặc biệt là các tác vụ đòi hỏi khả năng học từ ngữ cảnh.

[Momentum: động lượng - kỹ thuật giúp tăng tốc độ hội tụ; Mini-batch: nhóm nhỏ dữ liệu; Large-batch: nhóm lớn dữ liệu; Muon: một thuật toán tối ưu mới]

Bài báo có tên là **"Learning to (Learn at Test Time)"** được công bố vào năm 2024. Tiêu đề bài báo rất khéo léo với cấu trúc "Learning to (Learn at Test Time)" - vừa nhấn mạnh việc học tại thời điểm test, vừa gợi ý đến **meta-learning**. Dấu ngoặc đơn trong tiêu đề tạo ra hai cách đọc: "Learning to Learn at Test Time" và "Learning at Test Time". Bài báo có ba đóng góp chính rất quan trọng:

**1. Khung lý thuyết mới:**
Lần đầu tiên, họ chứng minh rằng các bộ tối ưu như SGD thực chất là RNN. Khi cập nhật tham số qua nhiều bước, SGD tạo ra một chuỗi trạng thái - đây chính là bản chất của RNN. Phát hiện này mở ra cách nhìn hoàn toàn mới về mối quan hệ giữa tối ưu hóa và mô hình chuỗi.

**2. Thiết kế kiến trúc TTT-Linear:**
Họ đề xuất một kiến trúc cụ thể với hai thành phần chính. Phần "inner loop" sử dụng self-supervised learning để cập nhật tham số dựa trên dữ liệu hiện tại. Phần "outer loop" là mô hình chính sử dụng các tham số đã cập nhật để đưa ra dự đoán. Thiết kế này vừa đơn giản vừa hiệu quả.

**3. Chứng minh thực nghiệm:**
Bài báo cho thấy TTT-Linear ĐẠT HIỆU SUẤT TƯƠNG ĐƯƠNG TRANSFORMER TRÊN NHIỀU TÁC VỤ NGÔN NGỮ. Đặc biệt ấn tượng là khả năng **xử lý chuỗi dài** và **HỌC TỪ NGỮ CẢNH** - hai điểm yếu của RNN truyền thống.

[Self-supervised learning: học tự giám sát; Inner/Outer loop: vòng lặp trong/ngoài; RNN: Recurrent Neural Network - mạng nơron hồi quy]

---

TTT DONE RIGHT
--------------
- https://www.alphaxiv.org/abs/2505.23884
- https://youtu.be/5QxQUr-m_2w
- https://asap-seminar.github.io/assets/slides/Test-Time%20Training%20Done%20Right.pdf
- https://tianyuanzhang.com/projects/ttt-done-right

![](https://pbs.twimg.com/media/GuBGLtSXIAA6SQ9?format=jpg)
![](https://pbs.twimg.com/media/GuBHkvNWIAEMXWr?format=jpg)

Tác giả giải thích TTT nói chung dưới sự diễn đạt của attn để sử dụng 1 ngôn ngữ chung (có thể dễ hiểu cho nhiều người vì transformers đã quá phổ biến).

**Chuỗi đầu vào:**
Chuỗi x gồm các token x₁, x₂,..., xₜ, mỗi token là vector d chiều. Đây là dữ liệu text đã được mã hóa thành số mà mô hình sẽ xử lý.

**Tách thành Q, K, V:**
Mỗi token xᵢ được biến đổi thành ba phần - query (q), key (k), và value (v).

**Fast weight function fw(.):**
Đây là hàm ánh xạ từ Rᵈ sang Rᵈ với tham số `w`. Hàm này chính là **"mô hình nhỏ" mà TTT liên tục cập nhật**. Nó có thể đơn giản như hàm tuyến tính hoặc phức tạp như MLP hay Transformer.

**`w` như bộ nhớ thích ứng:**
Tham số `w` đóng vai trò như "bộ nhớ" của hệ thống. Nó được cập nhật liên tục khi xử lý chuỗi, lưu trữ thông tin từ các token đã thấy. Đây là điểm khác biệt chính với Transformer thông thường.

**cập nhật tham số:** `W = W - delta_W L(f_W(k), v)`, the loss is function between value `v` and `f_w(k)` => KEY-VALUE RECONSTRUCTION LOSS!

|![](https://pbs.twimg.com/media/GuBIc64XcAADFXx?format=jpg)|![](https://pbs.twimg.com/media/GuBJZX6WcAA3XEu?format=jpg)|
|-|-|

**Giới thiệu về Memory Module:**

- **Giả sử bạn có KV cache** - có thể bạn muốn đọc một cuốn sách và KV cache này là "sách KV cache". Điều bạn làm là mỗi token sẽ được tách thành key và value - đó là ký hiệu của attention. Bạn có key và value tương ứng được ghép cặp.

- **Điều bạn muốn làm** là nén cặp key-value, nén các liên kết key-value vào memory module. Ở đây bạn không cần nghĩ memory module là gì - chúng ta sẽ nói về điều đó sau. Bạn chỉ muốn nén cặp key-value vào memory module này.

- **Khi token mới đến** - khi bạn muốn trả lời câu hỏi trong tương lai, khi token tương lai nào đó đến, query token này sẽ không attend trực tiếp với K và V, nó sẽ attend với memory module và nhận output.


**Framework này có hai thao tác cơ bản:**

1. **Memory Update** - Bạn muốn cập nhật memory module để lưu trữ thông tin quá khứ

2. **Memory Query** - Khi token mới đến, bạn không chỉ cập nhật memory mà còn muốn truy vấn memory để lấy output

**Tôi nghĩ Test-Time Training đưa ra cách đơn giản hoặc có nguyên tắc để định nghĩa các thao tác memory update và memory query mới.**


**Ký hiệu và Định nghĩa:**

Giả sử bạn muốn xử lý một chuỗi token gồm n token. Mỗi token x₁, x₂, xᵢ là vector d chiều. Giả sử nó là causal theo thời gian.

Theo ký hiệu của attention, bạn muốn tách mỗi token x thành query, key, value - chúng được ghép cặp.

**Test-Time Training giới thiệu ký hiệu mới:** Bạn muốn điều chỉnh một phần mô hình với test. Những trọng số này thường được gọi là **fast weight** - thường là một phần của mạng nơ-ron, có thể là một lớp hoặc một MLP riêng biệt bên trong mạng.

Chúng ta thường gọi nó là **fast function** hay **fast weight function**. Fast function này có thể là linear layer hoặc MLP layer - chỉ là một hàm có tham số. Tham số là W, nhận đầu vào d chiều và xuất ra vector d chiều khác.

**Trọng số W của fast weight chỉ được điều chỉnh trong chuỗi hiện tại.** Chúng ta sẽ điều chỉnh trọng số này cho chuỗi đầu vào X, và trọng số đã điều chỉnh sẽ lưu trữ bộ nhớ của chuỗi đầu vào.

Fast weight này có thể là bất cứ thứ gì - có thể là mạng tuyến tính, MLP, hoặc thậm chí transformer nếu bạn muốn.

---

**Test-Time Training định nghĩa:**

**Memory Update như online gradient descent dưới các mục tiêu nhất định.** Bạn muốn điều chỉnh fast weight W thông qua gradient descent, và gradient descent cố gắng tối thiểu hóa hàm loss L.

Hàm loss L là hàm giữa value và f(K). Phần lớn công trình dùng ký hiệu này - loss được định nghĩa trên K và V.

**Hàm loss phổ biến là key-value associations:**
- L có thể là dot product âm giữa f(K) và V
- Hoặc L2 loss giữa f(K) và V

Tất cả những loss này được gọi là **key-value association loss** hoặc **key-value reconstruction loss**. Ý nghĩa là bạn muốn buộc mô hình (fast weight function) ghi nhớ ánh xạ từ K sang V - về cơ bản là ghi nhớ liên kết giữa key và value.


## Test-Time Training mở ra không gian thiết kế rộng lớn

Framework TTT tạo ra một không gian thiết kế cực kỳ phong phú - đây là một trong những đóng góp quan trọng nhất của bài báo TTT.

### Các chiều không gian thiết kế:

**1. Fast Function (Hàm nhanh):**
Bạn có thể sử dụng bất kỳ kiến trúc nào - từ đơn giản như linear layer đến phức tạp như mạng nơ-ron phi tuyến, thậm chí cả transformer. Miễn là nó có tham số có thể học được, bạn đều có thể dùng làm fast weight function.

**2. Hàm mục tiêu huấn luyện:**
TTT cho phép đa dạng hóa hàm loss:
- Dot product loss hoặc L2 loss cho key-value association
- Next token prediction loss 
- Denoising loss như trong các mô hình khử nhiễu
- Bất kỳ hàm loss tự giám sát nào khác

**3. Bộ tối ưu (Optimizer):**
Từ vanilla SGD đơn giản đến các phương pháp tinh vi hơn như gradient descent với momentum. Trong bài nói này, tác giả còn thử nghiệm với Muon optimizer - một bộ tối ưu mới.

### Về gradient bậc hai

Nhiều người thắc mắc liệu TTT có tính gradient bậc hai không. Thực tế, đây chỉ là một chuỗi chain rule dài hơn bình thường:

**Quy trình forward:**
1. Weight update: Cập nhật trọng số W
2. Memory query: Truy vấn để lấy output

**Quy trình backward:**
Gradient lan truyền ngược từ output → W và Q → fast weight ban đầu → K và V

Đây chỉ là chuỗi backpropagation dài hơn với nhiều phép nhân ma trận. Không có vector-Jacobian product hay các phép tính phức tạp của gradient bậc hai. Về bản chất, nó chỉ làm mô hình "sâu hơn" theo một nghĩa nào đó.

### Tối ưu phần cứng - Yếu tố then chốt

**Tensor Cores và hiệu năng:**
GPU H100 có khả năng xử lý lý thuyết khoảng 2,000 teraflops với BFloat16 tensor cores, tương đương 1,000 nghìn tỷ phép tính dấu phẩy động mỗi giây cho phép nhân ma trận dày đặc.

**Điểm quan trọng:** Con số này chỉ đạt được với phép nhân ma trận-ma trận, không phải ma trận-vector. Kích thước tối thiểu của ma trận là 16×16.

**Hệ quả thực tế:**
- Nếu chỉ làm phép nhân ma trận-vector, bạn phải padding vector lên 16 lần
- Điều này dẫn đến hiệu suất GPU chỉ đạt dưới 8%

**Nguyên tắc thiết kế:**
Khi triển khai TTT, cần đảm bảo mọi phép tính đều là nhân ma trận-ma trận càng nhiều càng tốt. Tránh xử lý từng token riêng lẻ với phép nhân ma trận-vector.

[Tensor cores: lõi tính toán chuyên biệt cho phép nhân ma trận; Teraflops: nghìn tỷ phép tính dấu phẩy động/giây; Chain rule: quy tắc dây chuyền trong tính đạo hàm]

|![](https://pbs.twimg.com/media/GuBSEhDaoAAUy62?format=jpg)|![](https://pbs.twimg.com/media/GuBS2-CWcAExuTC?format=jpg)|
|-|-|

![](https://pbs.twimg.com/media/GuBTF2QW8AAk_MI?format=jpg)

## Storing long contexts in tiny caches with self-study
- https://hazyresearch.stanford.edu/blog/2025-06-08-cartridges

|![]()|![]()|
|-|-|
|![]()|![]()|


---

Learning to (Learn at Test Time)
--------------------------------
https://www.alphaxiv.org/abs/2407.04620

...

