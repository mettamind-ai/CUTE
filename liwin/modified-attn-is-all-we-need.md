Dựa trên idea về tổ hợp của Nam anh thấy thế này. Hiện tại ở mỗi layer (hay module) của LLM người ta hay dùng kiến trúc đồng nhất. Ví dụ với channel mixer thì là 1 loại FFN cố định, với seq mixer thì là softmax attn chẳng hạn. Giờ mình phát triển tiếp theo hướng mixture ko chỉ là chia nhỏ tham số mà cả đa dạng thêm các loại chức năng (đa dạng cách biểu diễn) thì sao nhỉ? Ở mỗi layer / module mình có nhiều experts, mỗi expect có thể thuộc về 1 trong nhiều phương pháp học / biểu diễn. Có thể là FFN, softmax attenion, mamba, RNN và vô số cách học hiệu quả khác ... Nó là Mixture of Anything we can think about sẽ làm tăng khả năng kết hợp lên vô tận ...

Các block trong 1 module sẽ được tính toán song song và tổ hợp lại. Kết hợp cơ chế routing để token chọn block hoặc block chọn token để tăng tính chuyên môn hoá. Mỗi token sẽ có rất nhiều lựa chọn để kết hợp với các tokens trước nó chứ ko chỉ đơn thuần là softmax attenion nữa ... Mà có thể là trừu tượng hoá (nén), khử nhiễu (bỏ qua tokens hoặc channels / thông tin không quan trọng), error correction (à tới bước này đã có đủ thông tin để sửa lỗi cho các bước trước rồi) ... 

# MODIFIED ATTN IS ALL WE NEED

Tiếp tục suy nghĩ về các **cơ chế** trong token mixing (seq mixing / softmax attn / RNN /  linear attn như mamba2 ... đều là nó). Như mọi người biết trước (softmax) attn thì RNN (dạng hồi quy với số lượng trạng thái nhớ hữu hạn) là bá chủ. Sau attn thì nhờ tính song song hoá cao (lợi computing khi huấn luyện) và khả năng retrival chính xác nhờ trạng thái nhớ KV cache mở rộng cùng với contex length nên attn trở thành first choice và 8 năm nay chưa có đối thủ. Công thức attn gốc đẹp và đơn giản: `attn = softmax(Q @ K) @ V` có thể biến thể thành phần heads thành `MHA`, `GQA` hoặc biến thể phần **reused / compressed KV** thành `GQA` (llama, mistral, qwen ...), `MLA` (deepseek, kimi k2), `GTA` (grouped tied attn - nhóm của Trí mới đề xuất) ... và nhiều kỹ thuật *nén trực tiếp KV cache* trong lúc inference nữa.

Anh sẽ không đi sâu vào công thức toán, vì anh ko giỏi toán (nhìn công thức dài bị loạn) và công thức toán google hoặc hỏi AI là ra, anh muốn bàn luận với mọi người nhiều hơn về góc nhìn trực quan, và cách thức hoạt động của những **CƠ CHẾ** trong token mixing để từ đó có thể **chọn lọc** và **tổ hợp** để tạo nên nhiều sự kết hợp như những gì mình nói phía trên.


Quay trở lại `RNN vs attn` thì góc nhìn khi `inference` thì là bọn nó giống nhau ở chỗ là có 1 **trạng thái nhớ** (với rnn thường là cố định còn attn là mở rộng của ctxlen) và khi thêm 1 token mới thì token mới sẽ kết hợp với trạng thái nhớ hiện tại để tạo nên một trạng thái nhớ mới và tạo ra một output mới. Nếu nhìn như vậy thì hạn chế của rnn (truyền thống) so với attn chỉ là ở 2 điểm: 1/ trạng thái nhớ cố định 2/ khó song song hoá khi huấn luyện. Còn ưu điểm là a) inference rất nhanh vì trạng thái nhớ không đổi và không phải attend b) khi trạng thái nhớ không đổi buộc model phải học cách **compress** nhiều hơn từ đó khử nhiễu và học `abstraction` tốt hơn.

Câu hỏi đặt ra là `RNN học được gì từ Attn và ngược lại Attn học được gì từ RNN?` giờ mình sẽ đi sâu phân tích từng vế của câu hỏi trên.


## RNN học được gì từ Attn?

### Tính song song hoá
Bắt đầu từ paper Linear Attn của Apple (khoảng năm 2022 nếu nhớ không nhầm), họ biến đổi công thức softmax attn `attn = sofmax(Q @ K) @ V` bằng cách bỏ softmax đi `linear attn = (Q @ K) @ V` rồi hoán vị `linear attn = Q @ (K @ V)` khiến attn từ độ phức tạp `n^2` (do Q @ K tăng dần theo ctxlen) thành độ phức tạp cố định `d^2` (do K @ V chỉ phụ thuộc vào model dim là 1 gia trị cố định). Như vậy linear attn vừa song song hoá như attn lại vừa có độ phức tạp cố định (nên gọi là linear). Rất tuyệt vời đúng không nhưng cái gì cũng có cái giá của nó, linear attn hoạt động tệ so với attn. Tại sao?

Bằng 1 cách rất hay ho, họ biến đổi được công thức linear attn thành dạng hồi quy (recurrent) `St = St-1*xxx + yyy gì đó` thay đổi, công thức hồi quy cho chúng có thêm nhiều cơ chế mạnh mẽ (forget, gated, delta, mở rộng St - trạng thái nhớ ...) rồi sau đó quy ngược lại về công thức linear attn gốc (có thể cần biết đổi 1 chút) để công thức biến đổi có thể tính toán song song theo dạng chunk. Và từ đó RNN hiện đại ra đời, có thể tính toán ở cả dạng hồi quy và ở dạng song song theo chunk. Khi training  hoặc pre-fill thì dùng dạng song song, khi inference từng token thì dùng dạng hồi quy, rất linh hoạt.

Và nhờ việc thay đổi công thức linear attn nên các dạng RNN hiện đại cũng mạnh lên trông thấy, điển hình là RWKV, xLSTM, Mamba, DeltaNet, ... trong đó `Mamba` nổi lên như là 1 lựa chọn phổ biến để dùng riêng hoặc hybrid với attn, không phải là do nó tốt nhất (mỗi dạng token mixing sẽ tốt ở 1 khía cạnh cụ thể) mà (phần nhiều) là do nó được Trí Đào tối ưu phần kernel và implement nên tốc độ training tốt hơn các cơ chế kia.

Nhân tiện nói về training speed, các papers họ làm cho mọi người có ảo tưởng rằng ồ linear attn (hay RNN hiện đại) cần ít phép tính hơn và độ phức tạp không đổi nên nó LUÔN nhanh hơn (flash) attn. Theo kinh nghiệm của anh khi dùng khéo thì dưới 8k ctxlen, flash attn là vô địch về tốc độ do tận dụng GPU tối đa.


### RNN học được gì từ Attn? => Mở rộng trạng thái nhớ
Gần đây có paper `log linear attn` và hình như có vài paper sử dụng linear attn nhưng có năng lực mở rộng trạng thái nhớ theo thời gian (giống attn mở rộng KV cache theo ctxlen) và nhờ đó tăng năng lực retrival của (log) linear attn. Phần này anh chưa tìm hiểu nên nói ngắn gọn vậy thôi.

---

Nói thêm 1 chút về tính hồi quy trong LLM, thì nó có 3 dạng: token mixing (đã nói ở trên), dạng layers (model càng depth càng abstract tốt -  gọi nó là hierarchy cũng được), dạng loop lại bằng cách thêm token (CoT, reasoning model, trả lời với nhiều tokens hơn ... ) - có paper họ chỉ ra rằng chỉ cần thêm (dump) tokens vào prompt cũng khiến perf tăng thì nó là thuộc dạng hồi quy bằng cách loop bằng token.

Có người nói transformer thuần thuộc `TC0` (ko giải quyết đc bài toán phức tạp) thực ra họ đang nói cơ chế softmax attn còn khi thể hiện thành model (transformer) thì mọi chuyện không đơn giản như vậy. Do cơ chế hồi quy / phân tầng / loop token mà transformers được mở rộng năng lực mặc dù chỉ dùng softmax attn cho token mixing.

Đoạn này thì giống như so sánh speed của linear attn vs softmax attn, có thể vô tình (thuần lý luận) hoặc cố ý (đẹp paper) mà nhiều cách viết sẽ khiến mình hiểu nhầm là softmax attn thực ra không mạnh đến thế đâu và transformers thuần còn yếu lắm. Thực tế không đơn giản như vậy vì model không chỉ có mỗi softmax attn.


Vế 2 của câu hỏi gốc
## Attn học được gì từ RNN hiện đại (hay linear attn)?
### Giới hạn độ phức tạp
#### SWA  (sliding window attn) 
Nếu anh nhớ không nhầm thì Mistral là model đầu tiên ứng dụng mạnh SWA khi mà họ dùng 4k window ở mọi layers và từ đó độ phức tạp của attn không phải O(n^2) với n là độ dài context nữa mà là cố định O(window^2) với mọi context length. Hình như mistral có thể mở rộng ngữ cảnh lên 32k ctxlen. WHY? Là vì cơ chế hồi quy / xếp chồng layers khiến layer sau `nối dài` ctxlen cho layer trước việc tuy attn ở từng layer bị hạn chế ở 4k nhưng khi xếp chồng lên nhau lại hoạt động được ở ctxlen dài hơn. Hình như SWA lần đầu xuất hiện trong longformer paper, để xử lý long context mà không đội computing ... Về mặt toán học thì thay vì phải bỏ softmax thì mình thu liễm độ dài của Q và K là xong.

#### Sparse Attn
Có thể hiểu đây là mở rộng của SWA cũng được, vì nó có tác dụng cố định lại số tokens cần phải attend dù cho ctxlen có dài tới đâu đi chăng nữa. Hiện cách làm phổ biến là dùng sliding window tính attn score to từng block rồi sau đó dùng top-k để chọn ra k block có điểm số cao nhất và chỉ attend trong số `top-k blocks` đó thôi (số tokens cần attn là không đổi, giống SWA).

Khi window = 1 và sliding = 1 thì nó là cơ chế lựa chọn từng token một (độ phân giải mịn nhất của sparse attn). Nói đến đây thì nó lại có sự liên hệ nhẹ tới hàm kích hoạt thưa trong FFN, như ReLU hoặc top-k sparse ... Phần này cũng thú vị nhưng nói sau vì không liên quan tới chủ đề chính.


### (Attn học được gì từ RNN) các cơ chế CHỌN LỌC thông tin

Vì RNN gốc sử dụng hữu hạn trạng thái nhớ nên khi cập nhật thông tin mới (ở current step) thì nó buộc phải bỏ bớt thông tin cũ ra (nếu không sẽ bị NaN vì phép cộng / nhân thông tin mới với trạng thái cũ sẽ khiến state của nó mở rộng ra vô hạn) và vì thế nó phải bổ xung thêm các **cơ chế chọn lọc thông tin** cũ trước khi cập thông tin mới. Điển hình là 2 phép biến đổi:

- Quên `St = alpha * St-1 + tt_mới` với alpha là 1 hệ số < 1, có thể là fixed (data dependence) hoặc learnabe (data dependence). Tất nhiên là learnable tốt hơn và Mamba thuộc dạng này.

- Xoá với công thức delta rule (anh chưa hiểu rõ lắm nên ko viết ra), đại loại là trước khi cập nhật thông tin mới mà địa diện cặp `KV` hiện tại, thì nó tìm trong `St` thông tin chính xác liên quan tới current `K`, xoá nó đi rồi sau đó mới cập nhật current `V` vào.

Rồi thông qua việc quên / xoá (có chọn lọc) người ta nói RNN đang **nén** thông tin của toàn bộ context vào 1 trạng thái nhớ hữu hạn của nó và thế là global attn ra đời (như trong paper NSA của DeepSeek) thực tế nó là 1 phép nén dữ liệu của toàn bộ context vào 1 hidden vector sau đó cộng vector đó với các biến đổi khác để tạo ra output vector. Còn cơ chế nén để tạo ra vector đó thì vô cùng nhiều, mình sẽ không đi sâu vào chi tiết.

Đến đây thì mình đã thấy attn sau khi được bổ xung cơ chế nén (global attn) từ RNN, kết hợp với cơ chế chọn lọc token để attend (SWA và sparse attn) thì đã trở nên vừa mạnh mẽ hơn vừa tiết kiệm computing hơn *<= đọc NSA (Native Sparse Attention) paper để biết chi tiết*. Điều này theo anh là do 1/ dữ liệu lởm nhiều token kém chất lượng nên làm full attn bị distraction nên cần nhờ sparse hay còn gọi là selected attn giúp loại bỏ bớt tokens kém chất lượng 2/ cơ chế nén của RNN trong global attn.


## Attn học được gì từ RNN ... (tiếp)
Giờ mới tới phần anh cảm thấy thú vị nhất khi mà người ta mang những cơ chế hay ho của RNN áp dụng trực tiếp vào softmax attn. (mình tạm bỏ qua các cách hybrid bằng trộn layer hay kết hợp các cách token mixing trong cùng 1 layer nhé, vì nó chưa tinh vi bằng cách áp dụng trực tiếp này).

- `FoX` forgetting transformer, mang cơ chế **quên** từ rnn vào attn, chỉ cần modified một chút flash attn là đc, 1 góc nhìn khác nó giống Alibi nhưng hệ số decay học được nên khi dùng FoX thì không cần positional embedding nữa và việc mở rộng ctx trở nên dễ dàng hơn (Alibi đã chứng minh điều này). Theo paper thì FoX performance cũng tốt hơn attn truyền thống => Rất đáng để thử!

- `DeltaFormer` áp dụng delta rule vào `V`alue trong QKV của attn để tạo ra `U` va dùng `U` thay cho V, nhờ đó chả phải sửa flash attn luôn mà áp dụng trực tiếp luôn. Theo paper thì nhờ delta rule mà deltaformer có năng lực tracking state tốt hơn hẳn và có năng lực tự học mở rộng context bằng việc sắp xếp data theo thứ tự từ ngắn tới dài ... (paper này họ giải thích attn và ffn dưới góc nhìn associate memory rất thú vị).

- `PaTH` một cách positional embedding mới mạnh mẽ hơn RoPE nhiều khiến gia tăng năng lực cho attn. Có vẻ thú vị nhưng anh không đọc kỹ vì sẽ tính toán phức tạp hơn và với góc nhìn của người thực hành thì `FoX` thú vị hơn nhiều khi vừa gia tăng năng lực với công thức đơn giản vừa giúp bỏ được RoPE.


## Thế cuối cùng muốn nói điều gì?
Trong `FoX` họ có nói softmax attn có thể biểu diễn bằng công thức linear attn khi mà model dim kéo dài tới vô hạn. Điều đó chứng tỏ attn vẫn có sức mạnh vượt hơn linear attn, cộng thêm công thức đơn giản dẫn tới việc dễ chế độ / dễ triển khai / dễ tối ưu hoá ... nên nó vẫn là thứ thống trị hiện nay trong token mixing.

Mặt khác attn sau 8 năm đã không ngừng tiến hoá, từ MHA như trong paper `Attn is all you need` (2017), cho tới GQA, MLA, GTA ... rồi selected attn (như trong SWA, sparse attn), rồi nén (với global attention hoặc nén trực tiếp KV cache khi inference) nó đã trở thành `modified attn (maybe) is all we need`.

Ngoài ra còn có rất nhiều cơ chế mới đang được phát minh / tích hợp trực tiếp vào attn như `FoX`, `DeltaFormer`, `ttt` (test time training) ... giúp attn vẫn giữ được sức mạnh gốc và cải thiện các điểm yếu cố hữu (độ phức tạp bình phương, thiếu tính biểu đạt do không có cơ chế nén, dễ bị sao nhãng bởi tokens kém chất lượng ...)

Tương lai sẽ là áp dụng attention gốc + các cơ chế mới một cách linh hoạt. Tưởng tượng cũng là `flash attn` nhưng mình đổi đi 1 chút để nó thể **masking linh hoạt** (hay selected attn linh hoạt qua masking), **khả năng quên** điều chỉnh được qua hệ số, **delta rule** áp dụng vào `V`, **khả năng nén** vào trực tiếp token value hoặc nén global điều chỉnh được qua hệ số ... và *những thứ điều chỉnh được mình biến nó thành learnable params* thì tự nhiên sẽ có cơ chế tổ hợp như <@575939874411249664> đã chỉ ra, và để cho model tự chọn hàm lượng của từng cơ chế (qua learnable hệ số điều chỉnh của từng cơ chế) thì mình sẽ có được Mixture of Anthing (we can think about) một cách đơn giản, mạnh mẽ (vẫn là flash attn) và không tốn nhiều tham số như MoE (ko phải nhân lên nhiều experts mà chỉ là phối kết hợp các hyper params của hệ số điều chỉnh) ...



## GPU poor vẫn muốn pretrain thì sao nhỉ.
Nói như vậy không phải là MoE không hay, MoE rất hay khi GPU rich, còn với GPU poor và tận dụng gamming gpus thì sẽ cần tìm một lối đi riêng. Lối đi đó có thể là MoA (mixture of anything) như trình bày ở trên kết hợp với việc loop (hidden) token một cách linh hoạt (idea spiral và MPAS của <@575939874411249664>) sẽ khiến tạo ra một model nhỏ nhưng có võ ...

Một điểm nữa anh muốn bênh softmax attn là người ta hay xoáy vào KV cache để nói attn phức tạp không cần thiết nhưng hãy nhìn MoE mà xem, khi mà hiện tại K2 đã mở rộng lên 1T params nhưng chỉ có 32b params là active, vậy cái gì là dư thừa, là phức tạp tưởng như không cần thiết mà lại đem lại hiệu quả? Đó là số experts không được active ở step này nhưng vẫn nằm trong VRAM đó thôi. Thì nhìn lại KV cache cũng thế 😄 Nó là sự phức tạp / thừa thãi cần thiết để có 1 cơ chế retrieval mạnh, cũng như sự phức tạp / thừa thãi của MoE để có 1 cơ chế học có chọn lọc mạnh. Ngoài ra thì KV cache cũng đã có cực kỳ nhiều tiến hoá rồi nên gánh nặng không phải ở đó nữa đâu, nhiều paper họ so sánh perf trong phương pháp mới của họ với KV cache chưa tối ưu hoặc chưa đổi mới thì đó là bias.

Một lần nữa anh vẫn thấy (flash) attn là #1 khi thực hành pretrain LLM. Nó cần được **liên tục tiến hoá** để thêm những cơ chế hay ho khác, liên tục giảm sức ì của KV cache khi inference thì thiết nghĩ MODIFIED ATTN IS ALL WE NEED! có lẽ vẫn còn đúng trong 1 thời gian dài nữa. Hy vọng sẽ có sự đổi mới gốc rễ từ phần cứng để các cơ chế tinh vi hơn về mặt toán học có thể triển khia hiệu quả trên phân cứng thì lúc đó có lẽ attn sẽ bị thay thế! Và ta sẽ có những thế hệ model mới còn mạnh mẽ hơn nhiều!
