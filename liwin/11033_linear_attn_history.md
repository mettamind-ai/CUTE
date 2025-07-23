@online{kexuefm-11033,  
        title={线性注意力简史：从模仿、创新到反哺},  
        author={苏剑林},  
        year={2025},  
        month={Jun},  
        url={\url{https://www.kexue.fm/archives/11033}},  
}
s
Trong cộng đồng tiếng Trung, trang web này có thể coi là một trong những nơi sớm quan tâm đến Linear Attention. Khi viết bài blog đầu tiên vào năm 2020 "Khám phá Linear Attention: Attention có bắt buộc phải có Softmax không?", mọi người chủ yếu vẫn đang thảo luận về Softmax Attention liên quan đến BERT. Nhìn lại, việc xem xét Linear Attention trong thời đại BERT không phải là quyết định sáng suốt, vì độ dài huấn luyện lúc đó còn ngắn và mô hình chủ yếu là Encoder, nên sử dụng Linear Attention hầu như không mang lại lợi thế gì. Về vấn đề này, tác giả cũng đã từng viết bài "Linear Transformer có lẽ không phải là mô hình bạn đang chờ đợi" để bày tỏ quan điểm này.

Mãi đến khi ChatGPT ra đời, buộc mọi người phải chuyển sang làm mô hình sinh Decoder-only, điều này cực kỳ phù hợp với dạng RNN của Linear Attention. Đồng thời, việc theo đuổi độ dài huấn luyện lớn hơn cũng khiến điểm nghẽn độ phức tạp bậc hai của Softmax Attention ngày càng rõ rệt. Trong bối cảnh mới này, Linear Attention ngày càng thể hiện tính cạnh tranh, thậm chí còn **"đóng góp ngược trở lại" cho Softmax Attention**.

## Độ phức tạp bậc hai

Đầu tiên giới thiệu một số ký hiệu:
```js
qi,ki,vi,oi∈Rd×1
Q=[q1,q2,⋯,qn]⊤∈Rn×d
K=[k1,k2,⋯,kn]⊤∈Rn×d
V=[v1,v2,⋯,vn]⊤∈Rn×d
O=[o1,o2,⋯,on]⊤∈Rn×d (1)
```
  
Một mô hình Attention về bản chất là một ánh xạ từ `Q,K,V → O`. Bài viết này tập trung vào trường hợp Causal, nghĩa là `ot` chỉ liên quan tối đa tới `Q[:t], K[:t], V[:t]`. Về nguyên tắc, chiều `d` của `Q,K` có thể khác với `V,O`, nhưng việc đơn giản hóa chúng thành cùng kích thước không làm thay đổi bản chất vấn đề.

Softmax Attention tiêu chuẩn thường đề cập đến cơ chế Attention được giới thiệu trong bài báo "Attention is All You Need":
```js
O=softmax(QK⊤+logM)V (2)
```
  
Ở đây bỏ qua hệ số tỷ lệ 1/√d vì nó luôn có thể được tích hợp vào Q,K. Softmax thực hiện chuẩn hóa theo hàm mũ trên chiều thứ hai, còn M ∈ R^(n×n) là ma trận tam giác dưới, được gọi là ma trận mặt nạ (mask matrix), định nghĩa là
```js
Mi,j={1: i≥j; 0: i<j } (3)
```
  
logM nghĩa là lấy log từng phần tử của ma trận M, trong đó log0 = -∞. Softmax Attention khi viết dưới dạng từng thành phần sẽ là:
```js
ot = ∑(j=1→t) exp(qt⊤kj)vj / ∑(j=1→t) exp(qt⊤kj) (4)
```

Ở đây mẫu số chủ yếu có tác dụng ổn định số học, ngoài ra nếu áp dụng RMSNorm cho O thì mẫu số sẽ tự động triệt tiêu. Do đó, phần cốt lõi của Softmax Attention nằm ở tử số, cụ thể là:
```js
O=exp(QK⊤+logM)V=(exp(QK⊤)⊙M)V (5)
```

Trong đó ⊙ là tích Hadamard (elemenwise multiple), `exp` là phép lũy thừa từng phần tử. Có thể thấy mẫu số thực chất là thay V bằng ma trận toàn 1 kích thước n×1, nếu cần có thể bổ sung sau. Cách triển khai tiêu chuẩn của Softmax Attention yêu cầu tính toán ma trận `exp(QK⊤)` kích thước n×n, nên độ phức tạp không gian và thời gian đều tỷ lệ với n². **Flash Attention giúp giảm yêu cầu không gian** nhưng vẫn không tránh khỏi độ phức tạp bậc hai.

## Hình dáng ban đầu

Ý tưởng ban đầu của Linear Attention chủ yếu là mô phỏng và xấp xỉ Softmax Attention, trong đó phương án đơn giản nhất là bỏ qua exp:
```js
O=(QK⊤⊙M)V (6)
```

Để đơn giản, ta quy ước phép nhân ma trận có độ ưu tiên cao hơn tích Hadamard, từ đó bỏ được một cặp ngoặc. Tại sao dạng này được gọi là "tuyến tính"? Để hiểu nhanh, hãy xét phiên bản không Causal (bỏ ⊙M), khi đó: `O = (QK⊤)V = Q(K⊤V)`. Lưu ý độ phức tạp tính `K⊤V` là O(nd²), kết quả là ma trận `d×d`, sau đó nhân với Q cũng có độ phức tạp O(nd²), nên tổng độ phức tạp phụ thuộc tuyến tính vào n.

Đối với phiên bản Causal (6), chúng ta có thể hiểu dưới dạng thành phần như sau:
```js
ot = ∑(j=1→t) vj(kj⊤qt) = (∑(j=1→t) vjkj⊤)qt (7)
```

Nếu ký hiệu phần trong ngoặc là St, ta có:
```js
ot = Stqt, St = St-1 + vtk⊤t (8)
```

Qua đó thấy rằng Linear Attention dạng Causal có thể biểu diễn như một RNN tuyến tính với trạng thái `St`, với độ phức tạp mỗi bước là hằng số và tổng độ phức tạp tỷ lệ với độ dài chuỗi n. "RNN tuyến tính" là khái niệm tổng quát hơn, trong đó Linear Attention là một trường hợp đặc biệt. Các kiến trúc RNN tuyến tính như LRU, SSM đã phát triển độc lập trước đây, nhưng hiện nay các mô hình hiệu quả nhất đều có dạng Linear Attention.

Các phiên bản Linear Attention ban đầu thường bắt chước Softmax Attention, ví dụ thêm phần mẫu số để chuẩn hóa, yêu cầu `k⊤jqt` phải không âm bằng cách thêm hàm kích hoạt không âm cho Q,K. Các nghiên cứu như Performer, RFA tập trung xấp xỉ exp(QK⊤).

Tuy nhiên, nghiên cứu sau này (`The Devil in Linear Transformer`) chỉ ra rằng chuẩn hóa theo chiều dài chuỗi không hoàn toàn giải quyết vấn đề ổn định số học, thay vào đó nên **dùng chuẩn hóa hậu kỳ**:
```js
O = RMSNorm((QK⊤⊙M)V) (9)
```

Khi không cần chuẩn hóa, việc thêm hàm kích hoạt không âm cho Q,K trở nên không bắt buộc. Việc sử dụng hàm kích hoạt (không nhất thiết không âm) vẫn có thể mang lại hiệu quả trong một số trường hợp, nhưng không làm thay đổi bản chất của Linear Attention. Thực tế cho thấy các mô hình không sử dụng hàm kích hoạt vẫn hoạt động tốt.

## Cổng Quên Linh Hoạt

Từ công thức (8), chúng ta thấy Linear Attention hiện tại về bản chất là một phép cộng dồn (cumsum), nơi tất cả thông tin lịch sử được cộng với trọng số bằng nhau. Khi số lượng token tích lũy đủ lớn, tỷ trọng thông tin từ mỗi token sẽ trở nên rất nhỏ, khiến ma trận trạng thái St cố định không thể tái tạo chính xác bất kỳ token nào - giống như ký ức của mỗi token đều trở nên mờ nhạt.

Để giải quyết vấn đề này, RetNet đã giới thiệu cơ chế quên vào Linear Attention:
```js
ot = Stqt, St = γSt-1 + vtk⊤t (10)
```

Trong đó:
- γ ∈ (0,1) là hệ số suy giảm (thường là hằng số, có thể huấn luyện hoặc dạng ma trận chéo)
- RetNet là mô hình đầu tiên kết hợp cơ chế này với Linear Attention
- Cơ chế quên giúp tập trung vào thông tin gần hơn (Recency Bias), phù hợp với đặc tính ngôn ngữ

Một điểm đáng chú ý là RetNet còn áp dụng RoPE cho Q,K, mở rộng hệ số suy giảm thành số phức `γeiθ`. Các thí nghiệm gần đây (như TransXSSM) cho thấy việc **thêm RoPE vào Linear Attention mang lại hiệu quả tích cực**.

Các phát triển tiếp theo:
- Biến γ thành hàm theo vị trí t (γt)
- DFW, Mamba, Mamba2 phát triển thành "data-dependent decay"
- Gần giống forget gate trong GRU/LSTM nhưng giữ tính tuyến tính

Lý do ưa chuộng Linear RNN:
- Có thể song song hóa khi huấn luyện
- Hiệu quả huấn luyện và suy luận tương đương Softmax Attention
- Giải pháp song song hóa phổ biến: Chuyển đổi thành bài toán Prefix Sum và Associative Scan

Tuy nhiên, "giải pháp tổng quát" không phải là tối ưu cho GPU. Phép nhân ma trận mới là thao tác hiệu quả nhất trên GPU, do đó thuật toán song song tận dụng tối đa phép nhân ma trận là lý tưởng nhất. Thậm chí không cần song song hoàn toàn, chỉ cần tìm được định dạng đệ quy "Chunk by Chunk" sử dụng triệt để phép nhân ma trận cũng có thể cải thiện đáng kể hiệu suất huấn luyện. Điều này đặt ra yêu cầu cho kiến trúc mô hình - chỉ có cổng quên dạng tích ngoài mới đáp ứng được, điển hình như Mamba với cổng quên không phải tích ngoài đã không tận dụng hết hiệu năng GPU, dẫn đến các phiên bản cải tiến như Mamba2 và GLA.

## Huấn luyện khi Kiểm thử (TTT)

Quá trình phát triển của Linear Attention từ bắt chước Softmax Attention ban đầu, đến việc tích hợp hệ số suy giảm tĩnh và cả "data-dependent decay", đã hình thành nên những đặc trưng riêng và chứng minh được giá trị trong nhiều tác vụ. Tuy nhiên, hầu hết tiến bộ này đều dựa trên thiết kế thủ công theo kinh nghiệm. Câu hỏi đặt ra là: **Liệu có nguyên tắc tổng quát nào để định hướng thiết kế Linear Attention nói riêng và các mô hình chuỗi (Token-Mixer) nói chung?**

TTT (Test Time Training) đưa ra giải pháp bằng cách xem việc xây dựng mô hình chuỗi như một bài toán "Học Trực tuyến" (Online Learning), đề xuất sử dụng bộ tối ưu để xây dựng RNN (không nhất thiết tuyến tính). Cụ thể, nó xem cặp (K,V) như tập dữ liệu (k₁,v₁),(k₂,v₂),...,(kₜ,vₜ), từ đó huấn luyện mô hình v = f(Sₜ;k) và đầu ra oₜ = f(Sₜ;qₜ), với Sₜ là tham số mô hình - có cấu trúc tuỳ ý.

Mối liên hệ với RNN nằm ở chỗ: các bộ tối ưu như SGD, Adam về bản chất chính là RNN cho tham số mô hình! Quan điểm này không mới, đã xuất hiện từ thời Meta Learning năm 2017 khi nghiên cứu dùng RNN (LSTM) để mô phỏng bộ tối ưu tốt hơn (xem "Optimization as a Model for Few-Shot Learning").

Đến lượt mình, TTT đảo ngược cách tiếp cận - dùng bộ tối ưu để xây dựng RNN. Quy trình như sau: 
1. Tham số hiện tại Sₜ₋₁ 
2. Bộ tối ưu (SGD) nhận dữ liệu mới (kₜ,vₜ) 
3. Cập nhật tham số thành Sₜ 
4. Trả về kết quả dự đoán f(Sₜ₋₁;qₜ)

Công thức tổng quát của RNN trong TTT:

oₜ = f(Sₜ;qₜ),  
Sₜ = Sₜ₋₁ - ηₜ∇Sₜ₋₁L(f(Sₜ₋₁;kₜ),vₜ)  (11)

Với:
- L(f(Sₜ₋₁;kₜ),vₜ): hàm mất mát 
- ηₜ: hệ số học, có thể phụ thuộc dữ liệu như "data-dependent decay"

Công thức này bao quát nhiều dạng RNN, trong đó (8) và (10) là trường hợp đặc biệt:

(8) Linear Attention:  
Sₜ = Sₜ₋₁ + vₜkₜᵀ  
oₜ = Sₜqₜ  
f(S;k) = Sk  
L(f,v) = -vᵀ(Sk)  
ηₜ = 1

(10) RetNet:  
Sₜ = γSₜ₋₁ + vₜkₜᵀ  
oₜ = Sₜqₜ  
f(S;k) = Sk  
L(f,v) = -vᵀ(Sk) + (1-γ²)/2‖S‖²_F  
ηₜ = 1

  
TTT原文则致力于探索mini-batch下的非线性RNN，后来的Titans则给TTT的SGD加上了动量，再后面《Test-Time Training Done Right》则探索了large-batch的TTT用法，还探索了“TTT + Muon”的组合。注意，TTT只是利用优化器来构建RNN，RNN以外的参数如Q,K,VQ,K,V的可训练参数，还是将整个模型构建起来后用整体的优化器训练的。

一个更值得思考的问题是：为什么TTT可以成为构建RNN的“指导原则”呢？RNN的核心目标，是将历史数据有效地压缩到一个固定大小的State中，而模型参数正好是固定大小的，训练模型某种程度上就相当于把训练数据压缩到模型权重中，TTT正是利用了它跟RNN目标的高度契合性。说直白一点，如果将RNN视为一个压缩任务，TTT将模型ff视为“解压器”，它的权重则是“压缩包”，而压缩算法则是SGD，压缩率则是损失LL。

这样一来，我们就不用花心思构建递归格式了，转而构建模型ff和损失LL，一个RNN强不强、靠不靠谱，我们也只需看对应的ff和LL就可以心中有数。

除此之外，TTT用Online Learning构建RNN，意味着所得RNN必然非常契合ICL（In Context Learning）任务，这也是TTT作为“指导原则”的优势。此前《Why Can GPT Learn In-Context? Language Models Implicitly Perform Gradient Descent as Meta-Optimizers》甚至反过来，将Softmax Attention去掉Softmax成线性Attention来解释它的ICL能力，用现在的视角看它就是构造了对应的TTT出来。

## 除旧而迎新 #

例如，最早的线性Attention对应的损失函数是−v⊤(Sk)−v⊤(Sk)，这一看就是个不大靠谱的目标，因为它是无下界的，这可能会导致SS趋于无穷。相比之下，RetNet往损失函数加入了L2正则项，避免了这种风险，从优化角度看也缓解了过拟合的风险，从而得到一个更好的RNN。

然而，用内积作为损失函数虽然简洁且有一定道理，但它不是直接鼓励Sk=vSk=v，所以并非一个理想的回归损失。更好的目标函数应该是平方损失，即12∥Sk−v∥212‖Sk−v‖2，将它代入到TTT的公式(11)(11)得到  

ot=f(St;qt),St=St−1−ηt(St−1kt−vt)k⊤t∇St−112∥St−1kt−vt∥2(12)(12)ot=f(St;qt),St=St−1−ηt(St−1kt−vt)kt⊤⏟∇St−112‖St−1kt−vt‖2

  
这便是DeltaNet，这个名字出自《Parallelizing Linear Transformers with the Delta Rule over Sequence Length》，更早则是由《Linear Transformers Are Secretly Fast Weight Programmers》提出。留意到ηt(St−1kt−vt)k⊤t=(St−1(ηt−−√kt)−(ηt−−√vt))(ηt−−√kt)⊤ηt(St−1kt−vt)kt⊤=(St−1(ηtkt)−(ηtvt))(ηtkt)⊤，这意味着ηtηt总可以吸收到kt,vtkt,vt的定义中去，所以我们接下来的分析都只考虑ηt=1ηt=1的情况：  

St===St−1−(St−1kt−vt)k⊤tSt−1−(St−1kt)k⊤t+vtk⊤tSt−1(I−ktk⊤t)+vtk⊤t(13)(13)St=St−1−(St−1kt−vt)kt⊤=St−1−(St−1kt)kt⊤+vtkt⊤=St−1(I−ktkt⊤)+vtkt⊤

  
如果有需要，我们再把kt,vtkt,vt换成ηt−−√kt,ηt−−√vtηtkt,ηtvt，就可以将ηtηt恢复出来。对比线性Attention最早的形式(8)(8)，DeltaNet的区别是在加vtk⊤tvtkt⊤前多减了个(St−1kt)k⊤t(St−1kt)kt⊤，其中St−1ktSt−1kt可以理解为新输入ktkt在旧模型St−1St−1下的预测结果。

直观来想，“先减后加”就是先移除模型对ktkt的旧认知，然后根据(kt,vt)(kt,vt)补充新认知，达到“除旧迎新”的效果。这个规则称为“Delta Rule”，正是DeltaNet一词中“Delta”的来源。Delta Rule并不新鲜，它又称为Least Mean Square、Widrow-Hoff Algorithm等，已经是上个世纪60年代的产物了。事实上，这个领域完全新的东西很少，很多改动都可以追溯到某个“上古时期”的工作，目前的努力主要集中在挖掘其中能Scalable的部分。

另外需要指出的是，按照时间的顺序，是DeltaNet在前，TTT在后，从Online Learning角度理解RNN，其实在TTT之前已经零星地体现在一些工作中，但TTT系统地提出了这个“指导原则”，并且将它用于构建新RNN模型，所以我们把TTT放在前面，使得整个介绍更加流畅自然一些。

有些读者可能疑问：DeltaNet还算线性RNN吗？答案是肯定的。我们所说的线性RNN，是指递归公式对State变量的依赖关系是线性的，但对输入或q,k,vq,k,v的依赖可以是非线性的（当然不同依赖形式的并行效率会有所不同），从式(13)(13)可以看出，等号右端始终只是出现了St−1St−1的一次方，所以它满足线性的定义。

## 求逆与推广 #

前面我们说了，线性RNN最理想的（即GPU高效的）并行算法是充分使用矩阵乘法的形式。为了完成这一目标，我们先将DeltaNet写成  

St=St−1+(vt−St−1kt)k⊤t(14)(14)St=St−1+(vt−St−1kt)kt⊤

  
记ut=vt−St−1ktut=vt−St−1kt，那么St=St−1+utk⊤tSt=St−1+utkt⊤，也就是说它只是在最早的线性Attention基础上把VV换成了U=[u1,u2,⋯,un]⊤U=[u1,u2,⋯,un]⊤，将它迭代t−1t−1次，我们有  

St−1=∑j=1t−1ujk⊤j⇒ut=vt−(∑j=1t−1ujk⊤j)kt=vt−∑j=1t−1uj(k⊤jkt)(15)(15)St−1=∑j=1t−1ujkj⊤⇒ut=vt−(∑j=1t−1ujkj⊤)kt=vt−∑j=1t−1uj(kj⊤kt)

  
最后的等式写成矩阵形式是U=V−(KK⊤⊙M−)UU=V−(KK⊤⊙M−)U，其中M−=M−IM−=M−I，这是一个线性方程组，它的解可以直接表示为  

U=(I+KK⊤⊙M−记为B)−1V(16)(16)U=(I+KK⊤⊙M−⏟记为B)−1V

  
这里出现了(I+B)−1(I+B)−1，一个n×nn×n矩阵的逆，标准复杂度是O(n3)O(n3)，比Softmax Attention还高！不过好在我们不需要显式的逆而是只要UU，这可以转化为解方程组(I+B)U=V(I+B)U=V，复杂度降到O(n2)O(n2)。进一步地，利用I+BI+B是下三角阵以及BB的低秩结构，可以将复杂度降到线性，写成分块矩阵乘法后就可以充分利用GPU。这些细节只能请大家阅读原论文了，本文先把主要数学原理介绍清楚。

DeltaNet之后，Gated DeltaNet（GDN）进一步地将遗忘门引入到DeltaNet之中，这倒是可以预料的变化。Gated DeltaNet的原始引入方式是  

St=αtSt−1(I−βtktk⊤t)+βtvtk⊤t(17)(17)St=αtSt−1(I−βtktkt⊤)+βtvtkt⊤

  
但个人认为，这个提法其实显式打破了Delta Rule，更好的提法应该是像Comba一样，只乘到第一个St−1St−1上：  

St=γtSt−1+ηt(vt−St−1kt)k⊤t(18)(18)St=γtSt−1+ηt(vt−St−1kt)kt⊤

  
它相当于将损失函数取12∥Sk−v∥2+1−γη∥S∥2F12‖Sk−v‖2+1−γη‖S‖F2。当然，从数学上来说，这两个提法都是等价的：  

αtSt−1(I−βtktk⊤t)+βtvtk⊤t=αtSt−1+αtβt(vt/αt−St−1kt)k⊤t(19)(19)αtSt−1(I−βtktkt⊤)+βtvtkt⊤=αtSt−1+αtβt(vt/αt−St−1kt)kt⊤

  
即γt=αt,ηt=αtβtγt=αt,ηt=αtβt然后把1/αt1/αt吸收到vtvt就可以转化为后者了。所以说，这两个形式在数学上并没有区别，由于多数αtαt会接近于1，所以能力上估计也没啥区别（Comba说(18)(18)会好一点），只不过后者更直观地保留了Delta Rule的样子。

从理论上来说，Gated DeltaNet也可以写成DeltaNet的形式，因为只需要定义α¯t=∏tj=1αtα¯t=∏j=1tαt，那么式(17)(17)两边同时除以α¯tα¯t，就得到  

α¯−1tSt=α¯−1t−1St−1(I−βtktk⊤t)+βt(α¯−1tvt)k⊤t(20)(20)α¯t−1St=α¯t−1−1St−1(I−βtktkt⊤)+βt(α¯t−1vt)kt⊤

  
然后结合ot=Stqt=(α¯−1tSt)(α¯tqt)ot=Stqt=(α¯t−1St)(α¯tqt)，可以发现只需要分别将α¯tqt,α¯−1tvtα¯tqt,α¯t−1vt设置为新的qt,vtqt,vt，那么就能简化成DeltaNet的形式。不过，这个结果只有在某些情况下具有理论推导的价值（比如推导下一节的Attention矩阵），因为实际计算中，不管怎么参数化，对于足够大的tt，α¯tα¯t和α¯−1tα¯t−1之一必有溢出的风险。

DeltaNet之后还有另一个推广DeltaProduct，它是将k,vk,v扩展若干倍后再做DeltaNet或者Gated DeltaNet，试图增强模型的状态追踪能力。不过，就笔者的审美而言，与其像DeltaProduct那样扩展常数倍，还不如像《时空之章：将Attention视为平方复杂度的RNN》一样尝试平方复杂度的RNN，看有没有机会超越Softmax Attention。

## 反哺进行时 #

说到超越Softmax Attention，开头提到，如今的线性Attention不仅能与Softmax Attention一较高低，甚至开始“反哺”它。这看似不可思议，但细思之下并不难理解。某种意义上，这些年Softmax Attention一直在退步，从MHA、GQA到MQA都是为了压缩KV Cache而做减法。而线性Attention没有KV Cache问题，所以一直往更好的方向前进。

为了更好看出这一点，我们不妨将前面提到的Attention机制都以矩阵形式写出来：  

Softmax Attention最早的线性Attention加入遗忘门后DeltaNetGated DeltaNet公式(exp(QK⊤)⊙M)V(QK⊤⊙M)V(QK⊤⊙Γ)V(QK⊤⊙M)(I+KK⊤⊙M−)−1V((QK⊤⊙M)(I+KK⊤⊙M−)−1⊙Γ)V=(QK⊤⊙Γ)(I+KK⊤⊙Γ−)−1V公式Softmax Attention(exp⁡(QK⊤)⊙M)V最早的线性Attention(QK⊤⊙M)V加入遗忘门后(QK⊤⊙Γ)VDeltaNet(QK⊤⊙M)(I+KK⊤⊙M−)−1VGated DeltaNet((QK⊤⊙M)(I+KK⊤⊙M−)−1⊙Γ)V=(QK⊤⊙Γ)(I+KK⊤⊙Γ−)−1V

  
其中  

Γi,j=⎧⎩⎨⎪⎪⎪⎪⎪⎪⎪⎪⎪⎪⎪⎪∏τ=j+1iγτ,1,0,i>ji=ji<j(21)(21)Γi,j={∏τ=j+1iγτ,i>j1,i=j0,i<j

  
以及Γ−=Γ−IΓ−=Γ−I。这样看来，Softmax Attention的形式还仅停留在最早的线性Attention那会（当然这也证明了它的强大）。那“反哺”怎么实现呢？首先我们需要一种方法把Softmax Attention转化为线性Attention，这个并不难，早在《Transformer升级之路：5、作为无限维的线性Attention》我们就总结了三种将Softmax Attention转化为 _无限维_ 线性Attention的方案。

总之，就是存在一个映射ϕϕ，将Q,KQ,K从n×dn×d映射到n×∞n×∞，满足exp(QK⊤)=ϕ(Q)ϕ(K)⊤exp⁡(QK⊤)=ϕ(Q)ϕ(K)⊤，这称为“核技巧”。那接下来的事情就简单了，我们只需将上述表格中的线性Attention的Q,KQ,K换成ϕ(Q),ϕ(K)ϕ(Q),ϕ(K)，最后再设法恢复expexp并归一化，就得到新的Softmax Attention变体了。例如，代入到遗忘门的公式，我们有  

(ϕ(Q)ϕ(K)⊤⊙Γ)V=exp(QK⊤+logΓ)V(22)(22)(ϕ(Q)ϕ(K)⊤⊙Γ)V=exp⁡(QK⊤+log⁡Γ)V

  
如果γtγt取常数，那么其实就是《Train Short, Test Long: Attention with Linear Biases Enables Input Length Extrapolation》所提的ALIBI，而如果γtγt是依赖于输入的，那么就是《Forgetting Transformer: Softmax Attention with a Forget Gate》所提的FoX。

一个更有意思的结果是《Understanding Transformer from the Perspective of Associative Memory》所提的DeltaFormer，顾名思义它是Softmax Attention的DeltaNet版本。将DeltaNet的Q,KQ,K换成ϕ(Q),ϕ(K)ϕ(Q),ϕ(K)，我们有  

=(ϕ(Q)ϕ(K)⊤⊙M)(I+ϕ(K)ϕ(K)⊤⊙M−)−1Vexp(QK⊤+logM)记为A(I+exp(KK⊤+logM−)记为B)−1V(23)(23)(ϕ(Q)ϕ(K)⊤⊙M)(I+ϕ(K)ϕ(K)⊤⊙M−)−1V=exp⁡(QK⊤+log⁡M)⏟记为A(I+exp⁡(KK⊤+log⁡M−)⏟记为B)−1V

  
如果要归一化，我们将expexp换成softmaxsoftmax即可。相比Softmax Attention，DeltaFormer将原本的AVAV改成了A(I+B)−1VA(I+B)−1V，注意到  

A(I+B)−1V==A(I−B+B2−B3+⋯)VA(V−BV+B2V−B3V+⋯)(24)(24)A(I+B)−1V=A(I−B+B2−B3+⋯)V=A(V−BV+B2V−B3V+⋯)

  
所以DeltaFormer相当于先用K,K,VK,K,V算多次Attention，将结果叠加起来后作为新的VV，再跟Q,KQ,K算一次Attention，这个特性让它对Multi-Hop的任务有奇效（比如Code）。此外，DeltaFormer的这个特点还意味着它跟MQA特别搭配，因为(I+B)−1V(I+B)−1V这部分只有K,VK,V参与，而对于MQA来说K,VK,V只有Single-Head，计算量相比MHA会明显降低。

不过，在笔者看来，这种固定系数的叠加可能是“没有免费午餐”，比如笔者的实验结果显示，DeltaFormer的语言模型损失并无太大变化，这意味着如果某些任务的损失明显降低，必然有另一些任务的损失上升了。

## 硬核编码术 #

还有一个值得关注的反哺工作是PaTH Attention，出自《PaTH Attention: Position Encoding via Accumulating Householder Transformations》，它从位置编码的角度将DeltaNet反哺到Softmax Attention。

我们在《Transformer升级之路：6、旋转位置编码的完备性分析》指出，对于任何正交矩阵ΩΩ，Rm=ΩmRm=Ωm都是广义的RoPE。除了旋转矩阵，还有哪些容易构建的正交矩阵呢？PaTH用的是Householder矩阵：设ww是任意模长为2−−√2的列向量，那么I−ww⊤I−ww⊤是一个正交矩阵，这我们在《从一个单位向量变换到另一个单位向量的正交矩阵》也推导过，几何意义是镜面反射。

容易看出，这跟DeltaNet中St−1St−1所乘的I−ktk⊤tI−ktkt⊤是一样的，所以PaTH干脆把这部分照搬过来，即放弃ΩmΩm这个形式，也放弃ww模长为2−−√2的约束，直接用一系列I−ww⊤I−ww⊤连乘来表达位置信息：  

q⊤ikj→q⊤i(I−wiw⊤i)(I−wi−1w⊤i−1)⋯(I−wj+1w⊤j+1)记为Ri,jkj(25)(25)qi⊤kj→qi⊤(I−wiwi⊤)(I−wi−1wi−1⊤)⋯(I−wj+1wj+1⊤)⏟记为Ri,jkj

  
将Ri,jRi,j写成递归形式是Ri,j=(I−wiw⊤i)Ri−1,j,Rj,j=IRi,j=(I−wiwi⊤)Ri−1,j,Rj,j=I。对比DeltaNet的式(13)(13)，上式相当于vtvt恒等于零，但初值S0S0不再是零。使用“求逆来相助”一节同样的过程，我们可以得到  

Ri,j=I−W⊤[j:i](I+W[j:i]W⊤[j:i]⊙M−)−1W[j:i](26)(26)Ri,j=I−W[j:i]⊤(I+W[j:i]W[j:i]⊤⊙M−)−1W[j:i]

  
其中W=[w1,w2,⋯,wn]⊤W=[w1,w2,⋯,wn]⊤，切片按Numpy来理解，如W[j:i]=[wj+1,wj+2,⋯,wi]⊤W[j:i]=[wj+1,wj+2,⋯,wi]⊤，切片优先级高于转置。注意求逆的是下三角阵，三角阵有一个重要特性，逆矩阵的对角线元素等于原矩阵对角线元素的倒数，如果是分块三角阵则对角块也满足这个特性，于是我们可以写出  

(I+W[j:i]W⊤[j:i]⊙M−)−1=((I+WW⊤⊙M−)−1记为J)[j:i,j:i](27)(27)(I+W[j:i]W[j:i]⊤⊙M−)−1=((I+WW⊤⊙M−)−1⏟记为J)[j:i,j:i]

  
接下来的变换，写成分量形式可能好理解一些  

Ai,j======q⊤iRi,jkjq⊤ikj−q⊤iW⊤[j:i]J[j:i,j:i]W[j:i]kjq⊤ikj−∑p=1d∑l=j+1i∑r=j+1i∑s=1dQi,pWl,pJl,rWr,sKj,sq⊤ikj−∑p=1d∑l=1i∑r=j+1n∑s=1dQi,pWl,pJl,rWr,sKj,sq⊤ikj−∑p=1d∑l=1n∑r=1n∑s=1dQi,pWl,pχl≤iJl,rχr≥j+1Wr,sKj,sq⊤ikj−∑l=1n∑r=1n(χl≤i∑p=1dQi,pWl,p)QW⊤⊙MJl,r(χr≥j+1∑s=1dWr,sKj,s)WK⊤⊙M−(28)(28)Ai,j=qi⊤Ri,jkj=qi⊤kj−qi⊤W[j:i]⊤J[j:i,j:i]W[j:i]kj=qi⊤kj−∑p=1d∑l=j+1i∑r=j+1i∑s=1dQi,pWl,pJl,rWr,sKj,s=qi⊤kj−∑p=1d∑l=1i∑r=j+1n∑s=1dQi,pWl,pJl,rWr,sKj,s=qi⊤kj−∑p=1d∑l=1n∑r=1n∑s=1dQi,pWl,pχl≤iJl,rχr≥j+1Wr,sKj,s=qi⊤kj−∑l=1n∑r=1n(χl≤i∑p=1dQi,pWl,p)⏟QW⊤⊙MJl,r(χr≥j+1∑s=1dWr,sKj,s)⏟WK⊤⊙M−

  
这里有几个关键点：比较巧妙的是第4个等号，它利用了JJ是下三角矩阵这一点，所以l<rl<r时Jl,rJl,r自动为零；第5个等号，χχ为示性函数，满足下标的条件时为1，否则为0；第6个等号，当我们分别处理p,sp,s两部分求和时，结果是QW⊤QW⊤和WK⊤WK⊤，而乘χl≤iχl≤i刚好表示保留QW⊤QW⊤的下三角部分（连同对角线），而乘χr≥j+1χr≥j+1则表示保留WK⊤WK⊤的下三角部分（不包括对角线）。

至此，我们可以把整个（Softmax之前的）注意力矩阵写出来：  

A=QK⊤⊙M−(QW⊤⊙M)(I+WW⊤⊙M−)−1(WK⊤⊙M−)(29)(29)A=QK⊤⊙M−(QW⊤⊙M)(I+WW⊤⊙M−)−1(WK⊤⊙M−)

  
有没有被震惊到？这还没完。直接求逆复杂度是O(n3)O(n3)，这肯定无法接受，还要想办法利用WW⊤WW⊤的低秩特点将复杂度降低到O(n2)O(n2)，然后还要推反向传播，最后写成类似Flash Attention的高效实现，这些细节大家只能看原论文挖掘了，总之全程都非常硬核。

从位置编码的角度看，PaTH是CoPE（Contextual Position Encoding）的一种，它的位置并不是编号1,2,3,⋯1,2,3,⋯，而是根据上下文内容自动生成的位置信号。类似地，FoX也可以看成是Contextual版的ALIBI。上下文相关的位置信息是当前线性Attention的主要特征，也可能是反哺Softmax Attention的主要方向。

## 化简乐无穷 #

我们不妨再深入点探讨一下PaTH，这不仅有助于我们了解PaTH，也能帮助我们更熟悉DeltaNet，两者本身就是高度相关的。这一节我们从PaTH的两个特例入手，它可以帮助我们更好地理解PaTH与DeltaNet的关联。

第一个特例是W=KW=K，代入到(29)(29)得到  

A==(QK⊤⊙M)(I−(I+KK⊤⊙M−)−1(KK⊤⊙M−))(QK⊤⊙M)(I+KK⊤⊙M−)−1(注:I−(I+A)−1A=(I+A)−1)(30)(30)A=(QK⊤⊙M)(I−(I+KK⊤⊙M−)−1(KK⊤⊙M−))=(QK⊤⊙M)(I+KK⊤⊙M−)−1(注:I−(I+A)−1A=(I+A)−1)

  
有没有觉得有点熟悉？这刚好就是DeltaNet的Attention矩阵！从这个特例看来，PaTH和DeltaFormer的区别就在于，DeltaFormer基于核技巧，给DeltaNet的QK⊤QK⊤和KK⊤KK⊤分别加上expexp，而PaTH直接给DeltaNet的Attention矩阵加上expexp。

第二个特例是重新引入∥w∥=2−−√‖w‖=2这个约束，此时I−ww⊤I−ww⊤是正交矩阵，我们引入  

Ri≜==(I−wiw⊤i)(I−wi−1w⊤i−1)⋯(I−w1w⊤1)I−W⊤[:i](I+W[:i]W⊤[:i]⊙M−)−1W[:i]Ri,0(31)(31)Ri≜(I−wiwi⊤)(I−wi−1wi−1⊤)⋯(I−w1w1⊤)=I−W[:i]⊤(I+W[:i]W[:i]⊤⊙M−)−1W[:i]=Ri,0

  
那么Ri,j=RiR⊤jRi,j=RiRj⊤。这个等式意味着我们可以像RoPE一样，用绝对位置的方式实现相对位置的PaTH，即只需要给每个q⊤i,k⊤iqi⊤,ki⊤都乘上RiRi，然后套用Softmax Attention的实现就行。那么乘RiRi是什么运算呢？重复上一节的展开过程，我们有  

(q⊤iRi)s=====(q⊤i−q⊤iW⊤[:i]J[:i,:i]W[:i])sQi,s−∑p=1d∑l=1i∑r=1iQi,pWl,pJl,rWr,sQi,s−∑p=1d∑l=1i∑r=1nQi,pWl,pJl,rWr,sQi,s−∑p=1d∑l=1n∑r=1nχl≤iQi,pWl,pJl,rWr,sQi,s−∑l=1nχl≤i∑p=1dQi,pWl,pQW⊤⊙M∑r=1nJl,rWr,sJW(32)(32)(qi⊤Ri)s=(qi⊤−qi⊤W[:i]⊤J[:i,:i]W[:i])s=Qi,s−∑p=1d∑l=1i∑r=1iQi,pWl,pJl,rWr,s=Qi,s−∑p=1d∑l=1i∑r=1nQi,pWl,pJl,rWr,s=Qi,s−∑p=1d∑l=1n∑r=1nχl≤iQi,pWl,pJl,rWr,s=Qi,s−∑l=1nχl≤i∑p=1dQi,pWl,p⏟QW⊤⊙M∑r=1nJl,rWr,s⏟JW

  
写成矩阵形式就是  

Q−(QW⊤⊙M)(I+WW⊤⊙M−)−1W(33)(33)Q−(QW⊤⊙M)(I+WW⊤⊙M−)−1W

  
是不是又觉得有点熟悉？其实第二部分就是DeltaNet(Q,W,W)DeltaNet(Q,W,W)！所以这种情况下PaTH实现的效果等价于是  

SoftmaxAttention(Q−DeltaNet(Q,W,W)Q~,K−DeltaNet(K,W,W)K~,V)(34)(34)SoftmaxAttention⁡(Q−DeltaNet⁡(Q,W,W)⏟Q~,K−DeltaNet⁡(K,W,W)⏟K~,V)

  
也就是用DeltaNet给Q,KQ,K加位置编码。这样看PaTH（在∥w∥=2−−√‖w‖=2这个约束下）就相当于Softmax Attention与DeltaNet的某种层内混合。当然我们也可以考虑放弃前面的推导，即便∥w∥≠2−−√‖w‖≠2时也按照上式来实现，这就类似于通过Canon Layers的方案，用卷积给Q,KQ,K加位置信息了，只不过这里的卷积不再是短卷积，而是DeltaNet这种长卷积。

## 剑走偏锋法 #

最后，我们再看最近的一个同样值得关注的线性Attention模型——MesaNet（还有一个大同小异的同期工作Atlas）。TTT的Online Learning视角告诉我们，DeltaNet其实就是在用SGD优化目标函数12∥Sk−v∥212‖Sk−v‖2，而我们仔细观察就会发现，SkSk只是kk的线性函数，所以这实际上只是一个线性回归问题，线性回归是有解析解的！  

St=GtH−1t,Gt=∑j=1tvjk⊤j,Ht=∑j=1tkjk⊤j(35)(35)St=GtHt−1,Gt=∑j=1tvjkj⊤,Ht=∑j=1tkjkj⊤

  
MesaNet就是利用这个解析解来构建序列模型的，其想法起源于《Uncovering mesa-optimization algorithms in Transformers》，高效训练则是由《MesaNet: Sequence Modeling by Locally Optimal Test-Time Training》实现。MesaNet在上述公式基础上给Gt,HtGt,Ht加入遗忘门，然后求时加上对角阵ΛtΛt避免不可逆，总的模型是  

ot=Gt(Ht+Λt)−1qt,Gt=γtGt−1+vtk⊤t,Ht=γtHt−1+ktk⊤t(36)(36)ot=Gt(Ht+Λt)−1qt,Gt=γtGt−1+vtkt⊤,Ht=γtHt−1+ktkt⊤

  
很明显，Gt,HtGt,Ht关于序列长度的复杂度是线性的，所以otot的计算复杂度也是线性的，因此MesaNet仍然属于线性Attention的范畴，并且由于解析解的缘故，基本上可以保证大多数情况下它优于DeltaNet甚至Gated DeltaNet。从信号处理的角度看，MesaNet与DeltaNet是Recursive Least Square和Least Mean Square的区别。

看上去都是优点，为啥笔者会将它归入“剑走偏锋”呢？在笔者看来，MesaNet“成也解析解，败也解析解”，解析解使得它通常优于DeltaNet，但也给人一种“到此为止”的感觉，因为只要稍变一下就几乎没有机会求得解析解了。纵观整个数学史，所有依赖于解析解的分支在今天几乎已经都没落了，因为解析解实在太稀罕、太没有代表性了。

从实现上来看，MesaNet需要求逆的矩阵Ht+ΛtHt+Λt并不是三角阵，尽管(Ht+Λt)−1qt(Ht+Λt)−1qt仍然可以转化为解方程而不需要显式逆，但非三角阵仍使得它求解复杂度会增加不少。如何尽可能低成本地并行计算全体(Ht+Λt)−1qt(Ht+Λt)−1qt将会是MesaNet长期的难点，目前论文用到的是“共轭梯度法”求近似解，能用但并不完美。

再就是从理论能力上看，MesaNet也并非严格优于DeltaNet。这是因为MesaNet的Gt,HtGt,Ht更新规则还是简单的滑动平均，它的求逆也不涉及到Token之间的交互，所以它的能力极限大概不如拥有Delta Rule的DeltaNet。直观理解就是，MesaNet会尽力记住全体k,vk,v，“全都要”可能会导致比较模糊的记忆，而DeltaNet的原则是“除旧迎新”，因为“除旧”的缘故，它可以实现长期、精准地记忆某些内容。

我们还可以从一个特殊例子来理解这个非最优性，那就是目前为止除MesaNet的所有Attention，都允许K、V共享的选择，“允许”的意思是不一定最优，但至少能训出非平凡的结果，然而MesaNet并不行，因为K、V相同的话，MesaNet的StSt就恒为单位阵了。

总的来说，MesaNet是一个让人赏心悦目的模型，但解析解也增加了它的复杂性和限制了它的灵活性，留下了不少亟待探索的空间。如果读者想要了解更多基于线性回归来构建序列模型的内容，还可以阅读TTR，它对各种线性回归目标下的序列模型做了详细讨论。

## 方兴未艾路 #

本文简要梳理了线性Attention的发展脉络，并介绍了部分模型的数学原理。线性Attention从模仿Softmax Attention起步，逐渐发展出自身特色，如今已成为极具竞争力的序列建模方案，甚至反过来为Softmax Attention的发展提供了新思路，这一过程本身充满了趣味性和启发性。

_**转载到请包括本文地址：**https://www.kexue.fm/archives/11033_

_**更详细的转载事宜请参考：**_《科学空间FAQ》

**如果您还有什么疑惑或建议，欢迎在下方评论区继续讨论。**

**如果您觉得本文还不错，欢迎分享/打赏本文。打赏并非要从中获得收益，而是希望知道科学空间获得了多少读者的真心关注。当然，如果你无视它，也不会影响你的阅读。再次表示欢迎和感谢！**

打赏

微信打赏

支付宝打赏

因为网站后台对打赏并无记录，因此欢迎在打赏时候备注留言。  
你还可以**点击这里**或在下方评论区留言来告知你的建议或需求。

**如果您需要引用本文，请参考：**
