# LongCE: Long-contextCross-Entropy loss
- https://www.alphaxiv.org/abs/2410.23771v4
- https://github.com/PKU-ML/LongPPL

Large Language Models (LLMs) have made significant progress in processing longer contexts, with some modern models capable of handling inputs of 100K tokens or more. However, as these context windows have expanded, researchers have noted a puzzling phenomenon: models with better perplexity (PPL) scores don't necessarily perform better on long-context tasks. This discrepancy raises questions about the reliability of perplexity as an evaluation metric for long-context capabilities.

We develop an efficient long-context training strategy by emphasizing key tokens. Specifically, we propose the **LongCE (Long-context Cross-Entropy) loss** that `upweights the key tokens`, which can be estimated by the model itself. In this way, LongCE can bootstrap its long-context abilities by alternating between estimating key tokens and optimizing key tokens. Experimental results across multiple LLMs show that **LongCE consistently improves over the conventional CE loss, with a maximum accuracy gain of 22% on LongEval**.

## Section 2 - A FINE-GRAINED ANALYSIS OF PERPLEXITY

LSD là viết tắt của "Long-Short Difference" (Khác biệt Dài-Ngắn), đây là một phương pháp đo lường mức độ ảnh hưởng của ngữ cảnh dài đối với mỗi token trong mô hình ngôn ngữ. LSD được tính toán như sau: "Để đo lường ảnh hưởng của ngữ cảnh dài đối với mỗi token xi, chúng tôi thực hiện một can thiệp về độ dài ngữ cảnh. Cụ thể, với một chuỗi x và một mô hình ngôn ngữ Pθ (có khả năng ngữ cảnh dài mạnh), đối với mỗi token xi có ngữ cảnh dài, chúng tôi tính toán sự khác biệt giữa xác suất log của nó dưới ngữ cảnh dài đầy đủ li = (x1, . . . , xi−1) và xác suất log dưới ngữ cảnh ngắn bị cắt bớt si = (xi−K, . . . , xi−1) (trong đó K là độ dài ngắn, ví dụ 64)."

Công thức LSD được định nghĩa là: `LSDθ(xi) = log Pθ(xi|li) − log Pθ(xi|si)`

"Chúng tôi gọi nó là Long-Short Difference (LSD), đo lường sự cải thiện độ chính xác dự đoán được tạo ra hoàn toàn bởi ngữ cảnh dài." Từ góc độ nhân quả, si đóng vai trò như ngữ cảnh phản thực được tạo ra bởi can thiệp (loại bỏ ngữ cảnh dài), và LSD ước tính hiệu ứng điều trị cá nhân của ngữ cảnh dài. Giá trị LSD cao cho thấy ngữ cảnh dài đóng vai trò quan trọng trong việc dự đoán token xi, khiến chúng trở thành các token chủ chốt cần xem xét để đánh giá hiệu suất ngữ cảnh dài.

[LSD: Long-Short Difference - Khác biệt Dài-Ngắn, phương pháp đo ảnh hưởng của ngữ cảnh dài; token: đơn vị nhỏ nhất trong xử lý ngôn ngữ tự nhiên; log probability: xác suất logarit; ITE: Individual Treatment Effect - Hiệu ứng điều trị cá nhân]

"only less than 10% tokens are highly influenced by long context and represent long-context abilities" từ phân tích trên GovReport dataset. Đa số tokens có LSD thấp (dưới 0.5), trong khi chỉ có một phần nhỏ tokens có LSD cao (trên 2) thực sự phụ thuộc vào thông tin ngữ cảnh dài.

Do việc trung bình hóa đều nhau, "perplexity computed equally over all tokens do not represent long-context performance." Điều này dẫn đến hiện tượng "huge discrepancy between perplexity and actual performance on long-context tasks" được quan sát thấy trong Figure 1(b).

## Section 3.2 - IMPROVING LONG-CONTEXT CAPABILITIES WITH LONGCE

"Due to the massive computational cost of pre-training an LLM from scratch on long texts, current long-context LLMs are pretrained on short contexts and then fine-tuned on longer contexts. By default, the long-context fine-tuning process adopts the **Cross Entropy (CE) loss** as in pre-training, which adopts a uniform average of all tokens, akin to standard perplexity: `CE(x; θ) = −(1/n)∑[i=1 to n] log Pθ(xi|x<i)`

Nevertheless, this de facto paradigm has the same issues that we discussed for perplexity in Section 2. We show that most tokens in a sequence are not influenced by the long context, while **ONLY A FEW KEY TOKENS require long-context information**; and in turn, the model's long-context performance depends crucially on its prediction on these key tokens (as measured in LongPPL, Section 3.1).

We propose the **LongCE (Long-context Cross Entropy) loss** that reweights every token xi w.r.t. its gain Isoft(xi; θ) from long context:

> `LongCE(x; θ) = −(1/n)∑[i=1 to n] Isoft(xi; θ) log Pθ(xi|x<i)`

For the ease of differentiable optimization using all tokens, we adopt a **soft long-context influence function** Isoft : X → [0, γ] based on the likelihood ratio between the long-context probability Pθ(xi|li) and short-context probability Pθ(xi|si):

> `Isoft(xi; θ) = min (exp (LSDθ(xi)), γ) = min (Pθ(xi|li)/Pθ(xi|si), γ)`

Here, γ > 0 is a hyper-parameter that sets a threshold on the maximal influence to avoid numerical instability. As a consequence of this reweighting term, **too easy tokens** (both short and long context give accurate prediction) and **too hard tokens** (neither short or long context predicts correctly) will have a weight around 1, while those **long-context-dependent tokens** (high Pθ(xi|li) and low Pθ(xi|si)) will be upweighted above 1, proportionally to the context informativeness."

**LongCE leverages the same model to evaluate the influence for training efficiency**. Therefore, LongCE training does not require a separate evaluator model, but uses the model itself for long-context evaluation. In this way, **LongCE bootstraps the model's long-context capabilities in an EM (expectation-maximization) way**: the language model Pθ first uses itself to estimate long-context influence of each token Isoft (Equation 7); and then this estimate is used to update the model parameters by optimizing the LongCE loss function."

## **Section 4.2 - FINE-TUNE WITH LONGCE LOSS:**

**Experimental Setup:**
"We primarily use Llama-2-7B as the base model to perform long-context finetuning. We also conduct experiments on Mistral-7B-v0.1 and Llama-2-13B. We use PG-19, a book dataset sourced from a library, and Pile-arxiv, a dataset consisting of Arxiv papers, as the training dataset. The training sequences are organized to be the context length with 32k tokens. **For the calculation of LongCE, we set γ = 5 in Equation 7** and use the same sliding window approach as described in Section 4.1 to improve training efficiency. The context length of si is set to be K = 4096."

**Kết quả LongCE Outperforms CE:**
"As shown in Table 3, we present the long-context capabilities of models fine-tuned with **LongCE loss and CE loss** under different fine-tuning strategies and training datasets. We also test the effectiveness of LongCE using different base models in Table 4. We find that **models fine-tuned with LongCE loss consistently outperform those fine-tuned with CE loss across nearly all settings**. This suggests that the **LongCE loss, with its re-weighting strategy based on long-context token importance, can be applied as a plug-and-play module** which can effectively improve the model's long-context performance."

**Training Efficiency:**
"In addition to the performance improvement brought by the LongCE loss, we also pay attention to the changes in training efficiency. **In LongCE, we need an extra forward pass to calculate the probability under short context Pθ(xi|si), which introduces additional computation costs**. By using a sliding window technique (as detailed in Appendix A.1), **the computational overhead of training the model with LongCE is controlled to about 80% that of training with CE loss**."

**Time Performance Analysis:**
"We visualize in Figure 7 how the long-context performance of models fine-tuned with **LongCE and CE changes over the course of training time**. Most of the time, **fine-tuning with LongCE loss is a more efficient method**. Additionally, in Appendix B.2, we find that by changing the hyperparameters of LongCE, i.e., the short context-length K and the sliding window length d, **this overhead can be further reduced to 36%, with almost no loss in model performance**."

## **Các Tables về kết quả LongCE:**

**Table 3 - Main Results:**
Shows LongCE consistently outperforming CE across different settings:
- Setting A (PG-19 + EABF): 	 **LongCE gains up to +22.0% on LongEval**
- Setting B (PG-19 + PI):   	 **LongCE gains up to +18.0% on LongEval** 
- Setting C (Pile-arxiv + EABF): **LongCE shows improvements across all metrics**

**Table 4 - Different Base Models:**
- **Mistral-7B-v0.1**: 	LongCE gains +10.0% to +16.0% on LongEval
- **Llama-2-13B**: 		LongCE gains +4.0% to +6.0% on LongEval

## **Appendix of Implementation Details:**

**A.2 - Implementation Details of LongCE:**
"**Fine-tuning strategies**: For EABF, we adopt the identical settings in the original paper, with a RoPE base of 500k. For PI, we set the scaling factor to 8 since we want to extend the context window from 4k to 32k.

**Training details**: We use a learning rate of `2 × 10⁻⁵` for Llama and `1 × 10⁻⁶` for Mistral, with no weight decay and a linear warmup of 20 steps along with AdamW with β₁ = 0.9 and β₂ = 0.95. We apply a global batch of 64 on PG-19 and 8 on Pile-arxiv."

**Hyperparameter Ablation (Table 7):** Shows that **LongCE hyperparameters K and d can be optimized** to reduce training time from +79% to +36% while maintaining performance.

**Performance on Non-Long-Context Tasks (Table 10): LongCE does not cause any additional loss in the model's performance on normal-length tasks** - performance nearly identical to CE baseline on MMLU, ARC-C, RACE, etc.

[ EM: Expectation-Maximization - thuật toán tối ưu hóa xen kẽ; EABF: Entropy-aware Adjusted Base Frequency - phương pháp điều chỉnh tần số cơ sở; PI: Position Interpolation - nội suy vị trí; RoPE: Rotary Position Embedding - mã hóa vị trí xoay; AdamW: Adam with weight decay - thuật toán tối ưu Adam có phân rã trọng số ]

Bài báo nghiên cứu chi tiết về việc tối ưu hóa các hyperparameters K và d của LongCE, trong đó K là độ dài cửa sổ ngữ cảnh ngắn (short context window length) và d là độ dài cửa sổ trượt (sliding window length).

**Về ảnh hưởng của K và d đến hiệu quả tính toán:**

### Tham số K và d

**K và d không xuất hiện trực tiếp trong công thức toán học của LongCE**, nhưng chúng có vai trò quan trọng trong **định nghĩa và implementation**. **Công thức LongCE chính:**
```
LongCE(x; θ) = −(1/n)∑[i=1 to n] Isoft(xi; θ) log Pθ(xi|x<i)
Isoft(xi; θ) = min (Pθ(xi|li)/Pθ(xi|si), γ)
```
**K được định nghĩa trong việc tạo short context si:**
```
si = (xi−K, . . . , xi−1) (where K is a short length, e.g., 64)
```
- **li**: long context = toàn bộ ngữ cảnh dài (x1, ..., xi−1)  
- **si**: short context = chỉ K tokens gần nhất (xi−K, ..., xi−1)

**K ảnh hưởng đến LongCE như thế nào:**
- K càng lớn → si càng dài  → Pθ(xi|si) càng chính xác → Isoft càng nhỏ
- K càng nhỏ → si càng ngắn → Pθ(xi|si) kém hơn        → Isoft càng lớn cho key tokens

**d chỉ là hyperparameter implementation để tối ưu tính toán:**
Từ Appendix A.1: "we introduce a step size d, which is smaller than the truncation length K (we set it to d = 1024). When calculating the short-context probabilities of xi to xi+d−1, we set the starting token of the context uniformly."
**d không ảnh hưởng đến kết quả toán học của LongCE**, chỉ ảnh hưởng đến:
- **Tốc độ tính toán**: d lớn → ít forward passes hơn → nhanh hơn
- **Độ chính xác**: d lớn → ít chính xác hơn (vì approximation)

**Tóm lại:**
- **K**: hyperparameter ảnh hưởng **trực tiếp** đến giá trị Isoft thông qua định nghĩa si
- **d**: hyperparameter chỉ ảnh hưởng đến **implementation efficiency**, không thay đổi kết quả toán học
- **γ**: hyperparameter xuất hiện trực tiếp trong công thức để clip maximum value

"Bằng cách thay đổi các hyperparameters của LongCE, tức là độ dài ngữ cảnh ngắn K và độ dài cửa sổ trượt d, computing overhead có thể được giảm thêm xuống 36%, mà hầu như không mất mát hiệu suất mô hình."

"Kết quả cho thấy rằng, một mặt, việc tăng K (context window) hoặc giảm d (sliding step) cải thiện đáng kể hiệu quả của LongCE (từ +79% xuống +36%/+43%). Mặt khác, dưới các cài đặt này, mặc dù hiệu suất của mô hình trên các tác vụ thực tế (LongBench) giảm nhẹ, nhưng nó đạt được những cải thiện đáng kể trên các tác vụ tổng hợp (LongEval, RULER)."

**Kết quả cụ thể từ thử nghiệm:** Với các cài đặt khác nhau trong Table 7:
- **LongCE (K=4k, d=1k, mặc định)**: Thời gian huấn luyện +79%, hiệu suất tốt trên tất cả benchmark
- **LongCE (K=1k, d=1k)**: Thời gian huấn luyện +43%, hiệu suất vẫn mạnh trên LongEval và RULER
- **LongCE (K=4k, d=4k)**: Thời gian huấn luyện +36%, cải thiện đáng kể trên synthetic tasks
- **LongCE (K=4k, d=512)**: Thời gian huấn luyện +150%, hiệu suất cao nhất trên một số benchmark

**Kết luận về tối ưu hóa:**

"Điều này cho thấy LongCE vẫn có tiềm năng cho việc nâng cao hiệu quả hơn nữa." Cụ thể, việc giảm K từ 4k xuống 1k hoặc tăng d từ 1k lên 4k có thể giảm đáng kể overhead tính toán trong khi vẫn duy trì hiệu suất tốt, đặc biệt trên các tác vụ synthetic như LongEval và RULER.

Ngoài ra "only less than 10% tokens are highly influenced by long context" => tính sparse cao! => có thể pre-filter hoặc sample một cách thông minh?

## Practical Implications
The findings from this paper have several important implications for the field:

**Improved Training**: The LongCE training strategy offers a more efficient way to improve long-context abilities without requiring architectural changes or specialized pre-training.

**Understanding Context Usage**: The key token identification method provides insights into how models utilize long-range context information, potentially guiding future architecture design.

**Resource Efficiency**: By focusing on key tokens, both evaluation and training can be made more efficient, requiring less computation for comparable or better results.

**Application Guidance**: The results suggest that applications requiring long-context understanding should `focus on how well models process specific key information` rather than overall perplexity.

## Conclusion
This research makes a significant contribution to our understanding of long-context language modeling by:

1. Demonstrating why standard perplexity fails as a metric for long-context capabilities
2. Introducing LongPPL, a new metric that strongly correlates with actual performance
3. Providing a method to identify key tokens that require long-context information
4. Developing LongCE, a training strategy that improves long-context abilities

The strong correlation between LongPPL and performance across multiple benchmarks (with correlation coefficients ranging from -0.84 to -0.96) validates the approach and offers a much more reliable way to evaluate long-context models than standard perplexity.

By **focusing on key tokens—those that actually require information from the distant context** — both evaluation and training can be made more effective.

## Các hướng phát triển tiếp

1. Hướng phát triển đầu tiên và cấp bách nhất là tối ưu hóa hiệu quả tính toán. Vấn đề mà bài báo gặp phải là chi phí tính toán tăng lên 80% so với phương pháp thông thường, điều này cần được giải quyết để có thể ứng dụng rộng rãi. Một cách tiếp cận là tìm cách xác định các từ quan trọng tiềm năng trước khi tính toán chi tiết, thay vì phải tính toán cho tất cả các từ. Điều này có thể giảm đáng kể độ phức tạp tính toán. Ngoài ra, có thể áp dụng phương pháp tính toán phân cấp, bắt đầu với độ phân giải thô rồi tinh chỉnh ở những vùng có biến động cao, tương tự như kỹ thuật kết xuất tiến dần trong đồ họa máy tính.

2. Hướng thứ hai là mở rộng nền tảng lý thuyết để hiểu sâu hơn về bản chất của vấn đề. Bài báo hiện tại chưa giải thích được tại sao các từ quan trọng lại có vai trò then chốt từ góc độ lý thuyết thông tin. Cần nghiên cứu về lượng thông tin chung giữa các từ quan trọng và ngữ cảnh dài, thiết lập các giới hạn lý thuyết cho việc lựa chọn từ quan trọng tối ưu. Điều này sẽ cung cấp hướng dẫn có nguyên tắc cho việc chọn các thông số thay vì chỉ điều chỉnh dựa trên thực nghiệm. Đồng thời, **cần phân tích mối quan hệ giữa các từ quan trọng và các mẫu chú ý (attention patterns) trong kiến trúc transformer, vì có thể các từ quan trọng trùng với những vùng có độ chú ý cao.**

3. Hướng thứ ba là phát triển các phương pháp nhận biết kiến trúc mới. Thay vì tính toán bên ngoài, có thể tích hợp việc xác định từ quan trọng trực tiếp vào kiến trúc mô hình. Ví dụ, **thêm các đầu chú ý chuyên biệt để phát hiện các từ phụ thuộc ngữ cảnh trong quá trình huấn luyện**. Cũng có thể **kết hợp với kiến trúc hỗn hợp chuyên gia**, **điều hướng các từ quan trọng qua các mạng chuyên gia được huấn luyện riêng cho việc hiểu ngữ cảnh dài**.

4. Hướng thứ tư tập trung vào ứng dụng thực tế trong triển khai thực tế. Cần phát triển **tính toán LongCE tăng dần cho các tình huống trực tuyến**, vì phương pháp hiện tại đòi hỏi toàn bộ ngữ cảnh nhưng các ứng dụng thực tế cần xử lý dữ liệu dòng. Khái niệm từ quan trọng cũng có thể được mở rộng sang các phương thức khác như các vùng hình ảnh quan trọng trong mô hình thị giác-ngôn ngữ hoặc các đoạn âm thanh quan trọng trong xử lý giọng nói.

5. Hướng thứ năm là nghiên cứu liên ngành kết hợp với các lĩnh vực khác. Có thể nghiên cứu các mẫu đọc của con người để hiểu cơ sở sinh học của việc xử lý thông tin quan trọng. Các nghiên cứu theo dõi mắt có thể tiết lộ các mẫu từ quan trọng tự nhiên mà con người tập trung vào. Việc **lựa chọn từ quan trọng cũng có mối quan hệ mật thiết với lý thuyết nén**, có thể phát triển các phương pháp dựa trên nén để xác định từ quan trọng.

Trong số những hướng này, cải thiện hiệu quả tính toán (1) và tích hợp kiến trúc (3) có tác động tiềm năng cao nhất cho việc áp dụng ngay lập tức, trong khi nền tảng lý thuyết (5) sẽ tạo ra những đột phá dài hạn. Sự kết hợp của nhiều hướng sẽ tạo ra những tiến bộ đáng kể nhất.

### attention patterns

Vấn đề đầu tiên khi tính attention scores từ multiple layers là transformer có nhiều lớp và mỗi lớp lại có nhiều attention heads khác nhau, mỗi head học các mẫu attention khác nhau. Ví dụ với một mô hình 12 lớp, mỗi lớp có 12 heads, ta có tới 144 attention matrices khác nhau cho cùng một đầu vào. Việc tổng hợp tất cả này không đơn giản vì các lớp khác nhau có vai trò khác nhau, các lớp đầu thường học các mẫu cú pháp trong khi các lớp sau học các mối quan hệ về nghĩa.

Ngoài ra, trọng số attention không hoàn toàn phản ánh luồng thông tin thực sự. Nghiên cứu gần đây cho thấy trọng số attention có thể gây hiểu lầm vì chúng chỉ cho biết mô hình "nhìn" đâu, không phải thông tin nào thực sự được sử dụng để tạo ra kết quả đầu ra. Hiện tượng này được gọi là "attention is not explanation" trong tài liệu nghiên cứu.

Có nhiều phương pháp để tổng hợp attention scores. Cách đơn giản nhất là tính trung bình trọng số attention across tất cả heads và layers, tuy nhiên phương pháp này coi tất cả layers như nhau, không phù hợp vì different layers có different semantic meanings. Phương pháp weighted aggregation theo từng lớp gán trọng số khác nhau cho từng lớp dựa trên mức độ liên quan đến nhiệm vụ. Với việc hiểu ngữ cảnh dài, có thể các lớp cuối quan trọng hơn vì chúng xử lý thông tin semantic cấp cao hơn.

Cũng có thể phân tích từng attention head riêng biệt vì mỗi head có thể chuyên biệt cho các hiện tượng ngôn ngữ khác nhau. Một số heads tập trung vào mối quan hệ cú pháp, một số khác tập trung vào phụ thuộc về nghĩa.

Các phương pháp tiên tiến hơn bao gồm **attribution dựa trên gradient, tính gradient của output loss đối với attention weights** để xác định which attention connections thực sự đóng góp vào dự đoán cuối cùng. Điều này có thể identify key tokens hiệu quả hơn việc averaging attention đơn giản. Attention rollout và flow trace luồng thông tin từ input tokens qua multiple layers để xác định accumulated attention effects, tính đến việc thông tin có thể truyền gián tiếp qua multiple hops.

Rất thú vị là có thể kết hợp phân tích attention với tính toán LSD để tạo ra hybrid approach. Thay vì chỉ dựa vào Long-Short Difference, có thể sử dụng attention patterns như tín hiệu bổ sung để tinh chỉnh việc xác định key token. Cụ thể, có thể tính correlation giữa Attention và LSD để xác định liệu high-attention tokens cũng có high LSD scores hay không. Nếu correlation mạnh, có thể sử dụng attention patterns như efficient proxy cho LSD computation, giảm computational overhead đáng kể mà vẫn duy trì độ chính xác.

#### Hãy tham khảo Paper5: VEGAD - Vocabulary Expansion via GrADients
**trực giác**: "các nhóm token thể hiện gradient lớn hơn được coi là quan trọng hơn đối với nhiệm vụ và nên được tích hợp vào từ vựng như các thuật ngữ chuyên lĩnh vực". Tính gradient cho cả hai thành phần quan trọng:

- Embedding layer: G^embed = ∂L_lm/∂α		=> khả năng hiểu token
- lmhead (unembedding): G^lmhead = ∂L_lm/∂β => khả năng dự đoán token

"tầng language modeling head cũng quan trọng đặc biệt đối với các nhiệm vụ sinh văn bản", điều mà logits đơn thuần không thể capture được. Trong phần ablation study, khi loại bỏ gradient của LMHead layer, "đối với các bộ dữ liệu yêu cầu sinh văn bản, 'w/o LMHead' bị giảm đáng kể"

--------------------------------------------------
Như vậy ngoài LSD (long-short difference score) ta còn có grad_score để tính điểm cho từng token.
Bạn nghĩa grad_score này có ý nghĩa gì và có thể kết hợp được với LSD hay LongCE không?
--------------------------------------------------


| Gradient \ LSD    | **High LSD** | **Low LSD** |
|-------------------|--------------|-------------|
| **High Gradient** | **Super key tokens**<br/>*Token quan trọng cho task VÀ phụ thuộc long context* | **Local key tokens**<br/>*Token quan trọng nhưng có thể predict từ short context*|
| **Low Gradient**  | **Context-sensitive non-critical**<br/>*Token ít ảnh hưởng loss nhưng cần long context* | **Ignore**<br/>*Token không quan trọng*   |

Ví dụ về Context-sensitive non-critical; Trong task QA về document dài:
- Question: "What year was the company founded?"
- Answer: "1995"

Trong long document có thể có token "xxx" ("furthermore" chẳng hạn) ở đoạn giữa:
- `High LSD`: Token này cần long context để predict (vì context xung quanh nó phức tạp)
- `Low Gradient`: Nhưng việc predict đúng/sai "xxx" không ảnh hưởng đến việc trả lời đúng "1995"

Ví dụ thực tế từ paper:
```
"Sarah has a dog named Buddy.
[...]
Sarah feels happy to play with Buddy."
```
- "Buddy" có LSD = 2.08 (High) → cần long context để biết tên con chó
- "feels" có LSD = 0.00 (Low) → có thể đoán từ short context

---

Gradient score đo "`task relevance`" tổng quát của token đối với objective function. Cụ thể, G^embed đo mức độ sensitive của loss khi thay đổi embedding representation của token, phản ánh khả năng hiểu token, trong khi G^lmhead đo mức độ sensitive của loss khi thay đổi output weights cho token, phản ánh khả năng sinh token. Trực giác đằng sau là token có gradient lớn nghĩa là nếu ta thay đổi representation hoặc prediction weights của nó một chút, loss sẽ thay đổi nhiều, cho thấy token này critical cho task performance.

Khi so sánh gradient score với LSD score, ta thấy chúng đo lường những khía cạnh khác nhau. Gradient score đo general task importance trong khi LSD score đo long-context dependency. Về mặt tính toán, gradient score chỉ cần một backward pass và tận dụng gradients có sẵn nên cost thấp, trong khi LSD score cần nhiều forward passes nên cost cao hơn. Điều quan trọng là gradient score và LSD score không equivalent mà capture các orthogonal dimensions.

Có bốn trường hợp kết hợp có thể xảy ra (bảng trên). Token có high gradient và high LSD là super key tokens, quan trọng cho task và phụ thuộc long context. Token có high gradient nhưng low LSD là local key tokens, quan trọng nhưng có thể predict từ short context. Token có low gradient nhưng high LSD là Context-sensitive non-critical, ít ảnh hưởng loss nhưng cần long context. Token có cả low gradient và low LSD thì không quan trọng và có thể ignore.

Khả năng kết hợp với LongCE rất tiềm năng. Ta có thể tạo hybrid key token selection bằng cách combine các scores với trọng số learnable để balance giữa task importance và context dependency, đồng thời áp dụng multi-dimensional filtering chỉ chọn tokens high trên cả hai dimensions. Về computational efficiency, có thể sử dụng two-stage filtering approach, pre-filter using gradient scores với cost thấp, sau đó compute LSD chỉ cho high-gradient tokens. Điều này có thể dramatically reduce computational overhead từ O((n-K)K²) xuống O((αn-K)K²) với α << 1.

Enhanced LongCE loss có thể được thiết kế với I_hybrid function kết hợp cả gradient score và LSD score. Function này có thể là multiplicative khi cần cả hai scores cao, additive cho flexible weighting, hoặc gated khi gradient làm gate cho LSD computation nếu vượt qua threshold.

Việc kết hợp gradient-LSD mang lại nhiều advantages. Gradient scores có thể identify semantic importance mà LSD miss, ví dụ trong câu "The capital of France is Paris", token "Paris" có high gradient vì critical cho correctness nhưng chỉ medium LSD vì có thể guess được từ "capital of France" ngay cả với short context. Gradient patterns thay đổi theo domain và task, cho phép adaptive key token selection, với legal documents có high gradient cho legal terms, code có high gradient cho function names và keywords, medical texts có high gradient cho disease names và symptoms. Vì gradients được compute anyway trong backpropagation, việc **extract gradient scores almost free**, chỉ cần storage và aggregation.

Implementation strategy thực tế sẽ bao gồm gradient-based pre-filtering để `CHỌN TOP-K% TOKENS THEO GRADIENT SCORES`, then **LSD computation chỉ cho filtered tokens**, hybrid weighting để combine cả hai scores cho final I_hybrid, và adaptive thresholds để điều chỉnh các trọng số theo domain và task. Điều này có potential giảm computational cost của current LongCE while improving accuracy qua better semantic understanding. (có thể giảm 20%-30% cost ít nhất ở đoạn tính final loss vì CE đc tính per token)

Kết luận là gradient score từ VEGAD complement perfectly với LSD approach, offering both computational efficiency và semantic richness mà pure LSD-based methods thiếu. Đây là direction rất promising cho next-generation long-context training methods!
