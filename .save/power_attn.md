# Scaling Context Requires Rethinking Attention

###### Abstract

We argue that neither transformers nor sub-quadratic architectures are well suited to training at long sequence lengths: the cost of processing the context is too expensive in the former, too inexpensive in the latter. Approaches such as sliding window attention which reduce the cost-per-token of a transformer impair in-context learning, and so are also unsuitable. To address these limitations, we introduce power attention, an architectural layer for linear-cost sequence modeling whose state size can be adjusted independently of parameters, unlocking the advantages of linear attention on practical domains. We develop and open-source a set of GPU kernels for efficient power attention, identifying a novel pattern of operation fusion to avoid memory and bandwidth bottlenecks. Our experiments on the in-context learning of power attention shows that these models dominate both exponential attention and linear attention at long-context training.

##  1 Introduction

Many techniques to improve the performance of language models involve adding tokens to the context. One popular approach is to include reference material, such as by adding the content of a codebase to the context of a coding assistant [Jimenez et al., 2023]. Another approach is to introduce tokens sampled from the model itself, as is done by chain-of-thought LLMs [DeepSeek-AI et al., 2025, Wei et al., 2022]. A third approach is to use LLM agents, which iteratively interact with the world via tool use and adapt to feedback via context tokens [Yang et al., 2024, He et al., 2024, Schick et al., 2023]. If these context scaling techniques continue to pay off, one might expect a future where contexts regularly contain millions or even billions of tokens.

However, it remains unclear what architectures are best suited for training with long contexts. It is commonly argued that, despite their ubiquity, transformers [Vaswani et al., 2023] are poorly suited to long-context training due to their use of self-attention, whose compute cost grows quadratically with context length. The fact that modern transformer-based LLMs are trained primarily on context lengths between 4k and 32k tokens [Grattafiori et al., 2024, Meta, 2025, Google et al., 2025], with long-context training relegated to post-training (if at all), lends credence to this position. These concerns have motivated research on so-called subquadratic sequence architectures such as those proposed by Sun et al. [2023], Peng et al. [2023], Gu and Dao [2024]. These architectures primarily utilize variants of linear attention, an operation similar to the attention layer of transformers except that it allows for a recurrent linear-cost formulation.

In Section 3 we argue that any strong long-context architecture must possess three attributes:

  1. 1.

A balanced weight-to-state ratio at long contexts.

  2. 2.

Admits an efficient hardware-aware implementation on tensor cores.

  3. 3.

Good in-context learning (ICL) ability.

We then show that neither attention-based architectures, nor existing subquadratic architectures, meet these criteria. Table 1 summarizes our perspective.

In Section 4 we introduce power attention, a powerful variant of linear attention. Power attention possesses a hyperparameter p𝑝pp which controls the state size independently of the parameter count, enabling us to balance the weight-state FLOPs ratio for architectures of any scale. It also admits a hardware-aware implementation for training on GPUs. Section 4.1 describes the implementation of our open-source kernels, which enable real wall-clock speedups over Flash Attention in practical settings (e.g. p=2𝑝2p=2p = 2 is 8.6x faster at 64k context). Furthermore, our kernels still lag behind Flash Attention [Dao, 2023] in terms of hardware utilization, and so we expect future engineering efforts which close this gap to result in even larger speedups.

Architecture | Balance | Efficiency | ICL  
---|---|---|---  
Transformer | ✗ | ✓ | ✓  
Classic RNNs | ✗ | ✗ | ✓  
Modern RNNs | ✗ | ✓ | ✓  
Windowed Attention | ✓ | ✓ | ✗  
Power Attention | ✓ | ✓ | ✓  
Table 1:  Comparison of approaches. Section 3 justifies the importance of these criteria, and explains why each architecture passes or fails.  We evaluate power attention empirically in Section 5. Experiments in Section 5.1 show that power attention has better in-context learning than other balanced architectures. In Section 5.3, we show that when training on contexts of length 65536, power attention dominates both exponential and linear attention in terms of loss-per-FLOP.

These results have two main limitations. Firstly, our experiments are limited to measuring negative log likelihood on a dataset of generic natural language text. We did not study other domains, modalities, or downstream tasks. Secondly, in our setting, the compute-optimal context grows relatively slowly, diminishing the value of long-context training. We leave to future work the replication of these results across a variety of settings and metrics, and exploration to identify domains with long compute-optimal contexts (perhaps tasks that require chain-of-thought reasoning, or modalities such as audio).

##  2 Background

Sequence modeling. Let 𝒳𝒳\mathcal{X}caligraphic_X denote a finite set of tokens, referred to as the vocabulary. Let 𝒳t 𝒳𝑡\mathcal{X}^{t}caligraphic_X t denote the set of length-t𝑡tt sequences over 𝒳𝒳\mathcal{X}caligraphic_X, the documents. Given some distribution 𝔻∈Dist⁢(𝒳t)𝔻Dist 𝒳𝑡\mathbb{D}\in\text{Dist}(\mathcal{X}^{t})blackboard_D ∈ Dist ( caligraphic_X t ) we are concerned with finding a model assigning the maximum probability to documents sampled from 𝔻𝔻\mathbb{D}blackboard_D. A common approach is causal sequence modeling, based on a model fθ 𝑓𝜃f_{\theta}f θ  mapping sequences 𝒳i 𝒳𝑖\mathcal{X}^{i}caligraphic_X i of arbitrary length i∈ℕ𝑖ℕi\in\mathbb{N}i ∈ blackboard_N to distributions over next-tokens, Dist⁢(𝒳)Dist𝒳\text{Dist}(\mathcal{X})Dist ( caligraphic_X ), where θ∈Θ𝜃Θ\theta\in\Thetaθ ∈ roman_Θ denotes the parameters of the model. Implicitly, such a model defines a distribution over the space of all documents x∈𝒳t𝑥 𝒳𝑡x\in\mathcal{X}^{t}x ∈ caligraphic_X t via the autoregressive factorization:

 | fθ⁢(x)=fθ⁢(x1,…,xt)=∏i=1tfθ⁢(xi∣x<i) 𝑓𝜃𝑥 𝑓𝜃 𝑥1… 𝑥𝑡 product𝑖1𝑡 𝑓𝜃conditional 𝑥𝑖 𝑥absent𝑖\displaystyle f_{\theta}(x)=f_{\theta}(x_{1},\dots,x_{t})=\prod_{i=1}^{t}f_{% \theta}(x_{i}\mid x_{<i})f θ  ( x ) = f θ  ( x 1  , … , x t  ) = ∏ i = 1  t f θ  ( x i  ∣ x < i  ) |  | (1)  
---|---|---|---  
  
The goal of causal sequence modeling is to learn parameters θ𝜃\thetaθ such that the induced distribution fθ 𝑓𝜃f_{\theta}f θ  matches the data distribution, where error is typically measured by the cross-entropy loss:

 | ℒD⁢(θ)=𝔼x∼𝔻⁢[−log⁡fθ⁢(x)] ℒ𝐷𝜃 𝔼similar-to𝑥𝔻delimited-[] 𝑓𝜃𝑥\displaystyle\mathcal{L}_{D}(\theta)=\mathbb{E}_{x\sim\mathbb{D}}\left[-\log f% _{\theta}(x)\right]caligraphic_L D  ( θ ) = blackboard_E x ∼ blackboard_D  [ - roman_log f θ  ( x ) ] |  | (2)  
---|---|---|---  
  
##### Recurrent neural networks.

RNNs [Elman, 1990, Hochreiter and Schmidhuber, 1997, Cho et al., 2014] are models fθ 𝑓𝜃f_{\theta}f θ  which can be expressed using a Markovian state, Si∈ℝn 𝑆𝑖 ℝ𝑛S_{i}\in\mathbb{R}^{n}S i  ∈ blackboard_R n, which summarizes the information of the entire input history x≤i=x1,⋯,xi 𝑥absent𝑖 𝑥1⋯ 𝑥𝑖x_{\leq i}=x_{1},\cdots,x_{i}x ≤ i  = x 1  , ⋯ , x i . The output of an RNN can be expressed as yi=gθ⁢(xi,Si) 𝑦𝑖 𝑔𝜃 𝑥𝑖 𝑆𝑖y_{i}=g_{\theta}(x_{i},S_{i})y i  = g θ  ( x i  , S i  ) and the state evolves according to a recurrent relation Si+1=hθ⁢(xi,Si) 𝑆𝑖1 ℎ𝜃 𝑥𝑖 𝑆𝑖S_{i+1}=h_{\theta}(x_{i},S_{i})S i + 1  = h θ  ( x i  , S i  ).

##### Attention.

The causal self-attention layer, a critical piece of the transformer architecture [Vaswani et al., 2023], is defined as follows. Let Q,K∈ℝt×d,V∈ℝt×vformulae-sequence𝑄𝐾 ℝ𝑡𝑑𝑉 ℝ𝑡𝑣Q,K\in\mathbb{R}^{t\times d},\;V\in\mathbb{R}^{t\times v}Q , K ∈ blackboard_R t × d , V ∈ blackboard_R t × v be the query, key and value matrices. We can also think of them as sequences of vectors Qi,Ki∈ℝd 𝑄𝑖 𝐾𝑖 ℝ𝑑Q_{i},K_{i}\in\mathbb{R}^{d}Q i  , K i  ∈ blackboard_R d and Vi∈ℝv 𝑉𝑖 ℝ𝑣V_{i}\in\mathbb{R}^{v}V i  ∈ blackboard_R v. The output of the attention layer is a matrix attnexp⁢(Q,K,V)∈ℝt×v attnexp𝑄𝐾𝑉 ℝ𝑡𝑣\text{attn}_{\text{exp}}(Q,K,V)\in\mathbb{R}^{t\times v}attn exp  ( Q , K , V ) ∈ blackboard_R t × v defined as

 | attnexp⁢(Q,K,V)i=∑j=1ieQiT⁢Kj⁢Vj attnexp 𝑄𝐾𝑉𝑖 𝑗1𝑖 𝑒  𝑄𝑇𝑖 𝐾𝑗 𝑉𝑗\displaystyle\text{attn}_{\text{exp}}(Q,K,V)_{i}=\sum_{j=1}^{i}e^{Q^{T}_{i}K_{% j}}V_{j}attn exp  ( Q , K , V ) i  = ∑ j = 1  i e Q T i  K j  V j  |  | (3)  
---|---|---|---  
  
This can be implemented efficiently in matrix form by using a mask M∈ℝt×t𝑀 ℝ𝑡𝑡M\in\mathbb{R}^{t\times t}M ∈ blackboard_R t × t where Mi⁢j=𝟏i≤j 𝑀𝑖𝑗 1𝑖𝑗M_{ij}=\mathbf{1}_{i\leq j}M i j  = bold_1 i ≤ j ,

 | attnexp⁢(Q,K,V)=(exp⁢(Q⁢KT)⊙M)⁢V attnexp𝑄𝐾𝑉direct-productexp𝑄 𝐾𝑇𝑀𝑉\displaystyle\text{attn}_{\text{exp}}(Q,K,V)=\left(\text{exp}(QK^{T})\odot M% \right)Vattn exp  ( Q , K , V ) = ( exp ( Q K T ) ⊙ M ) V |  | (4)  
---|---|---|---  
  
where exp⁡(A)𝐴\exp(A)roman_exp ( A ) denotes element-wise exponentiation of the matrix A𝐴AA.

Attention can be expressed in an RNN-like form. The outputs Yi 𝑌𝑖Y_{i}Y i  depend only on a state Si=(K≤i,V≤i)∈ℝt×d⊕ℝt×v 𝑆𝑖 𝐾absent𝑖 𝑉absent𝑖direct-sum ℝ𝑡𝑑 ℝ𝑡𝑣S_{i}=\left(K_{\leq i},V_{\leq i}\right)\in\mathbb{R}^{t\times d}\oplus\mathbb% {R}^{t\times v}S i  = ( K ≤ i  , V ≤ i  ) ∈ blackboard_R t × d ⊕ blackboard_R t × v, commonly called the KV cache. The main difference from conventional RNNs is that the state of attention does not have a fixed dimensionality; it grows with sequence length.

##### Normalization.

To stabilize learning, attention usually requires normalization. The original (and most common) approach to normalization is to divide by the sum of the attention scores, turning them into a probability distribution [Vaswani et al., 2023]. We use this normalization throughout this work. One limitation of this approach is that it requires positive attention scores. Other approaches have been proposed [Gu et al., 2024, Ramapuram et al., 2024], but we do not consider them.

Sliding window attention. A variant of attention which chooses a window size w𝑤ww, and truncates the KV cache to this length using a first-in-first-out approach [Child et al., 2019]. The formula for the outputs is ∑j=i−wieQiT⁢Kj⁢Vj 𝑗𝑖𝑤𝑖 𝑒  𝑄𝑇𝑖 𝐾𝑗 𝑉𝑗\sum_{j=i-w}^{i}e^{Q^{T}_{i}K_{j}}V_{j}∑ j = i - w  i e Q T i  K j  V j .

Linear attention. Katharopoulos et al. [2020] removes the exponential from attention and projects the keys and queries using ϕ:ℝd→ℝD:italic-ϕ→ ℝ𝑑 ℝ𝐷\phi:\mathbb{R}^{d}\to\mathbb{R}^{D}ϕ : blackboard_R d → blackboard_R D,

 | attnlinϕ⁢(Q,K,V)=(ϕ⁢(Q)⁢ϕ⁢(K)T⊙M)⁢V attnlinitalic-ϕ𝑄𝐾𝑉direct-productitalic-ϕ𝑄italic-ϕ 𝐾𝑇𝑀𝑉\displaystyle\text{attn}_{\text{lin}}^{\phi}(Q,K,V)=\left(\phi(Q)\phi(K)^{T}% \odot M\right)Vattn lin  ϕ ( Q , K , V ) = ( ϕ ( Q ) ϕ ( K ) T ⊙ M ) V |  | (5)  
---|---|---|---  
  
where ϕ⁢(A)∈ℝt×Ditalic-ϕ𝐴 ℝ𝑡𝐷\phi(A)\in\mathbb{R}^{t\times D}ϕ ( A ) ∈ blackboard_R t × D denotes application of ϕitalic-ϕ\phiϕ to the rows of A∈ℝt×d𝐴 ℝ𝑡𝑑A\in\mathbb{R}^{t\times d}A ∈ blackboard_R t × d. The key property of linear attention is that it admits an alternative to the KV cache, a constant-size state Si∈ℝv×D 𝑆𝑖 ℝ𝑣𝐷S_{i}\in\mathbb{R}^{v\times D}S i  ∈ blackboard_R v × D unrolled via the recurrence relation:

 | attnlinϕ⁢(Q,K,V)i=Si⁢ϕ⁢(Qi)Si=Si−1+Vi⁢ϕ⁢(Ki)Tformulae-sequence attnlinitalic-ϕ 𝑄𝐾𝑉𝑖 𝑆𝑖italic-ϕ 𝑄𝑖 𝑆𝑖 𝑆𝑖1 𝑉𝑖italic-ϕ 𝐾𝑖𝑇\displaystyle\text{attn}_{\text{lin}}^{\phi}(Q,K,V)_{i}=S_{i}\phi(Q_{i})\qquad S% _{i}=S_{i-1}+V_{i}\phi(K_{i})^{T}attn lin  ϕ ( Q , K , V ) i  = S i  ϕ ( Q i  ) S i  = S i - 1  + V i  ϕ ( K i  ) T |  | (6)  
---|---|---|---  
  
The array of t𝑡tt outputs can be computed with cost O⁢(t⁢D⁢v)𝑂𝑡𝐷𝑣O(tDv)O ( t D v ). For this reason, for long sequences, the recurrent form is preferred over the attention form on the KV cache, which has cost O⁢(t2⁢(d+v))𝑂 𝑡2𝑑𝑣O\left(t^{2}(d+v)\right)O ( t 2 ( d + v ) ). This recurrent form also highlights the motivation behind the inclusion of ϕitalic-ϕ\phiϕ. Since S∈ℝD⁢v𝑆 ℝ𝐷𝑣S\in\mathbb{R}^{Dv}S ∈ blackboard_R D v, the choice of ϕitalic-ϕ\phiϕ can be used to adjust the state size, known as state expansion [Schlag et al., 2021].

##### Chunked form.

The recurrent form of linear transformers is rarely useful in practice. The states Si∈ℝv×D 𝑆𝑖 ℝ𝑣𝐷S_{i}\in\mathbb{R}^{v\times D}S i  ∈ blackboard_R v × D are typically large, so having to compute and store in memory every state in the sequence becomes a major bottleneck. The chunked form [Buckman and Gelada, a, Sun et al., 2023] interpolates between the recurrent form and the attention form, capturing benefits of both. The key idea is to compute only a subset of all states: S0,Sc,S2⁢c,⋯ 𝑆0 𝑆𝑐 𝑆2𝑐⋯S_{0},S_{c},S_{2c},\cdotsS 0  , S c  , S 2 c  , ⋯, for some appropriately chosen chunk size c∈ℕ𝑐ℕc\in\mathbb{N}c ∈ blackboard_N. The chunked form is given by the following equation:

 | Yn⁢c+m=Sn⁢c⁢ϕ⁢(Qn⁢c+m)+∑j=n⁢c+1n⁢c+m(Qn⁢c+mT⁢Kj)⁢Vj 𝑌𝑛𝑐𝑚 𝑆𝑛𝑐italic-ϕ 𝑄𝑛𝑐𝑚 𝑗𝑛𝑐1𝑛𝑐𝑚 𝑄𝑛𝑐𝑚𝑇 𝐾𝑗 𝑉𝑗Y_{nc+m}=S_{nc}\phi(Q_{nc+m})+\sum_{j=nc+1}^{nc+m}(Q_{nc+m}^{T}K_{j})V_{j}Y n c + m  = S n c  ϕ ( Q n c + m  ) + ∑ j = n c + 1  n c + m ( Q n c + m  T K j  ) V j  |   
---|---|---  
  
For any i𝑖ii there exist 0≤n0𝑛0\leq n0 ≤ n and 0≤m<c0𝑚𝑐0\leq m<c0 ≤ m < c such that i=n⁢c+m𝑖𝑛𝑐𝑚i=nc+mi = n c + m. So Yn⁢c+m 𝑌𝑛𝑐𝑚Y_{nc+m}Y n c + m  can be computed with an interaction with the state Sc⁢n 𝑆𝑐𝑛S_{cn}S c n  of cost O⁢(v⁢D)𝑂𝑣𝐷O(vD)O ( v D ) and an intra-chunk attention of cost O⁢(c⁢d)𝑂𝑐𝑑O(cd)O ( c d ). Thus, the cost of the entire output sequence is O⁢(t⁢D⁢v+t⁢c⁢d)𝑂𝑡𝐷𝑣𝑡𝑐𝑑O(tDv+tcd)O ( t D v + t c d ).

##### Gating.

On long-context tasks, it is common to give a mechanism for the network to directly avoid attending to old data. Originally, this was done at a fixed rate using techniques such as ALiBi [Press et al., 2021]. More recently, Lin et al. [2025] propose a learned gating value per timestep, which is the approach we adopt in this work. Gating has been demonstrated to be particularly important in linear attention [Zhang et al., 2024, Yang et al., 2023, Gu and Dao, 2024].

##### Architectures.

Self-attention layers are merely one piece of a broader transformer architecture, which typically alternates between attention and MLP layers [Vaswani et al., 2023]. Modern architectures often also include components such as rotary embeddings [Su et al., 2024] and local convolutions [Yang et al., 2025]. In this work, we focus our study only on the attention layer, and in general do not modify other architectural components. We use the FLA codebase [Yang and Zhang, 2024] for all architectures.

##  3 What does long-context attention require?

In this section, we provide a framework for understanding what attributes of attention techniques make them suitable for long-context training. We focus on classic attention and linear attention, and conclude that neither is suitable for this setting. All experiments are conducted on LongCrawl64 [Buckman, 2024], a dataset containing 6M documents each of length 64k tokens.

###  3.1 Long-context attention requires a large state.

(a) Context length 32. (b) Context length 1024. (c) Context length 8192. (d) All curves, per-FLOP. Figure 1: Exponential attention (blue) vs linear attention (red). In Figure 1 we compare the performance of classic exponential attention with that of linear attention at context lengths 32, 1024, and 8192. To check that our conclusions about attention are broadly applicable, we run these experiments across a range of architectures, each modified to use either exponential attention (blue) or linear attention (red). See Appendix D for the full experimental details.

At short context lengths, both forms of attention perform equivalently. But as context length grows, exponential attention gains an advantage. We hypothesize a simple explanation for these observations: state scaling improves performance. In this setting, at context length 32 (Figure 1(a)), the state size of linear and exponential attention is the same, explaining their equivalent performance per-update. Whereas at context length 8k (Figure 1(c)), the state size of exponential attention is 256x larger, explaining its better performance per-update.

This additional performance comes at a cost: the larger state requires additional FLOPs per update. Linear attention is often claimed to be superior to exponential attention because it reduces this cost [Sun et al., 2023]. However, Figure 1(d) reveals that this is misleading, as the best performance for any FLOP budget can be achieved by training with exponential attention.

###  3.2 Long-context attention requires state-weight balance.

We now explore the implications of the importance of state size on compute-optimal sequence architectures. The computations of a sequence model can be divided into weight FLOPs, which involve an activation and a parameter, and state FLOPs, which involve an activation and a state.111In this work, we only consider architectures whose weight FLOPs are proportional to parameter count, and whose state FLOPs are proportional to state size. However, note that techniques such as mixture-of-experts [Shazeer et al., 2017] produce a distinction between parameter count and weight FLOPs, and would require more nuanced analysis. We have seen in the previous section that long-context performance scales with state size, and it is well-established that performance scales with parameter count [Kaplan et al., 2020].

Figure 2: Compute-optimal transformers have a balanced WSFR. See Appendix D for details. Attention | Context Length | WSFR  
---|---|---  
Exponential | 1 024 | 8:1  
Exponential | 8 192 | 1:1  
Exponential | 65 536 | 1:8  
Exponential | 1 000 000 | 1:125  
Linear | 1 024 | 30:1  
Linear | 8 192 | 30:1  
Linear | 65 536 | 30:1  
Linear | 1 000 000 | 30:1  
Window-8192 | 1 024 | 8:1  
Window-8192 | 8 192 | 1:1  
Window-8192 | 65 536 | 1:1  
Window-8192 | 1 000 000 | 1:1  
  
Table 2: WSFR comparison between attention techniques at various context lengths. Balanced architectures are in bold. We refer to as the relative proportion of these two types of FLOPs as the weight-state FLOP ratio (WSFR), and we argue that for compute-optimal models, the WSFR should be somewhat close to 1:1. This is because, for any model with a skewed WSFR (for example 100:1), doubling the smaller dimension will be effectively free in terms of total FLOPs. Since both the state and weight scales have a large impact on model performance, it is unwise to not take advantage of free scaling, and doing so will cause the WSFR to approach 1:1.

In Figure 2, we explore this empirically. We train a 400M GPT-2 model on context length 4096, as well as two other models with approximately the same total FLOPs: a small model with a large state, and a large model with a small state. 222Since we adjust the state size via the context length, we also adjust the batch size inversely, to keep tokens-per-update identical between runs. This set of models is therefore nearly identical except for WSFR, allowing us to isolate the impact of balance. We confirm that the most balanced architecture has the best performance.

Table 2 shows the WSFR of 124M-parameter GPT-2 models with various attention techniques and context lengths. Exponential attention is balanced for intermediate context lengths, but unbalanced for long context lengths, where it does far more state FLOPs than weight FLOPs. Linear attention, in contrast, is unbalanced at all context lengths in the opposite direction: far more weight FLOPs than state FLOPs. Thus, neither architecture is well-suited for long-context training.

How can we resolve this imbalance? One natural approach is to reduce the state size of exponential attention. In fact, many recent works in the transformer literature can be interpreted through this lens: hybrid architectures [Lieber et al., 2024] reduce the size of the state along the layer dimension, sparse attention [Child et al., 2019] reduces the size of the state along the time dimension, multi-query attention [Shazeer, 2019] reduces the size of the state along the head dimension, and latent attention [Liu et al., 2024] reduces the size of the state along the feature dimension. We use windowed attention to exemplify this family of reduced-state exponential attention approaches. Table 2 shows that windowed attention architectures have balanced WSFR for large context lengths, given appropriate selection of window size.

###  3.3 Long-context attention requires in-context learning.

  
Figure 3: Comparing learning curves if linear vs window-32 attention for several architectures and context lengths. (a) Window-32 attention. (b) Linear attention. Figure 4: In-context learning across training for RWKV variants at context length 1024.  In Section 3.1, we saw exponential attention outperform linear attention, and attributed this success to its larger state. In Figure 3, we perform a more fair comparison, by juxtaposing linear attention with windowed attention of equal state size. See Appendix D for details. We see a reversal of the previous trend: it is now linear attention that dominates at all context lengths and FLOP budgets. This indicates that linear attention makes better use of its state than windowed attention.

We can explain this gap using an in-context learning (ICL) curve of the training loss, which plots the negative log-likelihood at each context length throughout training. In Figure 4, we compare the in-context learning ability of RWKV with windowed attention to linear attention up to context length 1024. Figure 4(a) shows that no in-context learning occurs beyond 100 tokens for window-32 attention.333Note that this is a 12-layer model, so the effective context window is 12∗32=384123238412*32=38412 ∗ 32 = 384 tokens. In contrast, linear attention can be seen in Figure 4(b) to demonstrate consistent in-context learning across the entire sequence.

Ultimately, a long-context model only has value if the extra context improves its predictions, so these results tell us that windowed attention, despite being balanced at all context lengths, is nonetheless a poor choice for a long-context architecture. We hypothesize that this limitation will be shared by the other reduced-state exponential attention approaches discussed in Section 3.2, although thorough investigation of this hypothesis is left to future work. Instead, in Section 4, we introduce a technique for the other natural approach to balanced long-context attention, expanded-state linear attention.

##  4 Power attention

If one substitutes the exponential in the classic attention formula by the p𝑝pp-th power, the result is power attention, variants of which have been studied by Arora et al. [2025], Kacham et al. [2024].

 | attnpowp⁢(Q,K,V)i=∑j=1i(QiT⁢Kj)p⁢Vj attnpow𝑝 𝑄𝐾𝑉𝑖 𝑗1𝑖  𝑄𝑇𝑖 𝐾𝑗𝑝 𝑉𝑗\displaystyle\text{attn}_{\text{pow}}^{p}(Q,K,V)_{i}=\sum_{j=1}^{i}\left(Q^{T}% _{i}K_{j}\right)^{p}V_{j}attn pow  p ( Q , K , V ) i  = ∑ j = 1  i ( Q T i  K j  ) p V j  |  | (7)  
---|---|---|---  
  
Power attention is a special case of linear attention because there exist functions ϕitalic-ϕ\phiϕ s.t. ϕ⁢(Qi)T⁢ϕ⁢(Kj)=(QiT⁢Kj)pitalic-ϕ 𝑄𝑖𝑇italic-ϕ 𝐾𝑗  𝑄𝑇𝑖 𝐾𝑗𝑝\phi(Q_{i})^{T}\phi(K_{j})=\left(Q^{T}_{i}K_{j}\right)^{p}ϕ ( Q i  ) T ϕ ( K j  ) = ( Q T i  K j  ) p, granting the computational advantages of linear attention discussed in Section 2. Its simple inner-product attention form gives it an important computational advantage over other proposed state-expanded linear transformers, such as DPFP described in Schlag et al. [2021], which require the explicit expansion of ϕ⁢(q),ϕ⁢(k)italic-ϕ𝑞italic-ϕ𝑘\phi(q),\phi(k)ϕ ( q ) , ϕ ( k ) in the attention form. When large intermediate objects (such as expanded keys) are involved, the fused attention algorithms pioneered by Dao [2023] do not work, meaning such algorithms have poor hardware utilization in practice.

###### Lemma 4.1

The function tpowp:ℝd→ℝdp: tpow𝑝→ ℝ𝑑 ℝ 𝑑𝑝\textsc{tpow}_{p}:\mathbb{R}^{d}\to\mathbb{R}^{d^{p}}tpow p  : blackboard_R d → blackboard_R d p defined as

 | tpowp⁢(x)=[x1⁢⋯⁢x1x1⁢⋯⁢x2⋮xd⁢⋯⁢xd]=[⋮∏kxik⋮](i1,⋯,ip)∈ℕd×p tpow𝑝𝑥matrix 𝑥1⋯ 𝑥1 𝑥1⋯ 𝑥2⋮ 𝑥𝑑⋯ 𝑥𝑑 matrix⋮ product𝑘 𝑥 𝑖𝑘⋮ 𝑖1⋯ 𝑖𝑝 ℕ𝑑absent𝑝\displaystyle\textsc{tpow}_{p}(x)=\begin{bmatrix}x_{1}\cdots x_{1}\\\ x_{1}\cdots x_{2}\\\ \vdots\\\ x_{d}\cdots x_{d}\\\ \end{bmatrix}=\begin{bmatrix}\vdots\\\ \prod_{k}x_{i_{k}}\\\ \vdots\\\ \end{bmatrix}_{(i_{1},\cdots,i_{p})\in\mathbb{N}_{d}^{\times p}}tpow p  ( x ) = [ start_ARG start_ROW start_CELL x 1  ⋯ x 1  end_CELL end_ROW start_ROW start_CELL x 1  ⋯ x 2  end_CELL end_ROW start_ROW start_CELL ⋮ end_CELL end_ROW start_ROW start_CELL x d  ⋯ x d  end_CELL end_ROW end_ARG ] = [ start_ARG start_ROW start_CELL ⋮ end_CELL end_ROW start_ROW start_CELL ∏ k  x i k   end_CELL end_ROW start_ROW start_CELL ⋮ end_CELL end_ROW end_ARG ] ( i 1  , ⋯ , i p  ) ∈ blackboard_N d  × p  |  | (8)  
---|---|---|---  
  
Then, for q,k∈ℝd𝑞𝑘 ℝ𝑑q,k\in\mathbb{R}^{d}q , k ∈ blackboard_R d the following property holds tpowp⁢(q)T⁢tpowp⁢(k)=(qT⁢k)p tpow𝑝 𝑞𝑇 tpow𝑝𝑘  𝑞𝑇𝑘𝑝\textsc{tpow}_{p}(q)^{T}\textsc{tpow}_{p}(k)=(q^{T}k)^{p}tpow p  ( q ) T tpow p  ( k ) = ( q T k ) p

We therefore have that attnpowp⁢(Q,K,V)=attnlintpowp⁢(Q,K,V) attnpow𝑝𝑄𝐾𝑉 attnlin tpow𝑝𝑄𝐾𝑉\text{attn}_{\text{pow}}^{p}(Q,K,V)=\text{attn}_{\text{lin}}^{\textsc{tpow}_{p% }}(Q,K,V)attn pow  p ( Q , K , V ) = attn lin  tpow p  ( Q , K , V ). The proofs for this section can be found in Appendix B.

However, a major disadvantage of tpowp⁢(x) tpow𝑝𝑥\textsc{tpow}_{p}(x)tpow p  ( x ) is that it contains redundant entries. The theory of symmetric powers can be used to address this issue.

###### Lemma 4.2

For any d,p∈ℕ𝑑𝑝ℕd,p\in\mathbb{N}d , p ∈ blackboard_N denote the set of non-decreasing multi-indices as NDMIdp={(i1,⋯,ip)∈ℕd×p|i1≤⋯≤ip}  NDMI𝑝𝑑conditional-set 𝑖1⋯ 𝑖𝑝 ℕ𝑑absent𝑝 𝑖1⋯ 𝑖𝑝{\text{NDMI}^{p}_{d}}=\\{(i_{1},\cdots,i_{p})\in\mathbb{N}_{d}^{\times p}\;|\;i% _{1}\leq\cdots\leq i_{p}\\}NDMI p d  = { ( i 1  , ⋯ , i p  ) ∈ blackboard_N d  × p | i 1  ≤ ⋯ ≤ i p  }. Define spowp:ℝd→ℝD: spow𝑝→ ℝ𝑑 ℝ𝐷\textsc{spow}_{p}:\mathbb{R}^{d}\to\mathbb{R}^{D}spow p  : blackboard_R d → blackboard_R D to be the function

 | spowp⁢(x)=[⋮p!∏khistk⁢(i)!⁢∏kxik⋮]i∈NDMIdp spow𝑝𝑥 matrix⋮𝑝 product𝑘 hist𝑘𝑖 product𝑘 𝑥 𝑖𝑘⋮𝑖  NDMI𝑝𝑑\displaystyle\textsc{spow}_{p}(x)=\begin{bmatrix}\vdots\\\ \sqrt{\frac{p!}{\prod_{k}\text{hist}_{k}(i)!}}\;\prod_{k}x_{i_{k}}\\\ \vdots\\\ \end{bmatrix}_{i\in{\text{NDMI}^{p}_{d}}}spow p  ( x ) = [ start_ARG start_ROW start_CELL ⋮ end_CELL end_ROW start_ROW start_CELL square-root start_ARG divide start_ARG p ! end_ARG start_ARG ∏ k  hist k  ( i ) ! end_ARG end_ARG ∏ k  x i k   end_CELL end_ROW start_ROW start_CELL ⋮ end_CELL end_ROW end_ARG ] i ∈ NDMI p d   |  | (9)  
---|---|---|---  
  
Where histk⁢(i1,⋯,i2)=∑j=1p1⁢(ij=k) hist𝑘 𝑖1⋯ 𝑖2 𝑗1𝑝1 𝑖𝑗𝑘\text{hist}_{k}(i_{1},\cdots,i_{2})=\sum_{j=1}^{p}1(i_{j}=k)hist k  ( i 1  , ⋯ , i 2  ) = ∑ j = 1  p 1 ( i j  = k ) is simply the function that counts how many times the index k𝑘kk occurs across the the multi index. Then, the following statements hold:

  1. 1.

The dimensionality D𝐷DD is given by (d+p−1p)binomial𝑑𝑝1𝑝\binom{d+p-1}{p}( FRACOP start_ARG d + p - 1 end_ARG start_ARG p end_ARG ) (the binomial n choose k)

  2. 2.

The inner products spowp⁢(q)T⁢spowp⁢(k)=(qT⁢k)p spow𝑝 𝑞𝑇 spow𝑝𝑘  𝑞𝑇𝑘𝑝\textsc{spow}_{p}(q)^{T}\textsc{spow}_{p}(k)=(q^{T}k)^{p}spow p  ( q ) T spow p  ( k ) = ( q T k ) p

A few concrete examples might be helpful:

 | spow2⁢([x1x2])=[x1⁢x12⁢x1⁢x2x2⁢x2]spow3⁢([x1x2])=[x1⁢x1⁢x13⁢x1⁢x1⁢x23⁢x1⁢x2⁢x2x2⁢x2⁢x2]formulae-sequence spow2matrix 𝑥1 𝑥2matrix 𝑥1 𝑥12 𝑥1 𝑥2 𝑥2 𝑥2 spow3matrix 𝑥1 𝑥2matrix 𝑥1 𝑥1 𝑥13 𝑥1 𝑥1 𝑥23 𝑥1 𝑥2 𝑥2 𝑥2 𝑥2 𝑥2\textsc{spow}_{2}\left(\begin{bmatrix}x_{1}\\\ x_{2}\end{bmatrix}\right)=\begin{bmatrix}\;\;\;\;\;\;x_{1}x_{1}\\\ \sqrt{2}\;x_{1}x_{2}\\\ \;\;\;\;\;\;x_{2}x_{2}\\\ \end{bmatrix}\qquad\textsc{spow}_{3}\left(\begin{bmatrix}x_{1}\\\ x_{2}\end{bmatrix}\right)=\begin{bmatrix}\;\;\;\;\;\;x_{1}x_{1}x_{1}\\\ \sqrt{3}\;x_{1}x_{1}x_{2}\\\ \sqrt{3}\;x_{1}x_{2}x_{2}\\\ \;\;\;\;\;\;x_{2}x_{2}x_{2}\\\ \end{bmatrix}spow 2  ( [ start_ARG start_ROW start_CELL x 1  end_CELL end_ROW start_ROW start_CELL x 2  end_CELL end_ROW end_ARG ] ) = [ start_ARG start_ROW start_CELL x 1  x 1  end_CELL end_ROW start_ROW start_CELL square-root start_ARG 2 end_ARG x 1  x 2  end_CELL end_ROW start_ROW start_CELL x 2  x 2  end_CELL end_ROW end_ARG ] spow 3  ( [ start_ARG start_ROW start_CELL x 1  end_CELL end_ROW start_ROW start_CELL x 2  end_CELL end_ROW end_ARG ] ) = [ start_ARG start_ROW start_CELL x 1  x 1  x 1  end_CELL end_ROW start_ROW start_CELL square-root start_ARG 3 end_ARG x 1  x 1  x 2  end_CELL end_ROW start_ROW start_CELL square-root start_ARG 3 end_ARG x 1  x 2  x 2  end_CELL end_ROW start_ROW start_CELL x 2  x 2  x 2  end_CELL end_ROW end_ARG ] |   
---|---|---  
  
Table 3 compares the dimensions dp 𝑑𝑝d^{p}d p and (d+p−1p)binomial𝑑𝑝1𝑝\binom{d+p-1}{p}( FRACOP start_ARG d + p - 1 end_ARG start_ARG p end_ARG ) for tpow and spow respectively. For large p𝑝pp, these reductions in D𝐷DD have a large impact on the runtime and memory utilization of chunked power attention.

Ultimately, spowp spow𝑝\textsc{spow}_{p}spow p  is a state expansion that increases the state size by a factor of (d+p−1p)dbinomial𝑑𝑝1𝑝𝑑\frac{\binom{d+p-1}{p}}{d}divide start_ARG ( FRACOP start_ARG d + p - 1 end_ARG start_ARG p end_ARG ) end_ARG start_ARG d end_ARG without introducing any parameters. For example, for a model with head size 64, p=2𝑝2p=2p = 2 increases the state size by a factor of approximately 32, p=3𝑝3p=3p = 3 by about 700, and p=4𝑝4p=4p = 4 by about 12000.

p𝑝pp | tpow⁢Dtpow𝐷\textsc{tpow}\;Dtpow D | spow⁢Dspow𝐷\textsc{spow}\;Dspow D | Savings  
---|---|---|---  
2 | 4096 | 2080 | 49%  
3 | 262144 | 45760 | 82%  
4 | 16777216 | 766480 | 95%  
5 | 1073741824 | 10424128 | 99%  
6 | 68719476736 | 119877472 | 99.8%  
Table 3: State size comparison for tensor vs. symmetric power embeddings for d=64𝑑64d=64d = 64. ###  4.1 Hardware-aware implementation

An efficient implementation of chunked power attention requires careful consideration to the sizes of relevant objects. The main quantities that appear are the key dimension d𝑑dd, the value dimension v𝑣vv, the sequence dimension t𝑡tt, and the expanded key dimension D𝐷DD. At large problem sizes, d,v𝑑𝑣d,vd , v typically stay small, whereas t,D𝑡𝐷t,Dt , D typically become large. For example, in Llama 3 [Grattafiori et al., 2024], the largest d𝑑dd is 128128128128, whereas the largest t𝑡tt is 128000128000128000128000.

The inputs and outputs of attention, Q,K,V,Y𝑄𝐾𝑉𝑌Q,K,V,YQ , K , V , Y, are all in either ℝt×d ℝ𝑡𝑑\mathbb{R}^{t\times d}blackboard_R t × d or ℝt×v ℝ𝑡𝑣\mathbb{R}^{t\times v}blackboard_R t × v. But some intermediate objects, most notably ϕ⁢(Q),ϕ⁢(K)italic-ϕ𝑄italic-ϕ𝐾\phi(Q),\phi(K)ϕ ( Q ) , ϕ ( K ), live in ℝt×D ℝ𝑡𝐷\mathbb{R}^{t\times D}blackboard_R t × D. If materialized, these objects dominate memory consumption, and their IO bottlenecks computation and reduces arithmetic intensity. This is reminiscent of how in standard attention, memory and IO is dominated by the attention matrix, an intermediate object in ℝt×t ℝ𝑡𝑡\mathbb{R}^{t\times t}blackboard_R t × t. This problem was addressed by Flash Attention [Dao et al., 2022] via operator fusion, whose central algorithmic innovation was the design of a kernel that avoids materializing the attention matrix in HBM.

We apply the same principles to design efficient kernels for chunked power attention. We factorize the algorithm as follows:

 | update-state⁢(S,K,V)=S+VT⁢ϕ⁢(K)query-state⁢(S,Q)=ϕ⁢(Q)⁢STformulae-sequenceupdate-state𝑆𝐾𝑉𝑆 𝑉𝑇italic-ϕ𝐾query-state𝑆𝑄italic-ϕ𝑄 𝑆𝑇\displaystyle\text{update-state}(S,K,V)=S+V^{T}\phi(K)\qquad\text{query-state}% (S,Q)=\phi(Q)S^{T}update-state ( S , K , V ) = S + V T ϕ ( K ) query-state ( S , Q ) = ϕ ( Q ) S T |  | (10)  
---|---|---|---  
  
Each of these functions centers around a matrix multiplication between a large expanded object and a smaller object. This computational structure can be exploited via a fused expand-MMA kernel, a matrix multiplication where the tiles of one operand are expanded on-the-fly.

Our implementation of expand-MMA uses Triton [Tillet et al., 2019] with a custom templating system to handle multiple values of p𝑝pp. A complete implementation of chunked power attention also requires a kernel for intra-chunk attention and another for the cumulative gated sum of states (see Appendix F). We use Flash Attention [Dao et al., 2022] for the former and a simple CUDA kernel for the latter.

We have released our kernels open-source444https://github.com/m-a-n-i-f-e-s-t/power-attention to allow others to use power attention, and to enable research on other applications of the symmetric power in deep learning. In Appendix F, we provide more details on our implementation.

####  4.1.1 tspow

Based on Table 3, one would expect that chunked linear attention using spow would run faster than tpow, because its smaller D𝐷DD translates into fewer FLOPs. However, modern GPUs are mainly optimized for matrix multiplications, and the tpow expansion is more compatible with the computational structure of a matmul. tpow calculations can be easily partitioned for parallel processing amongst CTAs/threads using a standard tiling approach. In contrast, the less-regular structure of a symmetric tensor is less compatible with standard GPU operations. The correction term p!∏khistk⁢(i)!𝑝 product𝑘 hist𝑘𝑖\sqrt{\frac{p!}{\prod_{k}\text{hist}_{k}(i)!}}square-root start_ARG divide start_ARG p ! end_ARG start_ARG ∏ k  hist k  ( i ) ! end_ARG end_ARG must be computed on slower CUDA cores, causing thread divergence due to branching, and the jagged memory access patterns lead to share memory bank conflicts.

 (a) tpow  (b) spow  (c) tspow Figure 5: Illustration of tpow, spow, and tspow. Our approach is to use the idea of tiling to interpolate between tpow and spow, harnessing benefits of both. Our proposed tiled symmetric power expansion, tspow, operates on tiles of data (providing the GPU-friendly structure of tpow) but only computes tiles of data with non-decreasing multi-indices (reducing data duplication like spow). Figure 5 paints the basic picture for p=2𝑝2p=2p = 2. The dimension of every tile is d⁢-⁢t⁢i⁢l⁢ep𝑑-𝑡𝑖𝑙 𝑒𝑝d{\text{-}tile}^{p}d - t i l e p, and the number of tiles with non-decreasing multi indices is (d/d⁢-⁢t⁢i⁢l⁢e+p−1p)binomial𝑑𝑑-𝑡𝑖𝑙𝑒𝑝1𝑝\binom{d/d{\text{-}tile}+p-1}{p}( FRACOP start_ARG d / d - t i l e + p - 1 end_ARG start_ARG p end_ARG ). This means the dimension D=(d/d⁢-⁢t⁢i⁢l⁢e+p−1p)⁢d⁢-⁢t⁢i⁢l⁢ep𝐷binomial𝑑𝑑-𝑡𝑖𝑙𝑒𝑝1𝑝𝑑-𝑡𝑖𝑙 𝑒𝑝D=\binom{d/d{\text{-}tile}+p-1}{p}d{\text{-}tile}^{p}D = ( FRACOP start_ARG d / d - t i l e + p - 1 end_ARG start_ARG p end_ARG ) d - t i l e p. Empirically, we find that d⁢-⁢t⁢i⁢l⁢e=8𝑑-𝑡𝑖𝑙𝑒8d{\text{-}tile}=8d - t i l e = 8 is a good choice for p=2𝑝2p=2p = 2. For p=3𝑝3p=3p = 3 a smaller d⁢-⁢t⁢i⁢l⁢e=4𝑑-𝑡𝑖𝑙𝑒4d{\text{-}tile}=4d - t i l e = 4 seems preferable.

(a) Throughput comparison at head size 64. (b) Throughput comparison at head size 32. (c) Effect of chunk size on total execution time. Figure 6: Hardware efficiency of Power Attention kernels. ####  4.1.2 Benchmarks

To benchmark our progress, we compare the throughput (tokens per second) between Power Attention and Flash Attention kernels, with a batch size of 8 and 12 heads, on an A100 GPU. For short contexts, the attention form achieves higher throughput, but as the context size grows, Power Attention switches to the chunk form and retains a constant throughput from then on. On the other hand, the throughput of flash attention decays proportional to t𝑡tt. At context length 65536, degree-2 Power Attention achieves 3.3x (for head size 64) and 8.6x (for head size 32) higher throughput than Flash Attention.

Note that the performance of Power Attention is highly dependent on the chunk size c𝑐cc. Figure 6 shows the total execution time is broken down into its component operations for various c𝑐cc.

##  5 Empirical evaluation of power attention

In this section, we evaluate power attention on the basis of its in-context learning ability and long-context performance. To ensure that the dataset contains documents with true long-term structure 555This is not true of many common benchmark datasets. For example, most sequences in OpenWebText [Liu et al., 2019] have length less than 1k. Figure 10 in Appendix C shows the document length distribution., all of our experiments are conducted on LongCrawl64 [Buckman, 2024]. For these experiments, we use power attention with per-head gating, and normalize by the sum of the attention weights.

###  5.1 In-context learning comparison

(a) Window-1k attention. (b) p=2𝑝2p=2p = 2 power attention. (c) Close-up on ICL curves at 50k. Figure 7: Power attention demonstrates more ICL per FLOP than equivalent windowed attention. In Section 3, we saw that linear attention has better in-context learning than windowed attention. Now, we investigate whether this is also true for weight-state balanced linear attention, by using power attention. Figure 7 shows the progression through training of the ICL curves of two models with balanced weight-state ratios on context length 4096: window-1024 attention and power attention with p=2𝑝2p=2p = 2. Both models are based on the RKWV architecture with the attention layer swapped for their respective attention mechanism (see Appendix D for full experimental details). In this setting, we see that the power attention architecture has the steepest ICL curve throughout training. Furthermore, as a result of its better in-context learning ability, power attention outperforms (per FLOP) an equivalent transformer in this setting (see Appendix E).

###  5.2 Factors impacting in-context learning

(a) Gradient updates. (b) Documents per batch. (c) Parameter count. (d) Context length. Figure 8: The impact of conventional scaling axes on in-context learning of power attention. Figure 8 shows the context-wise loss curve of a p=2𝑝2p=2p = 2 power transformer when varying four axes: number of gradient updates, documents per batch, parameter count, and context length. See Appendix D for experimental details. In all cases, the ICL curve becomes steeper as we scale the respective axis. This indicates that long-context predictions benefit more from scale than short-context predictions.

One phenomenon of note is that scaling context, as shown in Figure 8(d), has two effects: additional opportunity for ICL and additional tokens-per-update (this second effect is similar to that of scaling the batch size). The dashed grey lines on this plot ablate these two factors by sampling long sequences but reshaping into a larger batch of shorter sequences, which removes the effect of ICL and so isolates the effect of the additional tokens. We see that the additional tokens are responsible for nearly all of the improvement, and so the same effect could have been achieved by scaling the batch size. 666Although we do not explore it in depth in this work, we note that increasing the batch size typically increases the diversity of the tokens more quickly than increasing the context length does. This translates into better gradient estimates and improved learning, including improved in-context learning. The takeaway is that increasing the context length is not always the best way to improve the in-context learning ability of a model, since all axes of scale improve in-context learning.

###  5.3 Long-context training

(a) Heldout best-context loss across training. (b) ICL after 3e8 TeraFLOPs. Figure 9: Comparison between different forms of attention on long context. The dashed line in 9(a) indicates the position of ICL measurement in 9(b). We now turn to the question of what architecture is best for training on long contexts. We compare three architectures, all based on RWKV, but with different attention layers: native RWKV linear attention, classic exponential attention, and p=2𝑝2p=2p = 2 power attention. Updates are computed on batches of 32 documents, each of length 65536. See Appendix D for experimental details. We see in Figure 9(a) that power attention dominates other architectures in terms of loss-per-FLOP.

The gap in performance between power attention and exponential attention can be attributed to the difference in cost: exponential attention is much more expensive than power attention on long contexts, so the power attention architecture has the opportunity to perform many more steps of gradient descent. In contrast, the gap in performance between power attention and the original RWKV can by attributed to the difference between their in-context learning abilities, as seen in Figure 9(b). RWKV obtains almost no benefit from additional context beyond 2000 tokens. Power attention allows RWKV to in-context learn nearly as well as exponential attention, while still retaining a large advantage in cost.

We note some limitations of this result. Firstly, a context length of 65536 is far larger than is compute-optimal in this setting, meaning that the dominance of power attention demonstrated here does not directly motivate its use on this dataset. Secondly, note that while power attention dominates in the FLOP regime of our experiments,777Our 109 10910^{9}10 9 TeraFLOPs corresponds to about 1000 H100-hours at 30% flop utilization. we expect that given sufficient training FLOPs, the attention model would overtake the power attention model, thanks to its larger state size.

##  6 Conclusions & future work

Our results indicate that linear attention with a hardware-efficient state expansion is the most effective architecture on long-context training, thanks to state-weight balance and strong in-context learning. We have proposed power attention, which is one such approach, and hope future architectural research will continue to study a variety of attention variants and state expansions. For example, one limitation of power attention as currently proposed is its use of the normalization from Vaswani et al. [2023], which requires positive inner products. This means only even powers are supported, and so the parameter-free adjustments to the state size enabled by adjusting p𝑝pp are coarse.

As discussed in Section 3.2 there are many techniques in the literature which reduce the state size of a transformer, including hybrid models, sparse attention, multi-query attention, and latent attention. We investigated one such approach, windowed attention, and found its in-context learning abilities to be worse than linear attention models of the same state size. In the future, a more comprehensive comparison to existing methods would be valuable. Furthermore, a complete characterization of the performance of these algorithms merits rigorous investigation under the framework of scaling laws [Kaplan et al., 2020]. Future work should quantitatively explore the impact of state size, context size, and in-context learning on model performance, with the aim of fitting scaling laws dependent on these factors.

Our initial implementation uses Triton [Tillet et al., 2019], a high-level tool for writing GPU kernels that allows quick, Pythonic prototyping. However, without the flexibility provided by CUDA, our kernels cannot be optimized as thoroughly as e.g. Flash Attention [Dao, 2023]. As a result, our implementation, Power Attention, is not yet as dominant in wall-clock comparisons as FLOPs comparisons would indicate. Future implementations of Power Attention will move from Triton to CUDA in order to push wall-clock performance further.

Our experiments are limited to measuring negative log likelihood on a dataset of generic natural language text. We did not study other domains, modalities, or downstream tasks. In the future, we hope to validate our findings in these settings. Furthermore, we have observed that autoregressive prediction of natural language is largely dominated by short-context dependencies, even on long documents. This diminishes the value of long-context training in this setting. In future work, we hope to discover domains which are dominated by long-term dependencies. For example, we plan to explore tasks that require chain-of-thought reasoning, tool use, and modalities such as audio and video. In domains where performance is heavily dependent on long-term dependencies, the compute-optimal context will be large, and we expect that the dominance of power attention on long contexts will be of practical importance.

##  Appendix A Derivation of chunked algorithm

Here we we prove that the chunked form of linear attention is equivalent to the attention form. Recall that the chunk-form says

 | Y(i)c=Sc⁢i⁢Q(i)c+V(i)c⁢(Q(i)c⁢K(i)cT⊙M)Sc⁢(i+1)=Sc⁢i+V(i)c⁢K(i)cTformulae-sequence 𝑌 𝑖𝑐 𝑆𝑐𝑖 𝑄 𝑖𝑐 𝑉 𝑖𝑐direct-product 𝑄 𝑖𝑐 𝐾 𝑖𝑐𝑇𝑀 𝑆𝑐𝑖1 𝑆𝑐𝑖 𝑉 𝑖𝑐 𝐾 𝑖𝑐𝑇\displaystyle Y_{(i)_{c}}=S_{ci}Q_{(i)_{c}}+V_{(i)_{c}}\left(Q_{(i)_{c}}K_{(i)% _{c}}^{T}\odot M\right)\;\;\;\;\;\;\;\;\;\;\;\;S_{c(i+1)}=S_{ci}+V_{(i)_{c}}K_% {(i)_{c}}^{T}Y ( i ) c   = S c i  Q ( i ) c   + V ( i ) c   ( Q ( i ) c   K ( i ) c   T ⊙ M ) S c ( i + 1 )  = S c i  + V ( i ) c   K ( i ) c   T |  | (11)  
---|---|---|---  
  
Because this is in matrix form, if we look at output at each position i𝑖ii, it becomes

 | Yi 𝑌𝑖\displaystyle Y_{i}Y i  | =Sc⁢i⁢Qi+∑j=⌊ic⌋⁢c+1i(Qi⁢KjT)⁢Vjabsent 𝑆𝑐𝑖 𝑄𝑖 𝑗𝑖𝑐𝑐1𝑖 𝑄𝑖  𝐾𝑇𝑗 𝑉𝑗\displaystyle=S_{ci}Q_{i}+\sum_{j=\lfloor\frac{i}{c}\rfloor c+1}^{i}(Q_{i}K^{T% }_{j})V_{j}= S c i  Q i  + ∑ j = ⌊ divide start_ARG i end_ARG start_ARG c end_ARG ⌋ c + 1  i ( Q i  K T j  ) V j  |  | (12)  
---|---|---|---|---  
|  | =(Sc⁢(i−1)+V(i)c⁢K(i)cT)⁢Qi+∑j=⌊ic⌋⁢c+1i(Qi⁢KjT)⁢Vjabsent 𝑆𝑐𝑖1 𝑉 𝑖𝑐 𝐾 𝑖𝑐𝑇 𝑄𝑖 𝑗𝑖𝑐𝑐1𝑖 𝑄𝑖  𝐾𝑇𝑗 𝑉𝑗\displaystyle=(S_{c(i-1)}+V_{(i)_{c}}K_{(i)_{c}}^{T})Q_{i}+\sum_{j=\lfloor% \frac{i}{c}\rfloor c+1}^{i}(Q_{i}K^{T}_{j})V_{j}= ( S c ( i - 1 )  + V ( i ) c   K ( i ) c   T ) Q i  + ∑ j = ⌊ divide start_ARG i end_ARG start_ARG c end_ARG ⌋ c + 1  i ( Q i  K T j  ) V j  |  | (13)  
|  | =Sc⁢(i−1)⁢Qi+∑j=⌊i−1c⌋⁢c+1⌊ic⌋⁢cVj⁢KjT⁢Qi+∑j=⌊ic⌋⁢c+1i(Qi⁢KjT)⁢Vjabsent 𝑆𝑐𝑖1 𝑄𝑖 𝑗𝑖1𝑐𝑐1𝑖𝑐𝑐 𝑉𝑗 𝐾𝑗𝑇 𝑄𝑖 𝑗𝑖𝑐𝑐1𝑖 𝑄𝑖  𝐾𝑇𝑗 𝑉𝑗\displaystyle=S_{c(i-1)}Q_{i}+\sum_{j=\lfloor\frac{i-1}{c}\rfloor c+1}^{% \lfloor\frac{i}{c}\rfloor c}V_{j}K_{j}^{T}Q_{i}+\sum_{j=\lfloor\frac{i}{c}% \rfloor c+1}^{i}(Q_{i}K^{T}_{j})V_{j}= S c ( i - 1 )  Q i  + ∑ j = ⌊ divide start_ARG i - 1 end_ARG start_ARG c end_ARG ⌋ c + 1  ⌊ divide start_ARG i end_ARG start_ARG c end_ARG ⌋ c V j  K j  T Q i  + ∑ j = ⌊ divide start_ARG i end_ARG start_ARG c end_ARG ⌋ c + 1  i ( Q i  K T j  ) V j  |  | (14)  
|  | =Sc⁢(i−1)⁢Qi+∑j=⌊i−1c⌋⁢c+1⌊ic⌋⁢c(KjT⁢Qi)⁢Vj+∑j=⌊ic⌋⁢c+1i(Qi⁢KjT)⁢Vjabsent 𝑆𝑐𝑖1 𝑄𝑖 𝑗𝑖1𝑐𝑐1𝑖𝑐𝑐 𝐾𝑗𝑇 𝑄𝑖 𝑉𝑗 𝑗𝑖𝑐𝑐1𝑖 𝑄𝑖  𝐾𝑇𝑗 𝑉𝑗\displaystyle=S_{c(i-1)}Q_{i}+\sum_{j=\lfloor\frac{i-1}{c}\rfloor c+1}^{% \lfloor\frac{i}{c}\rfloor c}(K_{j}^{T}Q_{i})V_{j}+\sum_{j=\lfloor\frac{i}{c}% \rfloor c+1}^{i}(Q_{i}K^{T}_{j})V_{j}= S c ( i - 1 )  Q i  + ∑ j = ⌊ divide start_ARG i - 1 end_ARG start_ARG c end_ARG ⌋ c + 1  ⌊ divide start_ARG i end_ARG start_ARG c end_ARG ⌋ c ( K j  T Q i  ) V j  + ∑ j = ⌊ divide start_ARG i end_ARG start_ARG c end_ARG ⌋ c + 1  i ( Q i  K T j  ) V j  |  | (15)  
|  | =Sc⁢(i−1)⁢Qi+∑j=⌊i−1c⌋⁢c+1⌊ic⌋⁢c(Qi⁢KjT)⁢Vj+∑j=⌊ic⌋⁢c+1i(Qi⁢KjT)⁢Vjabsent 𝑆𝑐𝑖1 𝑄𝑖 𝑗𝑖1𝑐𝑐1𝑖𝑐𝑐 𝑄𝑖 𝐾𝑗𝑇 𝑉𝑗 𝑗𝑖𝑐𝑐1𝑖 𝑄𝑖  𝐾𝑇𝑗 𝑉𝑗\displaystyle=S_{c(i-1)}Q_{i}+\sum_{j=\lfloor\frac{i-1}{c}\rfloor c+1}^{% \lfloor\frac{i}{c}\rfloor c}(Q_{i}K_{j}^{T})V_{j}+\sum_{j=\lfloor\frac{i}{c}% \rfloor c+1}^{i}(Q_{i}K^{T}_{j})V_{j}= S c ( i - 1 )  Q i  + ∑ j = ⌊ divide start_ARG i - 1 end_ARG start_ARG c end_ARG ⌋ c + 1  ⌊ divide start_ARG i end_ARG start_ARG c end_ARG ⌋ c ( Q i  K j  T ) V j  + ∑ j = ⌊ divide start_ARG i end_ARG start_ARG c end_ARG ⌋ c + 1  i ( Q i  K T j  ) V j  |  | (16)  
|  | =Sc⁢(i−1)⁢Qi+∑j=⌊i−1c⌋⁢c+1i(Qi⁢KjT)⁢Vjabsent 𝑆𝑐𝑖1 𝑄𝑖 𝑗𝑖1𝑐𝑐1𝑖 𝑄𝑖  𝐾𝑇𝑗 𝑉𝑗\displaystyle=S_{c(i-1)}Q_{i}+\sum_{j=\lfloor\frac{i-1}{c}\rfloor c+1}^{i}(Q_{% i}K^{T}_{j})V_{j}= S c ( i - 1 )  Q i  + ∑ j = ⌊ divide start_ARG i - 1 end_ARG start_ARG c end_ARG ⌋ c + 1  i ( Q i  K T j  ) V j  |  | (17)  
| ⋮⋮\displaystyle\vdots⋮ |  | (18)  
|  | =S0⁢Qi+∑j=⌊0c⌋⁢c+1i(Qi⁢KjT)⁢Vjabsent 𝑆0 𝑄𝑖 𝑗0𝑐𝑐1𝑖 𝑄𝑖  𝐾𝑇𝑗 𝑉𝑗\displaystyle=S_{0}Q_{i}+\sum_{j=\lfloor\frac{0}{c}\rfloor c+1}^{i}(Q_{i}K^{T}% _{j})V_{j}= S 0  Q i  + ∑ j = ⌊ divide start_ARG 0 end_ARG start_ARG c end_ARG ⌋ c + 1  i ( Q i  K T j  ) V j  |  | (19)  
|  | =∑j=1i(Qi⁢KjT)⁢Vjassuming initial state is 0 𝑗1𝑖 𝑄𝑖  𝐾𝑇𝑗 𝑉𝑗assuming initial state is 0\displaystyle=\sum_{j=1}^{i}(Q_{i}K^{T}_{j})V_{j}\quad\text{assuming initial % state is 0}= ∑ j = 1  i ( Q i  K T j  ) V j  assuming initial state is 0 |  | (20)  
  
This concludes the proof.

##  Appendix B Derivation of spow

###  B.1 Tensor product and tensor power

A convenient way to define the tensor product is, given vectors x,y∈ℝd𝑥𝑦 ℝ𝑑x,y\in\mathbb{R}^{d}x , y ∈ blackboard_R d, their tensor product x⊗y=x⁢yT∈ℝd×dtensor-product𝑥𝑦𝑥 𝑦𝑇 ℝ𝑑𝑑x\otimes y=xy^{T}\in\mathbb{R}^{d\times d}x ⊗ y = x y T ∈ blackboard_R d × d. The generic tensor product of p𝑝pp vectors in ℝd ℝ𝑑\mathbb{R}^{d}blackboard_R d can be written as T=⨂k=1pxk∈ℝd×⋯×d𝑇 tensor-product𝑘1𝑝 𝑥𝑘 ℝ𝑑⋯𝑑T=\bigotimes_{k=1}^{p}x_{k}\in\mathbb{R}^{d\times\cdots\times d}T = ⨂ k = 1  p x k  ∈ blackboard_R d × ⋯ × d where, evaluated at a multi-index (i1⁢⋯⁢ip)∈ℕd×p 𝑖1⋯ 𝑖𝑝 ℕ𝑑absent𝑝(i_{1}\cdots i_{p})\in\mathbb{N}_{d}^{\times p}( i 1  ⋯ i p  ) ∈ blackboard_N d  × p, the tensor T𝑇TT has value Ti=∏kxk,ik 𝑇𝑖 product𝑘 𝑥𝑘 𝑖𝑘T_{i}=\prod_{k}x_{k,i_{k}}T i  = ∏ k  x k , i k  . (For example, if T=a⊗b⊗c𝑇tensor-product𝑎𝑏𝑐T=a\otimes b\otimes cT = a ⊗ b ⊗ c then T1,2,3=a1⁢b2⁢c3 𝑇123 𝑎1 𝑏2 𝑐3T_{1,2,3}=a_{1}b_{2}c_{3}T 1 , 2 , 3  = a 1  b 2  c 3 .)

In this work, a central focus is on the p𝑝ppth tensor power, defined as taking the tensor product of a vector with itself p𝑝pp times, which we denote using x⊗p 𝑥tensor-productabsent𝑝x^{\otimes p}x ⊗ p . We can define the helpful tpow⁢(x,p)=flat⁢(x⊗p)∈ℝdptpow𝑥𝑝flat 𝑥tensor-productabsent𝑝 ℝ 𝑑𝑝\textsc{tpow}(x,p)=\text{flat}\left(x^{\otimes p}\right)\in\mathbb{R}^{d^{p}}tpow ( x , p ) = flat ( x ⊗ p ) ∈ blackboard_R d p, which gives us the flattened tensor power as a vector.

 | tpow⁢(x,p)=[x1⁢⋯⁢x1x1⁢⋯⁢x2⋮xd⁢⋯⁢xd]=[⋮∏kxik⋮](i1,⋯,ip)∈ℕd×ptpow𝑥𝑝matrix 𝑥1⋯ 𝑥1 𝑥1⋯ 𝑥2⋮ 𝑥𝑑⋯ 𝑥𝑑 matrix⋮ product𝑘 𝑥 𝑖𝑘⋮ 𝑖1⋯ 𝑖𝑝 ℕ𝑑absent𝑝\displaystyle\textsc{tpow}(x,p)=\begin{bmatrix}x_{1}\cdots x_{1}\\\ x_{1}\cdots x_{2}\\\ \vdots\\\ x_{d}\cdots x_{d}\\\ \end{bmatrix}=\begin{bmatrix}\vdots\\\ \prod_{k}x_{i_{k}}\\\ \vdots\\\ \end{bmatrix}_{(i_{1},\cdots,i_{p})\in\mathbb{N}_{d}^{\times p}}tpow ( x , p ) = [ start_ARG start_ROW start_CELL x 1  ⋯ x 1  end_CELL end_ROW start_ROW start_CELL x 1  ⋯ x 2  end_CELL end_ROW start_ROW start_CELL ⋮ end_CELL end_ROW start_ROW start_CELL x d  ⋯ x d  end_CELL end_ROW end_ARG ] = [ start_ARG start_ROW start_CELL ⋮ end_CELL end_ROW start_ROW start_CELL ∏ k  x i k   end_CELL end_ROW start_ROW start_CELL ⋮ end_CELL end_ROW end_ARG ] ( i 1  , ⋯ , i p  ) ∈ blackboard_N d  × p  |  | (21)  
---|---|---|---  
  
The central property that makes tpow useful to us is:

 | tpow⁢(x,p)T⁢tpow⁢(y,p)tpow 𝑥𝑝𝑇tpow𝑦𝑝\displaystyle\textsc{tpow}(x,p)^{T}\textsc{tpow}(y,p)tpow ( x , p ) T tpow ( y , p ) | =∑(i1,⋯)∈ℕd×pxi1⁢⋯⁢xip⁢yi1⁢⋯⁢yipabsent  𝑖1⋯ ℕ𝑑absent𝑝 𝑥 𝑖1⋯ 𝑥 𝑖𝑝 𝑦 𝑖1⋯ 𝑦 𝑖𝑝\displaystyle=\sum_{(i_{1},\cdots)\in\mathbb{N}_{d}^{\times p}}x_{i_{1}}\cdots x% _{i_{p}}y_{i_{1}}\cdots y_{i_{p}}= ∑ ( i 1  , ⋯ ) ∈ blackboard_N d  × p  x i 1   ⋯ x i p   y i 1   ⋯ y i p   |  | (22)  
---|---|---|---|---  
|  | =∑i1∈ℕdxi1⁢yi1⁢∑i2∈ℕdxi2⁢yi2⁢⋯absent  𝑖1 ℕ𝑑 𝑥 𝑖1 𝑦 𝑖1  𝑖2 ℕ𝑑 𝑥 𝑖2 𝑦 𝑖2⋯\displaystyle=\sum_{i_{1}\in\mathbb{N}_{d}}x_{i_{1}}y_{i_{1}}\sum_{i_{2}\in% \mathbb{N}_{d}}x_{i_{2}}y_{i_{2}}\ \cdots= ∑ i 1  ∈ blackboard_N d   x i 1   y i 1   ∑ i 2  ∈ blackboard_N d   x i 2   y i 2   ⋯ |  | (23)  
|  | =(xT⁢y)pabsent  𝑥𝑇𝑦𝑝\displaystyle=(x^{T}y)^{p}= ( x T y ) p |  | (24)  
  
Thus, letting ϕ=tpowitalic-ϕtpow\phi=\textsc{tpow}ϕ = tpow we see that power attention can be expressed as a special case of linear attention, Yiattnpowp=Yiattnlintpow⁢(⋅,p) 𝑌𝑖 attnpow𝑝 𝑌𝑖 attnlintpow⋅𝑝Y_{i}^{\text{attn}_{\text{pow}}^{p}}=Y_{i}^{\text{attn}_{\text{lin}}^{\textsc{% tpow}(\cdot,p)}}Y i  attn pow  p = Y i  attn lin  tpow ( ⋅ , p ). Power attention therefore inherits all of the desirable properties of linear attention described in Section 2, including a constant-size state and parallelizable chunked form. tpow is a state expansion, mapping keys and queries into ℝdp ℝ 𝑑𝑝\mathbb{R}^{d^{p}}blackboard_R d p, and so power attention possesses a state of size dp⁢v 𝑑𝑝𝑣d^{p}vd p v.

###  B.2 Symmetric power

Here we prove that symmetric power spow is a mathematically equivalent state expansion function to tpow.

Recall from Lemma 4.2 that

 | spowp⁢(x)=[⋮p!histk⁢(i)!⁢∏kxik⋮]i∈N⁢D⁢M⁢Idp spow𝑝𝑥 matrix⋮𝑝 hist𝑘𝑖 product𝑘 𝑥 𝑖𝑘⋮𝑖𝑁𝐷𝑀 𝐼𝑑𝑝\displaystyle\textsc{spow}_{p}(x)=\begin{bmatrix}\vdots\\\ \sqrt{\frac{p!}{\text{hist}_{k}(i)!}}\prod_{k}x_{i_{k}}\\\ \vdots\end{bmatrix}_{i\in NDMI_{d}^{p}}spow p  ( x ) = [ start_ARG start_ROW start_CELL ⋮ end_CELL end_ROW start_ROW start_CELL square-root start_ARG divide start_ARG p ! end_ARG start_ARG hist k  ( i ) ! end_ARG end_ARG ∏ k  x i k   end_CELL end_ROW start_ROW start_CELL ⋮ end_CELL end_ROW end_ARG ] i ∈ N D M I d  p  |  | (25)  
---|---|---|---  
  
Where each i=(i1,…,ip)𝑖 𝑖1… 𝑖𝑝i=(i_{1},...,i_{p})i = ( i 1  , … , i p  ) is the set of non-decreasing-multi-indices that determines a given entry in the embedded vector. One can use a different set of multi-indices α=(α1,…,αd)𝛼 𝛼1… 𝛼𝑑\alpha=(\alpha_{1},...,\alpha_{d})α = ( α 1  , … , α d  ) to represent the same embedding, where

 | αj={1if ⁢∃k∈{1,2,…,p},s.t.⁢ik=j0otherwise 𝛼𝑗cases1formulae-sequenceif 𝑘12…𝑝s.t. 𝑖𝑘𝑗0otherwise\displaystyle\alpha_{j}=\begin{cases}1&\text{if }\exists k\in\\{1,2,...,p\\},% \text{s.t.}i_{k}=j\\\ 0&\text{otherwise}\end{cases}α j  = { start_ROW start_CELL 1 end_CELL start_CELL if ∃ k ∈ { 1 , 2 , … , p } , s.t. i k  = j end_CELL end_ROW start_ROW start_CELL 0 end_CELL start_CELL otherwise end_CELL end_ROW |  | (26)  
---|---|---|---  
  
in other words, we can include all the dimensions of x𝑥xx in each entry of the expanded vector, and mask out unnecessary dimension by raising them to a power of 0.

With this setup, let x,y∈ℝd,spowp⁢(x)∈ℝ(d+p−1p)formulae-sequence𝑥𝑦 ℝ𝑑 spow𝑝𝑥 ℝbinomial𝑑𝑝1𝑝x,y\in\mathbb{R}^{d},\textsc{spow}_{p}(x)\in\mathbb{R}^{\binom{d+p-1}{p}}x , y ∈ blackboard_R d , spow p  ( x ) ∈ blackboard_R ( FRACOP start_ARG d + p - 1 end_ARG start_ARG p end_ARG ) be the symmetric power embedding indexed by the multi-indixes α=(α1,…⁢αd)𝛼 𝛼1… 𝛼𝑑\mathbf{\alpha}=(\alpha_{1},...\alpha_{d})α = ( α 1  , … α d  ), satisfying ∑i=1dαi=p 𝑖1𝑑 𝛼𝑖𝑝\sum_{i=1}^{d}\alpha_{i}=p∑ i = 1  d α i  = p.

 | spowp⁢(x)α=p!α1!⁢⋯⁢αd!⁢x1α1⁢⋯⁢xdαd. spow𝑝 𝑥𝛼𝑝 𝛼1⋯ 𝛼𝑑 𝑥1 𝛼1⋯ 𝑥𝑑 𝛼𝑑\displaystyle\textsc{spow}_{p}(x)_{\mathbf{\alpha}}=\sqrt{\frac{p!}{\alpha_{1}% !\cdots\alpha_{d}!}}\;x_{1}^{\,\alpha_{1}}\cdots x_{d}^{\,\alpha_{d}}.spow p  ( x ) α  = square-root start_ARG divide start_ARG p ! end_ARG start_ARG α 1  ! ⋯ α d  ! end_ARG end_ARG x 1  α 1  ⋯ x d  α d  . |  | (27)  
---|---|---|---  
  
Then, by the multinomial theorem

 | ⟨spowp⁢(x),spowp⁢(y)⟩ spow𝑝𝑥 spow𝑝𝑦\displaystyle\bigl{\langle}\textsc{spow}_{p}(x),\,\textsc{spow}_{p}(y)\bigr{\rangle}⟨ spow p  ( x ) , spow p  ( y ) ⟩ | =∑αp!α1!⁢⋯⁢αd!⁢x1α1⁢⋯⁢xdαd⁢y1α1⁢⋯⁢ydαdabsent 𝛼𝑝 𝛼1⋯ 𝛼𝑑 𝑥1 𝛼1⋯ 𝑥𝑑 𝛼𝑑 𝑦1 𝛼1⋯ 𝑦𝑑 𝛼𝑑\displaystyle=\sum_{\mathbf{\alpha}}\frac{p!}{\alpha_{1}!\cdots\alpha_{d}!}\;x% _{1}^{\,\alpha_{1}}\cdots x_{d}^{\,\alpha_{d}}\;y_{1}^{\,\alpha_{1}}\cdots y_{% d}^{\,\alpha_{d}}= ∑ α  divide start_ARG p ! end_ARG start_ARG α 1  ! ⋯ α d  ! end_ARG x 1  α 1  ⋯ x d  α d  y 1  α 1  ⋯ y d  α d  |  | (28)  
---|---|---|---|---  
|  | =∑αp!α1!⁢⋯⁢αd!⁢(x1⁢y1)α1⁢⋯⁢(xd⁢yd)αdabsent 𝛼𝑝 𝛼1⋯ 𝛼𝑑 𝑥1 𝑦1 𝛼1⋯ 𝑥𝑑 𝑦𝑑 𝛼𝑑\displaystyle=\sum_{\mathbf{\alpha}}\frac{p!}{\alpha_{1}!\cdots\alpha_{d}!}(x_% {1}y_{1})^{\alpha_{1}}\cdots(x_{d}y_{d})^{\alpha_{d}}= ∑ α  divide start_ARG p ! end_ARG start_ARG α 1  ! ⋯ α d  ! end_ARG ( x 1  y 1  ) α 1  ⋯ ( x d  y d  ) α d  |  | (29)  
|  | =(x1⁢y1+⋯+xd⁢yd)p 𝑥1 𝑦1⋯ 𝑥𝑑 𝑦𝑑𝑝\displaystyle=\bigl{(}x_{1}y_{1}+\dots+x_{d}y_{d}\bigr{)}^{p}= ( x 1  y 1  + ⋯ + x d  y d  ) p |  | (30)  
|  | =(xT⁢y)pabsent  𝑥𝑇𝑦𝑝\displaystyle=(x^{T}y)^{p}= ( x T y ) p |  | (31)  
  
Therefore

 | ⟨spowp⁢(x),spowp⁢(y)⟩=(xT⁢y)p=⟨tpowp⁢(x),tpowp⁢(y)⟩ spow𝑝𝑥 spow𝑝𝑦  𝑥𝑇𝑦𝑝 tpow𝑝𝑥 tpow𝑝𝑦\displaystyle\langle\textsc{spow}_{p}(x),\textsc{spow}_{p}(y)\rangle=(x^{T}y)^% {p}=\langle\textsc{tpow}_{p}(x),\textsc{tpow}_{p}(y)\rangle⟨ spow p  ( x ) , spow p  ( y ) ⟩ = ( x T y ) p = ⟨ tpow p  ( x ) , tpow p  ( y ) ⟩ |  | (32)  
---|---|---|---  
  
##  Appendix C LongCrawl64

(a) Openwebtext document length distribution. (b) GPT-2 trained with 8192 context. Figure 10:  LongCrawl64 [Buckman, 2024] consists of 6,661,465 pre-tokenized documents, each of which is 65,536 tokens long, for a total token count of 435 billion. The data is sourced from the Common Crawl, a typical source for language modeling datasets, but pretokenized and filtered down to only include long sequences. This is a prerequisite for in-context learning. In Figure 10, it is clear that there is little potential for in-context learning beyond the lengths of the majority of documents in the dataset. The figure above and the discussion on the need for long documents were originally presented by Buckman and Gelada [b].

##  Appendix D Experimental details

Our experiments were implemented in PyTorch [Paszke et al., 2019], based around the FLA codebase [Yang and Zhang, 2024] whose implementations of all architectures we use. Since this work is focused specifically on the attention layer, we typically separate out the architecture (which we consider everything except the attention) from the attention itself. Also, since comparing loss between models of different context lengths can be nuanced, we report the best-context loss as described by Buckman and Gelada [b] whenever plotting a scalar-valued loss for a training curve. When selecting the best context, and also when plotting in-context learning curves, we smooth across the sequence dimension by binning, using exponentially-growing bins so that the bin widths are equal on a log plot.

Unless otherwise noted, we used the following hyperparameters for training: LongCrawl64 (train set for training and report losses on the heldout set), batch size 32, context length 1024, learning rate 3e-4 with a 2000 step warmup from 0 and a cosine decay over the full range of training to 1e-5, one epoch of training (or until convergence), AdamW with weight decay of .1, beta1 of .9, beta2 of .999, gradient clipping of 1, bf16 training, and activation checkpointing for memory reduction. We use a common set of model sizes, taken from Radford et al. [2019]. Small models have width 768, 12 hidden layers, 12 heads, and a MLP ratio of 4. Medium models have width 1024, 24 hidden layers, 16 heads, and a MLP ratio of 4. Large models have width 1280, 36 hidden layers, 20 heads, and a MLP ratio of 4. For experiments involving the RWKV architecture, we use RWKV7.

Figure 1 and Figure 3 contain a variety of architectures, labeled in the legend; all are small models with default hyperparameters. Figure 4 contains two small RWKV models with default hyperparameters and differing attention layers.

In Figure 2, all three curves are RWKV + attention models. The model with 1:12 WSFR has 6 heads, 8 layers, head size of 64, hidden dimension of 512, a context of 65536, and a batch size of 32. The model with 4:1 WSFR is a medium model with batch size 512 and context size 4096. The model with 99:1 WSFR has 26 layers, 20 heads, head size of 64, hidden dimension of 1280, batch size 32768, and context size 64.

Figure 7 and Table 4 contain the resuls of experiments run using the RWKV architeture at large (750M) size. The transformer was run using a batch size of 7680 and context of 1024. The other runs used batch size 1920 and context size 4096.

Figure 8(a) consists of one large p=2𝑝2p=2p = 2 transformer. Figure 8(b) consists of small p=2𝑝2p=2p = 2 transformer models evaluated at iteration 64k. Figure 8(c) consists of large p=2𝑝2p=2p = 2 transformers evaluated at iteration 64k. The curve corresponding to the smallest model scale has 512 width, 6 layers, and 8 heads; the other three are small, medium, and large models. Figure 8(d) involves a sweep over context lengths on a p=2𝑝2p=2p = 2 transformer with 512 width, 6 layers, and 8 heads, evaluated at 170k iterations. The grey curves are the same, but after sampling documents (of any length), the data is reshaped to have a block size of 1024 before training. This means the actual tokens are kept constant, but the model has no opportunity to use long context to learn.

Figure 9(a) shows three small RWKV models with different attention layers, trained on context length of 65536 tokens at a batch size of 32.

All experiments in this work were run on Nvidia H100 GPUs, typically on nodes of 8 GPUs. In the course of conducting the experimental portion of this work, we had access to between 32 and 300 H100s for two months. The majority of the compute was spent on research, with only about 20% of compute spent on experiments in this paper.

##  Appendix E Power attention is compute-optimal under inference latency constraints

To a first-order approximation, inference latency is proportional to the sum of the parameter count and the state size. This is relevant when choosing an optimal training strategy for a model whose ultimate usage will have inference constraints, for example a speech model model [Wang et al., 2021]. A reasonable approach is to choose the parameter count and state size to be at some tolerable scale, and spend the compute budget scaling other axes of training, such as batch size and context length.

| Best context length | Loss  
---|---|---  
Window (1k) | 1878 | 1.638  
Attention | 1024 | 1.631  
Power (p=2𝑝2p=2p = 2) | 4096 | 1.613  
Table 4: Power attention is compute-optimal given sufficient train FLOPs, when inference latency is equal. In this setting, we can compare the best achievable performance of a transformer to that of other models, keeping all variants equivalent in terms of state size, parameter count, tokens per update, and total FLOPs. In Table 4, we construct three such models, and compare the final best-context loss. The transformer is trained on batches of 7680 document of length 1024 (a context length which keeps its state size equivalent to the other approaches), while the windowed transformer and power attention transformer use batches of 1920 documents of length 4096. Power attention is the only architecture to outperform the transformer. See Appendix D for experimental details.

##  Appendix F Algorithms in power attention

In this section we present the four algorithms used in power attention.

###  F.1 Attention

As shown below, the attention kernel in Power Attention is very similar to Flash Attention, apart from an extra step of log-space power and an extra output for the normalization term l𝑙ll. We chose to raise the attention score matrix to a power p𝑝pp in the log space because it is more numeric stable than performing the power operation directly.

Step | Flash Attention | Power Attention (attention form)  
---|---|---  
1\. Query-Key Inner Product | S=Q⁢KT𝑆𝑄 𝐾𝑇S=QK^{T}S = Q K T | S=Q⁢KT𝑆𝑄 𝐾𝑇S=QK^{T}S = Q K T  
2\. Softmax Scaling | S=S⊙scale𝑆direct-product𝑆scaleS=S\odot\text{scale}S = S ⊙ scale | S=S⊙scale𝑆direct-product𝑆scaleS=S\odot\text{scale}S = S ⊙ scale  
3\. Log-space Power |  | S=p⁢log⁡(|S|+ϵ)𝑆𝑝𝑆italic-ϵS=p\log(|S|+\epsilon)S = p roman_log ( | S | + ϵ )  
4\. Row max scaling | S=S−rowmax⁢(S)𝑆𝑆rowmax𝑆S=S-\text{rowmax}(S)S = S - rowmax ( S ) | S=S−rowmax⁢(S)𝑆𝑆rowmax𝑆S=S-\text{rowmax}(S)S = S - rowmax ( S )  
5\. Masked Exponential | P=exp⁢(S⊙M)𝑃expdirect-product𝑆𝑀P=\text{exp}(S\odot M)P = exp ( S ⊙ M ) | P=exp⁢(S⊙M)𝑃expdirect-product𝑆𝑀P=\text{exp}(S\odot M)P = exp ( S ⊙ M )  
6\. Normalization | P=P⊙D−1⁢(rowsum⁢(P))𝑃direct-product𝑃 𝐷1rowsum𝑃P=P\odot D^{-1}(\text{rowsum}(P))P = P ⊙ D - 1 ( rowsum ( P ) ) | ζ=rowsum⁢(P)𝜁rowsum𝑃\zeta=\text{rowsum}(P)ζ = rowsum ( P )  
7\. Matmul with Value | O=P⁢V𝑂𝑃𝑉O=PVO = P V | O=P⁢V𝑂𝑃𝑉O=PVO = P V  
8\. Output | O𝑂OO | O,ζ𝑂𝜁O,\zetaO , ζ  
Table 5: Procedural comparison between Flash Attention and Power Attention (attention form). Q,K∈Rt×d𝑄𝐾 𝑅𝑡𝑑Q,K\in R^{t\times d}Q , K ∈ R t × d, V∈Rt×v𝑉 𝑅𝑡𝑣V\in R^{t\times v}V ∈ R t × v; p𝑝pp stands for the degree of power; ϵitalic-ϵ\epsilonϵ is a small constant to avoid taking the log of zero; rowmax(P) refers to the operation of taking the max of the t×t𝑡𝑡t\times tt × t attention score matrix, an often-used techniques for stabilizing softmax [Milakov and Gimelshein, 2018]; rowsum⁢(P)rowsum𝑃\text{rowsum}(P)rowsum ( P ) refers to the operation of summing up each row of the softmax matrix; D−1 𝐷1D^{-1}D - 1 refers to the operation of converting a vector into a diagonal matrix and take its inverse; ζ𝜁\zetaζ is the normalization term (sum of attention scores) used for combining attention output and query-state output ###  F.2 Update state

The update-state operation concerns with creating a new state Si+1∈RD×v 𝑆𝑖1 𝑅𝐷𝑣S_{i+1}\in R^{D\times v}S i + 1  ∈ R D × v based on the past state Si∈RD×v 𝑆𝑖 𝑅𝐷𝑣S_{i}\in R^{D\times v}S i  ∈ R D × v and all the keys Ki∈Rc×d 𝐾𝑖 𝑅𝑐𝑑K_{i}\in R^{c\times d}K i  ∈ R c × d and values Vi∈Rc×v 𝑉𝑖 𝑅𝑐𝑣V_{i}\in R^{c\times v}V i  ∈ R c × v in the current chunk. There are many variants to this formulation in modern RNNs. Specifically, past state Si 𝑆𝑖S_{i}S i  are usually gated with a decay factor γi 𝛾𝑖\gamma_{i}γ i , which often depend on input as well.

 | update-state⁢(Si,Ki,Vi)update-state 𝑆𝑖 𝐾𝑖 𝑉𝑖\displaystyle\text{update-state}(S_{i},K_{i},V_{i})update-state ( S i  , K i  , V i  ) | =Si+ϕ⁢(Ki)T⁢Viabsent 𝑆𝑖italic-ϕ 𝐾𝑖𝑇 𝑉𝑖\displaystyle=S_{i}+\phi(K_{i})^{T}V_{i}= S i  + ϕ ( K i  ) T V i  |  | (33)  
---|---|---|---|---  
| gated-update-state⁢(Si,Ki,Vi)gated-update-state 𝑆𝑖 𝐾𝑖 𝑉𝑖\displaystyle\text{gated-update-state}(S_{i},K_{i},V_{i})gated-update-state ( S i  , K i  , V i  ) | =Si⊙γi+ϕ⁢(Ki)T⁢Viabsentdirect-product 𝑆𝑖 𝛾𝑖italic-ϕ 𝐾𝑖𝑇 𝑉𝑖\displaystyle=S_{i}\odot\gamma_{i}+\phi(K_{i})^{T}V_{i}= S i  ⊙ γ i  + ϕ ( K i  ) T V i  |  | (34)  
  
Regardless the exact state evolution formula, the fusion of state expansion and a subsequent matrix multiplication is the fundamental building block, which we termed fused spow-mma (expand M) kernel. We use expand M here as M, N, K are commonly used to denote the 3 dimensions of a matrix multiplication problem, and this kernel is expanding the state along M axis (KiT∈Rd×c→ϕ⁢(Ki)T∈RD×c 𝐾𝑖𝑇 𝑅𝑑𝑐→italic-ϕ 𝐾𝑖𝑇 𝑅𝐷𝑐K_{i}^{T}\in R^{d\times c}\rightarrow\phi(K_{i})^{T}\in R^{D\times c}K i  T ∈ R d × c → ϕ ( K i  ) T ∈ R D × c). We might also use the term update-state interchangeably with fused spow-mma (expand M) kernel, as the gated summation is done in the discumsum kernel. Note that in practice, the update-state kernel would also produce a normalization term (a.k.a. sum of expanded keys) γ𝛾\gammaγ, which is used to combine the outputs of chunked attention and query-state such that the output is normalized. We denote this kernel with fused-update-state.

 | fused-update-state⁢(Si,Ki,Vi)fused-update-state 𝑆𝑖 𝐾𝑖 𝑉𝑖\displaystyle\text{fused-update-state}(S_{i},K_{i},V_{i})fused-update-state ( S i  , K i  , V i  ) | =(ϕ⁢(Ki)T⁢Vi,ϕ⁢(Ki)T⁢𝟏)absentitalic-ϕ 𝐾𝑖𝑇 𝑉𝑖italic-ϕ 𝐾𝑖𝑇1\displaystyle=(\phi(K_{i})^{T}V_{i},\phi(K_{i})^{T}\mathbf{1})= ( ϕ ( K i  ) T V i  , ϕ ( K i  ) T bold_1 ) |  | (35)  
---|---|---|---|---  
  
Algorithm 1 Fused Update State

1:Matrices A𝐴AA of size 𝐝×𝐊𝐝𝐊\mathbf{d}\times\mathbf{K}bold_d × bold_K, B𝐵BB of size 𝐊×𝐍𝐊𝐍\mathbf{K}\times\mathbf{N}bold_K × bold_N

2:Output matrix C𝐶CC of size 𝐃×𝐍𝐃𝐍\mathbf{D}\times\mathbf{N}bold_D × bold_N, normalization factor γ𝛾\gammaγ of size 𝐃𝐃\mathbf{D}bold_D

3:Define degree of power 𝐩𝐩\mathbf{p}bold_p, tile size for expansion 𝐝𝐭𝐢𝐥𝐞 𝐝𝐭𝐢𝐥𝐞\mathbf{d_{tile}}bold_d bold_tile , expanded tile size 𝐃𝐭𝐢𝐥𝐞=𝐝𝐭𝐢𝐥𝐞𝐩 𝐃𝐭𝐢𝐥𝐞 𝐝𝐭𝐢𝐥𝐞𝐩\mathbf{D_{tile}}=\mathbf{d_{tile}}^{\mathbf{p}}bold_D bold_tile  = bold_d bold_tile  bold_p

4:Denote the ordered list of non-decreasing-multi-indices 𝐍𝐃𝐌𝐈𝐝/𝐝𝐭𝐢𝐥𝐞𝐩  𝐍𝐃𝐌𝐈𝐩𝐝 𝐝𝐭𝐢𝐥𝐞\mathbf{NDMI}^{\mathbf{p}}_{\mathbf{d}/\mathbf{d_{tile}}}bold_NDMI bold_p bold_d / bold_d bold_tile   with λ𝜆\mathbf{\lambda}λ, of size 𝐋×𝐩𝐋𝐩\mathbf{L}\times\mathbf{p}bold_L × bold_p

5:Divide A into NA=⌈𝐊𝐓𝐊⌉ 𝑁𝐴𝐊𝐓𝐊N_{A}=\lceil\mathbf{\frac{K}{TK}}\rceilN A  = ⌈ divide start_ARG bold_K end_ARG start_ARG bold_TK end_ARG ⌉ tiles, A1,…,ANA 𝐴1… 𝐴 𝑁𝐴A_{1},...,A_{N_{A}}A 1  , … , A N A  , each of size 𝐝×𝐓𝐊𝐝𝐓𝐊\mathbf{d}\times\mathbf{TK}bold_d × bold_TK; divide each Ak 𝐴𝑘A_{k}A k  further into 𝐍𝐝=𝐝𝐝𝐭𝐢𝐥𝐞 𝐍𝐝𝐝 𝐝𝐭𝐢𝐥𝐞\mathbf{N_{d}=\frac{d}{d_{tile}}}bold_N bold_d  = divide start_ARG bold_d end_ARG start_ARG bold_d bold_tile  end_ARG subtiles, Ak1,…,Ak𝐝/𝐝𝐭𝐢𝐥𝐞 𝐴𝑘1… 𝐴𝑘𝐝 𝐝𝐭𝐢𝐥𝐞A_{k}^{1},...,A_{k}^{\mathbf{d/d_{tile}}}A k  1 , … , A k  bold_d / bold_d bold_tile , each of size 𝐝𝐭𝐢𝐥𝐞×𝐓𝐊 𝐝𝐭𝐢𝐥𝐞𝐓𝐊\mathbf{d_{tile}\times TK}bold_d bold_tile  × bold_TK

6:Divide B into NB=⌈𝐍𝐓𝐍⌉ 𝑁𝐵𝐍𝐓𝐍N_{B}=\lceil\mathbf{\frac{N}{TN}}\rceilN B  = ⌈ divide start_ARG bold_N end_ARG start_ARG bold_TN end_ARG ⌉ tiles, B1,…,BNB 𝐵1… 𝐵 𝑁𝐵B_{1},...,B_{N_{B}}B 1  , … , B N B  , each of size 𝐊×𝐓𝐍𝐊𝐓𝐍\mathbf{K}\times\mathbf{TN}bold_K × bold_TN; divide each Bj 𝐵𝑗B_{j}B j  further into ⌈𝐊𝐓𝐊⌉𝐊𝐓𝐊\lceil\mathbf{\frac{K}{TK}}\rceil⌈ divide start_ARG bold_K end_ARG start_ARG bold_TK end_ARG ⌉ subtitles, Bj1,…,Bj⌈𝐊/𝐓𝐊⌉ 𝐵𝑗1… 𝐵𝑗𝐊𝐓𝐊B_{j}^{1},...,B_{j}^{\lceil\mathbf{K/TK}\rceil}B j  1 , … , B j  ⌈ bold_K / bold_TK ⌉, each of size 𝐓𝐊×𝐓𝐍𝐓𝐊𝐓𝐍\mathbf{TK}\times\mathbf{TN}bold_TK × bold_TN

7:for  1≤l≤𝐋1𝑙𝐋1\leq l\leq\mathbf{L}1 ≤ l ≤ bold_L, in parallel  do

8:     for  1≤j≤NB1𝑗 𝑁𝐵1\leq j\leq N_{B}1 ≤ j ≤ N B , in parallel  do

9:         Initialize accumulation registers: Cl,j←0← 𝐶𝑙𝑗0C_{l,j}\leftarrow 0C l , j  ← 0 of shape 𝐃𝐭𝐢𝐥𝐞×𝐓𝐍 𝐃𝐭𝐢𝐥𝐞𝐓𝐍\mathbf{D_{tile}\times TN}bold_D bold_tile  × bold_TN, D

10:         Initialize register for matrix multiplication A^k←0← ^𝐴𝑘0\hat{A}_{k}\leftarrow 0over^ start_ARG A end_ARG k  ← 0 of shape 𝐃𝐭𝐢𝐥𝐞×𝐓𝐊 𝐃𝐭𝐢𝐥𝐞𝐓𝐊\mathbf{D_{tile}}\times\mathbf{TK}bold_D bold_tile  × bold_TK

11:         Initialize register for normalization factor: γl 𝛾𝑙\gamma_{l}γ l  of shape 𝐃𝐭𝐢𝐥𝐞 𝐃𝐭𝐢𝐥𝐞\mathbf{D_{tile}}bold_D bold_tile 

12:         for 1≤k≤⌈𝐊𝐓𝐊⌉1𝑘𝐊𝐓𝐊1\leq k\leq\lceil\mathbf{\frac{K}{TK}}\rceil1 ≤ k ≤ ⌈ divide start_ARG bold_K end_ARG start_ARG bold_TK end_ARG ⌉ do

13:              Load Ak 𝐴𝑘A_{k}A k  from global memory to on-chip SRAM

14:              Load Bjk 𝐵𝑗𝑘B_{j}^{k}B j  k from global memory to on-chip SRAM

15:              for 1≤z≤𝐩1𝑧𝐩1\leq z\leq\mathbf{p}1 ≤ z ≤ bold_p do

16:                  Load Akλ⁢(l,z) 𝐴𝑘𝜆𝑙𝑧A_{k}^{\mathbf{\lambda}(l,z)}A k  λ ( l , z ) from on-chip SRAM into registers

17:              end for

18:              Ak^←Akλ⁢(l,1)⊗⋯⊗Akλ⁢(l,𝐩)←^ 𝐴𝑘tensor-product 𝐴𝑘𝜆𝑙1⋯ 𝐴𝑘𝜆𝑙𝐩\hat{A_{k}}\leftarrow A_{k}^{\mathbf{\lambda}(l,1)}\otimes\cdots\otimes A_{k}^% {\mathbf{\lambda}(l,\mathbf{p})}over^ start_ARG A k  end_ARG ← A k  λ ( l , 1 ) ⊗ ⋯ ⊗ A k  λ ( l , bold_p )

19:              γl←rowsum⁢(Ak^)+γl← 𝛾𝑙rowsum^ 𝐴𝑘 𝛾𝑙\gamma_{l}\leftarrow\text{rowsum}(\hat{A^{k}})+\gamma_{l}γ l  ← rowsum ( over^ start_ARG A k end_ARG ) + γ l 

20:              Cl,j←Ak^⁢Bjk+Cl,j← 𝐶𝑙𝑗^ 𝐴𝑘 𝐵𝑗𝑘 𝐶𝑙𝑗C_{l,j}\leftarrow\hat{A_{k}}B_{j}^{k}+C_{l,j}C l , j  ← over^ start_ARG A k  end_ARG B j  k + C l , j 

21:         end for

22:         Write Cl,j,γl 𝐶𝑙𝑗 𝛾𝑙C_{l,j},\gamma_{l}C l , j  , γ l  to global memory

23:     end for

24:end for

###  F.3 Discumsum

The discumsum operation involves discounting and accumulative-summing states S∈Rn×D×d𝑆 𝑅𝑛𝐷𝑑S\in R^{n\times D\times d}S ∈ R n × D × d for each chunk in a sequence (hence the name), where n=⌈tc⌉𝑛𝑡𝑐n=\lceil\frac{t}{c}\rceiln = ⌈ divide start_ARG t end_ARG start_ARG c end_ARG ⌉. It takes the output produced by update-state kernel, and a gating factor λ∈Rn𝜆 𝑅𝑛\lambda\in R^{n}λ ∈ R n and produced the discounted accumulative sum. Discounting is necessary when gating is involved. The discumsum kernel used in in paper was implemented by a custom CUDA kernel.

| discumsum⁢(S,λ)discumsum𝑆𝜆\displaystyle\text{discumsum}(S,\lambda)discumsum ( S , λ ) | =[S1S1⊙λ1+S2⋯S1⊙λ1+⋯+Si⊙∏j=1iλj+Sn]absentmatrix 𝑆1direct-product 𝑆1 𝜆1 𝑆2⋯direct-product 𝑆1 𝜆1⋯direct-product 𝑆𝑖 product𝑗1𝑖 𝜆𝑗 𝑆𝑛\displaystyle=\begin{bmatrix}S_{1}\\\ S_{1}\odot\lambda_{1}+S_{2}\\\ \cdots\\\ S_{1}\odot\lambda_{1}+\cdots+S_{i}\odot\prod_{j=1}^{i}\lambda_{j}+S_{n}\end{bmatrix}= [ start_ARG start_ROW start_CELL S 1  end_CELL end_ROW start_ROW start_CELL S 1  ⊙ λ 1  + S 2  end_CELL end_ROW start_ROW start_CELL ⋯ end_CELL end_ROW start_ROW start_CELL S 1  ⊙ λ 1  + ⋯ + S i  ⊙ ∏ j = 1  i λ j  + S n  end_CELL end_ROW end_ARG ] |  | (36)  
---|---|---|---|---  
  
###  F.4 Query state

The query-state kernel involves querying the past state Si 𝑆𝑖S_{i}S i  using the queries Qi∈Rc×d 𝑄𝑖 𝑅𝑐𝑑Q_{i}\in R^{c\times d}Q i  ∈ R c × d in the current chunk.

 | query-state⁢(Si,Qi)=ϕ⁢(Q)⁢Siquery-state 𝑆𝑖 𝑄𝑖italic-ϕ𝑄 𝑆𝑖\displaystyle\text{query-state}(S_{i},Q_{i})=\phi(Q)S_{i}query-state ( S i  , Q i  ) = ϕ ( Q ) S i  |  | (37)  
---|---|---|---  
  
Notice that as opposed to the update-state kernel, the query-state kernel expands the queries along the dimension of reduction in matrix multiplication. Therefore in the inner loop of the kernel, we go through all the nondecreasing-multi-indices.

In practice, we also fuse the summation of the intra-chunk output from attention Y∈Rc×v𝑌 𝑅𝑐𝑣Y\in R^{c\times v}Y ∈ R c × v and ϕ⁢(Q)⁢Siitalic-ϕ𝑄 𝑆𝑖\phi(Q)S_{i}ϕ ( Q ) S i  into the query-state kernel itself. We also chose to fuse the normalization into it. The algorithm for fused-query-state is shown below.

 | fused-query-state⁢(Si,Qi,Yi,ζi,γi)=Yi+ϕ⁢(Qi)⁢Siζi+ϕ⁢(Qi)⁢γifused-query-state 𝑆𝑖 𝑄𝑖 𝑌𝑖 𝜁𝑖 𝛾𝑖 𝑌𝑖italic-ϕ 𝑄𝑖 𝑆𝑖 𝜁𝑖italic-ϕ 𝑄𝑖 𝛾𝑖\displaystyle\text{fused-query-state}(S_{i},Q_{i},Y_{i},\zeta_{i},\gamma_{i})=% \frac{Y_{i}+\phi(Q_{i})S_{i}}{\zeta_{i}+\phi(Q_{i})\gamma_{i}}fused-query-state ( S i  , Q i  , Y i  , ζ i  , γ i  ) = divide start_ARG Y i  + ϕ ( Q i  ) S i  end_ARG start_ARG ζ i  + ϕ ( Q i  ) γ i  end_ARG |  | (38)  
---|---|---|---  
  
Algorithm 2 Fused Query State

1:Matrices A𝐴AA of size 𝐌×𝐝𝐌𝐝\mathbf{M}\times\mathbf{d}bold_M × bold_d, B𝐵BB of size 𝐃×𝐍𝐃𝐍\mathbf{D}\times\mathbf{N}bold_D × bold_N, Y𝑌YY of size 𝐌×𝐍𝐌𝐍\mathbf{M}\times\mathbf{N}bold_M × bold_N, γ𝛾\gammaγ of size 𝐃𝐃\mathbf{D}bold_D, ζ𝜁\zetaζ of size 𝐌𝐌\mathbf{M}bold_M

2:Output matrix C𝐶CC of size 𝐌×𝐍𝐌𝐍\mathbf{M}\times\mathbf{N}bold_M × bold_N

3:Define degree of power 𝐩𝐩\mathbf{p}bold_p, tile size for expansion 𝐝𝐭𝐢𝐥𝐞 𝐝𝐭𝐢𝐥𝐞\mathbf{d_{tile}}bold_d bold_tile , expanded tile size 𝐃𝐭𝐢𝐥𝐞=𝐝𝐭𝐢𝐥𝐞𝐩 𝐃𝐭𝐢𝐥𝐞 𝐝𝐭𝐢𝐥𝐞𝐩\mathbf{D_{tile}}=\mathbf{d_{tile}}^{\mathbf{p}}bold_D bold_tile  = bold_d bold_tile  bold_p

4:Denote the ordered list of non-decreasing-multi-indices 𝐍𝐃𝐌𝐈𝐝/𝐝𝐭𝐢𝐥𝐞𝐩  𝐍𝐃𝐌𝐈𝐩𝐝 𝐝𝐭𝐢𝐥𝐞\mathbf{NDMI}^{\mathbf{p}}_{\mathbf{d}/\mathbf{d_{tile}}}bold_NDMI bold_p bold_d / bold_d bold_tile   with λ𝜆\mathbf{\lambda}λ, of size 𝐋×𝐩𝐋𝐩\mathbf{L}\times\mathbf{p}bold_L × bold_p

5:Divide A into NA=⌈MT⁢M⌉ 𝑁𝐴𝑀𝑇𝑀N_{A}=\lceil\frac{M}{TM}\rceilN A  = ⌈ divide start_ARG M end_ARG start_ARG T M end_ARG ⌉ tiles, A1,…,ANA 𝐴1… 𝐴 𝑁𝐴A_{1},...,A_{N_{A}}A 1  , … , A N A  , each of size 𝐓𝐌×𝐝𝐓𝐌𝐝\mathbf{TM}\times\mathbf{d}bold_TM × bold_d; divide each Ai 𝐴𝑖A_{i}A i  further into Nd=𝐝𝐝𝐭𝐢𝐥𝐞 𝑁𝑑𝐝 𝐝𝐭𝐢𝐥𝐞N_{d}=\frac{\mathbf{d}}{\mathbf{d_{tile}}}N d  = divide start_ARG bold_d end_ARG start_ARG bold_d bold_tile  end_ARG subtitles, Ai1,…,AiNd 𝐴𝑖1… 𝐴𝑖 𝑁𝑑A_{i}^{1},...,A_{i}^{N_{d}}A i  1 , … , A i  N d , each of size 𝐓𝐌×𝐝𝐭𝐢𝐥𝐞𝐓𝐌 𝐝𝐭𝐢𝐥𝐞\mathbf{TM}\times\mathbf{d_{tile}}bold_TM × bold_d bold_tile 

6:Divide B into NB=⌈NT⁢N⌉ 𝑁𝐵𝑁𝑇𝑁N_{B}=\lceil\frac{N}{TN}\rceilN B  = ⌈ divide start_ARG N end_ARG start_ARG T N end_ARG ⌉ tiles, B1,…,BNB 𝐵1… 𝐵 𝑁𝐵B_{1},...,B_{N_{B}}B 1  , … , B N B  , each of size 𝐃×𝐓𝐍𝐃𝐓𝐍\mathbf{D}\times\mathbf{TN}bold_D × bold_TN; divide each Bi 𝐵𝑖B_{i}B i  further into 𝐋𝐋\mathbf{L}bold_L subtitles, Bi1,…,Bi𝐋 𝐵𝑖1… 𝐵𝑖𝐋B_{i}^{1},...,B_{i}^{\mathbf{L}}B i  1 , … , B i  bold_L, each of size 𝐃𝐭𝐢𝐥𝐞×𝐓𝐍 𝐃𝐭𝐢𝐥𝐞𝐓𝐍\mathbf{D_{tile}}\times\mathbf{TN}bold_D bold_tile  × bold_TN

7:Divide Y𝑌YY into NA 𝑁𝐴N_{A}N A  tiles, Y1,…⁢YNA 𝑌1… 𝑌 𝑁𝐴Y_{1},...Y_{N_{A}}Y 1  , … Y N A  , each of size 𝐓𝐌×𝐍𝐓𝐌𝐍\mathbf{TM}\times\mathbf{N}bold_TM × bold_N; divide each Yi 𝑌𝑖Y_{i}Y i  further into NB 𝑁𝐵N_{B}N B  subtiles, Yi1,…,YiNB 𝑌𝑖1… 𝑌𝑖 𝑁𝐵Y_{i}^{1},...,Y_{i}^{N_{B}}Y i  1 , … , Y i  N B , each of size 𝐓𝐌×𝐓𝐍𝐓𝐌𝐓𝐍\mathbf{TM}\times\mathbf{TN}bold_TM × bold_TN

8:Divide γ𝛾\gammaγ into 𝐋𝐋\mathbf{L}bold_L tiles, γ1,…⁢γL 𝛾1… 𝛾𝐿\gamma_{1},...\gamma_{L}γ 1  , … γ L , each of size 𝐃𝐭𝐢𝐥𝐞 𝐃𝐭𝐢𝐥𝐞\mathbf{D_{tile}}bold_D bold_tile ; divide ζ𝜁\zetaζ into NA 𝑁𝐴N_{A}N A  tiles, ζ1,…,ζNA 𝜁1… 𝜁 𝑁𝐴\zeta_{1},...,\zeta_{N_{A}}ζ 1  , … , ζ N A  , each of size 𝐓𝐌𝐓𝐌\mathbf{TM}bold_TM

9:for  1≤i≤NA1𝑖 𝑁𝐴1\leq i\leq N_{A}1 ≤ i ≤ N A , in parallel  do

10:     for  1≤j≤NB1𝑗 𝑁𝐵1\leq j\leq N_{B}1 ≤ j ≤ N B , in parallel  do

11:         Initialize accumulation registers: Ci,j←0← 𝐶𝑖𝑗0C_{i,j}\leftarrow 0C i , j  ← 0 of shape 𝐓𝐌×𝐓𝐍𝐓𝐌𝐓𝐍\mathbf{TM\times TN}bold_TM × bold_TN

12:         Initialize register for matrix multiplication A^i←0← ^𝐴𝑖0\hat{A}_{i}\leftarrow 0over^ start_ARG A end_ARG i  ← 0, of shape 𝐓𝐌×𝐃𝐭𝐢𝐥𝐞𝐓𝐌 𝐃𝐭𝐢𝐥𝐞\mathbf{TM}\times\mathbf{D_{tile}}bold_TM × bold_D bold_tile 

13:         Initialize register for normalization s←0←𝑠0s\leftarrow 0s ← 0, of shape 𝐓𝐌𝐓𝐌\mathbf{TM}bold_TM

14:         Load Ai 𝐴𝑖A_{i}A i  from global memory to on-chip SRAM

15:         for 1≤l≤𝐋1𝑙𝐋1\leq l\leq\mathbf{L}1 ≤ l ≤ bold_L do

16:              Load Bjl,γl 𝐵𝑗𝑙 𝛾𝑙B_{j}^{l},\gamma_{l}B j  l , γ l  from global memory to on-chip SRAM

17:              for 1≤z≤𝐩1𝑧𝐩1\leq z\leq\mathbf{p}1 ≤ z ≤ bold_p do

18:                  Load Aiλ⁢(l,z) 𝐴𝑖𝜆𝑙𝑧A_{i}^{\mathbf{\lambda}(l,z)}A i  λ ( l , z ) from on-chip SRAM into registers

19:              end for

20:              Ai^←Aiλ⁢(l,1)⊗⋯⊗Aiλ⁢(l,𝐩)←^ 𝐴𝑖tensor-product 𝐴𝑖𝜆𝑙1⋯ 𝐴𝑖𝜆𝑙𝐩\hat{A_{i}}\leftarrow A_{i}^{\mathbf{\lambda}(l,1)}\otimes\cdots\otimes A_{i}^% {\mathbf{\lambda}(l,\mathbf{p})}over^ start_ARG A i  end_ARG ← A i  λ ( l , 1 ) ⊗ ⋯ ⊗ A i  λ ( l , bold_p )

21:              s←Ai^⁢λl+s←𝑠^ 𝐴𝑖 𝜆𝑙𝑠s\leftarrow\hat{A_{i}}\lambda_{l}+ss ← over^ start_ARG A i  end_ARG λ l  + s

22:              Ci,j←Ai^⁢Bjl+Ci,j← 𝐶𝑖𝑗^ 𝐴𝑖 𝐵𝑗𝑙 𝐶𝑖𝑗C_{i,j}\leftarrow\hat{A_{i}}B_{j}^{l}+C_{i,j}C i , j  ← over^ start_ARG A i  end_ARG B j  l + C i , j 

23:         end for

24:         Load Yij,ζi 𝑌𝑖𝑗 𝜁𝑖Y_{i}^{j},\zeta_{i}Y i  j , ζ i  from global memory to on-chip SRAM

25:         Ci,j←Yij+Ci,jζi+s← 𝐶𝑖𝑗 𝑌𝑖𝑗 𝐶𝑖𝑗 𝜁𝑖𝑠C_{i,j}\leftarrow\frac{Y_{i}^{j}+C_{i,j}}{\zeta_{i}+s}C i , j  ← divide start_ARG Y i  j + C i , j  end_ARG start_ARG ζ i  + s end_ARG

26:         Write Ci,j 𝐶𝑖𝑗C_{i,j}C i , j  to global memory

27:     end for

28:end for
