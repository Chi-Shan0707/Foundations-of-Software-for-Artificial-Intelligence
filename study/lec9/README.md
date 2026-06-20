# L9: 大语言模型

> 本讲系统介绍大语言模型（Large Language Models, LLMs）的核心原理与主流实现路径。大语言模型是一类以语言建模任务为核心、在大规模语料上训练的神经网络模型，其目标是通过自回归方式学习自然语言中词元（token）之间的统计与语义结构，从而实现文本的理解与生成。现代大语言模型普遍采用 **Decoder-only Transformer** 架构，并以预测下一个词元作为训练目标。本讲内容依次覆盖词嵌入、分词、注意力机制、Transformer 架构变体、Decoder-only 模型的训练与生成，以及 Google 多语言翻译实验对隐空间可解释性的启示。

---

## 目录

1. [词嵌入（Word Embedding）](#1-词嵌入word-embedding)
2. [分词（Tokenization）](#2-分词tokenization)
3. [注意力机制（Attention）](#3-注意力机制attention)
4. [Transformer 架构变体](#4-transformer-架构变体)
5. [Decoder-only Transformer 结构与实现](#5-decoder-only-transformer-结构与实现)
6. [训练目标与自回归生成](#6-训练目标与自回归生成)
7. [Google 多语言翻译实验：Interlingua 与零样本翻译](#7-google-多语言翻译实验interlingua-与零样本翻译)
8. [中文分词与 BPE](#8-中文分词与-bpe)
9. [关键知识点总结](#9-关键知识点总结)

---

## 1. 词嵌入（Word Embedding）

### 1.1 从离散符号到连续向量

对语言进行建模的第一步，是将语言转化为可计算的数学表示。神经网络只能处理数值输入，因此自然语言中的词或符号需要被映射到向量空间中。这种将离散符号映射为连续向量表示的方法称为**词嵌入**（word embedding）。在词嵌入中，每个词元被表示为一个 $d$ 维实向量，所有词元共同嵌入于同一个向量空间 $\mathbb{R}^d$ 中。

### 1.2 One-hot 表示的缺陷

最原始的词元表示方式是 **one-hot 向量**。设词表大小为 $V$，则每个词元被表示为一个 $V$ 维向量，其中仅有一个位置为 1，其余位置为 0。该方式存在两个明显缺陷：

1. **维度高且稀疏**：词表通常包含数万乃至数十万项，one-hot 向量随之维度巨大且仅有一个非零元素；
2. **缺乏语义关联**：任意两个不同词的 one-hot 向量之间均正交，无法反映词与词之间的相似性或语义关系。

### 1.3 词嵌入的线性变换视角

词嵌入本质上可以视为对 one-hot 向量的一次线性变换。设嵌入矩阵为 $W \in \mathbb{R}^{V \times d}$，则词元 $i$ 的向量表示为：

$$
\mathbf{e}_i = \mathbf{o}_i W
$$

其中 $\mathbf{o}_i$ 为对应的 one-hot 向量。由于 $\mathbf{o}_i$ 仅在第 $i$ 个位置取值为 1，上式等价于直接取矩阵 $W$ 的第 $i$ 行：

$$
\mathbf{e}_i = W[i, :]
$$

因此在实际实现中，并不需要显式构造 one-hot 向量，而是通过查表（lookup）操作直接取出对应行的向量。PyTorch 中 `nn.Embedding` 即采用这一等价实现，以避免稀疏矩阵乘法带来的开销。

词嵌入的核心优势在于其**稠密性**与**可学习性**：与 one-hot 表示不同，嵌入向量是低维稠密的，且其数值通过训练自动学习得到，从而能够在向量空间中编码词语之间的语义与语法关系。

### 1.4 向量空间中的语义性质

若词嵌入训练效果良好，不同词之间的语义与语法关联可以在向量空间中得到体现，并表现为向量运算上的几何关系：

- **性别关系**：$\text{King} - \text{Man} + \text{Woman} \approx \text{Queen}$，表示 King 与 Queen 之间的性别差异可通过向量差表达；
- **年龄关系**：$\text{King} - \text{Prince} \approx \text{Queen} - \text{Princess}$，表示成年与年轻之间的年龄差异在王室词汇中同样可被捕捉。

这类线性可加性是判断词嵌入质量的重要经验性证据。

### 1.5 端到端训练

构建有效词嵌入空间的核心思想是：不人为规定词向量的具体含义，而是通过学习任务让模型自动发现这些表示。在现代大语言模型中，词嵌入一般不再作为独立模块预训练，而是作为整个模型的第一层参数，与后续网络层共同进行端到端优化。训练目标通常为语言建模（如预测下一个 token 的概率），通过最小化损失函数同步更新嵌入矩阵 $W$ 及后续所有参数。

词嵌入只是信息表示的起点。经过多层非线性变换（例如自注意力机制），每个词元会被进一步映射为**上下文相关**的表示。因此，同一个词在不同语境中的最终向量表示并不相同。这种上下文相关的稠密向量才是大模型真正用于理解与生成语言的核心表示。

---

## 2. 分词（Tokenization）

### 2.1 词元作为建模单位

在大语言模型中，词嵌入的输入单位通常并非完整单词，而是更细粒度的**词元**（token）。词表中既包含单个字符，也包含完整单词或介于二者之间的子词（subword）。

### 2.2 最长匹配原则

现代大模型普遍采用基于子词的分词方法，其核心原则之一是**最长匹配**：在词表中优先寻找能够匹配当前文本前缀的最长词元。

例如，假设词表中同时存在字符 `p, l, a, y, i, n, g` 以及子词 `play` 和 `ing`。当输入字符串为 `playing` 时，分词器会优先匹配最长的词元 `play` 与 `ing`，而非逐字符拆分为 `p, l, a, y, i, n, g`。

这种分词方式的好处在于：在有限的词表规模下，同时兼顾以下目标：

- 高频词能够整体保留，从而降低序列长度；
- 低频词能够拆分为更小的子词，避免词表无限膨胀；
- 未登录词（OOV）仍可被表示，因为任何未知符号最终可被拆解为已知的最小单元；
- 对拼写错误或错词具有一定鲁棒性。

具体的子词构造算法（如 BPE）将在第 8 节展开。

---

## 3. 注意力机制（Attention）

Transformer 的核心是注意力机制。其基本思想是通过计算序列中各词元之间的相关性，实现信息的加权聚合。在该机制中，每个输入向量（词嵌入）会被映射为 Query（Q）、Key（K）和 Value（V）三种角色，并基于 Query 与 Key 的相似度计算注意力权重，进而对 Value 进行加权求和。

### 3.1 Q / K / V 投影

设输入序列为 $x_1, x_2, \dots, x_n \in \mathbb{R}^{d_{\text{model}}}$，其中 $d_{\text{model}}$ 为词嵌入维度。在进入注意力计算前，每个 token 通过三组独立的线性映射分别生成 Query、Key 和 Value：

$$
\mathbf{q}_i = \mathbf{x}_i W_Q, \qquad \mathbf{k}_i = \mathbf{x}_i W_K, \qquad \mathbf{v}_i = \mathbf{x}_i W_V
$$

其中投影矩阵为：

$$
W_Q \in \mathbb{R}^{d_{\text{model}} \times d_k}, \quad
W_K \in \mathbb{R}^{d_{\text{model}} \times d_k}, \quad
W_V \in \mathbb{R}^{d_{\text{model}} \times d_v}.
$$

由此：

- $\mathbf{q}_i, \mathbf{k}_i \in \mathbb{R}^{d_k}$，被映射到用于相似度计算的子空间；
- $\mathbf{v}_i \in \mathbb{R}^{d_v}$，位于用于信息聚合的子空间。

### 3.2 缩放点积注意力

标准的**缩放点积注意力**（Scaled Dot-Product Attention）定义为：

$$
\text{Attention}(Q, K, V) = \text{softmax}\!\left(\frac{QK^T}{\sqrt{d_k}}\right) V
$$

其中：

$$
Q \in \mathbb{R}^{n \times d_k}, \quad
K \in \mathbb{R}^{n \times d_k}, \quad
V \in \mathbb{R}^{n \times d_v}
$$

$n$ 为序列长度，$d_k$ 为 Query/Key 的特征维度，$d_v$ 为 Value 的特征维度。

以第 $i$ 个词元为 Query 的注意力计算过程可分解为四步：

1. **线性映射**：对当前 token $x_i$ 生成 $\mathbf{q}_i = \mathbf{x}_i W_Q$；对所有历史 token $j \le i$ 生成 $\mathbf{k}_j = \mathbf{x}_j W_K$ 与 $\mathbf{v}_j = \mathbf{x}_j W_V$。
2. **相似度计算**：计算 Query 与 Key 的点积相似度 $s_{ij} = \mathbf{q}_i \cdot \mathbf{k}_j$。
3. **归一化**：通过 softmax 得到注意力权重 $\alpha_{ij} = \text{softmax}\!\left(\frac{s_{ij}}{\sqrt{d_k}}\right)$，满足 $\sum_{j=1}^{i} \alpha_{ij} = 1$。
4. **加权聚合**：对 Value 进行加权求和，得到输出 $\text{out}_i = \sum_{j=1}^{i} \alpha_{ij} \mathbf{v}_j \in \mathbb{R}^{d_v}$。

### 3.3 缩放因子的作用

公式中的缩放因子 $\sqrt{d_k}$ 用于避免点积结果随维度增大而过大，从而稳定 softmax 的梯度分布。若 $Q$ 与 $K$ 的各分量相互独立且均值为零、方差为 1，则 $QK^T$ 中每个元素的理论方差为 $d_k$。当 $d_k$ 较大时，点积数值会显著增大，使 softmax 输出趋近于 one-hot 分布，梯度几乎为零，训练难以进行。除以 $\sqrt{d_k}$ 将方差归一化回 1，保证梯度稳定。

### 3.4 因果掩码（Causal Masking）

在 Decoder-only 架构中，每个位置只允许访问自身及之前的历史 token，不能"看到未来"。实现方式是在计算注意力分数时，引入一个下三角掩码，将 $j > i$ 的位置对应的分数置为 $-\infty$，使其在 softmax 后权重为 0：

$$
\text{mask}_{ij} = \begin{cases} 0, & j \le i \\ -\infty, & j > i \end{cases}
$$

这一操作使信息流被强制为单向，满足自回归建模的约束条件。

### 3.5 多头注意力（Multi-Head Attention）

单个注意力头只能在一种相似度模式下进行信息聚合。为使模型能够同时关注序列中不同位置、不同子空间上的关系，Transformer 采用**多头注意力**机制：将 $Q$、$K$、$V$ 分别沿特征维度切分为 $h$ 份，每一份独立进行一次缩放点积注意力，最后将结果拼接并通过线性映射融合回原维度。

形式上，第 $k$ 个头的计算为：

$$
\text{head}_k = \text{Attention}(Q W_Q^{(k)},\ K W_K^{(k)},\ V W_V^{(k)})
$$

$$
\text{MultiHead}(Q, K, V) = \text{Concat}(\text{head}_1, \dots, \text{head}_h)\, W_O
$$

多头机制使不同子空间并行学习不同模式的关系（如语法依赖、共指消解、长距离依赖等），从而显著提升模型表达能力。

---

## 4. Transformer 架构变体

最初的 Transformer（见论文《Attention is All You Need》）为 Encoder-Decoder 结构，主要用于机器翻译。后续研究发现，并非所有任务都需要"先理解再转换"的完整流程。根据是否保留编码器与解码器，可派生出三种独立架构。

### 4.1 核心组件定义

**Encoder（编码器）**：接收整个输入序列，通过多层 Self-Attention 让每个 token 与序列中所有其他 token 交互。关键特性是**双向上下文**（Bidirectional）——计算第 3 个词时能看到第 1–2 个词和后面的词。输出是一系列特征向量，包含语义、词性、句法结构等深层信息。

**Decoder（解码器）**：逐词生成输出。关键特性是**单向/因果掩码**（Causal Masking）——生成第 $n$ 个词时不能看到第 $n+1$ 个词。核心模块包括 Masked Self-Attention（只看已生成历史）和 Cross-Attention（查看 Encoder 输出）。

### 4.2 三种独立架构对比

| 架构 | 核心能力 | 信息流动 | 训练目标 | 典型用途 | 代表模型 |
|------|---------|---------|---------|---------|---------|
| **Encoder-Only** | 理解输入内容 | 双向 | Masked LM（完形填空） | 填空、分类、提取特征 | BERT, RoBERTa |
| **Decoder-Only** | 续写/生成内容 | 单向（Causal/Masked） | Causal LM（预测下一词） | 对话、写作、推理 | GPT, LLaMA, Qwen |
| **Encoder-Decoder** | 转换输入到输出 | 编码双向 + 解码单向 | Seq2Seq | 翻译、摘要 | T5, BART |

**Encoder-Only**：只保留 Transformer 编码器部分。训练目标为掩码语言模型（Masked LM），类似完形填空——将句子中随机若干 token 用 `[MASK]` 替换，让模型利用左右上下文预测被掩盖的词。擅长理解任务，但不擅长生成。

**Decoder-Only**：去掉 Cross-Attention，仅保留解码器。训练目标为因果语言模型（Causal LM），即预测下一个词。架构更简单、更易于规模化；大规模参数下涌现出通用泛化能力与零样本学习能力，是当前 LLM 的主流选择（详见 4.4）。

**Encoder-Decoder**：保留编码器与解码器分工——Encoder 提取输入语义，Decoder 根据编码结果生成目标形式，并通过 Cross-Attention 与编码器输出交互。适用于输入输出长度差异较大的映射任务，如翻译、摘要。

### 4.3 Embedding 与 Encoder 的区别

Decoder-only 架构同样包含 Embedding，但 Embedding 只是数据的静态映射——将离散 token ID 变成连续向量，并不等同于 Encoder 架构。三者关系如下：

- **Embedding** = 将单词映射为向量（所有 Transformer 模型均有此步骤）；
- **Encoder（架构）** = 允许信息双向流动的特征提取器；
- **Decoder（架构）** = 强制信息单向流动的生成器。

Decoder-only 将"理解输入"与"生成输出"统一为同一种操作——基于历史上下文预测下一词。其处理输入的方式是单向的：一边读题一边写答案，而非反复审题。

### 4.4 Decoder-Only 为何成为 LLM 主流

尽管 Encoder-only 在理解任务上效率更高，但研究表明，在大规模参数与数据下，Decoder-only 架构通过简单的"预测下一个词"任务，能涌现出极强的通用泛化能力与零样本学习能力。其优势主要体现在：

1. **架构简洁**：去掉了 Cross-Attention，仅保留 Masked Self-Attention 与 FFN，参数效率高，便于规模化扩展；
2. **任务统一**：将翻译、问答、写作、推理等不同任务统一为"续写"范式，无需针对不同任务设计专用结构；
3. **涌现能力**：参数量达到一定规模后，简单的下一词预测任务即可让模型习得逻辑推理、代码生成等复杂能力；
4. **零样本与少样本学习**：通过 Prompt 即可完成新任务，无需针对每个下游任务重新训练。

---

## 5. Decoder-only Transformer 结构与实现

### 5.1 整体结构

Decoder-only Transformer 由多层相同的 **Decoder Block** 堆叠而成。模型的总体数据流为：

1. 输入端：将离散 token ID 序列通过 token embedding 映射为连续向量，并叠加位置编码（positional encoding）以引入序列顺序信息；
2. 中间层：表示向量依次通过多个 Decoder Block 进行逐层特征变换；
3. 输出端：通过 LayerNorm 对最终隐藏状态归一化，再经线性层映射到词表空间，得到 logits 分布，用于预测下一词元。

由于 Transformer 中 Self-Attention 本身对位置不敏感，位置编码是引入序列顺序信息的必要手段。常见实现包括可学习的位置嵌入、正弦位置编码以及相对位置编码（如 RoPE）。

### 5.2 Decoder Block

每个 Decoder Block 包含两个子层，并结合残差连接（Residual Connection）与 LayerNorm 以稳定深层网络的训练：

1. **Masked Self-Attention**：通过因果掩码限制信息流，使每个 token 仅能访问自身及其之前的历史 token；
2. **Feed-Forward Network（FFN）**：逐位置的两层全连接网络，通常配合非线性激活（如 ReLU、GELU），形式为 $\text{FFN}(x) = \text{ReLU}(xW_1 + b_1)W_2 + b_2$。

Post-Norm 结构下，前向计算为：

$$
x = \text{LayerNorm}\!\left(x + \text{SelfAttn}(x)\right)
$$

$$
x = \text{LayerNorm}\!\left(x + \text{FFN}(x)\right)
$$

### 5.3 残差连接与 LayerNorm 的作用

残差连接允许梯度直接回传到浅层网络，缓解深层 Transformer 的梯度消失问题；LayerNorm 对每个样本的特征维度进行归一化，使各层输入分布保持稳定。二者结合是训练数十乃至上百层 Transformer 的关键技巧。

---

## 6. 训练目标与自回归生成

### 6.1 因果语言模型

Decoder-only Transformer 采用**下一个词元预测**（next-token prediction）作为训练目标。给定一个长度为 $T$ 的 token 序列 $x_{1:T}$，模型在每个位置 $t$ 基于历史上下文 $x_{<t}$ 建模条件概率 $P(x_t \mid x_{<t})$。整个序列的似然为：

$$
P(x_{1:T}) = \prod_{t=1}^{T} P(x_t \mid x_{<t})
$$

训练目标即最大化该似然，等价于最小化交叉熵损失。

### 6.2 交叉熵损失

设模型在每个位置输出词表上的 logits 分布 $\mathbf{z}_t \in \mathbb{R}^{V}$，经 softmax 得到预测概率 $\hat{p}_t = \text{softmax}(\mathbf{z}_t)$，真实目标为 $y_t = x_{t+1}$。则单个位置的交叉熵损失为：

$$
\mathcal{L}_t = -\log \hat{p}_t[y_t] = -\log \frac{\exp(z_t[y_t])}{\sum_{v=1}^{V} \exp(z_t[v])}
$$

对于一个 mini-batch 中的所有位置，平均损失为：

$$
\mathcal{L} = -\frac{1}{B \cdot T} \sum_{b=1}^{B} \sum_{t=1}^{T} \log \hat{p}_{b,t}[y_{b,t}]
$$

实际实现中，PyTorch 的 `F.cross_entropy` 接收形状为 `(B*T, V)` 的 logits 与形状为 `(B*T,)` 的目标序列，自动完成 softmax 与负对数似然的计算。

### 6.3 训练样本构造

训练样本采用**滑动窗口**方式从长序列中截取固定长度 $T$（block_size）的连续片段，构造样本对 $(x, y)$：

- 输入序列 $x = \text{data}[i : i + T]$；
- 目标序列 $y = \text{data}[i + 1 : i + T + 1]$，即 $x$ 在时间维度上右移一位的对齐结果，满足 $y_t = x_{t+1}$。

由于因果掩码的存在，单次前向传播即可同时完成序列中所有位置的下一词预测训练，无需逐 token 串行计算，训练效率高。

### 6.4 自回归生成

推理阶段采用**自回归**（autoregressive）方式生成文本，流程如下：

1. 将输入提示（prompt）编码为 token ID 序列；
2. 截取最近 $T$ 个 token 作为上下文窗口（受最大上下文长度限制），前向传播得到最后一层 logits；
3. 仅取最后一个时间步的预测分布，根据解码策略（贪心、采样、束搜索等）选取下一 token；
4. 将预测结果拼接到输入序列末尾，重复步骤 2–4，直至达到最大生成步数或生成 EOS token。

最简单的**贪心解码**（greedy decoding）每步选取概率最大的 token：

$$
x_{t+1} = \arg\max_{v} P(v \mid x_{\le t})
$$

贪心解码确定性高但容易陷入局部最优，实际应用中常采用温度采样、top-$k$、top-$p$ 等策略以提升生成多样性。

---

## 7. Google 多语言翻译实验：Interlingua 与零样本翻译

### 7.1 实验背景

Google 于 2016 年在 TACL 发表 Multilingual NMT 系统（《Google's Multilingual Neural Machine Translation System: Enabling Zero-Shot Translation》），提出使用**单一 Encoder-Decoder 模型**处理所有语言对。具体做法是：在输入句子开头添加目标语言 Token（如 `<2es>` 表示译为西班牙语，`<2ko>` 表示译为韩语），然后将所有语料混合训练。

传统机器翻译需为每对语言单独训练一个模型，若共有 $N$ 种语言，则需要训练 $N(N-1)$ 个模型，工程成本不可接受。统一模型将所有语言对纳入同一参数空间，为后续的零样本能力奠定了基础。

### 7.2 零样本翻译

模型在训练时仅见过以下两类语料：

- 葡萄牙语 → 英语
- 英语 → 西班牙语

然而，当输入葡萄牙语并将目标 Token 设为 `<2es>` 时，模型在从未见过任何葡萄牙语—西班牙语平行语料的情况下，成功实现了高质量的翻译。这就是**零样本翻译**（Zero-Shot Translation），证明了统一模型在共享隐空间中实现了语言间的语义迁移。

### 7.3 语言混杂实验：α 系数的含义

为探究统一模型内部发生了什么，Google 设计了一组探测实验：在解码阶段，不使用单一目标语言 Token 向量，而是将两种语言的 Token 向量做线性插值：

$$
E_{\text{mixed}} = \alpha \cdot E_{\text{langA}} + (1 - \alpha) \cdot E_{\text{langB}}
$$

其中 $\alpha$ 即为实验中的比例系数（如 0.3、0.5、0.7）。

**意大利语 vs 西班牙语（同语系）**：

| $\alpha$ | 输出特征 |
|----------|---------|
| 1.0 | 纯西班牙语 |
| 0.7 | 以西班牙语为主，渗透约 30% 意大利语特征 |
| 0.5 | 语法通顺但词汇对半混杂的"中间语言" |
| 0.3 | 意大利语占主导 |

**英语 vs 韩语（跨语系）**：在 0.5:0.5 混合时，文本呈现"局部英语、局部韩语"的交替坍塌。前半句使用 SVO 语序的英语结构，后半句突然坍塌为韩语词汇与助词。即使两种语言的语序与语系截然不同，模型内部语义表征依然共享。

### 7.4 结论：Interlingua 隐空间

该实验从两个层面验证了统一模型内部的可解释结构：

1. **隐空间的连续性**：语言在模型内部并非离散孤岛，而是一片连续的几何表面。调整 $\alpha$ 可在该表面上平滑滑动，从一种语言过渡到另一种语言；
2. **语义与形式的分离**：Encoder 将"纯粹语义"抽象为与具体语言无关的概念向量（Language-Independent Representation），Decoder 根据目标语言 Token 决定输出形式。若将目标 Token 拧到中间位置（如 $\alpha = 0.5$），语义会同时流向两种语言，形成混杂文本。

这一"中间通用语"（Interlingua）的存在，是 Encoder-Decoder 架构"语义咽喉"特性的直观体现：语义由 Encoder 提供，而语言形式由 Decoder 入口的 Token 向量控制，二者在数学上实现了分离。

---

## 8. 中文分词与 BPE

### 8.1 传统 NLP 中的分词

中文没有空格分隔词边界，传统方法包括：

- **基于词典**：最大匹配法（MM, RMM），从前向后或从后向前扫描，每次取词典中存在的最长匹配；
- **基于序列标注**：将分词转化为每个字的标注问题（B/M/E/S 分别表示词首、词中、词尾、单字词），用 HMM 或 CRF 建模，再用 Viterbi 解码求最优切分；
- **经典工具**：Jieba（前缀词典 + HMM）、HanLP、Stanza。

### 8.2 现代 LLM 的分词：BPE（Byte-Pair Encoding）

当代主流模型（Qwen、Baichuan、ChatGLM、LLaMA 等）不再依赖传统中文分词器，而是采用 **BPE**（Byte-Pair Encoding）或其变体 SentencePiece。BPE 的构造过程为：

1. **初始化**：将训练语料拆为最小单元（汉字或 UTF-8 字节）；
2. **统计迭代**：统计相邻单元的共现频次，将最高频的二元组合合并为一个新 Token；
3. **循环**：重复步骤 2，直到词表达到预设大小（如 Qwen 词表为 151,936）。

BPE 的关键特性：

- **动态粒度**：高频词整体保留为单个 Token，低频词被拆成更小的子词单元；
- **无损还原**：遇到未知符号时，可通过字节级 Token 兜底，不会出现 `[UNK]`，理论上可表示任意 Unicode 文本；
- **压缩率**：关键评价指标为 Token/字符比。早期 LLaMA 对中文不友好（一个汉字常被切分为 3 个字节 Token），国产模型通常一个 Token 平均代表 1.5–2 个汉字，中文压缩效率显著更高；
- **鲁棒性**：对拼写错误或错词具有一定容错能力，因为错词仍可被拆解为已知的子词组合。

### 8.3 频率压缩的本质

BPE 的核心思想是**频率压缩**——高频共现的片段被压缩为单个 Token，本质上是在对语料库进行熵编码：出现频率越高的模式，被赋予越短的编码长度。这与信息论中的霍夫曼编码思想相通，使得模型在固定词表规模下，能够以更少的 Token 数表示同样的文本，从而扩展有效上下文长度并降低计算开销。

---

## 9. 关键知识点总结

| 主题 | 核心要点 |
|------|---------|
| 词嵌入 | one-hot → 线性变换 → 稠密可学习向量；语义关系可由向量运算体现；端到端训练 |
| 分词 | 子词为建模单位；最长匹配；兼顾高频整体保留与低频拆分，处理 OOV |
| 注意力 | Q/K/V 投影；缩放点积公式 $\text{softmax}(QK^T/\sqrt{d_k})V$；缩放因子稳定梯度 |
| 因果掩码 | 下三角掩码置 $-\infty$，强制单向信息流，满足自回归约束 |
| 多头注意力 | 沿特征维度切分为 $h$ 份并行计算，使不同子空间学习不同关系模式 |
| 架构变体 | Encoder-only（BERT）/ Decoder-only（GPT, LLaMA）/ Encoder-Decoder（T5）|
| Decoder-only 主流 | 架构简洁、任务统一、涌现能力强、支持零/少样本学习 |
| 训练目标 | 因果语言模型；下一词预测；交叉熵损失；滑动窗口构造 $(x, y)$ 对 |
| 自回归生成 | 截取上下文 → 前向 → 取最后位 logits → 解码下一 token → 拼接循环 |
| 多语言实验 | 单一 Encoder-Decoder + 目标语言 Token；零样本翻译；α 插值揭示连续隐空间 |
| BPE | 频率压缩；动态粒度；字节级兜底避免 `[UNK]`；Token/字符比为压缩率核心指标 |

### 参考实现

本讲涉及的完整 PyTorch 实现包括：缩放点积自注意力（`SelfAttention`）、Decoder Block（`DecoderBlock`，含 Masked Self-Attention 与 FFN）、Decoder-only Transformer 整体模型（`DecoderOnlyTransformer`，含 token embedding、位置嵌入、多层 Decoder Block 与 lm_head 输出层）、训练样本构造（`NextTokenDataset`，基于滑动窗口）、训练循环（AdamW + 交叉熵）以及贪心解码生成函数（`generate`）。详细代码见课程教材第 9 章。


---

## 附录：代码文件索引

| 文件 | 说明 |
|------|------|
| `code/l9-train.py` | Decoder-only Transformer 训练脚本（AdamW + 交叉熵） |
| `code/l9-test.py` | 模型测试与文本生成（贪心解码） |
| `notes/L9-大语言模型.pdf` | 课程讲义原文 |
