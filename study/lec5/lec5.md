# L5: PyTorch 与 CNN 损失函数的数学本质

> 本节课核心：理解 CNN 中 Loss（损失函数）的计算原理，从数据流向、数学定义、概率论基础三个维度掌握交叉熵损失（Cross-Entropy Loss）的本质，并理解最大似然估计（MLE）如何为深度学习的优化目标提供严格的理论支撑。

---

## 目录

1. [前置知识：最大似然估计（MLE）](#1-前置知识最大似然估计mle)
2. [CNN 计算 Loss 的标准数据流](#2-cnn-计算-loss-的标准数据流)
3. [核心损失函数的数学表达与计算](#3-核心损失函数的数学表达与计算)
4. [数学本质：为什么使用交叉熵？](#4-数学本质为什么使用交叉熵)
5. [现代深度学习框架中的计算优化](#5-现代深度学习框架中的计算优化)
6. [代码实战：LeNet-5 与 SimpleRNN 中的 Loss](#6-代码实战lenet-5-与-simplernn-中的-loss)

---

## 1. 前置知识：最大似然估计（MLE）

在深入理解 CNN 的 Loss 之前，我们必须先理解**最大似然估计（Maximum Likelihood Estimation, MLE）**。因为交叉熵损失的本质，就是 MLE 在分类任务中的具体实现形式。

### 1.1 什么是"似然"（Likelihood）？

**似然（Likelihood）**与**概率（Probability）**虽然数学表达式相同，但语义截然不同：

| | 概率（Probability） | 似然（Likelihood） |
|--|---------------------|---------------------|
| **固定什么？** | 固定模型参数 $\theta$，观察数据 $x$ 的变化 | 固定已观测数据 $x$，观察参数 $\theta$ 的变化 |
| **含义** | 在给定模型下，数据出现的可能性有多大？ | 在给定数据下，哪组参数最可能产生了这些数据？ |
| **表达式** | $P(x \mid \theta)$ | $L(\theta) = P(x \mid \theta)$ |

用一个直觉例子来理解：假设你有一枚硬币，抛了 10 次得到 7 次正面、3 次反面。

- **概率问题**：如果这枚硬币是公平的（$\theta = 0.5$），观察到"7正3反"的概率是多少？→ 这是概率。
- **似然问题**：观察到"7正3反"这个结果后，这枚硬币出现正面的概率 $\theta$ 最可能是多少？→ 这是似然。

### 1.2 似然函数的构造

假设我们有 $N$ 个**独立同分布（i.i.d.）**的观测样本 $\{(x^{(1)}, y^{(1)}), (x^{(2)}, y^{(2)}), \dots, (x^{(N)}, y^{(N)})\}$，模型的参数为 $\theta$。

由于样本之间相互独立，根据概率论的乘法法则，**联合概率**（Joint Probability）等于各个样本概率的乘积：

$$
P(Y \mid X; \theta) = \prod_{k=1}^{N} P(y^{(k)} \mid x^{(k)}; \theta)
$$

这个关于 $\theta$ 的函数就是**似然函数（Likelihood Function）**：

$$
L(\theta) = \prod_{k=1}^{N} P(y^{(k)} \mid x^{(k)}; \theta)
$$

### 1.3 从"最大化似然"到"最小化负对数似然"

MLE 的目标是找到最优参数 $\theta^*$，使得似然函数最大化：

$$
\theta^* = \arg\max_{\theta} \prod_{k=1}^{N} P(y^{(k)} \mid x^{(k)}; \theta)
$$

但直接优化这个连乘形式存在两个严重的**计算问题**：

1. **下溢出（Underflow）**：当 $N$ 很大时，每个概率 $P \in [0, 1]$，几百项连乘会迅速趋近于 0，超出浮点数精度（float64 约能表示到 $10^{-308}$）。
2. **求导困难**：连乘的导数需要使用乘积法则（Product Rule），展开后极为复杂。

**解决方案：取对数（Logarithm）**。对数函数 $\log(\cdot)$ 是**严格单调递增**的，因此：

$$
\arg\max_{\theta} f(\theta) \iff \arg\max_{\theta} \log f(\theta)
$$

取对数后，**连乘变连加**，乘积法则变加法法则：

$$
\log L(\theta) = \sum_{k=1}^{N} \log P(y^{(k)} \mid x^{(k)}; \theta)
$$

再取负号，将"最大化"转为"最小化"（这是优化领域的惯例，梯度下降算法默认做最小化）：

$$
\theta^* = \arg\min_{\theta} \left[ -\sum_{k=1}^{N} \log P(y^{(k)} \mid x^{(k)}; \theta) \right]
$$

这个表达式就是**负对数似然（Negative Log-Likelihood, NLL）**。**在分类任务中，NLL 的形式与交叉熵损失完全一致——这就是交叉熵的理论根基。**

### 1.4 MLE 的小结

```
MLE 的完整推导链：

  观测数据 → 构造似然函数 L(θ) → 取对数（避免下溢出+简化求导）
                                         ↓
                                 最大化对数似然
                                         ↓
                                 取负号 → 最小化 NLL
                                         ↓
                                 NLL = 交叉熵（在分类任务中）
                                         ↓
                          ⭐ 这就是 CNN 训练中优化 Loss 的本质
```

**学习要点**：理解这一点后，你就不会再把"交叉熵损失"看作一个凭空选择的公式——它是从"让模型最可能产生观测数据"这个朴素思想出发，经过严格数学推导得出的必然结论。

---

## 2. CNN 计算 Loss 的标准数据流

在实际的 CNN 架构（例如基于 ResNet 骨干网络的分类器，或课程中的 LeNet-5）中，一张图片输入网络后，计算 Loss 经历以下严格步骤：

### 2.1 特征提取与全连接映射（Logits）

输入图像经过多层卷积（Convolution）、池化（Pooling）、非线性激活（ReLU）后被展平（Flatten），最终通过全连接层（Fully Connected Layer）。

此时网络的输出是一个**无界的实数向量**，称为 **Logits**（记为 $z \in \mathbb{R}^C$，其中 $C$ 为类别数量）。

> **为什么叫 Logits？**
> Logit 来源于统计学，定义为某事件发生概率 $p$ 的对数几率（log-odds）：
> $\text{logit}(p) = \log \frac{p}{1-p}$
> 在深度学习中，Logits 被泛化为分类器最后一层的原始输出，它尚未经过归一化，可以取任意实数值。Logits 中的相对大小表达了网络对不同类别的"偏好程度"，但它们本身并不直接等于概率。

以 LeNet-5（代码见 `code/l5-lenet5.py`）为例，对于 MNIST 的 10 类分类任务，网络最终输出一个 10 维向量 $z \in \mathbb{R}^{10}$：

```python
# LeNet-5 的前向传播最终输出
x = self.fc2(x)  # 输出 shape: (batch_size, 10) → 这就是 Logits
return x
```

### 2.2 概率归一化（Activation）

为了将 Logits 转化为符合概率公理的值（非负且总和为 1），通常会引入激活函数：

#### Softmax 函数（多分类任务）

$$
\hat{y}_i = \frac{e^{z_i}}{\sum_{j=1}^{C} e^{z_j}}, \quad i = 1, 2, \dots, C
$$

**性质详解**：

- **非负性**：由于 $e^{z_i} > 0$，所以 $\hat{y}_i > 0$，满足概率的非负要求。
- **归一性**：$\sum_{i=1}^{C} \hat{y}_i = \frac{\sum_{i} e^{z_i}}{\sum_{j} e^{z_j}} = 1$，满足概率的总和为 1。
- **单调性**：$z_i$ 越大，$\hat{y}_i$ 越大，保留了 Logits 中的排序信息。
- **指数放大的"赢家通吃"效应**：Softmax 对最大的 Logit 值给予不成比例的高概率。例如，如果 $z = [2, 1, 0]$，则 $\hat{y} \approx [0.665, 0.244, 0.090]$；如果 $z = [10, 1, 0]$，则 $\hat{y} \approx [0.9999, 0.0001, 0.0000]$。这种特性有助于网络做出更"自信"的预测。

#### Sigmoid 函数（二分类任务）

$$
\hat{y} = \sigma(z) = \frac{1}{1 + e^{-z}}
$$

**性质详解**：

- **值域**：$\hat{y} \in (0, 1)$，天然表示正类的概率。
- **对称中心**：当 $z = 0$ 时，$\hat{y} = 0.5$，即"不确定"状态。
- **单调递增**：$z \to +\infty$ 时 $\hat{y} \to 1$，$z \to -\infty$ 时 $\hat{y} \to 0$。
- **与 Softmax 的关系**：二分类 Softmax 退化为 Sigmoid。当 $C = 2$ 时，设 $z_1 = z, z_2 = 0$，则 $\hat{y}_1 = \frac{e^z}{e^z + 1} = \frac{1}{1 + e^{-z}} = \sigma(z)$。

### 2.3 计算样本损失（Sample Loss）

将网络的预测概率分布 $\hat{y}$ 与真实标签分布 $y$ 代入预定的损失函数公式，得到该样本的损失值 $L$。

真实标签 $y$ 通常被编码为 **One-hot 向量**（例如，MNIST 中数字 "3" 的标签为 $[0, 0, 0, 1, 0, 0, 0, 0, 0, 0]$）。

> **为什么使用 One-hot 编码？**
> One-hot 编码将离散的类别标签转化为向量空间的表示，使得标签可以直接与网络的连续值输出进行数学运算（如交叉熵公式中的逐元素乘法）。此外，One-hot 编码隐含了类别之间的"互斥性"——一个样本只属于一个类别。

### 2.4 计算批次损失（Batch Loss）

在实际训练中，我们使用小批量梯度下降（Mini-batch SGD），计算一个 Batch 内 $N$ 个样本 Loss 的平均值作为最终的优化目标：

$$
J = \frac{1}{N} \sum_{k=1}^{N} L^{(k)}
$$

**为什么要取平均而不是求和？**

1. **学习率与 Batch Size 解耦**：如果用求和，增大 Batch Size 会使梯度的量级成比例增大，需要相应减小学习率。取平均后，梯度的量级与 Batch Size 无关，使得超参数调优更简单。
2. **梯度估计的方差更小**：平均操作本质上是一种蒙特卡洛估计（Monte Carlo Estimation），Batch Size 越大，对真实梯度的估计越精确、方差越小。

### 2.5 完整数据流总结

```
┌─────────────────────────────────────────────────────────────────────┐
│                     CNN Loss 计算的完整数据流                        │
├─────────────────────────────────────────────────────────────────────┤
│                                                                     │
│  ┌──────────┐    ┌──────────────┐    ┌────────────┐                │
│  │ 输入图像  │───→│ Conv+Pool+FC │───→│   Logits   │                │
│  │ x ∈ R^HWC│    │  (特征提取)   │    │ z ∈ R^C   │                │
│  └──────────┘    └──────────────┘    └─────┬──────┘                │
│                                            │                        │
│                                            ▼                        │
│                                     ┌────────────┐                 │
│                                     │  Softmax   │                 │
│                                     │ (概率归一化) │                 │
│                                     └─────┬──────┘                 │
│                                           │                        │
│                                           ▼                        │
│                                     ┌────────────┐                 │
│                                     │  ŷ ∈ R^C   │                 │
│                                     │ (预测概率)   │                 │
│                                     └─────┬──────┘                 │
│                                           │                        │
│  ┌──────────┐                            │                        │
│  │ 真实标签  │                            │                        │
│  │ y (One-hot) ─────────────────────────→ │                        │
│  └──────────┘                            ▼                        │
│                                     ┌────────────┐                 │
│                                     │ CrossEntropy│                 │
│                                     │ (损失计算)   │                 │
│                                     └─────┬──────┘                 │
│                                           │                        │
│                                           ▼                        │
│                                     ┌────────────┐                 │
│                                     │   Loss L   │                 │
│                                     │ (标量)      │                 │
│                                     └─────┬──────┘                 │
│                                           │                        │
│                                           ▼                        │
│                                     ┌────────────┐                 │
│                                     │ loss.backward()              │
│                                     │ (反向传播)   │                 │
│                                     └────────────┘                 │
│                                                                     │
└─────────────────────────────────────────────────────────────────────┘
```

---

## 3. 核心损失函数的数学表达与计算

在 CNN 中，最常用于分类任务的损失函数是**交叉熵损失（Cross-Entropy Loss）**。

### 3.1 二元交叉熵（Binary Cross-Entropy, BCE）

当 CNN 处理二分类问题时，假设真实标签 $y \in \{0, 1\}$，模型预测样本为正类的概率为 $\hat{y}$。BCE 的计算公式为：

$$
L_{\text{BCE}} = - \left( y \log(\hat{y}) + (1 - y) \log(1 - \hat{y}) \right)
$$

**逐项解析**：

- 当 $y = 1$（真实为正类）：$L = -\log(\hat{y})$
  - $\hat{y} \to 1$：$L \to 0$，预测正确，损失极小。
  - $\hat{y} \to 0$：$L \to +\infty$，预测完全错误，产生强烈惩罚。
  - 这正是 $-\log$ 函数在 $(0, 1]$ 上的行为：越接近 0 惩罚越重。

- 当 $y = 0$（真实为负类）：$L = -\log(1 - \hat{y})$
  - $\hat{y} \to 0$：$L \to 0$，预测正确。
  - $\hat{y} \to 1$：$L \to +\infty$，预测完全错误。

**数值安全提醒**：当 $\hat{y} = 0$ 或 $\hat{y} = 1$ 时，$\log(0)$ 未定义。实际实现中通常对 $\hat{y}$ 进行 clip 操作（如 $\hat{y} = \max(\epsilon, \min(1-\epsilon, \hat{y}))$，$\epsilon$ 取 $10^{-7}$ 级别）来避免数值错误。

### 3.2 多类别交叉熵（Categorical Cross-Entropy, CCE）

当 CNN 处理 $C$ 个类别的互斥分类任务时，真实标签 $y$ 是 One-hot 向量，预测概率为 $\hat{y}$。CCE 的计算公式为：

$$
L_{\text{CCE}} = - \sum_{i=1}^{C} y_i \log(\hat{y}_i)
$$

**关键简化——One-hot 的威力**：

由于 $y$ 是 One-hot 编码（假设第 $c$ 类是正确类别，即 $y_c = 1$，其余 $y_{i \neq c} = 0$），公式中的求和项绝大多数被消解：

$$
L_{\text{CCE}} = - \sum_{i=1}^{C} y_i \log(\hat{y}_i) = - \left( 0 \cdot \log(\hat{y}_1) + \cdots + 1 \cdot \log(\hat{y}_c) + \cdots + 0 \cdot \log(\hat{y}_C) \right) = -\log(\hat{y}_c)
$$

最终退化为：

$$
\boxed{L_{\text{CCE}} = -\log(\hat{y}_c)}
$$

这个简化形式被称为**对数似然损失（Negative Log-Likelihood, NLL）**。它只关注网络对**真实类别**预测出的概率值 $\hat{y}_c$。

**直觉理解**：

| 网络对正确类别的预测概率 $\hat{y}_c$ | Loss $= -\log(\hat{y}_c)$ | 含义 |
|:--:|:--:|:--|
| 0.99 | 0.01 | 网络非常自信且正确，损失极小 |
| 0.90 | 0.105 | 网络比较正确，损失较小 |
| 0.50 | 0.693 | 网络和随机猜测差不多 |
| 0.10 | 2.303 | 网络基本判错，损失较大 |
| 0.01 | 4.605 | 网络极度自信地判错，惩罚极大 |
| → 0 | → $+\infty$ | 灾难性错误，损失趋向无穷 |

### 3.3 回归任务中的损失函数：MSE / L2 Loss

如果 CNN 不用于分类，而是用于预测连续数值（例如回归任务中的坐标框预测、深度图估计），则常用**均方误差（Mean Squared Error, MSE）**：

$$
L_{\text{MSE}} = \frac{1}{C} \sum_{i=1}^{C} (y_i - \hat{y}_i)^2
$$

**MSE 的统计含义**：在假设噪声服从高斯分布（正态分布）$N(0, \sigma^2)$ 的前提下，最小化 MSE 等价于对该高斯模型做 MLE。这再次印证了"损失函数 = MLE 的具体化"这一统一框架。

课程中的 SimpleRNN（`code/l5-rnn.py`）正是使用 MSE 来训练正弦波预测模型的：

```python
criterion = nn.MSELoss()   # 回归任务用 MSE
```

### 3.4 损失函数选择速查表

| 任务类型 | 标签形式 | 激活函数 | PyTorch Loss | 本质 |
|---------|---------|---------|-------------|------|
| 多分类（互斥） | 整数标签 `0, 1, ..., C-1` | 无需手动加（内置 Softmax） | `nn.CrossEntropyLoss` | NLL |
| 多分类（互斥） | One-hot 向量 | Softmax | `NLLLoss(log_softmax(z))` | NLL |
| 多标签分类 | 多热向量 `[1, 0, 1, ...]` | Sigmoid | `nn.BCEWithLogitsLoss` | MLE |
| 二分类 | 标量 `0` 或 `1` | Sigmoid | `nn.BCEWithLogitsLoss` | MLE |
| 回归 | 连续实数 | 无 | `nn.MSELoss` | 高斯 MLE |

---

## 4. 数学本质：为什么使用交叉熵？

从高等数学和统计学的视角来看，CNN 分类任务使用交叉熵并非经验之谈，而是有严格的理论支撑的。它可以从两个等价的视角推导出来：

### 4.1 视角一：极大似然估计（MLE）

回到 [第 1 节](#1-前置知识最大似然估计mle) 的推导框架。假设模型参数为 $\theta$，每个样本标签的生成服从**多项分布**（Multinomial Distribution）——即在 $C$ 个类别中选择一个，选第 $i$ 类的概率为 $P(y=i \mid x; \theta)$。

我们希望找到一组参数 $\theta^*$，使得观测到现有训练数据的联合概率最大化：

$$
\theta^* = \arg\max_{\theta} \prod_{k=1}^{N} P(y^{(k)} \mid x^{(k)}; \theta)
$$

取负对数后：

$$
\theta^* = \arg\min_{\theta} \left[ -\sum_{k=1}^{N} \log P(y^{(k)} \mid x^{(k)}; \theta) \right]
$$

对于分类任务，$P(y^{(k)} = c \mid x^{(k)}; \theta)$ 就是网络输出的概率 $\hat{y}_c^{(k)}$，因此：

$$
-\sum_{k=1}^{N} \log \hat{y}_{c^{(k)}}^{(k)} = -\sum_{k=1}^{N} \sum_{i=1}^{C} y_i^{(k)} \log \hat{y}_i^{(k)} = \sum_{k=1}^{N} L_{\text{CCE}}^{(k)}
$$

**结论：最小化交叉熵 = 对模型参数做极大似然估计。**

这意味着当你在 PyTorch 中写 `loss = CrossEntropyLoss()(output, target); loss.backward()` 时，你实际上是在执行一个统计推断过程——寻找最可能产生训练数据的模型参数。

### 4.2 视角二：KL 散度与信息论

#### 信息熵（Shannon Entropy）

信息熵度量了一个概率分布的"不确定性"：

$$
H(P) = -\sum_{i=1}^{C} P(i) \log P(i)
$$

- 如果 $P$ 是均匀分布（$P(i) = 1/C$），$H(P) = \log C$，不确定性最大。
- 如果 $P$ 是退化分布（某个 $P(c) = 1$，其余为 0），$H(P) = 0$，完全确定。

#### 交叉熵（Cross-Entropy）

交叉熵度量了用分布 $Q$ 来编码来自分布 $P$ 的数据时，平均所需的"信息量"：

$$
H(P, Q) = -\sum_{i=1}^{C} P(i) \log Q(i)
$$

#### KL 散度（Kullback-Leibler Divergence）

KL 散度度量了两个概率分布之间的"非对称差异"：

$$
D_{\text{KL}}(P \| Q) = \sum_{i=1}^{C} P(i) \log \left( \frac{P(i)}{Q(i)} \right)
$$

展开这个公式：

$$
D_{\text{KL}}(P \| Q) = \underbrace{\sum_{i=1}^{C} P(i) \log P(i)}_{= -H(P)} - \underbrace{\sum_{i=1}^{C} P(i) \log Q(i)}_{= -H(P, Q)} = -H(P) + H(P, Q)
$$

重新排列：

$$
H(P, Q) = H(P) + D_{\text{KL}}(P \| Q)
$$

**这是核心等式。** 其中：
- $H(P)$ 是真实分布的熵，对给定训练集而言是**常数**。
- $D_{\text{KL}}(P \| Q)$ 衡量模型分布 $Q$ 与真实分布 $P$ 的偏离程度。
- $D_{\text{KL}}(P \| Q) \geq 0$，当且仅当 $P = Q$ 时取等号（Gibbs 不等式）。

因此：

$$
\min_{\theta} H(P, Q_\theta) \iff \min_{\theta} D_{\text{KL}}(P \| Q_\theta)
$$

**结论：最小化交叉熵等价于最小化模型预测分布与真实数据分布之间的 KL 散度。**

当 $D_{\text{KL}}(P \| Q) = 0$ 时，$P = Q$，即模型的预测分布完美拟合了真实的标签分布。

### 4.3 两个视角的统一

```
┌─────────────────────────────────────────────────────────────────────┐
│               交叉熵损失的两种等价推导                                │
├─────────────────────────────────────────────────────────────────────┤
│                                                                     │
│  ┌─────────────────────────┐    ┌─────────────────────────┐        │
│  │   统计学视角：MLE        │    │  信息论视角：KL 散度      │        │
│  │                         │    │                         │        │
│  │  最大化：               │    │  最小化：                │        │
│  │  ∏ P(y|x;θ)           │    │  H(P,Q) = H(P)+KL(P||Q) │        │
│  │         ↓  取负对数     │    │         ↓  H(P)是常数    │        │
│  │  最小化：               │    │  等价于：                │        │
│  │  -Σ log P(y|x;θ)       │    │  最小化 KL(P||Q)        │        │
│  │         ↓              │    │         ↓              │        │
│  │  = Cross-Entropy Loss  │    │  = Cross-Entropy Loss  │        │
│  └────────────┬────────────┘    └────────────┬────────────┘        │
│               │                              │                    │
│               └──────────┬───────────────────┘                    │
│                          ▼                                        │
│               ┌─────────────────────┐                             │
│               │   两者殊途同归       │                             │
│               │   交叉熵损失是       │                             │
│               │   理论上最优的选择    │                             │
│               └─────────────────────┘                             │
│                                                                     │
└─────────────────────────────────────────────────────────────────────┘
```

---

## 5. 现代深度学习框架中的计算优化

### 5.1 Log-Sum-Exp 技巧

在如 PyTorch 这样的框架中，计算 Loss 时通常不会让你手动先算 Softmax 再算 Log。例如 `torch.nn.CrossEntropyLoss` 会将 Softmax 和 NLLLoss 合并在一起，直接接收**未经激活的 Logits**。

这样做的原因是**数值稳定性（Numerical Stability）**。

**问题**：直接用指数函数计算 Softmax 时，如果某个 $z_i$ 非常大（例如 $z_i = 1000$），则 $e^{z_i} = e^{1000}$ 远超 float64 的表示范围（$\approx 1.8 \times 10^{308}$），产生 **Overflow**。即使不溢出，大数相除也会导致严重的精度损失。

**解决方案**：现代底层算子利用 **Log-Sum-Exp（LSE）技巧**进行数学恒等变换。

首先，将 Softmax 和 Log 合并：

$$
\log(\hat{y}_c) = \log \left( \frac{e^{z_c}}{\sum_{j=1}^{C} e^{z_j}} \right) = z_c - \log \left( \sum_{j=1}^{C} e^{z_j} \right)
$$

关键步骤——从所有 $z_j$ 中减去最大值 $m = \max(z)$：

$$
\log \left( \sum_{j=1}^{C} e^{z_j} \right) = \log \left( e^{m} \sum_{j=1}^{C} e^{z_j - m} \right) = m + \log \left( \sum_{j=1}^{C} e^{z_j - m} \right)
$$

因此：

$$
\log(\hat{y}_c) = z_c - m - \log \left( \sum_{j=1}^{C} e^{z_j - m} \right)
$$

由于 $z_j - m \leq 0$，所以 $e^{z_j - m} \in (0, 1]$，**指数项永远不会溢出**。

### 5.2 梯度的数值稳定性

Log-Sum-Exp 不仅保护了前向传播，还使得反向传播中的梯度更加稳定。

交叉熵关于 Logits 的梯度为：

$$
\frac{\partial L}{\partial z_i} = \hat{y}_i - y_i
$$

这个梯度形式极为简洁——它就是**预测概率与真实标签的差值**。

- 对于正确类别 $c$：$\frac{\partial L}{\partial z_c} = \hat{y}_c - 1$（因为 $y_c = 1$）。由于 $\hat{y}_c < 1$，梯度为负，训练过程会增大 $z_c$。
- 对于错误类别 $i \neq c$：$\frac{\partial L}{\partial z_i} = \hat{y}_i - 0 = \hat{y}_i$。梯度为正，训练过程会减小 $z_i$。

这个梯度公式不会出现梯度消失或爆炸的问题（梯度始终在 $[-1, 1]$ 之间），这就是交叉熵相比 MSE 更适合分类任务的另一个重要原因。

### 5.3 PyTorch 中的对应实现

| PyTorch 类 | 包含的操作 | 输入 | 等价手动操作 |
|-----------|-----------|------|------------|
| `nn.CrossEntropyLoss` | LogSoftmax + NLLLoss | 未经激活的 Logits | `F.nll_loss(F.log_softmax(z, dim=1), target)` |
| `nn.NLLLoss` | 仅 NLLLoss | 已取 log 的概率 | `F.nll_loss(log_prob, target)` |
| `nn.BCEWithLogitsLoss` | Sigmoid + BCE | 未经激活的 Logits | `F.binary_cross_entropy(F.sigmoid(z), target)` |

> **最佳实践**：永远使用 `CrossEntropyLoss` 或 `BCEWithLogitsLoss`，而不是手动先算 Softmax/Sigmoid 再算 Loss。前者在数值上更稳定，计算上也更高效（一次前向传播完成两步操作）。

---

## 6. 代码实战：LeNet-5 与 SimpleRNN 中的 Loss

### 6.1 LeNet-5 中的交叉熵损失

在 `code/l5-lenet5.py` 中，LeNet-5 使用 `CrossEntropyLoss` 进行 MNIST 手写数字分类：

```python
# 损失函数：多分类交叉熵（内部集成了 LogSoftmax + NLLLoss）
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

for epoch in range(num_epochs):
    model.train()
    for batch_idx, (inputs, labels) in enumerate(train_loader):
        optimizer.zero_grad()

        outputs = model(inputs)       # shape: (batch_size, 10) — 这是 Logits
        loss = criterion(outputs, labels)  # 自动完成 Softmax + Log + NLL

        loss.backward()               # 反向传播，计算 ∂L/∂z_i = ŷ_i - y_i
        optimizer.step()              # 更新参数
```

**关键观察**：

- `model(inputs)` 输出的是 **Logits**（未经 Softmax 的原始值），而非概率。
- `labels` 是整数张量（如 `tensor([3, 7, 1, ...])`），PyTorch 内部会自动将其视为 One-hot 处理。
- `CrossEntropyLoss` 一次性完成了 LogSoftmax → NLL 的计算，并利用 Log-Sum-Exp 保证了数值稳定性。

### 6.2 SimpleRNN 中的 MSE 损失

在 `code/l5-rnn.py` 中，SimpleRNN 使用 `MSELoss` 进行正弦波预测（回归任务）：

```python
# 损失函数：均方误差（回归任务）
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.01)

for epoch in range(num_epochs):
    model.train()
    for inputs, targets in train_loader:
        optimizer.zero_grad()

        outputs = model(inputs.float())           # shape: (batch_size, 1) — 预测的 sin 值
        loss = criterion(outputs, targets.float()) # MSE: mean((ŷ - y)²)

        loss.backward()
        optimizer.step()
```

**为什么回归任务用 MSE 而不是交叉熵？**

- 交叉熵要求输出是一个合法的概率分布（非负、求和为 1），而回归任务的输出是任意实数。
- MSE 直接度量预测值与真实值之间的欧氏距离，语义清晰。
- 从 MLE 视角：如果假设观测噪声服从高斯分布 $N(0, \sigma^2)$，则最小化 MSE 恰好等价于 MLE。

### 6.3 分类 vs 回归：Loss 选择对照

```
┌────────────────────┬──────────────────────┬──────────────────────┐
│                    │  LeNet-5 (分类)       │  SimpleRNN (回归)     │
├────────────────────┼──────────────────────┼──────────────────────┤
│ 任务               │ MNIST 10分类          │ 正弦波值预测           │
│ 输出               │ Logits ∈ R^10        │ 标量 ∈ R             │
│ 真实标签           │ 整数 {0,...,9}        │ 连续实数              │
│ 损失函数           │ CrossEntropyLoss     │ MSELoss              │
│ 理论基础           │ 多项分布 MLE          │ 高斯分布 MLE          │
│ 梯度形式           │ ŷ_i - y_i            │ 2(ŷ - y)             │
│ 激活函数           │ 内置 LogSoftmax       │ 无（线性输出）         │
└────────────────────┴──────────────────────┴──────────────────────┘
```

---

## 总结

| 主题 | 核心要点 |
|------|---------|
| **MLE** | 找到使观测数据出现概率最大的参数；取负对数后转化为最小化 NLL |
| **交叉熵** | 分类任务的 NLL；最小化交叉熵 = 最小化 KL 散度 = 拟合真实分布 |
| **BCE** | 二分类的交叉熵；$-\log \hat{y}$ 对错误预测惩罚极重 |
| **CCE** | 多分类的交叉熵；One-hot 编码使其退化为 $-\log \hat{y}_c$ |
| **MSE** | 回归任务的标准损失；等价于高斯噪声假设下的 MLE |
| **Log-Sum-Exp** | 通过减去最大值防止指数溢出，保证前向和反向传播的数值稳定性 |
| **PyTorch 实践** | 优先使用 `CrossEntropyLoss` / `BCEWithLogitsLoss`，不要手动算 Softmax |
