# L1: 使用 Python 编写神经网络

> 课程：AIE310008 人工智能的软件基础（复旦大学）
> 讲者：徐辉 (xuh@fudan.edu.cn)

本章以「用神经网络预测阶乘」为载体，在不依赖任何深度学习框架的前提下，使用纯 Python 从零实现一个前馈神经网络，覆盖模型构建、前向传播、反向传播、参数更新与泛化评估的完整流程。

---

## 目录

1. [学习目标](#1-学习目标)
2. [问题定义与分析](#2-问题定义与分析)
3. [模型构建](#3-模型构建)
4. [模型训练](#4-模型训练)
5. [实验结果与分析](#5-实验结果与分析)
6. [泛化与过拟合](#6-泛化与过拟合)
7. [练习](#7-练习)
- [附录：代码文件索引](#附录代码文件索引)

---

## 1. 学习目标

- 理解神经网络的基本工作原理，包括网络结构、前向传播与反向传播机制。
- 能够阅读并理解使用 Python 编写的基础神经网络模型代码。
- 能够利用神经网络对简单数学函数进行建模与拟合。

---

## 2. 问题定义与分析

### 2.1 问题

给定任意一个自然数 $x$，其阶乘定义为

$$
x! = \prod_{i=1}^{x} i.
$$

目标：构造一个神经网络模型，使其能够由输入 $x$ 预测 $x!$。从数学角度，阶乘函数是自然数集合到自然数集合的映射

$$
f: \mathbb{N} \to \mathbb{N}, \quad f(x) = x!.
$$

### 2.2 对数变换：为什么拟合 log(x!)

直接拟合 $x!$ 会遇到严重的数值问题：阶乘增长极快，$10! = 3\,628\,800$，$20!$ 已接近浮点数上限，输出尺度的剧烈变化使损失函数被大值主导、梯度极不平衡，训练难以稳定。

为缓解这一困难，对目标函数取对数，将问题转化为对 $\log(x!)$ 的预测：

$$
\log(x!) = \log\!\Big(\prod_{i=1}^{x} i\Big) = \sum_{i=1}^{x} \log i.
$$

该变换将乘法转化为加法，显著抑制输出尺度的增长，使网络拟合的是一个增长平缓、量级适中的目标值。预测完成后，再通过指数运算 $\exp(\hat{y})$ 还原为原始阶乘值。因此，神经网络实际学习的是映射

$$
x \;\mapsto\; \log f(x).
$$

### 2.3 一般步骤

用神经网络求解此类函数拟合问题，通常包括三步：

1. **模型构建**：设计输入形式、网络结构与输出表示。
2. **模型训练**：基于训练数据调整参数，最小化预测值与真实值的差异。
3. **效果验证**：评估预测准确度与泛化能力。

---

## 3. 模型构建

### 3.1 网络结构

采用前馈神经网络（feed-forward neural network）：信息自输入层单向传播至输出层。本章模型由一个输入节点、一个隐藏层（含 $n$ 个神经元 $h_1, h_2, \dots, h_n$）以及一个输出节点构成。

隐藏层每个神经元执行非线性变换：

$$
h_i = \sigma(w_{1i} \cdot x + b_{1i}),
$$

其中 $\sigma(\cdot)$ 为激活函数。输出层为线性形式，对所有隐藏层输出加权求和：

$$
\hat{y} = \sum_{i=1}^{n} w_{2i} \cdot h_i + b_2.
$$

### 3.2 激活函数

激活函数为模型引入非线性能力。若不引入激活函数，即使堆叠多个神经元，整个网络仍等价于一个线性模型。

**Sigmoid**

$$
\sigma(z) = \frac{1}{1 + e^{-z}}
$$

输出严格落在 $(0, 1)$ 区间，常被解释为概率值。

**ReLU（Rectified Linear Unit）**

$$
\sigma(z) = \max(0, z)
$$

输入为负时输出零，输入为正时线性增长；计算简单、收敛快，是现代深度网络的主流选择。

### 3.3 Python 实现：模型初始化

使用纯 Python 实现模型，将其封装为类 `FactorialNN`。`__init__` 方法接收 `hidden_size` 参数（默认 10），随机初始化各层权重和偏置。

```python
import random, math

class FactorialNN:
    def __init__(self, hidden_size=10):
        self.hidden_size = hidden_size
        # 输入层 -> 隐藏层：权重 w1 与偏置 b1
        self.w1 = [random.uniform(-1, 1) for _ in range(hidden_size)]
        self.b1 = [random.uniform(-1, 1) for _ in range(hidden_size)]
        # 隐藏层 -> 输出层：权重 w2 与偏置 b2（标量）
        self.w2 = [random.uniform(-1, 1) for _ in range(hidden_size)]
        self.b2 = random.uniform(-1, 1)
```

权重与偏置均初始化为 $[-1, 1]$ 区间内的均匀随机数。初始化区间过大会使前向传播时激活值过大，将非线性函数推入饱和区，反向传播时产生梯度消失；过小或全部相同则使神经元行为雷同，削弱表达能力。

### 3.4 前向传播

`forward` 方法描述数据从输入层经隐藏层到输出层的逐层计算：

```python
def forward(self, x):
    z, h = [], []
    for i in range(self.hidden_size):
        zi = self.w1[i] * x + self.b1[i]     # 线性变换
        z.append(zi)
        h.append(relu(zi))                    # 非线性激活（可换为 sigmoid）
    y = sum(self.w2[i] * h[i] for i in range(self.hidden_size)) + self.b2
    return z, h, y
```

返回值同时包含隐藏层激活前的线性输出 `z`、隐藏层输出 `h` 与预测值 `y`，其中 `z` 与 `h` 在反向传播中用于计算激活函数的导数。

---

## 4. 模型训练

训练的目标是得到一组参数 $\theta$，使预测值 $\hat{y}$ 尽可能接近真实值 $y$。该过程通过**梯度下降**最小化损失函数，并由**反向传播**计算损失对各参数的梯度。

### 4.1 损失函数

回归任务中常用均方误差（MSE）：

$$
L_{\text{MSE}} = \frac{1}{N} \sum_{i=1}^{N} (\hat{y}_i - y_i)^2.
$$

平方运算消除误差正负号影响，并对较大偏差给予更高惩罚；同时保证损失函数处处可导，便于与梯度下降结合。单样本平方误差对预测值的偏导为

$$
\frac{\partial L}{\partial \hat{y}} = \hat{y} - y.
$$

（常数系数 2 通常省略。）

此外，平均绝对误差（MAE）也是一种常用度量，$L_{\text{MAE}} = \frac{1}{N}\sum_i |\hat{y}_i - y_i|$，但其在 $\hat{y}=y$ 处不可导，且梯度仅取 $\pm 1$，不利于精细优化。

### 4.2 反向传播：链式求导

反向传播的本质是利用链式法则，将输出误差逐层向前回传，计算各参数的梯度。本章模型需要计算 $\frac{\partial L}{\partial w_{2i}}, \frac{\partial L}{\partial b_2}, \frac{\partial L}{\partial w_{1i}}, \frac{\partial L}{\partial b_{1i}}$。

**输出层参数**。$w_{2i}$ 的影响路径为 $w_{2i} \to \hat{y} \to L$：

$$
\frac{\partial L}{\partial w_{2i}} = \frac{\partial L}{\partial \hat{y}} \cdot \frac{\partial \hat{y}}{\partial w_{2i}} = (\hat{y} - y) \cdot h_i,
\qquad
\frac{\partial L}{\partial b_2} = \hat{y} - y.
$$

**隐藏层参数**。$w_{1i}$ 的影响路径更长 $w_{1i} \to z_i \to h_i \to \hat{y} \to L$，链式法则给出：

$$
\frac{\partial L}{\partial w_{1i}} = \frac{\partial L}{\partial \hat{y}} \cdot \frac{\partial \hat{y}}{\partial h_i} \cdot \frac{\partial h_i}{\partial z_i} \cdot \frac{\partial z_i}{\partial w_{1i}}.
$$

逐项展开：

$$
\frac{\partial L}{\partial \hat{y}} = \hat{y} - y,\quad
\frac{\partial \hat{y}}{\partial h_i} = w_{2i},\quad
\frac{\partial h_i}{\partial z_i} = \sigma'(z_i),\quad
\frac{\partial z_i}{\partial w_{1i}} = x.
$$

因此

$$
\frac{\partial L}{\partial w_{1i}} = (\hat{y} - y) \cdot w_{2i} \cdot \sigma'(z_i) \cdot x,
\qquad
\frac{\partial L}{\partial b_{1i}} = (\hat{y} - y) \cdot w_{2i} \cdot \sigma'(z_i).
$$

### 4.3 激活函数的导数

**ReLU**：

$$
\frac{d}{dz}\text{ReLU}(z) = \begin{cases} 1, & z > 0 \\ 0, & z < 0 \end{cases}
$$

$z=0$ 处不可导，工程上约定 $\text{ReLU}'(0)=0$。这使 ReLU 在反向传播中呈现「门控」机制：神经元激活（$z>0$）时梯度无衰减通过；未激活（$z \le 0$）时梯度被截断。

**Sigmoid**：由 $\sigma(z) = (1+e^{-z})^{-1}$ 求导可得其简洁形式

$$
\frac{d}{dz}\sigma(z) = \sigma(z)\big(1 - \sigma(z)\big).
$$

由于反向传播时已知隐藏层输出 $h = \sigma(z)$，导数常直接写为 $\frac{dh}{dz} = h(1-h)$。Sigmoid 导数最大值为 0.25，在深层网络中易造成梯度衰减。

### 4.4 参数更新

设学习率为 $\eta$，按梯度下降更新各参数：

$$
w_{2i} \leftarrow w_{2i} - \eta \frac{\partial L}{\partial w_{2i}}, \quad
b_2 \leftarrow b_2 - \eta \frac{\partial L}{\partial b_2},
$$
$$
w_{1i} \leftarrow w_{1i} - \eta \frac{\partial L}{\partial w_{1i}}, \quad
b_{1i} \leftarrow b_{1i} - \eta \frac{\partial L}{\partial b_{1i}}.
$$

学习率是关键超参数：过大则损失震荡甚至发散，过小则收敛缓慢，需结合损失曲线调节。

### 4.5 训练方法实现

`train` 方法对单个样本执行「前向 → 反向 → 更新」一次完整过程：

```python
def relu(x):           return max(0.0, x)
def relu_derivative(z): return 1.0 if z > 0 else 0.0
def sigmoid(x):         return 1.0 / (1.0 + math.exp(-x))
def sigmoid_derivative(h): return h * (1 - h)

def train(self, x, target, lr=0.01):
    z, h, y = self.forward(x)
    dy = y - target                      # dL/dy（省略常数 2）
    mse = dy ** 2
    # 隐藏层梯度：链式法则
    dh = [dy * self.w2[i] * relu_derivative(z[i]) for i in range(self.hidden_size)]
    # Sigmoid 替代写法：dy * self.w2[i] * sigmoid_derivative(h[i])
    # 更新隐藏层 -> 输出层
    for i in range(self.hidden_size):
        self.w2[i] -= lr * dy * h[i]
    self.b2 -= lr * dy
    # 更新输入层 -> 隐藏层
    for i in range(self.hidden_size):
        self.w1[i] -= lr * dh[i] * x
        self.b1[i] -= lr * dh[i]
    return mse
```

### 4.6 数据准备与训练循环

训练样本由 $n \in [1, N]$ 的 $\log(n!)$ 构成，输入做归一化（除以 $N$）以稳定数值尺度：

```python
MAX_N = 10
def log_factorial(n):
    return math.log(math.factorial(n))

training_data = [(n / MAX_N, log_factorial(n)) for n in range(1, MAX_N + 1)]

nn = FactorialNN(hidden_size=10)
for epoch in range(20000):
    total_loss = 0.0
    for x, y in training_data:
        total_loss += nn.train(x, y)
    if epoch % 2000 == 0:
        print(f"Epoch {epoch}, Loss = {total_loss:.4f}")
```

每轮（epoch）遍历全部训练样本并更新参数，通过周期性输出总损失观察收敛情况。训练完成后用 `exp` 将预测的 $\log(n!)$ 还原为阶乘值。

---

## 5. 实验结果与分析

在同一网络结构下，ReLU 与 Sigmoid 两种激活函数在收敛速度与拟合能力上差异显著（取自讲义输出）。

| Epoch | ReLU Loss | Sigmoid Loss |
|------:|----------:|-------------:|
| 0     | 661.51    | 705.75       |
| 2000  | 5.08      | 0.07         |
| 10000 | 4.89      | 0.05         |
| 18000 | 4.89      | 0.04         |

Sigmoid 在该任务上最终损失更低、对 $\log(n!)$ 的拟合更精确，预测还原后的阶乘值与真值更为接近；ReLU 收敛较快但稳定在较高损失平台，预测偏差较大（尤其大 $n$ 处）。这说明激活函数的选择与任务特性、输出尺度密切相关，并非「ReLU 一定优于 Sigmoid」。

---

## 6. 泛化与过拟合

**训练 / 测试划分**。仅观察训练误差无法判断模型真实能力。应将数据划分为训练集与测试集：训练集用于参数优化，测试集由模型未见过的样本构成，用于评估泛化能力。本章代码中，训练集覆盖 $n \in [1, N]$，测试集取 $n \in [N+1, N+5]$ 等训练时未出现的样本。

**过拟合**。当模型容量过大或训练轮数过多时，网络可能记忆训练样本的具体数值而非学到内在规律，表现为训练误差极低、测试误差反而很高。阶乘函数在训练区间外增长极快，泛化测试区间稍作外推即可暴露外推能力的不足，是观察过拟合的直观案例。

实践原则：**训练损失低 ≠ 泛化好**，必须以测试误差为最终评估依据。

---

## 7. 练习

1. 设计神经网络预测第 $x$ 个斐波那契数，并分析效果。
2. 设计神经网络预测 $y = \sin(x)$，分析拟合与外推效果。
3. 比较隐藏层神经元数量（如 2、10、20）对拟合能力的影响。
4. 若将拟合目标改回 $x!$ 本身而非 $\log(x!)$，训练会出现何种现象？如何解释？

---

## 附录：代码文件索引

本目录包含两份独立实现，对应不同激活函数与训练配置，可分别运行观察差异：

| 文件 | 激活函数 | 说明 |
|------|---------|------|
| `lec1-fnn.py` | ReLU | 显式链式求导，训练集与测试集分别输出预测值与误差 |
| `lec1-factorial_nn.py` | Sigmoid | `sigmoid_derivative(h)=h(1-h)`，训练 10000 轮后取 `exp` 还原阶乘 |

两份代码的网络结构、初始化方式与训练流程一致，主要差异在于激活函数及其导数，可作为同一模型在不同非线性单元下的对比实验。
