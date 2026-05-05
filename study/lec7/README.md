# L7: GPU 算子开发与显存层次

> 本节课核心：理解 GPU 的显存层次结构（Global Memory、Shared Memory、Register），掌握 Triton 编程模型如何抽象这些底层硬件细节，并通过向量化乘法、矩阵乘法与 LeNet-5 全连接层替换的实战，理解共享内存优化与 Autotune 的作用。

---

## 课程概览

本节课从 GPU 硬件架构出发，讲解 GPU 的存储层次（Registers → Shared Memory → L1/L2 Cache → Global Memory）及其对算子性能的影响。在此基础上，引入 **Triton**——一种基于 Python 的 GPU Kernel 编程语言，它以"Program"为单位抽象线程块，自动管理 Shared Memory 和内存合并。通过三个由浅入深的代码实战，逐步展示 Triton 的编程范式与优化手段。

---

## 知识点导航

### 第一部分：GPU 架构与显存

| 主题 | 要点 |
|------|------|
| [GPU 架构基础](lec7.md#1-gpu-架构基础) | SM（Streaming Multiprocessor）、Warp（32 线程）、SIMT 执行模型 |
| [显存层次结构](lec7.md#2-gpu-显存层次结构) | Registers（~1 cycle）→ Shared Mem（~20 cycles）→ L1/L2 → Global Mem（~800 cycles）；速度与容量呈反比 |
| [共享内存详解](lec7.md#3-共享内存shared-memory详解) | Block 内共享、Tiling 分块策略、Bank 机制与 Bank Conflict |
| [内存合并](lec7.md#4-内存合并与-bank-conflict) | Coalesced Access 要求同 Warp 线程访问连续地址 |

### 第二部分：Triton 编程模型

| 主题 | 要点 |
|------|------|
| [Triton vs CUDA](lec7.md#51-triton-与-cuda-的关系) | 线程级 vs 程序级抽象；Triton 自动管理 Shared Memory 和内存合并 |
| [核心 API](lec7.md#52-triton-的核心抽象) | `tl.program_id`、`tl.arange`、`tl.load/store`、`tl.dot`、`@triton.jit`、`@triton.autotune` |
| [Grid 与 Program](lec7.md#53-grid-与-program-的对应关系) | 二维 Grid 将输出矩阵划分为 Tile，每个 Program 处理一个 Tile |

### 第三部分：代码实战

| 实战 | 文件 | 关键概念 |
|------|------|---------|
| 向量化乘法 | `code/l7-vecmul.py` | `@triton.jit`、`tl.load/store`、`mask`、`triton.cdiv`、Host-Device 封装 |
| 矩阵乘法 + Autotune | `code/l7-autotune.py` | Naive vs Autotune、`@triton.autotune`、`tl.dot`（Tensor Core）、`num_stages` 软件流水线、`cuda.synchronize()` 计时 |
| LeNet-5 Triton FC | `code/l7-lenet5.py` | 自定义 Triton Linear Kernel、`torch.autograd.Function`（forward + backward）、完整训练流程 |

---

## 代码文件说明

### 1. `code/l7-vecmul.py` — 向量化逐元素乘法

**最简 Triton 入门**：计算 $z_i = x_i \times y_i$，帮助理解 Triton 的基本编程范式。

```
Host 端：                            GPU 端（Kernel）：
  vecmul(x, y)                       vecmul_kernel[grid](...)
    ↓                                  ├─ pid = tl.program_id(0)
  构建 grid = (cdiv(N, BS),)           ├─ offsets = pid*BS + arange(BS)
  调用 kernel                          ├─ mask = offsets < N
    ↓                                  ├─ x, y = tl.load(ptr + offsets)
  返回 out                             ├─ out = x * y
                                       └─ tl.store(ptr + offsets, out)
```

**核心要点**：
- `BLOCK_SIZE=1024`：每个 Program 处理 1024 个元素，共启动 `ceil(N/1024)` 个 Program
- `tl.arange(0, BLOCK_SIZE)` 生成连续偏移，保证内存合并（Coalesced Access）
- `mask` 处理尾部越界，避免非法内存访问

### 2. `code/l7-autotune.py` — 矩阵乘法与自动调参

**核心实战**：对比 Naive 与 Autotune 两种矩阵乘法实现，展示 Triton 自动优化的威力。

**代码结构**：

```python
# 1. Naive 版本：逐元素遍历 K 维度
matmul_naive       # for k in range(K): a*b（标量乘法）

# 2. Autotune 版本：分块 + Tensor Core + 自动搜索
@triton.autotune(  # 自动遍历 5 种 Config，找到最快配置
    configs=[...],
    key=["M", "N", "K"],
)
matmul_autotune    # for i in range(num_k): tl.dot(a, b)（分块矩阵乘法）

# 3. Host 封装 + Benchmark（含 cuda.synchronize 正确计时）
benchmark(...)     # warmup → 多次执行 → 同步 → 取平均
```

**Naive vs Autotune 关键对比**：

| 维度 | Naive | Autotune |
|------|-------|----------|
| K 维遍历 | 逐元素（步长 1），每次加载 1 列 A + 1 行 B | 分块（步长 `BLOCK_K`），每次加载一个 Tile |
| 矩阵乘法 | `a * b`（标量乘法） | `tl.dot(a, b)`（Tensor Core 硬件加速） |
| Shared Memory | 未显式利用 | Triton 编译器自动将 Tile 缓存到 Shared Memory |
| 参数 | 固定 `BLOCK_M=64, BLOCK_N=64` | 自动搜索 5 种 Config 的最优组合 |
| 软件流水线 | 无 | `num_stages` 控制 Load 与 Compute 的重叠程度 |

**Autotune 搜索空间**：

| Config | BLOCK_M | BLOCK_N | BLOCK_K | num_warps | num_stages | 特点 |
|--------|---------|---------|---------|-----------|------------|------|
| 1 | 64 | 64 | 32 | 4 | 2 | 小 Tile，高 Occupancy |
| 2 | 128 | 64 | 32 | 8 | 2 | M 维度更大 |
| 3 | 64 | 128 | 32 | 8 | 2 | N 维度更大 |
| 4 | 128 | 128 | 64 | 8 | 3 | 大 Tile + 3 级流水线 |
| 5 | 256 | 64 | 32 | 8 | 2 | 宽 M，窄 N |

### 3. `code/l7-lenet5.py` — 用 Triton 替换 LeNet-5 的全连接层

**进阶实战**：将 LeNet-5 的 `nn.Linear` 替换为自定义 Triton Kernel 实现的 `TritonLinear`，演示如何将自定义算子融入 PyTorch 训练流程。

**代码结构**：

```python
# 1. Triton Linear Kernel（带 Autotune）
@triton.autotune(configs=[...], key=["in_features", "out_features"])
linear_kernel       # 本质是矩阵乘法 out = x @ W^T + bias

# 2. PyTorch 自定义 Function（实现 forward + backward）
class TritonLinearFunction(torch.autograd.Function):
    forward(ctx, x, w, b)     # 调用 Triton Kernel
    backward(ctx, grad_out)    # 梯度用 PyTorch 原生 matmul 计算

# 3. nn.Module 封装
class TritonLinear(nn.Module):
    weight, bias               # nn.Parameter
    forward(x)                 # TritonLinearFunction.apply(x, weight, bias)

# 4. LeNet-5 网络定义
class LeNet5(nn.Module):
    conv1, conv2, pool         # 仍使用 PyTorch 原生 nn.Conv2d
    fc1 = TritonLinear(400, 120)   # ← 替换为 Triton
    fc2 = TritonLinear(120, 84)    # ← 替换为 Triton
    fc3 = TritonLinear(84, 10)     # ← 替换为 Triton

# 5. MNIST 训练 + 测试（标准 PyTorch 流程）
```

**关键设计决策**：
- **Conv 层保持 PyTorch 原生**：卷积的优化极为复杂，CuDNN 已经高度优化，用 Triton 重写收益不大
- **FC 层替换为 Triton**：全连接层的计算本质是矩阵乘法，Triton 的 Autotune 可以针对具体维度找到优于通用 `nn.Linear` 的配置
- **`torch.autograd.Function`**：通过自定义 forward/backward 将 Triton Kernel 嵌入 PyTorch 的自动微分系统。forward 调用 Triton Kernel 计算，backward 用 PyTorch 原生 `@` 运算计算梯度（因为梯度计算也是矩阵乘法，直接用 PyTorch 即可）
- **`backward` 函数是算子融入训练流程的关键**：只有实现了 `backward`，PyTorch 才能通过 `loss.backward()` → `grad_out` → `backward(ctx, grad_out)` 计算参数梯度，进而 `optimizer.step()` 更新权重。没有 `backward` 的自定义算子只能用于推理（inference），无法参与训练

---

## 课堂授课要点（transcript 摘要）

以下为课堂讲授的关键内容摘录，与笔记相互补充：

### 1. GPU 体系结构与 CUDA 编程回顾

- GPU 由多个 **SM（Streaming Multiprocessor）** 组成，每个 SM 类似于 CPU 上的一个 core，但运算能力简单，只能做基本的运算
- 真正的调度以 **SM** 为单位，一个 SM 可以调度一个或多个 **Block**
- Block 内有多个 **Thread**，维度是三维的（方便矩阵运算的索引）
- Shared Memory 类似于 CPU 中的 **L1 Cache**，访问速度远快于 Global Memory
- 任务在 GPU 上执行需经历完整的数据搬运流程：

```
CPU 端（Host）                        GPU 端（Device）
┌──────────────┐                     ┌──────────────┐
│ 准备数据      │                     │              │
│ (torch.randn) │                     │              │
└──────┬───────┘                     │              │
       │ .cuda()                     │              │
       ▼                             │              │
┌──────────────┐    数据拷贝         │              │
│ 数据在 GPU 上 │ ─────────────────→ │ Global Mem   │
└──────┬───────┘   (CPU → GPU)      │              │
       │ kernel[grid](...)           │              │
       ▼                             │              │
       │ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─→│  Kernel 执行  │
       │        启动（异步）          │  (并行计算)   │
       │                             │              │
       │ .cpu()                      │              │
       ▼                             │              │
┌──────────────┐    数据拷贝         │              │
│ 结果回 CPU   │ ←───────────────── │ Global Mem   │
└──────────────┘   (GPU → CPU)      └──────────────┘
```

- 传统的 CUDA 编程需要编写 C++/CUDA C 代码，开发流程繁琐（手动管理 Shared Memory、线程同步、内存合并等），不利于快速迭代和实验

### 2. Triton 框架与算子开发

- Triton 是一个基于 Python 的 GPU Kernel 编程框架，以 **Python 装饰器**（`@triton.jit`）的形式简化算子开发
- 允许开发者用 Python 编写 Kernel 代码，由 Triton 编译器**自动翻译成高效的 CUDA PTX/SASS 代码**
- Triton 的核心抽象包括：
  - **Program**：对应一个 CUDA Block，是独立的计算单元，以"一组线程"的粒度编程
  - **Mask**：用于处理边界检查，避免越界读写，提高代码简洁性和安全性。Triton 的 `tl.load/store` 接受 `mask` 参数，自动跳过无效索引
  - **Broadcast**：支持向量化和广播操作，通过 `[:, None]` / `[None, :]` 等索引技巧实现维度扩展，简化矩阵运算的实现

### 3. 矩阵乘法算子实现

- 课程通过向量乘法和分块矩阵乘法两个案例，演示了 Triton 的使用方法
- 分块矩阵乘法通过将大矩阵划分为小块（Tile），利用 GPU 的共享内存进行块内计算，解决了单个 Block 的内存容量限制问题
- 介绍了 Triton 的 **Auto-tune** 功能，用于自动寻找最优的 Block 划分和线程数配置，以获得最佳性能

### 4. 算子集成与自动微分

- 讲解了如何将自定义的 Triton 算子封装成 PyTorch 可识别的模块（Layer）
- **实现 `backward` 函数是关键**：只有提供反向传播函数，自定义算子才能支持 PyTorch 的自动微分机制，从而无缝融入神经网络的训练流程（`loss.backward()` → 梯度计算 → 参数更新）

---

## 核心概念速查

### GPU 存储层次

```
速度：Registers (1 cycle) > Shared Mem (20 cycles) > L1 (30 cycles) > L2 (200 cycles) > Global Mem (800 cycles)
容量：Registers (KB) < Shared Mem (48-100 KB) < L1 (32-128 KB) < L2 (MB) < Global Mem (8-80 GB)
可见：线程私有          Block 内共享          SM 私有        全芯片共享        全局可见
```

### Triton 核心 API

| API | CUDA 对应 | 作用 |
|-----|-----------|------|
| `@triton.jit` | `__global__` | 声明 Kernel 函数 |
| `@triton.autotune` | — | 自动搜索最优 Block Size、num_warps、num_stages |
| `tl.program_id(axis)` | `blockIdx` | 获取当前 Program 在 Grid 中的索引 |
| `tl.arange(0, N)` | `threadIdx + blockDim` | 生成 Block 内线程偏移张量 |
| `tl.load(ptr + offsets, mask)` | `data[global_idx]` | 从 Global Memory 加载（自动合并） |
| `tl.store(ptr + offsets, val, mask)` | `data[global_idx] = val` | 写入 Global Memory（自动合并） |
| `tl.dot(a, b)` | Tensor Core MMA | 矩阵乘法，自动利用 Tensor Core |
| `triton.cdiv(a, b)` | `(a + b - 1) / b` | 向上取整除法 |

### Mask 与 Broadcast

Triton 中两个重要的编程范式，课堂上重点强调：

| 机制 | 作用 | 示例 |
|------|------|------|
| **Mask** | 边界检查，避免越界读写 | `mask = offsets < N`; `tl.load(ptr, mask=mask)` — 当数组长度不能被 BLOCK_SIZE 整除时，最后一个 Program 的部分线程需要被禁用 |
| **Broadcast** | 维度扩展，实现外积/矩阵运算 | `a[:, None] * b[None, :]` — 将一维向量扩展为二维，实现列向量 × 行向量的矩阵乘法 |

### Autotune 参数含义

| 参数 | 含义 | 增大的好处 | 增大的代价 |
|------|------|-----------|-----------|
| `BLOCK_M/N` | 输出 Tile 大小 | 每个 Program 做更多工作，减少调度开销 | Shared Memory 占用增大、Occupancy 下降 |
| `BLOCK_K` | K 维 Tile 大小 | 减少循环次数，提高数据复用 | Shared Memory 占用翻倍 |
| `num_warps` | Warp 数量 | 更高并行度 | 每线程寄存器减少 |
| `num_stages` | 流水线级数 | 更好地隐藏内存延迟 | Shared Memory 占用 = stages × 单级 |

---

## 文件索引

| 文件 | 说明 |
|------|------|
| `study/lec7/README.md` | 本文件：课程概览与代码导读 |
| `study/lec7/lec7.md` | 完整笔记：GPU 架构、显存层次、Triton 编程、代码分析 |
| `study/lec7/transcript.md` | 课堂录音转写 |
| `code/l7-vecmul.py` | 向量化逐元素乘法（Triton 入门） |
| `code/l7-autotune.py` | 矩阵乘法 Naive vs Autotune + Benchmark |
| `code/l7-lenet5.py` | LeNet-5 全连接层 Triton 实现 + MNIST 训练 |
| `notes/L7-算子开发.pdf` | 课堂讲义 PDF |

---

## 学习路线建议

```
1. GPU 架构基础
   理解 SM / Warp / Thread 的层次关系与 SIMT 执行模型
   ↓
2. 显存层次
   掌握 Registers → Shared Memory → Global Memory 的速度/容量差异
   理解"为什么 Shared Memory 是算子优化的核心"
   ↓
3. l7-vecmul.py（动手运行）
   理解 Triton 的 Program 抽象、Grid 定义、mask 机制
   ↓
4. l7-autotune.py（动手运行）
   对比 Naive 与 Autotune 的性能差异
   理解 Autotune 搜索空间设计与 Occupancy 权衡
   ↓
5. l7-lenet5.py（动手运行）
   理解 torch.autograd.Function 如何将 Triton Kernel 嵌入训练流程
   ↓
6. 进阶思考
   - 为什么 Conv 层不替换？哪些算子适合用 Triton 优化？
   - Autotune 的 key 设计如何影响缓存命中率？
   - 如何为一个新算子设计合理的 Autotune 搜索空间？
```

---

## 常见问题

### Q: 为什么 Triton 不需要手动管理 Shared Memory？

Triton 编译器在编译时会分析 Kernel 中的数据访问模式，自动将需要复用的数据（如 Tiling 中的 Tile）分配到 Shared Memory 中。在 CUDA 中这需要程序员显式地 `__shared__` 声明并手动 `__syncthreads()` 同步，而 Triton 将这些细节交给了编译器。

### Q: `tl.dot(a, b)` 和 `a * b` 有什么区别？

`a * b` 是逐元素乘法（Hadamard product），而 `tl.dot(a, b)` 是矩阵乘法，Triton 会将其编译为 GPU 的 **Tensor Core MMA 指令**（如 `wmma.mma.sync`），利用专用硬件以 2-8 倍的吞吐量完成计算。

### Q: `torch.cuda.synchronize()` 为什么必须？

GPU Kernel 的启动是**异步**的——Host 端调用后立即返回，实际计算在 GPU 上排队执行。`synchronize()` 会阻塞 Host 端，直到 GPU 完成所有已提交的任务。不调用它，`time.time()` 测量的只是 Kernel 提交开销（微秒级），而非实际执行时间（毫秒级）。

### Q: Triton 的 Autotune 缓存存在哪里？

缓存在本地文件系统的 `~/.triton/autotune/` 目录下，以 `key` 参数的值和 GPU 架构名为文件名。更换 GPU 型号后缓存会自动失效并重新搜索。

### Q: `l7-lenet5.py` 中为什么 backward 不用 Triton？

backward 的计算本质也是矩阵乘法（$grad\_x = grad\_out \times W$、$grad\_W = grad\_out^T \times x$），直接使用 PyTorch 的 `@` 运算即可获得 CuBLAS 优化的高性能。只有 forward 的线性层是"瓶颈算子"（因为我们要演示 Triton 的能力），backward 用 PyTorch 原生实现既简单又高效。

---

## 课后作业

1. **阅读资料**：课后阅读 Triton 官方文档与算子融合（Operator Fusion）相关资料，了解多个算子合并为一个 Kernel 以减少 Global Memory 访问的技术
2. **预习下节课**：下节课将讲解**算子融合**（如 Flash Attention 将 Softmax + Attention QKV 合并为单个 Kernel），思考以下问题：
   - 为什么将多个连续算子融合为一个 Kernel 能提升性能？
   - 算子融合与 Shared Memory 的关系是什么？
   - Triton 中如何实现算子融合？
