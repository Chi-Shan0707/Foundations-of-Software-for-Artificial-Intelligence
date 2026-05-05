# L7: GPU 算子开发与显存层次

> 本节课核心：理解 GPU 的显存层次结构（Global Memory、Shared Memory、Register），掌握 Triton 编程模型如何抽象这些底层硬件细节，并通过向量化乘法与矩阵乘法的实战理解共享内存优化与 Autotune 的作用。

---

## 目录

1. [GPU 架构基础](#1-gpu-架构基础)
2. [GPU 显存层次结构](#2-gpu-显存层次结构)
3. [共享内存（Shared Memory）详解](#3-共享内存shared-memory详解)
4. [内存合并与 Bank Conflict](#4-内存合并与-bank-conflict)
5. [Triton 编程模型](#5-triton-编程模型)
6. [代码实战：向量化乘法](#6-代码实战向量化乘法)
7. [代码实战：矩阵乘法与 Autotune](#7-代码实战矩阵乘法与-autotune)

---

## 1. GPU 架构基础

### 1.1 为什么需要 GPU？

CPU 擅长**低延迟**任务（分支预测、复杂控制流），但核心数量有限（通常 4-64 核）。深度学习的核心计算模式是**大规模并行**的矩阵运算，对延迟不敏感但对**吞吐量（Throughput）**要求极高。GPU 专为高吞吐量设计，拥有数千个轻量级核心。

### 1.2 GPU 的核心组成

```
┌──────────────────────────────────────────────────────────────────┐
│                         GPU Chip                                 │
│                                                                  │
│  ┌─────────────┐  ┌─────────────┐       ┌─────────────┐         │
│  │   SM 0      │  │   SM 1      │  ...  │   SM N      │         │
│  │ ┌─────────┐ │  │ ┌─────────┐ │       │ ┌─────────┐ │         │
│  │ │Warp 0   │ │  │ │Warp 0   │ │       │ │Warp 0   │ │         │
│  │ │Warp 1   │ │  │ │Warp 1   │ │       │ │Warp 1   │ │         │
│  │ │ ...     │ │  │ │ ...     │ │       │ │ ...     │ │         │
│  │ │Warp 31  │ │  │ │Warp 31  │ │       │ │Warp 31  │ │         │
│  │ └─────────┘ │  │ └─────────┘ │       │ └─────────┘ │         │
│  │ Shared Mem  │  │ Shared Mem  │       │ Shared Mem  │         │
│  │ Registers   │  │ Registers   │       │ Registers   │         │
│  └──────┬──────┘  └──────┬──────┘       └──────┬──────┘         │
│         │                │                      │                │
│         └────────────────┼──────────────────────┘                │
│                          ▼                                       │
│  ┌──────────────────────────────────────────────────────┐        │
│  │              L2 Cache（所有 SM 共享）                   │        │
│  └──────────────────────┬───────────────────────────────┘        │
│                          ▼                                       │
│  ┌──────────────────────────────────────────────────────┐        │
│  │              Global Memory（VRAM / 显存）              │        │
│  └──────────────────────────────────────────────────────┘        │
│                                                                  │
└──────────────────────────────────────────────────────────────────┘
```

**关键概念**：

| 概念 | 说明 |
|------|------|
| **SM（Streaming Multiprocessor）** | GPU 的核心计算单元，一个 GPU 有几十到上百个 SM。每个 SM 独立调度和执行线程。 |
| **Warp** | SM 调度的最小单元，由 32 个线程组成。同一 Warp 内的线程必须执行相同指令（SIMT 模型）。 |
| **Thread** | 最小的执行单元，每个线程有独立的寄存器和程序计数器。 |

### 1.3 SIMT 执行模型

GPU 采用 **SIMT（Single Instruction, Multiple Threads）** 模型：同一 Warp 中的 32 个线程在同一时钟周期执行**相同指令**，但操作**不同数据**。

这意味着：
- 如果 Warp 内线程走不同的 `if/else` 分支（**Warp Divergence**），两个分支必须**串行执行**，导致性能减半甚至更差。
- 编写 GPU 算子时应尽量让同一 Warp 内的线程执行相同的代码路径。

---

## 2. GPU 显存层次结构

GPU 的存储器分为多个层次，从快到慢、从小到大排列：

```
                    容量              带宽            延迟
              ┌─────────────┐   ─────────────────────────────
              │  Registers  │   ~数 KB/SM       ~1 cycle
              │  (寄存器)    │   ~19 TB/s
              ├─────────────┤
              │ Shared Mem  │   ~48-100 KB/SM   ~20-30 cycles
              │ (共享内存)   │   ~19 TB/s
              ├─────────────┤
              │   L1 Cache  │   ~32-128 KB/SM   ~30-40 cycles
              ├─────────────┤
              │   L2 Cache  │   ~几 MB/全芯片    ~200+ cycles
              ├─────────────┤
              │  Global Mem │   ~8-80 GB       ~400-800 cycles
              │  (显存/VRAM)│   ~0.5-2 TB/s
              └─────────────┘
```

### 2.1 各层存储器详解

#### 寄存器（Registers）

- **位置**：每个线程私有，位于 SM 内。
- **速度**：最快，1 个时钟周期即可访问。
- **容量**：每个线程通常分配 255 个 32-bit 寄存器。一个 SM 上的所有线程共享寄存器池（如 65536 个），线程数越多，每个线程分到的寄存器越少。
- **生命周期**：随线程存在，线程结束后自动释放。
- **无显式管理**：编译器自动分配，程序员无法直接控制。

> **重要性**：寄存器是 GPU 计算性能的基础。算子优化的重要方向之一就是减少寄存器使用，使 SM 能容纳更多活跃线程，从而提高占用率（Occupancy）。

#### 全局内存（Global Memory / VRAM）

- **位置**：独立于 GPU 芯片外的 DRAM（如 GDDR6、HBM3）。
- **容量**：通常 8-80 GB（H100 有 80 GB HBM3）。
- **速度**：最慢，访问延迟约 400-800 个时钟周期。
- **可见性**：所有线程、所有 SM 都可以访问。
- **用途**：存储模型参数、输入/输出数据、激活值等大块数据。

> **关键瓶颈**：GPU 计算极快（如 H100 算力达 ~2000 TFLOPS），但显存带宽有限（~3 TB/s）。根据 Roofline Model，当计算强度（Arithmetic Intensity, 即 FLOPs / Bytes）低于某阈值时，程序受限于**显存带宽（Memory Bound）**，而非计算能力（Compute Bound）。

#### L1 / L2 Cache

- **L1 Cache**：每个 SM 私有，约 32-128 KB。在 NVIDIA Ampere+ 架构中，L1 Cache 与 Shared Memory 共享同一块物理存储，可通过配置调整二者比例。
- **L2 Cache**：所有 SM 共享，约几 MB。用于缓存最近访问的 Global Memory 数据，减少对显存的直接访问。

#### 共享内存（Shared Memory）

- **位置**：SM 片上（On-chip），每个 SM 独立拥有一块。
- **容量**：通常 48-100 KB/SM（可通过编译参数调整）。
- **速度**：接近寄存器，约 20-30 个时钟周期。
- **可见性**：**同一 Block 内的所有线程共享**，其他 Block 的线程不可见。
- **生命周期**：Block 执行期间存在，Block 结束后自动释放。
- **用途**：作为 Global Memory 与寄存器之间的**用户管理的高速缓冲区**，用于数据复用（tiling/reuse）。

> **共享内存是算子优化的核心武器。** 由于 Global Memory 太慢而寄存器太分散（线程私有），共享内存提供了一个平衡点：足够快、又可以被同 Block 内的线程协同读写。

### 2.2 显存层次总结

```
┌────────────────────────────────────────────────────────────────────┐
│                    GPU 存储层次全景                                   │
├────────────────────────────────────────────────────────────────────┤
│                                                                    │
│  线程私有                                                          │
│  ┌──────────────┐                                                  │
│  │  Registers   │  ← 最快，编译器管理                                │
│  └──────────────┘                                                  │
│                                                                    │
│  Block 内共享                                                       │
│  ┌──────────────┐                                                  │
│  │ Shared Memory│  ← 快，程序员手动管理（Triton 自动管理）              │
│  │ + L1 Cache   │                                                  │
│  └──────────────┘                                                  │
│                                                                    │
│  全局共享                                                          │
│  ┌──────────────┐                                                  │
│  │  L2 Cache    │  ← 硬件自动管理                                    │
│  └──────────────┘                                                  │
│  ┌──────────────┐                                                  │
│  │ Global Memory│  ← 最慢，容量最大                                  │
│  │ (VRAM)       │                                                  │
│  └──────────────┘                                                  │
│                                                                    │
│  数据流动：Global Mem → Shared Mem → Registers → 计算               │
│  优化核心：减少 Global Mem 访问次数，增加 Shared Mem 数据复用         │
│                                                                    │
└────────────────────────────────────────────────────────────────────┘
```

---

## 3. 共享内存（Shared Memory）详解

### 3.1 为什么需要共享内存？

以矩阵乘法 $C = A \times B$ 为例，计算 $C[i,j]$ 需要 $A$ 的第 $i$ 行和 $B$ 的第 $j$ 列做内积。如果直接从 Global Memory 读取：

- 每计算一个 $C[i,j]$，从 Global Memory 读取 $K$ 个元素（$A[i,:]$ 和 $B[:,j]$）。
- 但 $A[i,:]$ 在计算 $C[i,0], C[i,1], \ldots, C[i,N-1]$ 时**被重复读取了 $N$ 次**。
- 同理 $B[:,j]$ 在计算 $C[0,j], C[1,j], \ldots, C[M-1,j]$ 时也被重复读取了 $M$ 次。

**共享内存的作用**：将需要复用的数据块（Tile）从 Global Memory 加载到 Shared Memory 中，然后在 Shared Memory 上完成计算，大幅减少对慢速 Global Memory 的访问次数。

### 3.2 Tiling（分块）策略

```
矩阵 A (M×K)          矩阵 B (K×N)          矩阵 C (M×N)

┌────────────┐       ┌────────────┐       ┌────────────┐
│            │       │            │       │            │
│  ┌──┐      │       │ ┌──┐ ┌──┐ │       │ ┌──┐      │
│  │A0│  ... │       │ │B0│ │B1│ │       │ │C0│  ... │
│  └──┘      │       │ └──┘ └──┘ │       │ └──┘      │
│            │       │            │       │            │
│  计算 C00 = │  ×    │  B 的列     │  =    │  C00 =    │
│  A0 × B0   │       │            │       │  A0·B0   │
│            │       │            │       │            │
└────────────┘       └────────────┘       └────────────┘

步骤：
1. 将 A 的一个 Tile (BLOCK_M × BLOCK_K) 从 Global Mem → Shared Mem
2. 将 B 的一个 Tile (BLOCK_K × BLOCK_N) 从 Global Mem → Shared Mem
3. 在 Shared Mem 上计算部分乘积累加到 Accumulator
4. 重复直到遍历完 K 维度
5. 将结果从 Accumulator 写回 Global Mem 的 C 对应位置
```

### 3.3 共享内存的硬件实现：Bank 机制

共享内存被划分为 **32 个 Bank**（与 Warp 大小一致），每个 Bank 的宽度为 4 字节。连续的 32 个 4-byte 地址分别映射到 32 个 Bank 上。

**Bank Conflict**：如果同一个 Warp 中的多个线程同时访问**同一个 Bank** 的不同地址，这些访问会被**串行化**（称为 Bank Conflict），降低带宽利用率。

| 访问模式 | 示例 | 性能 |
|---------|------|------|
| 无冲突 | 线程 $i$ 访问 Bank $i$ | 1 次事务，最优 |
| 多路冲突 | 多个线程访问同一 Bank 的不同地址 | 串行化，降速 |
| 广播（Broadcast） | 所有线程读取同一 Bank 的同一地址 | 1 次事务，无惩罚 |

**避免 Bank Conflict 的常见技巧**：

- 使用 **Padding**：在共享内存数组中插入填充列，打破地址对齐到相同 Bank 的模式。
- 使用合适的**数据布局**：确保访问模式在 Bank 间均匀分布。
- 在 Triton 中，编译器通常会自动处理 Bank Conflict 优化。

---

## 4. 内存合并与 Bank Conflict

### 4.1 内存合并（Memory Coalescing）

当线程从 Global Memory 读取数据时，硬件会尝试将同一 Warp 中多个线程的内存请求**合并（Coalesce）**为一个或少量事务（Transaction）。

```
良好合并（Coalesced Access）：
  Warp 中的 32 个线程读取连续的 32 个 float
  Thread 0 → addr 0, Thread 1 → addr 4, ..., Thread 31 → addr 124
  硬件将其合并为 1-2 个 128-byte 事务 ✓

未合并（Strided Access）：
  Thread 0 → addr 0, Thread 1 → addr 1024, ..., Thread 31 → addr 31744
  每个线程的请求需要独立的事务 ✗
  带宽利用率极低
```

**规则**：让同一 Warp 内的线程访问**连续的内存地址**，或至少在 128-byte 段内对齐。

### 4.2 Triton 中的内存访问优化

在 Triton 的向量化乘法示例（`code/l7-vecmul.py`）中：

```python
offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
x = tl.load(x_ptr + offsets, mask=mask)
```

- `tl.arange(0, BLOCK_SIZE)` 生成连续偏移量，线程 $i$ 访问地址 `x_ptr + pid * BLOCK_SIZE + i`。
- 同一 program（对应一个 Warp 或多个 Warp）内的线程访问**连续地址**，自动实现内存合并。
- `mask` 参数用于处理数组尾部可能越界的元素，避免非法访问。

---

## 5. Triton 编程模型

### 5.1 Triton 与 CUDA 的关系

| 维度 | CUDA | Triton |
|------|------|--------|
| **编程层级** | 线程级（每个线程单独编程） | 程序级（一组线程作为一个程序执行） |
| **内存管理** | 手动管理 Shared Memory、同步 | 编译器自动管理 Shared Memory |
| **SIMT 细节** | 需要理解 Warp、Bank Conflict | 编译器自动优化 |
| **代码量** | 矩阵乘法 ~200+ 行 | 矩阵乘法 ~50 行 |
| **可移植性** | 绑定 NVIDIA GPU | 支持 NVIDIA/AMD (ROCm) |
| **性能** | 可达到理论峰值 | 接近 CuBLAS 的 90%+ |

### 5.2 Triton 的核心抽象

Triton 将一组线程抽象为一个 **"Program"**，对应 CUDA 中的一个 Thread Block。核心概念：

- **`tl.program_id(axis)`**：当前 Program 在 grid 中的索引，类似 CUDA 的 `blockIdx`。
- **`tl.arange(0, N)`**：生成一个范围为 $[0, N)$ 的张量，表示 Block 内线程的索引，类似 CUDA 的 `threadIdx` + `blockDim` 的组合。
- **`tl.load(ptr + offsets, mask=mask)`**：从指针加载向量数据，自动处理内存合并和边界检查。
- **`tl.store(ptr + offsets, value, mask=mask)`**：向指针存储向量数据。
- **`tl.dot(a, b)`**：执行矩阵乘法，自动映射到 GPU 的 Tensor Core（如果可用）。
- **`@triton.jit`**：JIT 编译装饰器，将 Python 函数编译为 GPU Kernel。
- **`@triton.autotune`**：自动搜索最优参数配置（BLOCK_SIZE、num_warps、num_stages 等）。

### 5.3 Grid 与 Program 的对应关系

```
Host 端定义 Grid：

  grid = (G0, G1)  # 二维 grid

对应 GPU 上的 Program 分布：

  ┌───────────────────────────────────┐
  │  Program(0,0)  Program(1,0)  ...  │  ← axis=1
  │  Program(0,1)  Program(1,1)  ...  │
  │  ...                              │
  │              ↑ axis=0              │
  └───────────────────────────────────┘

  每个 Program 处理输出矩阵 C 的一个 Tile
  tl.program_id(0) → 对应 M 维度的 Block 索引
  tl.program_id(1) → 对应 N 维度的 Block 索引
```

---

## 6. 代码实战：向量化乘法

### 6.1 问题定义

给定两个向量 $x, y \in \mathbb{R}^N$，计算逐元素乘法 $z_i = x_i \times y_i$。

完整代码见 `code/l7-vecmul.py`。

### 6.2 Kernel 分析

```python
@triton.jit
def vecmul_kernel(x_ptr, y_ptr, out_ptr, N, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)                    # 当前 program 的索引

    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)  # 计算全局偏移
    mask = offsets < N                         # 边界检查掩码

    x = tl.load(x_ptr + offsets, mask=mask)   # 从 Global Mem 加载 x
    y = tl.load(y_ptr + offsets, mask=mask)   # 从 Global Mem 加载 y

    out = x * y                               # 逐元素乘法

    tl.store(out_ptr + offsets, out, mask=mask)  # 写回 Global Mem
```

**逐行解读**：

1. **`pid = tl.program_id(0)`**：获取当前 program 的 ID。例如 $N = 10240$，$BLOCK\_SIZE = 1024$，则共有 $\lceil 10240/1024 \rceil = 10$ 个 program，`pid` 取值 $0, 1, \ldots, 9$。

2. **`offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)`**：每个 program 负责处理 `BLOCK_SIZE` 个元素。`tl.arange(0, BLOCK_SIZE)` 生成一个一维张量 `[0, 1, 2, ..., BLOCK_SIZE-1]`，加上 `pid * BLOCK_SIZE` 后得到全局索引。

3. **`mask = offsets < N`**：当 $N$ 不能被 $BLOCK\_SIZE$ 整除时，最后一个 program 会越界。mask 标记哪些索引有效，`tl.load/store` 只处理 `mask=True` 的位置。

4. **`tl.load(x_ptr + offsets, mask=mask)`**：从 Global Memory 加载数据。Triton 编译器会自动将这些 load 指令优化为合并的内存访问，最大化带宽利用率。

### 6.3 Host 端封装

```python
def vecmul(x, y):
    N = x.numel()
    out = torch.empty_like(x)

    grid = lambda meta: (triton.cdiv(N, meta['BLOCK_SIZE']),)  # 向上取整

    vecmul_kernel[grid](x, y, out, N, BLOCK_SIZE=1024)
    return out
```

- **`triton.cdiv(N, BLOCK_SIZE)`**：计算 $\lceil N / BLOCK\_SIZE \rceil$，确保所有元素都被覆盖。
- **`grid = lambda meta: (...)`**：使用 lambda 以便 autotune 可以动态调整 `BLOCK_SIZE` 时 grid 也自动更新。

### 6.4 数据流图

```
Global Memory                    Program 0 (pid=0)           Global Memory
┌─────────────┐                ┌──────────────────────┐     ┌─────────────┐
│ x[0..1023]  │──tl.load()──→ │  x = [x0,...,x1023]  │     │             │
│ y[0..1023]  │──tl.load()──→ │  y = [y0,...,y1023]  │     │             │
└─────────────┘                │  out = x * y          │     │             │
                               │       ↓               │     │             │
                               └───────┼───────────────┘     │             │
                                       │ tl.store()          │             │
                                       └──────────────────→  │ z[0..1023]  │
                                                            │ z[1024..2047│
                                                            │ ...         │
                                                            └─────────────┘
```

---

## 7. 代码实战：矩阵乘法与 Autotune

### 7.1 问题定义

给定矩阵 $A \in \mathbb{R}^{M \times K}$ 和 $B \in \mathbb{R}^{K \times N}$，计算 $C = A \times B$。

完整代码见 `code/l7-autotune.py`。

### 7.2 Naive 版本

```python
@triton.jit
def matmul_naive(a_ptr, b_ptr, c_ptr, M, N, K, BLOCK_M, BLOCK_N):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    for k in range(K):                          # 逐元素遍历 K 维度
        a = tl.load(a_ptr + offs_m[:, None] * K + k, ...)
        b = tl.load(b_ptr + k * N + offs_n[None, :], ...)
        acc += a * b

    tl.store(c_ptr + offs_m[:, None] * N + offs_n[None, :], acc, ...)
```

**问题**：每次循环只加载 1 列 $A$ 和 1 行 $B$，Global Memory 访问次数为 $K$ 次，远未充分利用共享内存。

### 7.3 Autotune 版本（核心改进）

```python
@triton.autotune(
    configs=[
        triton.Config({"BLOCK_M": 64,  "BLOCK_N": 64,  "BLOCK_K": 32}, num_warps=4, num_stages=2),
        triton.Config({"BLOCK_M": 128, "BLOCK_N": 64,  "BLOCK_K": 32}, num_warps=8, num_stages=2),
        triton.Config({"BLOCK_M": 64,  "BLOCK_N": 128, "BLOCK_K": 32}, num_warps=8, num_stages=2),
        triton.Config({"BLOCK_M": 128, "BLOCK_N": 128, "BLOCK_K": 64}, num_warps=8, num_stages=3),
        triton.Config({"BLOCK_M": 256, "BLOCK_N": 64,  "BLOCK_K": 32}, num_warps=8, num_stages=2),
    ],
    key=["M", "N", "K"],
)
@triton.jit
def matmul_autotune(a_ptr, b_ptr, c_ptr, M, N, K, BLOCK_M, BLOCK_N, BLOCK_K):
    # ... 省略与 naive 类似的初始化 ...

    offs_k = tl.arange(0, BLOCK_K)
    num_k = tl.cdiv(K, BLOCK_K)

    for i in range(num_k):                     # 分块遍历 K 维度
        k = i * BLOCK_K + offs_k

        a = tl.load(a_ptr + offs_m[:, None] * K + k[None, :], ...)
        b = tl.load(b_ptr + k[:, None] * N + offs_n[None, :], ...)

        acc += tl.dot(a, b)                    # 使用 Tensor Core 加速

    tl.store(c_ptr + offs_m[:, None] * N + offs_n[None, :], acc, ...)
```

**关键改进**：

| 维度 | Naive | Autotune |
|------|-------|----------|
| K 维遍历粒度 | 逐元素（步长 1） | 分块（步长 BLOCK_K） |
| 矩阵乘法 | `a * b`（标量乘法） | `tl.dot(a, b)`（Tensor Core） |
| 参数选择 | 固定 | 自动搜索最优配置 |
| Pipeline | 无 | `num_stages` 控制软件流水线深度 |

### 7.4 Autotune 机制详解

`@triton.autotune` 的本质是一个**编译期 + 运行期的自动调参系统**，在 Kernel 首次执行时搜索最优配置。

#### 7.4.1 完整工作流

```
┌──────────────────────────────────────────────────────────────────────┐
│                  @triton.autotune 完整工作流                          │
├──────────────────────────────────────────────────────────────────────┤
│                                                                      │
│  1. 首次调用 Kernel（Warmup 阶段）                                     │
│     ┌──────────────────────────────────────────────────────┐         │
│     │  检查缓存：本地磁盘 ~/.triton/autotune/ 下是否已有     │         │
│     │  对应 key=(M,N,K) 的最优配置？                         │         │
│     └──────────────────┬───────────────────────────────────┘         │
│                    │                                                 │
│           ┌────────┴────────┐                                        │
│           ▼                 ▼                                        │
│     [命中缓存]         [未命中]                                       │
│     直接使用最优         遍历 configs 列表：                           │
│     配置执行            对每个 Config：                                │
│                          ├─ 编译 Kernel（JIT → PTX → SASS）          │
│                          ├─ 在实际 GPU 上运行多次（默认 25 次）       │
│                          ├─ 记录平均执行时间                           │
│                          └─ 标记最快的 Config 为 Best                  │
│                                                                      │
│  2. 后续调用（生产阶段）                                               │
│     ┌──────────────────────────────────────────────────────┐         │
│     │  直接使用已缓存的最优配置，零额外开销                    │         │
│     └──────────────────────────────────────────────────────┘         │
│                                                                      │
└──────────────────────────────────────────────────────────────────────┘
```

> **注意**：代码中的 `run_autotune(a, b)`（第 169 行）就是 warmup 调用。它在 benchmark 循环之前执行，目的是**提前触发 JIT 编译和 autotune 搜索**，避免将编译时间计入 benchmark 结果。

#### 7.4.2 `key` 参数与缓存键

`key=["M", "N", "K"]` 的含义：当 Kernel 函数接收到的参数 `M, N, K` 发生变化时，视为不同的优化问题，触发新的搜索。

```
  第一次调用：matmul_autotune[grid](a, b, c, M=1024, N=1024, K=1024)
  → key = (1024, 1024, 1024)
  → 触发 autotune 搜索，找到最优 Config，缓存

  第二次调用：matmul_autotune[grid](a, b, c, M=1024, N=1024, K=1024)
  → key = (1024, 1024, 1024)  ← 命中缓存，直接复用

  第三次调用：matmul_autotune[grid](a, b, c, M=2048, N=1024, K=512)
  → key = (2048, 1024, 512)   ← 未命中，重新搜索
```

**设计要点**：`key` 应选择影响最优配置的因素，通常是矩阵形状。如果只选 `["M"]`，则不同 `(N, K)` 但相同 `M` 的调用会复用同一个配置，可能不是最优。

#### 7.4.3 参数含义与硬件约束

每个 `triton.Config` 中的参数都对应 GPU 的硬件资源限制：

| 参数 | 含义 | 增大的好处 | 增大的代价 |
|------|------|-----------|-----------|
| `BLOCK_M` | M 维度 Tile 大小 | 每个 Program 计算更多输出元素，减少 Program 数量 | Shared Memory 占用增大（$O(BLOCK\_M \times BLOCK\_K)$）、寄存器压力增大 |
| `BLOCK_N` | N 维度 Tile 大小 | 同上 | 同上（$O(BLOCK\_K \times BLOCK\_N)$） |
| `BLOCK_K` | K 维度 Tile 大小 | 每轮循环计算更多乘积累加，减少循环次数 | Shared Memory 占用翻倍（$O(BLOCK\_M \times BLOCK\_K + BLOCK\_K \times BLOCK\_N)$） |
| `num_warps` | 每个 Program 的 Warp 数 | 更高并行度，计算吞吐更大 | 每个线程分到的寄存器减少，可能限制 BLOCK_SIZE |
| `num_stages` | 流水线级数 | 更好地隐藏内存延迟 | Shared Memory 占用 = `num_stages × 单级占用`，可能超出 SM 容量 |

**Shared Memory 预算约束**（以 float32 为例）：

$$
\text{Shared Mem per stage} = BLOCK\_M \times BLOCK\_K \times 4 + BLOCK\_K \times BLOCK\_N \times 4 \text{ bytes}
$$

$$
\text{Total Shared Mem} = num\_stages \times \text{Shared Mem per stage}
$$

例如 `BLOCK_M=128, BLOCK_N=128, BLOCK_K=64, num_stages=3`：

$$
\text{per stage} = 128 \times 64 \times 4 + 64 \times 128 \times 4 = 65536 \text{ bytes}
$$

$$
\text{total} = 3 \times 65536 = 196608 \text{ bytes} \approx 192 \text{ KB}
$$

而 Ampere A100 的 SM 只有 164 KB Shared Memory → **这个配置在 A100 上会编译失败**（OCC 驱动会报 shared memory exceeded）。因此 autotune 在搜索时会自动跳过超出硬件限制的配置。

#### 7.4.4 Occupancy（占用率）与参数选择的权衡

**Occupancy** 定义为：每个 SM 上同时活跃的 Warp 数占硬件最大支持 Warp 数的比例。Occupancy 越高，GPU 资源利用率越高。

Occupancy 受以下因素制约（取瓶颈项）：

```
┌─────────────────────────────────────────────────────────────────┐
│              Occupancy 的三个制约因素                               │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  1. Shared Memory                                                │
│     每个 SM 只有 ~48-164 KB                                      │
│     Shared Mem 用得越多 → 能同时运行的 Block 数越少                 │
│                                                                 │
│  2. Registers                                                    │
│     每个 SM 只有 ~65536 个 32-bit 寄存器                          │
│     BLOCK_SIZE 越大 → 每个线程需要的寄存器越多                     │
│     → 能同时运行的 Warp 数越少                                     │
│                                                                 │
│  3. 最大 Block / Warp 数                                          │
│     每个 SM 最多运行 32 个 Block 或 64 个 Warp（A100）             │
│     这是硬上限，无法突破                                           │
│                                                                 │
│  实际 Occupancy = min(受限于 Shared Mem, 受限于 Registers, 硬上限)  │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

**不存在"越大越好"的参数**。例如：
- 增大 `BLOCK_M` 和 `BLOCK_N` → 每个 Program 做更多工作 → Shared Memory 占用上升 → Occupancy 下降
- 但如果 Matrix 很小（如 $128 \times 128$），Grid 中 Program 数量本来就少，Occupancy 天然不足，此时**增大 Tile 反而有利**（减少调度开销）。
- 如果 Matrix 很大（如 $8192 \times 8192$），Grid 中 Program 数量充足，此时应**减小 Tile 以提高 Occupancy**。

这就是为什么需要 autotune——最优配置取决于矩阵形状和硬件能力的**复杂交互**，手工调参几乎不可能穷举。

#### 7.4.5 搜索空间的设计原则

在实际的算子开发中，`configs` 列表的设计是一门经验性学问：

**好的搜索空间**应满足：
1. **覆盖不同的资源瓶颈**：既有小 Tile（高 Occupancy）配置，也有大 Tile（高数据复用）配置。
2. **控制搜索数量**：每个 Config 需要编译 + benchmark，搜索空间太大会显著增加首次编译时间。实践中 10-30 个 Config 是合理的范围。
3. **考虑硬件特性**：`BLOCK_K` 通常是 16/32/64 的倍数（对应 Tensor Core 的 MMA 指令宽度）；`num_warps` 通常是 2 的幂。

**本课代码中的搜索空间分析**：

```python
configs=[
    # 小 Tile，低资源占用，高 Occupancy
    triton.Config({"BLOCK_M": 64,  "BLOCK_N": 64,  "BLOCK_K": 32}, num_warps=4,  num_stages=2),
    # 中等 Tile，M 维度更大
    triton.Config({"BLOCK_M": 128, "BLOCK_N": 64,  "BLOCK_K": 32}, num_warps=8,  num_stages=2),
    # 中等 Tile，N 维度更大
    triton.Config({"BLOCK_M": 64,  "BLOCK_N": 128, "BLOCK_K": 32}, num_warps=8,  num_stages=2),
    # 大 Tile，高数据复用，三级流水线
    triton.Config({"BLOCK_M": 128, "BLOCK_N": 128, "BLOCK_K": 64}, num_warps=8,  num_stages=3),
    # 宽 M，窄 N
    triton.Config({"BLOCK_M": 256, "BLOCK_N": 64,  "BLOCK_K": 32}, num_warps=8,  num_stages=2),
]
```

这 5 个 Config 覆盖了从小到大的 Tile 尺寸，从 2 级到 3 级流水线，以及不同的 M/N 比例。

#### 7.4.6 Benchmark 代码分析

回到 `code/l7-autotune.py` 中的 benchmark 函数：

```python
def benchmark(fn, a, b, iters=10):
    torch.cuda.synchronize()         # ← 关键！等待 GPU 所有任务完成
    start = time.time()
    for _ in range(iters):
        fn(a, b)
    torch.cuda.synchronize()         # ← 关键！等待最后一次 Kernel 完成
    return (time.time() - start) / iters
```

**为什么要 `torch.cuda.synchronize()`？**

GPU 执行是**异步**的。Host 端调用 `kernel[grid](...)` 后立即返回，实际计算在 GPU 上排队异步执行。如果不 `synchronize`，`time.time()` 测量的是**Kernel 启动时间**而非**Kernel 执行时间**。

```
错误的计时（无 synchronize）：
  Host:  t0 = time.time()
  Host:  kernel()        → 立即返回（只是提交任务到 GPU 队列）
  Host:  t1 = time.time()
  Host:  elapsed = t1 - t0  ← 测量的是 Kernel 提交开销（微秒级）

  GPU:   ... 还在执行 kernel ...  ← 但 Host 已经在计时了

正确的计时（有 synchronize）：
  Host:  torch.cuda.synchronize()  ← 等待 GPU 完成之前的任务
  Host:  t0 = time.time()
  Host:  kernel()
  Host:  kernel()
  Host:  torch.cuda.synchronize()  ← 等待所有 kernel 完成
  Host:  t1 = time.time()
  Host:  elapsed = t1 - t0  ← 测量的是真正的 Kernel 执行时间
```

#### 7.4.7 Autotune 的局限与进阶方向

| 局限 | 说明 | 进阶方向 |
|------|------|---------|
| **搜索空间需手工设计** | `configs` 需要开发者根据经验提供 | 使用自动搜索策略（如遗传算法、贝叶斯优化）来探索更大的空间 |
| **首次调用开销大** | 需要编译所有 Config 并 benchmark | 缓存机制已解决重复调用的问题；JIT 编译可预编译部署 |
| **缓存可能失效** | GPU 型号更换后缓存不适用 | Triton 使用 GPU 架构名作为缓存 key 的一部分 |
| **评估精度有限** | 默认 25 次 benchmark 可能不够稳定 | 可通过 `bench_fn` 自定义 benchmark 逻辑 |
| **不能自动发现新算法** | 只能在给定算法框架内调参 | 算法结构本身的优化（如 Flash Attention 的 IO-aware 算法）仍需人类设计 |

> **课程中的实践意义**：在 `main()` 函数中，`run_autotune(a, b)` 首次调用时会触发 autotune，整个过程可能需要几秒到十几秒。这就是为什么 warmup 步骤必不可少——在部署场景中，这部分开销可以通过持久化缓存（`~/.triton/autotune/`）来消除。

### 7.5 软件流水线（Software Pipelining）

`num_stages` 控制 Triton 生成的软件流水线深度。其核心思想是**让内存加载和计算重叠执行**：

```
无流水线（num_stages=1）：
  时间 →
  ┌───────┐ ┌───────┐ ┌───────┐
  │Load 0 │ │Load 1 │ │Load 2 │ ...
  └───────┘ └───┬───┘ └───┬───┘
                 │       │
                 ▼       ▼
              ┌─────┐ ┌─────┐
              │Dot 0│ │Dot 1│ ...
              └─────┘ └─────┘

多级流水线（num_stages=3）：
  时间 →
  ┌───────┬───────┬───────┐
  │Load 0 │Load 1 │Load 2 │Load 3 ...
  └───┬───┴───┬───┴───┬───┘
      │       │       │
      ▼       ▼       ▼
  ┌─────┐ ┌─────┐ ┌─────┐
  │Dot 0│ │Dot 1│ │Dot 2│ ...
  └─────┘ └─────┘ └─────┘

  → 内存加载与计算并行执行，延迟被隐藏
```

流水线级数越高，内存延迟被计算重叠得越充分，但需要更多 Shared Memory 来缓存预取的数据块。

**不同 `num_stages` 的 Shared Memory 开销对比**（以 $BLOCK\_M=128, BLOCK\_N=128, BLOCK\_K=32$, float32 为例）：

| num_stages | Shared Mem 占用 | Global Mem 访问效率 | 适用场景 |
|:----------:|:--------------:|:------------------:|---------|
| 1 | 32 KB | 低（串行） | Shared Mem 紧张的简单算子 |
| 2 | 64 KB | 中等 | 大多数情况的默认选择 |
| 3 | 96 KB | 高 | Shared Mem 充足、计算密集型算子 |
| 4 | 128 KB | 很高 | 大矩阵、高性能需求 |

> 注意：Shared Mem 占用与 `num_stages` 线性增长。如果总占用超出 SM 的 Shared Memory 容量，该 Config 会被 autotune 自动淘汰。

---

## 总结

| 主题 | 核心要点 |
|------|---------|
| **GPU 架构** | SM 是核心计算单元；Warp（32 线程）是最小调度单位；SIMT 模型要求同 Warp 线程执行相同指令 |
| **显存层次** | Registers > Shared Mem > L1 > L2 > Global Mem；速度与容量呈反比 |
| **Global Memory** | 容量大（GB级）但最慢（~800 cycles），带宽是主要瓶颈 |
| **Shared Memory** | SM 片上高速存储（~20 cycles），Block 内线程共享，是 Tiling 优化的核心 |
| **Memory Coalescing** | 同 Warp 线程应访问连续地址，以合并为少量事务 |
| **Bank Conflict** | 同 Warp 线程访问同一 Bank 不同地址会串行化，应通过 Padding 等方法避免 |
| **Triton** | 以 Program 为抽象单位，自动管理 Shared Memory、合并内存访问、优化 Bank Conflict |
| **Autotune** | 首次调用时遍历 configs 搜索最优配置，通过 `key` 缓存；需平衡 Tile 大小、Occupancy、Shared Mem 约束 |
| **Software Pipelining** | 通过 `num_stages` 使内存加载与计算重叠，隐藏内存延迟；级数越高延迟隐藏越好但 Shared Mem 开销越大 |
| **GPU 异步执行** | Host 调用 Kernel 后立即返回，必须用 `cuda.synchronize()` 才能准确计时 |
