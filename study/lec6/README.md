# L6: GPU 编程与 CUDA 详解

> GPU 架构、CUDA 编程模型、矩阵乘法优化（Tiling）及 PyTorch GPU 使用。

---

## 目录

1. [为什么需要 GPU？](#1-为什么需要-gpu)
2. [GPU 架构与 SIMT 执行模型](#2-gpu-架构与-simt-执行模型)
3. [CUDA 编程模型：Grid → Block → Warp → Thread](#3-cuda-编程模型grid--block--warp--thread)
4. [第一个 CUDA 程序：向量逐元素相乘](#4-第一个-cuda-程序向量逐元素相乘)
5. [CUDA 内存管理：Host ↔ Device 数据传输](#5-cuda-内存管理host--device-数据传输)
6. [二维并行：矩阵乘法的 Naive 实现](#6-二维并行矩阵乘法的-naive-实现)
7. [性能瓶颈：全局内存访问](#7-性能瓶颈全局内存访问)
8. [Tiling（分块）优化技术](#8-tiling分块优化技术)
9. [PTX 指令集与可移植性](#9-ptx-指令集与可移植性)
10. [完整基准测试代码解析](#10-完整基准测试代码解析)
11. [在 Kaggle / PyTorch 中使用 GPU](#11-在-kaggle--pytorch-中使用-gpu)
12. [课后实验与扩展](#12-课后实验与扩展)

---

## 1. 为什么需要 GPU？

### CPU vs GPU 的设计哲学

| 特性 | CPU | GPU |
|------|-----|-----|
| 核心数 | 少（几到几十个） | 极多（成千上万个） |
| 单核性能 | 强，高频率 | 弱，低频率 |
| 擅长任务 | 顺序执行、复杂控制流 | 大规模并行计算 |
| 典型应用 | 操作系统、通用计算 | 图形渲染、矩阵运算、深度学习 |

GPU（Graphics Processing Unit）最初为图形渲染设计，但由于其**大规模并行计算**的能力，逐渐被用于通用计算（GPGPU, General-Purpose GPU）。在深度学习中，矩阵乘法、卷积等操作具有天然的并行性，非常适合在 GPU 上加速。

---

## 2. GPU 架构与 SIMT 执行模型

### 2.1 NVIDIA GPU 硬件结构

NVIDIA GPU 由多个 **SM（Streaming Multiprocessor，流式多处理器）** 组成，每个 SM 包含多个 CUDA 核心（core）。

```
GPU
├── SM (Streaming Multiprocessor)
│   ├── core, core, core, ...  (多个 CUDA 核心)
│   ├── 共享内存 (Shared Memory)
│   └── 寄存器文件
├── SM
│   ├── core, core, core, ...
│   └── ...
└── 全局内存 (Global Memory, HBM/GDDR)
```

- **SM** 是 GPU 的基本调度单元，一个 GPU 有数十个 SM。
- 每个 SM 可以同时执行多个 **block**，但有资源上限（如最多 2048 个线程、最多 32 个 block）。

### 2.2 SIMT：单指令多线程

GPU 采用 **SIMT（Single Instruction, Multiple Threads）** 执行模型：

- 同一时刻，一组线程（32 个，称为一个 **Warp**）执行**相同的指令**。
- 如果线程之间出现分支（如 `if-else`），Warp 需要串行执行不同分支，导致**线程发散（Thread Divergence）**，降低效率。

---

## 3. CUDA 编程模型：Grid → Block → Warp → Thread

CUDA 是 NVIDIA 推出的 GPU 编程框架，通过 C/C++ 扩展实现。

### 3.1 四层执行层级

```
Grid                          ← 整个任务
├── Block[0, 0]
│   ├── Warp 0: Thread[0,0], Thread[1,0], ..., Thread[31,0]
│   ├── Warp 1: Thread[32,0], ...
│   └── ...
├── Block[1, 0]
│   └── ...
└── ...
```

| 层级 | 含义 | 关键限制 |
|------|------|----------|
| **Grid** | 所有 Block 的集合 | 一维/二维/三维 |
| **Block** | 一组线程，可共享 Shared Memory | 最多 1024 个线程（三维乘积） |
| **Warp** | 32 个线程为一组，是实际调度单位 | 固定 32 个线程 |
| **Thread** | 最细粒度的执行单元 | 拥有私有寄存器 |

### 3.2 关键内置变量

CUDA Kernel 中可以访问以下内置变量来确定自己的身份：

| 变量 | 含义 |
|------|------|
| `gridDim.x/y/z` | Grid 中 Block 的数量（各维度） |
| `blockIdx.x/y/z` | 当前 Block 在 Grid 中的索引 |
| `blockDim.x/y/z` | 当前 Block 中 Thread 的数量（各维度） |
| `threadIdx.x/y/z` | 当前 Thread 在 Block 中的索引 |

### 3.3 全局线程索引计算

**一维情况**：
```cpp
int i = blockIdx.x * blockDim.x + threadIdx.x;
```

**二维情况**（用于矩阵运算）：
```cpp
int row = blockIdx.y * blockDim.y + threadIdx.y;  // 行号
int col = blockIdx.x * blockDim.x + threadIdx.x;  // 列号
```

---

## 4. 第一个 CUDA 程序：向量逐元素相乘

### 4.1 Kernel 函数

```cpp
#include <cuda_runtime.h>

#define N 1024

__global__ void ele_mul(float* a, float* b, float* c) {
    // 计算全局线程索引
    int i = blockIdx.x * blockDim.x + threadIdx.x;

    // 边界检查，防止越界
    if (i < N) {
        c[i] = a[i] * b[i];
    }
}
```

**关键点**：
- `__global__` 修饰符：表示该函数在 **GPU 上执行**，由 **CPU 调用**。
- 每个线程独立计算一个元素，无需同步（除了启动时的隐式同步）。
- 必须做**边界检查**（`if (i < N)`），因为 Block 数量需要向上取整。

### 4.2 CPU 端调用流程

```cpp
int main() {
    // 1. 在 CPU（Host）上分配和初始化数据
    float *ha, *hb, *hc;
    ha = (float*)malloc(N * sizeof(float));
    hb = (float*)malloc(N * sizeof(float));
    hc = (float*)malloc(N * sizeof(float));
    for (int i = 0; i < N; i++) {
        ha[i] = 2.0f;
        hb[i] = 3.0f;
    }

    // 2. 在 GPU（Device）上分配内存
    float *da, *db, *dc;
    cudaMalloc(&da, N * sizeof(float));
    cudaMalloc(&db, N * sizeof(float));
    cudaMalloc(&dc, N * sizeof(float));

    // 3. 将数据从 Host 拷贝到 Device
    cudaMemcpy(da, ha, N * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(db, hb, N * sizeof(float), cudaMemcpyHostToDevice);

    // 4. 配置并启动 Kernel
    int threadsPerBlock = 256;
    int numBlocks = (N + threadsPerBlock - 1) / threadsPerBlock;  // 向上取整
    ele_mul<<<numBlocks, threadsPerBlock>>>(da, db, dc);

    // 5. 将结果从 Device 拷贝回 Host
    cudaMemcpy(hc, dc, N * sizeof(float), cudaMemcpyDeviceToHost);

    // 6. 释放资源
    cudaFree(da); cudaFree(db); cudaFree(dc);
    free(ha); free(hb); free(hc);
    return 0;
}
```

**CUDA 程序标准六步曲**：
1. Host 内存分配与初始化
2. Device 内存分配（`cudaMalloc`）
3. Host → Device 数据传输（`cudaMemcpy`）
4. 启动 Kernel（`<<<grid, block>>>`）
5. Device → Host 数据传输（`cudaMemcpy`）
6. 释放资源（`cudaFree` / `free`）

### 4.3 二维 Grid / Block 配置

对于矩阵运算，需要使用二维的线程组织：

```cpp
// 定义 16×16 的二维 Block
// dim3 是 CUDA 内置的三维向量类型，缺省 z=1
dim3 threadsPerBlock(16, 16);  // 每个 Block 256 个线程

// 计算 Grid 大小，向上取整
dim3 numBlocks(
    (N + threadsPerBlock.x - 1) / threadsPerBlock.x,  // X 维度
    (N + threadsPerBlock.y - 1) / threadsPerBlock.y   // Y 维度
);

// 启动二维 Kernel
matmul<<<numBlocks, threadsPerBlock>>>(dA, dB, dC);
```

---

## 5. CUDA 内存管理：Host ↔ Device 数据传输

### 5.1 两种内存空间

- **Host Memory**：CPU 主存（DRAM），通过 `malloc` / `free` 管理。
- **Device Memory**：GPU 显存（VRAM），通过 `cudaMalloc` / `cudaFree` 管理。

### 5.2 数据拷贝方向

```cpp
cudaMemcpy(dest, src, size, kind);
```

| `kind` 参数 | 方向 |
|-------------|------|
| `cudaMemcpyHostToDevice` | CPU → GPU |
| `cudaMemcpyDeviceToHost` | GPU → CPU |
| `cudaMemcpyDeviceToDevice` | GPU → GPU |

> **性能提示**：Host ↔ Device 数据传输是 PCI-e / NVLink 总线操作，带宽远低于显存带宽。应尽量减少数据传输次数，尽量让数据在 GPU 上完成全部计算。

---

## 6. 二维并行：矩阵乘法的 Naive 实现

### 6.1 问题描述

计算矩阵乘法 C = A × B，其中 A、B、C 均为 N×N 矩阵：

$$C_{i,j} = \sum_{k=0}^{N-1} A_{i,k} \times B_{k,j}$$

### 6.2 Naive GPU 实现

每个线程负责计算 C 中的一个元素：

```cpp
__global__ void matmul_naive(float* A, float* B, float* C, int N) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < N && col < N) {
        float sum = 0.0f;
        for (int k = 0; k < N; k++) {
            // A 按行访问，B 按列访问
            sum += A[row * N + k] * B[k * N + col];
        }
        C[row * N + col] = sum;
    }
}
```

**问题分析**：
- 每个线程计算一个元素，需要读取 A 的一行和 B 的一列。
- 对于 N×N 的矩阵，每个 A 的元素被读取 N 次（被 N 个不同的线程读取）。
- 所有访问都走**全局内存（Global Memory）**，延迟高、带宽有限。

---

## 7. 性能瓶颈：全局内存访问

### 7.1 GPU 内存层级

| 内存类型 | 速度 | 容量 | 可见范围 |
|----------|------|------|----------|
| 寄存器（Register） | 最快 | 极小 | 单个线程 |
| 共享内存（Shared Memory） | 很快 | 有限（如 48KB/Block） | 同一个 Block 内 |
| 全局内存（Global Memory） | 慢 | 大（显存容量） | 所有线程 |

### 7.2 内存访问模式

- **合并访问（Coalesced Access）**：同一个 Warp 中的线程访问连续的内存地址，可以合并为一次事务，效率最高。
- **非合并访问**：Warp 中线程访问分散的地址，导致多次内存事务，效率低下。

在 Naive 矩阵乘法中：
- A 的访问是连续的（`row * N + k`，k 递增），可以合并。
- B 的访问是跨行的（`k * N + col`，k 递增时跳跃 N 个元素），**无法合并**，成为性能瓶颈。

---

## 8. Tiling（分块）优化技术

### 8.1 核心思想

将大矩阵切分为小的 **Tile（分块）**，每个 Block 负责计算 C 中的一个 Tile。

关键优化：利用 **Shared Memory** 缓存当前 Block 需要的 A 和 B 的数据子集，大幅减少全局内存访问次数。

```
C 的 Tile (TILE_SIZE × TILE_SIZE)
┌─────────┐
│ Block   │  ← 需要 A 的一行 Tile 和 B 的一列 Tile
│ [i,j]   │
└─────────┘

A 按行切分 → 每个 Tile 是 TILE_SIZE × N 的子矩阵
B 按列切分 → 每个 Tile 是 N × TILE_SIZE 的子矩阵
```

### 8.2 Tiled Kernel 详解

```cpp
template <int TILE_SIZE>
__global__ void matmul_tiled(float* A, float* B, float* C, int N) {
    // ---------- 1. 声明共享内存 ----------
    __shared__ float As[TILE_SIZE][TILE_SIZE];  // A 的 Tile 缓存
    __shared__ float Bs[TILE_SIZE][TILE_SIZE];  // B 的 Tile 缓存

    // ---------- 2. 计算当前线程负责的 C 元素位置 ----------
    int row = blockIdx.y * TILE_SIZE + threadIdx.y;
    int col = blockIdx.x * TILE_SIZE + threadIdx.x;

    float value = 0.0f;  // 累加器

    // ---------- 3. 遍历所有 Tile ----------
    for (int t = 0; t < (N + TILE_SIZE - 1) / TILE_SIZE; t++) {

        // ---- 3a. 协作加载 A 的 Tile 到共享内存 ----
        // A[row][t*TILE_SIZE + threadIdx.x]
        if (row < N && t * TILE_SIZE + threadIdx.x < N)
            As[threadIdx.y][threadIdx.x] = A[row * N + t * TILE_SIZE + threadIdx.x];
        else
            As[threadIdx.y][threadIdx.x] = 0.0f;  // 越界填充 0

        // ---- 3b. 协作加载 B 的 Tile 到共享内存 ----
        // B[t*TILE_SIZE + threadIdx.y][col]
        if (col < N && t * TILE_SIZE + threadIdx.y < N)
            Bs[threadIdx.y][threadIdx.x] = B[(t * TILE_SIZE + threadIdx.y) * N + col];
        else
            Bs[threadIdx.y][threadIdx.x] = 0.0f;

        // ---- 3c. 等待 Block 内所有线程完成加载 ----
        __syncthreads();

        // ---- 3d. 在共享内存中计算当前 Tile 的乘积 ----
        for (int k = 0; k < TILE_SIZE; k++) {
            value += As[threadIdx.y][k] * Bs[k][threadIdx.x];
        }

        // ---- 3e. 等待所有线程完成计算，再加载下一个 Tile ----
        __syncthreads();
    }

    // ---------- 4. 将结果写回全局内存 ----------
    if (row < N && col < N) {
        C[row * N + col] = value;
    }
}
```

### 8.3 Tiling 的关键机制

1. **协作加载（Collaborative Load）**：Block 内的所有线程一起把 A 和 B 的各一个 Tile 加载到 Shared Memory。
2. **`__syncthreads()`**：Block 级同步屏障，确保所有线程都完成了数据加载/计算，才能进入下一阶段。
3. **复用数据**：一个 Tile 加载到 Shared Memory 后，Block 内的所有线程都可以高速访问，避免重复从全局内存读取。
4. **越界处理**：对于 N 不是 TILE_SIZE 整数倍的情况，用 `0.0f` 填充，保证计算正确性。

### 8.4 为什么 Tiling 更快？

- **Naive 版本**：每个线程计算一个 C 元素，需要读取 A 的 N 个元素和 B 的 N 个元素，全部走全局内存。
- **Tiled 版本**：将全局内存访问次数减少到 `N / TILE_SIZE` 次，中间计算全部在 Shared Memory 中完成，速度提升数倍到数十倍。

### 8.5 TILE_SIZE 的选择

`TILE_SIZE` 是一个编译时常量（通过模板参数传入），常见取值：

| TILE_SIZE | Block 线程数 | 适用场景 |
|-----------|-------------|----------|
| 8 | 64 | 小矩阵，或 SM 资源受限时 |
| 16 | 256 | 平衡选择，常用 |
| 32 | 1024 | 大矩阵，接近 Block 上限 |

**注意**：Shared Memory 大小有限（通常 48KB 或 96KB），两个 `float` 类型的 32×32 Tile 需要 `2 × 32 × 32 × 4 = 8192` 字节，足够用。但如果 Tile 太大，会导致 Shared Memory 不足，无法同时运行多个 Block。

---

## 9. PTX 指令集与可移植性

### 9.1 CUDA 编译流程

```
CUDA C/C++ (.cu)
      ↓  NVCC 编译器
    PTX (Parallel Thread Execution)  ← 虚拟指令集，机器无关
      ↓  JIT 即时编译
    SASS  ← 实际 GPU 机器码
```

### 9.2 PTX 的作用

- **PTX** 是 NVIDIA 定义的虚拟指令集，类似于 Java 字节码。
- 编译时生成 PTX，运行时由驱动 JIT 编译为当前 GPU 的实际机器码（SASS）。
- **好处**：PTX 可以跨不同代际的 GPU 运行，只要驱动支持 JIT 编译即可。

### 9.3 查看 PTX 代码

```bash
nvcc -ptx ele_mul.cu -o ele_mul.ptx
```

PTX 示例片段（对应 `ele_mul` Kernel）：

```ptx
// 读取参数
ld.param.u64 %rd1, [_Z7ele_mulPfS_S__param_0];  // a
ld.param.u64 %rd2, [_Z7ele_mulPfS_S__param_1];  // b
ld.param.u64 %rd3, [_Z7ele_mulPfS_S__param_2];  // c

// 计算线程索引: tid = blockIdx.x * blockDim.x + threadIdx.x
mov.u32 %r2, %ctaid.x;   // blockIdx.x
mov.u32 %r3, %ntid.x;    // blockDim.x
mov.u32 %r4, %tid.x;     // threadIdx.x
mad.lo.s32 %r1, %r2, %r3, %r4;  // 乘法累加

// 边界检查
setp.gt.s32 %p1, %r1, 1023;
@%p1 bra $L__BB0_2;  // 如果越界，跳转到结束

// 内存访问与计算
mul.wide.s32 %rd5, %r1, 4;  // tid * sizeof(float)
ld.global.f32 %f1, [%rd8];  // 加载 b[tid]
ld.global.f32 %f2, [%rd6];  // 加载 a[tid]
mul.f32 %f3, %f2, %f1;      // a[tid] * b[tid]
st.global.f32 [%rd10], %f3; // 写回 c[tid]

ret;  // 返回
```

**关键 PTX 指令**：
- `mov.u32`：数据移动
- `mad.lo.s32`：乘法累加（multiply-add）
- `ld.global.f32`：从全局内存加载 float
- `st.global.f32`：向全局内存存储 float
- `setp.gt.s32`：比较设置谓词（predicate）
- `bra`：分支跳转

---

## 10. 完整基准测试代码解析

### 10.1 文件：`code/l6-bench_mat.cu`

该文件实现了三种矩阵乘法，并进行性能对比：
1. **CPU 版本**：三重嵌套循环
2. **GPU Naive 版本**：每个线程计算一个元素，直接访问全局内存
3. **GPU Tiled 版本**：使用 Shared Memory 分块优化

```cpp
#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>
#include <sys/time.h>

#define N_SIZE 1024

// ========== CPU 版本 ==========
void matmul_cpu(float* A, float* B, float* C, int N) {
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            float sum = 0.0f;
            for (int k = 0; k < N; k++) {
                sum += A[i * N + k] * B[k * N + j];
            }
            C[i * N + j] = sum;
        }
    }
}

// ========== GPU Naive 版本 ==========
__global__ void matmul_naive(float* A, float* B, float* C, int N) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < N && col < N) {
        float sum = 0.0f;
        for (int k = 0; k < N; k++) {
            sum += A[row * N + k] * B[k * N + col];
        }
        C[row * N + col] = sum;
    }
}

// ========== GPU Tiled 版本（模板支持不同 TILE） ==========
template <int TILE_SIZE>
__global__ void matmul_tiled(float* A, float* B, float* C, int N) {
    __shared__ float As[TILE_SIZE][TILE_SIZE];
    __shared__ float Bs[TILE_SIZE][TILE_SIZE];

    int row = blockIdx.y * TILE_SIZE + threadIdx.y;
    int col = blockIdx.x * TILE_SIZE + threadIdx.x;

    float value = 0.0f;

    for (int t = 0; t < (N + TILE_SIZE - 1) / TILE_SIZE; t++) {
        if (row < N && t * TILE_SIZE + threadIdx.x < N)
            As[threadIdx.y][threadIdx.x] = A[row * N + t * TILE_SIZE + threadIdx.x];
        else
            As[threadIdx.y][threadIdx.x] = 0.0f;

        if (col < N && t * TILE_SIZE + threadIdx.y < N)
            Bs[threadIdx.y][threadIdx.x] = B[(t * TILE_SIZE + threadIdx.y) * N + col];
        else
            Bs[threadIdx.y][threadIdx.x] = 0.0f;

        __syncthreads();

        for (int k = 0; k < TILE_SIZE; k++) {
            value += As[threadIdx.y][k] * Bs[k][threadIdx.x];
        }

        __syncthreads();
    }

    if (row < N && col < N) {
        C[row * N + col] = value;
    }
}

// ========== CPU 计时 ==========
double cpu_time() {
    struct timeval tv;
    gettimeofday(&tv, NULL);
    return tv.tv_sec * 1000.0 + tv.tv_usec / 1000.0;
}

// ========== Main ==========
int main() {
    int N = N_SIZE;
    size_t size = N * N * sizeof(float);

    float *hA = (float*)malloc(size);
    float *hB = (float*)malloc(size);
    float *hC = (float*)malloc(size);

    for (int i = 0; i < N * N; i++) {
        hA[i] = 1.0f;
        hB[i] = 1.0f;
    }

    // CPU 测试
    double t1 = cpu_time();
    matmul_cpu(hA, hB, hC, N);
    double t2 = cpu_time();
    printf("CPU time: %f ms\n", t2 - t1);

    // GPU 初始化
    float *dA, *dB, *dC;
    cudaMalloc(&dA, size);
    cudaMalloc(&dB, size);
    cudaMalloc(&dC, size);

    cudaMemcpy(dA, hA, size, cudaMemcpyHostToDevice);
    cudaMemcpy(dB, hB, size, cudaMemcpyHostToDevice);

    // GPU 事件计时
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    // Naive GPU
    dim3 threads(16, 16);
    dim3 blocks((N + 15) / 16, (N + 15) / 16);

    cudaEventRecord(start);
    matmul_naive<<<blocks, threads>>>(dA, dB, dC, N);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    float time_naive;
    cudaEventElapsedTime(&time_naive, start, stop);
    printf("GPU Naive: %f ms\n", time_naive);

    // 不同 TILE_SIZE 测试（宏技巧）
    #define RUN_TILE(T) \
        { \
            dim3 tpb(T, T); \
            dim3 nb((N + T - 1) / T, (N + T - 1) / T); \
            cudaEventRecord(start); \
            matmul_tiled<T><<<nb, tpb>>>(dA, dB, dC, N); \
            cudaEventRecord(stop); \
            cudaEventSynchronize(stop); \
            float t_ms; \
            cudaEventElapsedTime(&t_ms, start, stop); \
            printf("GPU Tiled (%d): %f ms\n", T, t_ms); \
        }

    RUN_TILE(8);
    RUN_TILE(16);
    RUN_TILE(32);

    #undef RUN_TILE

    cudaFree(dA); cudaFree(dB); cudaFree(dC);
    free(hA); free(hB); free(hC);

    return 0;
}
```

### 10.2 性能对比结果

```
CPU time:         3682.18 ms    (≈ 3.7 秒)
GPU Naive:           1.73 ms    (≈ 2100 倍加速)
GPU Tiled (8):       1.73 ms
GPU Tiled (32):      1.04 ms    (≈ 3500 倍加速)
```

**结论**：
- GPU 相比 CPU 有数量级的加速。
- Tiling 进一步优化了 GPU 性能，更大的 TILE_SIZE（32）效果更明显（因为减少了全局内存访问次数）。
- 但 TILE_SIZE 不能无限增大，受限于 Shared Memory 容量和 Block 线程数上限。

### 10.3 `RUN_TILE` 宏的技巧

```cpp
#define RUN_TILE(T) \
    { \
        dim3 tpb(T, T); \
        dim3 nb((N + T - 1) / T, (N + T - 1) / T); \
        ... \
        matmul_tiled<T><<<nb, tpb>>>(dA, dB, dC, N); \
        ... \
    }
```

- `TILE_SIZE` 是模板参数，必须在**编译时**确定。
- 使用宏在编译时展开为多个版本的函数调用，避免写重复的代码。
- 这是一种常见的 CUDA 编程技巧，用于测试不同参数配置。

---

## 11. 在 Kaggle / PyTorch 中使用 GPU

### 11.1 PyTorch 自动使用 GPU

```python
import torch

# 检测设备
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# 创建模型并移至 GPU
model = LeNet5().to(device)

# 训练循环中，将输入数据也移至 GPU
for inputs, labels in train_loader:
    inputs = inputs.to(device)
    labels = labels.to(device)

    # 前向、反向传播自动在 GPU 上执行
    outputs = model(inputs)
    loss = criterion(outputs, labels)
    ...
```

**要点**：
- `model.to(device)`：将模型参数移到 GPU。
- `tensor.to(device)`：将数据张量移到 GPU。
- 所有参与计算的张量必须在同一设备上（全在 GPU 或全在 CPU），否则会报错。

### 11.2 在 Kaggle Notebook 中运行 CUDA

Kaggle 提供免费的 GPU 环境，可以直接在 Notebook Cell 中编译和运行 CUDA：

```python
# Cell 1: 检查 GPU
!nvidia-smi

# Cell 2: 写入 CUDA 源码
%%writefile ele_mul.cu
#include <cuda_runtime.h>
#define N 1024
__global__ void ele_mul(float* a, float* b, float* c) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < N) c[i] = a[i] * b[i];
}
int main() { ... }

# Cell 3: 编译
!nvcc ele_mul.cu -o ele_mul

# Cell 4: 运行
!./ele_mul

# Cell 5: 查看 PTX
!nvcc -ptx ele_mul.cu -o ele_mul.ptx
```

---

## 12. 课后实验与扩展

### 12.1 推荐实验

1. **修改 TILE_SIZE**：尝试 4、8、16、32、64，观察性能变化，找到最优值。
2. **改变矩阵大小**：测试 N = 256、512、1024、2048 时的加速比。
3. **非方阵乘法**：修改代码支持 M×K 乘以 K×N 的矩阵乘法。
4. **Bank Conflict 分析**：研究 Shared Memory 的 bank conflict 问题，尝试通过 padding 优化。
5. **Warp-level 优化**：了解 `__shfl_sync` 等 Warp 原语，实现更细粒度的数据共享。

### 12.2 常见陷阱

- **忘记 `__syncthreads()`**：在 Tiled 算法中，加载数据和计算之间必须同步，否则读到脏数据。
- **越界访问**：不做边界检查会导致访问非法内存，程序崩溃。
- **Host/Device 数据混淆**：在 CPU 上访问 GPU 指针（或反之）会导致段错误。
- **Kernel 启动失败不报错**：CUDA Kernel 启动是异步的，需要用 `cudaGetLastError()` 或 `cudaDeviceSynchronize()` 检查错误。

### 12.3 进阶方向

- **cuBLAS**：NVIDIA 官方优化的高性能矩阵运算库，生产环境直接使用。
- **CUTLASS**：基于 CUDA C++ 模板的高性能矩阵运算库，可自定义融合算子。
- **Triton**：OpenAI 开发的 Python 级 GPU 编程语言，简化 CUDA 编程。

---

## 附录：代码文件索引

| 文件 | 说明 |
|------|------|
| `code/l6-bench_mat.cu` | 矩阵乘法基准测试（CPU vs GPU Naive vs GPU Tiled） |
| `code/l6-ele_mul.cu` | 向量逐元素相乘的完整 CUDA 示例 |
| `study/lec6/lec6-cuda_kernel.cu` | 向量加法的 CUDA 实现 |
| `study/lec6/lec6-cuda_kernel.cpp` | 向量逐元素乘法的 CUDA C++ 代码 |
| `notes/L6-GPU编程.pdf` | 课程讲义原文 |

---

> **编译命令备忘**：
> ```bash
> nvcc l6-ele_mul.cu -o l6-ele_mul
> nvcc l6-bench_mat.cu -o l6-bench_mat
> nvcc -ptx ele_mul.cu -o ele_mul.ptx  # 生成 PTX
> ```
