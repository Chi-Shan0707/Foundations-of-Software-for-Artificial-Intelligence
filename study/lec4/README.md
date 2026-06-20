# L4: Python 并发编程

> Python 并发模型（线程、进程、协程），数据竞争问题及解决方案。

---

## 目录

1. [并发 vs 并行](#1-并发-vs-并行)
2. [Python 线程（threading）](#2-python-线程threading)
3. [数据竞争与同步](#3-数据竞争与同步)
4. [Python 进程（multiprocessing）](#4-python-进程multiprocessing)
5. [协程（asyncio）](#5-协程asyncio)
6. [OpenMP 并行化](#6-openmp-并行化)
7. [总结与最佳实践](#7-总结与最佳实践)

---

## 1. 并发 vs 并行

### 1.1 核心概念

| 概念 | 定义 | 关键点 | 典型场景 |
|------|------|--------|----------|
| **并发（Concurrency）** | 多个任务在同一时间段内**交替**执行 | 逻辑上的同时，物理上可能串行 | I/O 密集型（网络请求、文件读写） |
| **并行（Parallelism）** | 多个任务在同一时刻**真正同时**执行 | 物理上的同时，需要多核 CPU | CPU 密集型（科学计算、矩阵运算） |

### 1.2 形象类比

```
并发: 一个人同时吃三碗饭
     - 一口吃 A 碗，一口吃 B 碗，一口吃 C 碗
     - 看起来像同时吃三碗，实际是轮流吃

并行: 三个人各自吃一碗饭
     - 三个人同时往嘴里送饭
     - 真正的同时进行
```

### 1.3 Python 中的并发/并行模型

```
┌─────────────────────────────────────────────────────────────────────┐
│                    Python 并发/并行模型对比                         │
├─────────────────────────────────────────────────────────────────────┤
│                                                                     │
│  ┌─────────────────┐  ┌─────────────────┐  ┌─────────────────┐     │
│  │  threading      │  │ multiprocessing │  │  asyncio        │     │
│  │  (多线程)        │  │  (多进程)        │  │  (协程)          │     │
│  └─────────────────┘  └─────────────────┘  └─────────────────┘     │
│         │                     │                     │              │
│         ▼                     ▼                     ▼              │
│  ┌─────────────────┐  ┌─────────────────┐  ┌─────────────────┐     │
│  │  共享内存        │  │  独立内存        │  │  单线程事件循环  │     │
│  │  - 可直接通信    │  │  - 需 IPC 通信   │  │  - 显式让出执行权│     │
│  │  - 需要同步机制  │  │  - 天然隔离      │  │  - 适合 I/O 密集 │     │
│  │  - 有 GIL 限制  │  │  - 绕过 GIL     │  │                   │     │
│  └─────────────────┘  └─────────────────┘  └─────────────────┘     │
│         │                     │                     │              │
│         ▼                     ▼                     ▼              │
│  ┌─────────────────┐  ┌─────────────────┐  ┌─────────────────┐     │
│  │  适用: I/O 密集  │  │  适用: CPU 密集  │  │  适用: 高并发 I/O│     │
│  └─────────────────┘  └─────────────────┘  └─────────────────┘     │
│                                                                     │
└─────────────────────────────────────────────────────────────────────┘
```

---

## 2. Python 线程（threading）

### 2.1 基本用法

```python
import threading
import time

def worker(name: str, duration: float):
    """线程工作函数"""
    print(f"[{name}] 开始执行")
    time.sleep(duration)
    print(f"[{name}] 执行完毕")

# 创建线程
t1 = threading.Thread(target=worker, args=("线程1", 1.0))
t2 = threading.Thread(target=worker, args=("线程2", 0.5))

# 启动线程
t1.start()
t2.start()

# 等待线程结束
t1.join()
t2.join()

print("主线程继续执行")
```

**输出示例**：
```
[线程1] 开始执行
[线程2] 开始执行
[线程2] 执行完毕
[线程1] 执行完毕
主线程继续执行
```

### 2.2 关键方法解析

| 方法 | 作用 | 阻塞？ | 说明 |
|------|------|--------|------|
| `start()` | 启动线程 | 否 | 将线程交给操作系统调度 |
| `join()` | 等待线程结束 | 是 | 主线程在此阻塞，直到目标线程结束 |
| `is_alive()` | 检查线程是否运行中 | 否 | 返回布尔值 |

**join() 的意义**：

```python
t1.start()
t2.start()

t1.join()  # 主线程在这里等待 t1 结束
t2.join()  # 主线程在这里等待 t2 结束

# 两个 join() 都返回后，主线程才会继续执行下面的代码
print("所有线程都执行完毕")
```

### 2.3 线程调度的不确定性

**重要**：`start()` 的**调用顺序**不等于**执行顺序**！

```python
t1.start()  # 先启动
t2.start()  # 后启动
```

上面的代码**不能保证** `t1` 一定先于 `t2` 执行。原因：

- `start()` 只是把线程交给操作系统调度
- 真正谁先运行、谁先抢到 CPU 是**不确定的**
- 操作系统根据调度算法决定

如果需要严格顺序（`t1` 完全结束后再 `t2`）：

```python
t1.start()
t1.join()  # 等待 t1 完全结束
t2.start()  # 再启动 t2
t2.join()
```

### 2.4 Python 的 GIL（全局解释器锁）

**GIL（Global Interpreter Lock）** 是 Python 解释器的线程同步机制：

- 同一时刻只有一个线程能执行 Python 字节码
- 多线程在 Python 中**不能并行利用多核**

**GIL 的影响**：

| 场景 | 影响 | 原因 |
|------|------|------|
| CPU 密集型 | ❌ 加速有限 | GIL 导致线程不能真正并行 |
| I/O 密集型 | ✅ 可以加速 | I/O 时会释放 GIL，其他线程可运行 |

**为什么有 GIL？**

1. **历史原因**：Python 最初不是为多线程设计的
2. **内存管理安全**：Python 的引用计数不是线程安全的
3. **C 扩展兼容**：很多 C 扩展库依赖 GIL

---

## 3. 数据竞争与同步

### 3.1 什么是数据竞争？

**数据竞争（Data Race）**：两个或多个线程同时访问同一块内存，至少有一个是写操作，且没有同步机制。

**经典例子：计数器**

```python
import threading

cnt = 0
N = 10

def task():
    global cnt
    for _ in range(N):
        temp = cnt        # ① 读
        # time.sleep(0.00001)  # 故意延长间隔
        cnt = temp + 1    # ② 写

t1 = threading.Thread(target=task)
t2 = threading.Thread(target=task)

t1.start()
t2.start()

t1.join()
t2.join()

print(cnt)  # 期望值: 20，实际可能小于 20
```

**为什么结果不是 20？**

```
时间线：
t1: 读 cnt=0 → temp=0
t2: 读 cnt=0 → temp=0      ← 两个线程读到相同的旧值！
t1: 写 cnt=1
t2: 写 cnt=1               ← t2 的覆盖了 t1 的增量！
```

### 3.2 竞态窗口（Race Window）

```
正常情况（无竞态）：
┌─────┐     ┌─────┐
│读   │ →   │写   │
└─────┘     └─────┘
  ↑           ↑
原子操作（几乎同时完成）

竞态情况：
┌─────┐               ┌─────┐
│读   │               │写   │
└─────┘               └─────┘
  ↑                     ↑
  └─────────────────────┘
        竞态窗口
     （其他线程可能在此期间修改数据）
```

**GIL 的保护粒度：单条字节码，不是整个操作**

GIL（Global Interpreter Lock）保证任意时刻只有一个线程执行 Python 字节码。但关键在于：GIL 保护的是**单条字节码**的原子性，而非多条字节码组成的复合操作。

`n = n + 1` 在字节码层面被拆解为四条指令：

```
LOAD_GLOBAL  n    # 读取 n 的当前值到栈顶
LOAD_CONST   1    # 加载常量 1
INPLACE_ADD      # 栈顶两值相加
STORE_GLOBAL n   # 将结果写回 n
```

GIL 确保每一条指令完整执行，但线程切换可能发生在**任意两条指令之间**。这就是数据竞争的根本原因：

```
时间线（n 初始为 0，线程 A 和 B 都执行 n = n + 1）：

线程 A: LOAD_GLOBAL n → 读到 0
        ↑ 线程切换！GIL 被转交
线程 B: LOAD_GLOBAL n → 读到 0（A 还没写回）
        LOAD_CONST 1 → INPLACE_ADD → STORE_GLOBAL n → n = 1
        ↑ 线程切换！GIL 回到 A
线程 A: LOAD_CONST 1 → INPLACE_ADD → STORE_GLOBAL n → n = 1（覆盖了 B 的更新）

最终 n = 1，丢失了一次更新。
```


**`sleep()` 的作用**：

- 不加 `sleep`：竞态窗口很短，结果"经常"是 20，但不保证
- 加 `sleep`：延长竞态窗口，更容易暴露问题

### 3.3 锁（Lock）：保护临界区

**临界区（Critical Section）**：访问共享资源的代码段

```python
import threading

cnt = 0
N = 10
lock = threading.Lock()  # 创建锁

def task():
    global cnt
    for _ in range(N):
        with lock:        # 进入临界区
            temp = cnt
            time.sleep(0.00001)  # 即使有 sleep，其他线程也无法进入
            cnt = temp + 1
        # 离开临界区，锁自动释放

t1 = threading.Thread(target=task)
t2 = threading.Thread(target=task)

t1.start()
t2.start()

t1.join()
t2.join()

print(cnt)  # 稳定输出 20
```

### 3.4 锁的正确使用

**错误示例：各自使用不同的锁**

```python
t3 = threading.Thread(target=task_with_lock, args=(threading.Lock(),))
t4 = threading.Thread(target=task_with_lock, args=(threading.Lock(),))
```

**问题**：`t3` 和 `t4` 使用的是**不同的锁对象**，不会互斥！

**正确示例：共享同一把锁**

```python
shared_lock = threading.Lock()  # 全局共享的锁

t5 = threading.Thread(target=task_with_shared_lock, args=(shared_lock,))
t6 = threading.Thread(target=task_with_shared_lock, args=(shared_lock,))
```

### 3.5 锁的工作原理

```
无锁情况：
线程1: [读]────[写]
线程2:      [读]────[写]  ← 交叉执行，产生竞态

有锁情况：
线程1: [🔒获取锁]─[读]─[写]─[🔓释放锁]
线程2:               [🔒等待]──[🔒获取锁]─[读]─[写]─[🔓释放锁]
                            ↑
                    线程2 在这里阻塞，直到线程1 释放锁
```

### 3.6 常见的同步原语

| 原语 | 用途 | 特点 |
|------|------|------|
| `Lock` | 互斥访问 | 一次只允许一个线程 |
| `RLock` | 可重入锁 | 同一线程可多次获取 |
| `Semaphore` | 限制并发数 | 允许指定数量的线程同时访问 |
| `Event` | 线程间通信 | 等待/通知机制 |
| `Condition` | 复杂同步 | 结合锁和通知 |

---

## 4. Python 进程（multiprocessing）

### 4.1 进程 vs 线程

| 特性 | 线程（threading） | 进程（multiprocessing） |
|------|------------------|------------------------|
| **内存** | 共享地址空间 | 独立地址空间 |
| **通信** | 直接读写全局变量 | 需要 IPC（管道、队列等） |
| **创建开销** | 小 | 大 |
| **GIL 限制** | 受 GIL 限制 | 绕过 GIL，真正并行 |
| **数据安全** | 需要同步机制 | 天然隔离 |

### 4.2 基本用法

```python
import multiprocessing

counter = 0  # 主进程的全局变量

def worker(name: str):
    global counter
    print(f"[{name}] start, local counter = {counter}")

    # ⚠️ 这里只会修改当前子进程自己的 counter 副本
    # 不会回写到主进程的 counter
    counter += 1

    print(f"[{name}] end, local counter = {counter}")

if __name__ == "__main__":
    processes = []

    for i in range(2):
        p = multiprocessing.Process(target=worker, args=(f"p{i}",))

        # 这里传入的参数需要时元组，所以必须补一个逗号
        # 即便这里传了  ，所谓”主进程“的全局变量，那么，到def里头也还是一个”右值“传进去

        processes.append(p)
        p.start()

    for p in processes:
        p.join()

    # 主进程的 counter 从未被子进程修改，仍然是 0
    print(f"[main] counter = {counter}")
```

**输出**：
```
[p0] start, local counter = 0
[p0] end, local counter = 1
[p1] start, local counter = 0
[p1] end, local counter = 1
[main] counter = 0
```

### 4.3 进程间内存隔离

```
┌─────────────────────────────────────────────────────────────────┐
│                    进程内存布局对比                              │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  主进程                           子进程1      子进程2           │
│  ┌─────────────────┐            ┌─────────┐  ┌─────────┐       │
│  │ counter = 0     │            │counter=0│  │counter=0│       │
│  └─────────────────┘            └─────────┘  └─────────┘       │
│         │                           │            │             │
│         │                           │            │             │
│         └───────────────────────────┴────────────┘             │
│                     各自独立，互不影响                            │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

**关键理解**：

- 子进程创建时，会**复制**父进程的内存空间
- 子进程中的修改**不会**影响父进程
- 如果需要通信，需要使用专门的 IPC 机制

### 4.4 进程间通信（IPC）

```python
from multiprocessing import Process, Queue

def worker(q: Queue, name: str):
    q.put(f"来自 {name} 的消息")

if __name__ == "__main__":
    q = Queue()

    p1 = Process(target=worker, args=(q, "进程1"))
    p2 = Process(target=worker, args=(q, "进程2"))

    p1.start()
    p2.start()

    p1.join()
    p2.join()

    while not q.empty():
        print(q.get())
```

---

## 5. 协程（asyncio）

### 5.1 什么是协程？

**协程（Coroutine）**：用户态的并发单元，由程序自己控制调度，而不是操作系统。

**核心特点**：

- 协程默认在**一个线程**中运行
- 不依赖多核并行
- 依靠**主动让出执行权**来切换
- 切换通常发生在 `await` 处

### 5.2 协程 vs 线程

| 特性 | 线程 | 协程 |
|------|------|------|
| **调度者** | 操作系统 | 程序/事件循环 |
| **切换时机** | 不可控（操作系统决定） | 可控（`await` 点） |
| **切换开销** | 较大（用户态↔内核态） | 极小（用户态） |
| **内存开销** | MB 级别 | KB 级别 |
| **适用场景** | CPU 密集、多核并行 | I/O 密集、高并发 |

### 5.3 基本用法

```python
import asyncio

async def worker(name: str):
    print(f"[{name}] 开始")
    await asyncio.sleep(1)  # 模拟 I/O 操作
    print(f"[{name}] 结束")

async def main():
    # 创建并发任务
    t1 = asyncio.create_task(worker("协程1"))
    t2 = asyncio.create_task(worker("协程2"))

    # 等待所有任务完成
    await asyncio.gather(t1, t2)

# 运行协程
asyncio.run(main())
```

### 5.4 协程也会出现竞态！

```python
import asyncio

counter = 0
N = 5

async def worker(name: str):
    global counter
    for _ in range(N):
        # 故意把"读-改-写"拆开
        old = counter
        await asyncio.sleep(0)  # 让出执行权
        counter = old + 1
    print(f"[{name}] counter_snapshot = {counter}")

async def main():
    t1 = asyncio.create_task(worker("c1"))
    t2 = asyncio.create_task(worker("c2"))
    await asyncio.gather(t1, t2)
    print(f"expected = {2 * N}, actual = {counter}")

asyncio.run(main())
```

**输出**：
```
[c1] counter_snapshot = 4
[c2] counter_snapshot = 5
expected = 10, actual = 5
```

**原因**：

```
时间线：
c1: 读 counter=0 → old=0
c2: 读 counter=0 → old=0      ← await 时切换到 c2
c2: 写 counter=1
c1: 写 counter=1              ← c1 恢复，覆盖了 c2 的更新！
```

> . 协程 (Asyncio) 是 讲礼貌的“协作式调度” (Cooperative)。如果我们不加上这个`wait asyncio.sleep(0) `，就什么也不会发生。 是我们人为"纵容“了切换。


### 5.5 协程的同步：asyncio.Lock

```python
async def safe_worker(name: str, lock: asyncio.Lock):
    global safe_counter
    for _ in range(N):
        async with lock:  # 使用异步锁
            old = safe_counter
            await asyncio.sleep(0)
            safe_counter = old + 1

async def main():
    lock = asyncio.Lock()
    s1 = asyncio.create_task(safe_worker("s1", lock))
    s2 = asyncio.create_task(safe_worker("s2", lock))
    await asyncio.gather(s1, s2)
    print(f"expected = {2 * N}, actual = {safe_counter}")
```

### 5.6 轮询（Polling）

**轮询**：反复检查任务是否完成

```python
async def polling_demo():
    task = asyncio.create_task(asyncio.sleep(0.2))

    while not task.done():  # 反复检查
        print("[polling] task not done yet")
        await asyncio.sleep(0.05)

    print("[polling] task done")
```

**缺点**：

- 需要不断检查，存在额外开销
- 检查间隔太短浪费资源，太长增加延迟

**推荐方式**：

```python
# 直接 await 任务
await task

# 或使用 gather
await asyncio.gather(task1, task2, task3)
```

---

## 6. OpenMP 并行化

### 6.1 OpenMP 简介

**OpenMP**：面向共享内存并行编程的 API

- 主要用于 C/C++/Fortran
- 通过编译器指令（pragma）实现
- 适合多核 CPU 上的循环并行化

### 6.2 矩阵乘法示例

```c
#include <stddef.h>
#include <omp.h>

void matmul_blocked(const double* A, const double* B, double* C,
                    int M, int K, int N, int BS) {
    // 并行化输出块循环
    #pragma omp parallel for collapse(2) schedule(static)
    for (int ii = 0; ii < M; ii += BS) {      // 行块
        for (int jj = 0; jj < N; jj += BS) {  // 列块
            for (int kk = 0; kk < K; kk += BS) {  // 累加维度块
                // 处理边界
                int i_max = (ii + BS < M) ? ii + BS : M;
                int j_max = (jj + BS < N) ? jj + BS : N;
                int k_max = (kk + BS < K) ? kk + BS : K;

                for (int i = ii; i < i_max; i++) {
                    for (int j = jj; j < j_max; j++) {
                        double sum = C[i*N + j];
                        for (int k = kk; k < k_max; k++) {
                            sum += A[i*K + k] * B[k*N + j];
                        }
                        C[i*N + j] = sum;
                    }
                }
            }
        }
    }
}
```

### 6.3 OpenMP 指令解析

| 指令 | 作用 |
|------|------|
| `#pragma omp parallel for` | 并行化后面的 for 循环 |
| `collapse(2)` | 将两层循环合并成一个并行空间 |
| `schedule(static)` | 静态调度：循环迭代均匀分配给线程 |
| `schedule(dynamic)` | 动态调度：线程完成任务后领取新任务 |

### 6.4 编译与运行

```bash
# 编译
gcc -O3 -fopenmp -shared -fPIC -o libmatmul.so l4-matmul.c

# Python 调用
python l4-bench-matmul.py
```

### 6.5 性能对比

```python
# 测试结果（256×256 矩阵）
Python time:           2.3s
Blocked C (OpenMP):    0.02s   ← 115x 加速
NumPy (BLAS):          0.01s
```

---

## 7. 总结与最佳实践

### 7.1 三种并发模型对比

| 模型 | 适用场景 | 优点 | 缺点 |
|------|---------|------|------|
| **threading** | I/O 密集型 | 简单、共享内存 | GIL 限制、需要同步 |
| **multiprocessing** | CPU 密集型 | 绕过 GIL、真正并行 | 内存开销大、通信复杂 |
| **asyncio** | 高并发 I/O | 轻量、高并发 | 学习曲线、逻辑竞态 |

### 7.2 选择指南

```
┌─────────────────────────────────────────────────────────────────┐
│                    并发模型选择指南                              │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  你的任务是？                                                    │
│                                                                 │
│  ┌─────────────────┐  ┌─────────────────┐  ┌─────────────────┐  │
│  │  CPU 密集型      │  │  I/O 密集型      │  │  高并发 I/O     │  │
│  │  (计算密集)      │  │  (网络/磁盘)     │  │  (成千上万的连接)│  │
│  └─────────────────┘  └─────────────────┘  └─────────────────┘  │
│         │                     │                     │             │
│         ▼                     ▼                     ▼             │
│  ┌─────────────────┐  ┌─────────────────┐  ┌─────────────────┐  │
│  │ multiprocessing │  │  threading      │  │  asyncio        │  │
│  │  或 OpenMP      │  │                 │  │                 │  │
│  └─────────────────┘  └─────────────────┘  └─────────────────┘  │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

### 7.3 同步原语速查

| Python 线程 | Python 协程 | 用途 |
|------------|------------|------|
| `threading.Lock()` | `asyncio.Lock()` | 互斥锁 |
| `threading.RLock()` | - | 可重入锁 |
| `threading.Semaphore()` | `asyncio.Semaphore()` | 信号量 |
| `threading.Event()` | `asyncio.Event()` | 事件 |
| `threading.Condition()` | `asyncio.Condition()` | 条件变量 |
| `queue.Queue()` | `asyncio.Queue()` | 线程/协程安全队列 |

### 7.4 常见陷阱

1. **忘记 join()**：主线程可能提前结束
2. **各自用不同的锁**：锁不住
3. **死锁**：多个线程互相等待对方释放锁
4. **协程中的竞态**：`await` 前后共享变量仍需同步
5. **GIL 幻觉**：Python 多线程不能加速 CPU 密集任务

### 7.5 核心要点总结

1. **并发 ≠ 并行**：并发是逻辑上的同时，并行是物理上的同时
2. **join() 保证结束，不保证正确**：需要配合锁使用
3. **数据竞争的根本原因**：竞态窗口 + 共享状态 + 无同步
4. **进程隔离 vs 线程共享**：进程天然安全，线程需要同步
5. **协程更轻量但不免竞态**：单线程但逻辑上仍有并发问题
6. **OpenMP 适合数值计算**：真正的多核并行

---

## 8. 样例程序使用指南

本目录包含一系列教学样例程序，按学习顺序编号。建议按顺序运行，逐步理解 Python 并发编程的核心概念。

### 8.1 快速开始

```bash
cd study/lec4/

# 按顺序运行样例
python lec4-01_thread.py
python lec4-02_data_race.py
python lec4-03_with_lock.py
python lec4-04_multiprocessing.py
python lec4-05_coroutine.py
python lec4-06_coroutine_race.py
```

### 8.2 样例清单

| 文件 | 主题 | 演示内容 | 运行命令 |
|------|------|----------|----------|
| **lec4-01_thread.py** | 线程基础 | 创建、启动、join() | `python lec4-01_thread.py` |
| **lec4-02_data_race.py** | 数据竞态 | 共享变量冲突 | `python lec4-02_data_race.py` |
| **lec4-03_with_lock.py** | 锁机制 | 无锁 vs 有锁对比 | `python lec4-03_with_lock.py` |
| **lec4-04_multiprocessing.py** | 进程隔离 | 内存隔离、IPC | `python lec4-04_multiprocessing.py` |
| **lec4-05_coroutine.py** | 协程基础 | async/await、并发 | `python lec4-05_coroutine.py` |
| **lec4-06_coroutine_race.py** | 协程竞态 | 协程锁、轮询 vs 事件 | `python lec4-06_coroutine_race.py` |
| **lec4-07_matmul_bench.py** | 性能测试 | Python vs C+OpenMP | 需先编译 C 代码 |

### 8.3 详细的样例说明

#### lec4-01_thread.py - 线程基础

```bash
python lec4-01_thread.py
```

**输出示例**：
```
=== 创建两个线程 ===
[线程1] 开始执行
[线程2] 开始执行
主线程继续执行...
[线程2] 执行完毕
[线程1] 执行完毕
所有线程都执行完毕
```

**学习要点**：
- 如何使用 `threading.Thread()` 创建线程
- `start()` 和 `join()` 的作用
- 线程执行顺序的不确定性

---

#### lec4-02_data_race.py - 数据竞态演示

```bash
python lec4-02_data_race.py
```

**输出示例**：
```
=== 数据竞态演示 ===
期望结果: 20 (两个线程各增加 10 次)

--- 第 1 次运行 ---
期望: 20, 实际: 18, 结果: ✗
  原因：两个线程同时读取到相同的旧值，覆盖了对方的更新

--- 第 2 次运行 ---
期望: 20, 实际: 20, 结果: ✓
  （这次没有发生竞态，但不是保证的！）
```

**学习要点**：
- 数据竞态的产生原因
- 期望结果与实际结果的差异
- 竞态的随机性（多次运行结果不同）

**实验建议**：
- 取消代码中 `time.sleep(0.00001)` 的注释，观察竞态窗口扩大后的效果

---

#### lec4-03_with_lock.py - 使用锁解决竞态

```bash
python lec4-03_with_lock.py
```

**输出示例**：
```
=== 锁的作用演示 ===

【情况1】两个线程，没有锁
期望: 20, 实际: 17, 结果: ✗

【情况2】两个线程，各自使用不同的锁（错误示例）
期望: 20, 实际: 18, 结果: ✗
原因：使用的是不同的锁对象，不会互斥！

【情况3】两个线程，共享同一把锁（正确示例）
期望: 20, 实际: 20, 结果: ✓
原因：两个线程竞争同一把锁，临界区互斥执行！
```

**学习要点**：
- `with lock:` 语句的用法
- 共享同一把锁的重要性
- 锁如何保证临界区的互斥执行

---

#### lec4-04_multiprocessing.py - 进程内存隔离

```bash
python lec4-04_multiprocessing.py
```

**输出示例**：
```
=== 进程内存隔离演示 ===

【情况1】子进程修改不会影响父进程
[p0] start, local counter = 0
[p0] end, local counter = 1
[p1] start, local counter = 0
[p1] end, local counter = 1
[父进程] counter = 0
观察：父进程的 counter 没有被子进程修改

【情况2】使用 Queue 进行进程间通信
从 Queue 接收消息：
  - 来自 p3 的消息
  - p3 完成工作
  - 来自 p4 的消息
  - p4 完成工作
```

**学习要点**：
- 进程间的内存隔离特性
- 子进程修改不影响父进程
- 使用 `multiprocessing.Queue` 进行 IPC

**注意**：Windows 下必须将代码放在 `if __name__ == "__main__"` 下

---

#### lec4-05_coroutine.py - 协程基础

```bash
python lec4-05_coroutine.py
```

**输出示例**：
```
=== 协程基础演示 ===

=== 顺序执行 ===
[任务1] 开始
[任务1] 结束
[任务2] 开始
[任务2] 结束
总耗时约: 0.8 秒

=== 并发执行 ===
[任务1] 开始
[任务2] 开始
[任务2] 结束
[任务1] 结束
总耗时约: 0.5 秒（并行执行）
```

**学习要点**：
- `async def` 和 `await` 的语法
- `asyncio.create_task()` 创建并发任务
- `asyncio.gather()` 等待多个任务
- 顺序执行 vs 并发执行的时间差异

---

#### lec4-06_coroutine_race.py - 协程竞态与同步

```bash
python lec4-06_coroutine_race.py
```

**输出示例**：
```
=== 协程竞态演示 ===

【情况1】协程没有锁保护
[c1] counter_snapshot = 4
[c2] counter_snapshot = 5
期望: 10, 实际: 5, 结果: ✗

【情况2】协程有 asyncio.Lock 保护
[s1] safe_counter_snapshot = 10
[s2] safe_counter_snapshot = 10
期望: 10, 实际: 10, 结果: ✓

=== 轮询示例 ===
[polling] 任务未完成...
[polling] 任务未完成...
[polling] 任务未完成...
[polling] 任务完成！

=== 事件驱动示例 ===
等待任务完成...
任务完成！
```

**学习要点**：
- 协程也会出现数据竞态（单线程但逻辑并发）
- `asyncio.Lock` 的使用方法
- 轮询 vs 事件驱动的对比

---

#### lec4-07_matmul_bench.py - 矩阵乘法性能测试

**前置步骤**：编译 C 代码

```bash
# 进入 code 目录
cd ../../code

# 编译 OpenMP 版本
gcc -O3 -fopenmp -shared -fPIC -o libmatmul.so l4-matmul.c

# 返回 lec4 目录
cd ../study/lec4/
```

**运行测试**：

```bash
python lec4-07_matmul_bench.py
```

**输出示例**：
```
=== 矩阵乘法性能测试 (256×256 × 256×256) ===

【1】纯 Python 实现...
耗时: 2.314s

【2】C + OpenMP 实现...
耗时: 0.020s

【3】NumPy (BLAS) 实现...
耗时: 0.012s

=== 性能对比 ===
纯 Python:        2.314s  (1.0x)
C + OpenMP:       0.020s  (115.7x 加速)
NumPy (BLAS):     0.012s  (192.8x 加速)

=== 正确性验证 ===
Python 最大误差:  0.00e+00
C 实现最大误差:   2.22e-16
Python 正确:     ✓
C 实现正确:      ✓
```

**学习要点**：
- Python 纯实现与 C 实现的性能差距
- OpenMP 多核并行的加速效果
- NumPy 底层 BLAS 的强大性能

---

### 8.4 补充说明

所有示例代码均以 `lec4-NN_*.py` 命名（见 8.1–8.3 节）。多线程部分的三种情况（无锁、带锁、带锁+sleep）集中在 `lec4-03_with_lock.py` 中，可通过修改 `LOCK_MODE` 参数切换。

---

### 8.5 环境要求

- Python 3.7+
- 依赖包：`numpy`（仅 lec4-07_matmul_bench.py 需要）
- gcc 编译器（仅 lec4-07_matmul_bench.py 需要）
- OpenMP 支持（gcc 默认支持）

安装依赖：
```bash
pip install numpy
```

---

### 8.6 常见问题

**Q1: lec4-02_data_race.py 运行结果总是 20，为什么？**

A: 竞态窗口太小，冲突概率低。取消 `time.sleep(0.00001)` 的注释可以放大竞态窗口。

**Q2: Windows 下运行 multiprocessing 报错？**

A: 确保代码在 `if __name__ == "__main__"` 下运行。

**Q3: lec4-07_matmul_bench.py 提示找不到 libmatmul.so？**

A: 需要先编译 C 代码：
```bash
cd ../../code
gcc -O3 -fopenmp -shared -fPIC -o libmatmul.so l4-matmul.c
```

---

## 附录：代码文件索引

| 文件 | 说明 |
|------|------|
| `lec4-01_thread.py` | 线程基础示例 |
| `lec4-02_data_race.py` | 数据竞态演示 |
| `lec4-03_with_lock.py` | 锁机制演示 |
| `lec4-04_multiprocessing.py` | 进程内存隔离演示 |
| `lec4-05_coroutine.py` | 协程基础 |
| `lec4-06_coroutine_race.py` | 协程竞态与同步 |
| `lec4-07_matmul_bench.py` | 矩阵乘法性能测试 |
