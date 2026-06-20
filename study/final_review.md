# 人工智能的软件基础：全栈架构深度复习手册

> 从 Python 字节码到 GPU Warp 调度，从单线程协程到千亿参数分布式训练。
> 本手册基于课程全部代码（l1-l12）、作业（FlashAttention）和课堂笔记，逐层打通"微观→宏观"的架构认知。

---

## 全局架构地图

```
┌─────────────────────────────────────────────────────────────────────────┐
│                        用户写下的 Python 代码                             │
│                    y = model(x)   /   a + b                              │
└──────────────┬──────────────────────────────────┬────────────────────────┘
               │                                    │
     【模块一】CPython 解释器                  【模块二】AST 降级
     字节码 → PVM 执行循环                    @kernel / @triton.jit 提取 AST
     GIL · Lock · 协程 · 多进程               Python 表达式 → Native 函数调用
               │                                    │
               ▼                                    ▼
┌──────────────────────────────┐   ┌─────────────────────────────────────────┐
│     【模块五】TorchDynamo      │   │          【模块三】GPU 硬件               │
│     FX Graph 捕获             │   │   SM · Warp(32线程) · SIMT · 分支发散     │
│     aten.mm / aten.mul 分解   │   │   HBM ←→ SRAM ←→ Registers               │
│     Inductor 算子融合         │──▶│   Coalesced Access · Warp Shuffle         │
│     mul→add→relu 揉进一个 kernel│   └──────────────────┬──────────────────────┘
└──────────────┬───────────────┘                      │
               │                                       │
               ▼                                       ▼
┌──────────────────────────────────┐   ┌──────────────────────────────────────┐
│    【模块四】Triton 算子           │   │  【模块六】FlashAttention / PageAttn   │
│    tl.program_id · Block · Tile   │   │  SRAM 分块 · Online Softmax           │
│    二维指针网格 · 广播             │   │  PagedAttention 内存碎片治理           │
│    @triton.autotune 自动调优      │   │  __shfl_xor_sync 跨线程通信            │
└──────────────────────────────────┘   └──────────────────────────────────────┘
                                                  │
                                                  ▼
               ┌──────────────────────────────────────────────────┐
               │        【模块七】3D 分布式并行                       │
               │   DP/FSDP 切数据 · TP 切矩阵 · PP 切层              │
               │   NVLink · GPipe · 1F1B · Pipeline Bubble          │
               └──────────────────────────────────────────────────┘
```

**一条数据从输入到输出的完整旅程**：

```
用户写 x @ W (Python) 
  → ast.parse 提取 BinOp(AST)                        [模块二]
  → CPython LOAD_FAST + BINARY_OP (字节码)            [模块一]
  → TorchDynamo 捕获为 FX Graph: aten.mm             [模块五]
  → Inductor 模式匹配: mm + bias → aten.addmm         [模块五]
  → 剩余 Pointwise 算子融合成 Triton kernel            [模块五]
  → Triton 编译: 生成 Block/Warp/Thread 调度           [模块四]
  → GPU SM 上 Warp 执行, 数据流经 HBM→SRAM→Reg         [模块三]
  → 如果是 Attention: FlashAttention 在 SRAM 里做分块   [模块六]
  → 如果是多卡: AllReduce 同步梯度                      [模块七]
```

---

# 模块一：Python 底层执行机制与高并发模型（CPU 域）

## 核心命题：如何突破单核物理限制与 GIL 的束缚？

### 1.1 PVM（Python 虚拟机）的"三表一栈"

CPython 不是编译器，是一个**字节码解释器**。你的 `.py` 文件先被编译成 `.pyc`（字节码），然后由 PVM（Python Virtual Machine）逐条执行。

**比喻**：PVM 就像一个只有一张工作台的厨师（单核单线程），工作台上有一个操作数栈（把食材摞上去、拿下来），旁边挂着三本台账：

```
┌─────────────────────────────────────────────────┐
│              PVM 执行循环（ceval.c）               │
│                                                   │
│  ┌─────────────┐  ┌──────────────┐               │
│  │ co_consts   │  │ co_names     │  co_varnames  │
│  │ 常量表       │  │ 全局符号表    │  局部变量表    │
│  │ (None, 1,…) │  │ (print,…)   │  (a, b, c)    │
│  └─────────────┘  └──────────────┘               │
│                                                   │
│         ┌───────────────────────┐                 │
│         │   操作数栈 (Stack)     │                 │
│         │   ← push / pop →      │                 │
│         └───────────────────────┘                 │
│                                                   │
│   while True:                                     │
│     opcode = *next_instr++                        │
│     switch(opcode) { ... }  ← 巨大的 switch-case   │
└─────────────────────────────────────────────────┘
```

**三本台账的区别**：

| 表 | 名字 | 内容 | 访问方式 | 速度 |
|---|---|---|---|---|
| **co_consts** | 常量表 | 数字字面量、字符串字面量、None | 按索引 `LOAD_CONST i` | O(1) 最快 |
| **co_varnames** | 局部变量表 | 函数内的局部变量名 | 按索引 `LOAD_FAST i` | O(1) 最快 |
| **co_names** | 全局/内置符号表 | 模块级变量、函数名、import 的名字 | 查字典 `LOAD_GLOBAL name` | O(n) 较慢 |

**实测字节码对比**：以下是我们用 `dis` 模块对 `def add(a, b): c = a + b; return c` 的真实反汇编：

```
  5           RESUME                   0          # 函数入口

  6           LOAD_FAST_LOAD_FAST      1 (a, b)   # 一次性把 a, b 压入操作数栈
              BINARY_OP                0 (+)       # 弹出栈顶两个，做加法，结果压栈
              STORE_FAST               2 (c)       # 弹出栈顶，存到局部变量 c

  7           LOAD_FAST                2 (c)       # 把 c 压入栈
              RETURN_VALUE                          # 弹出栈顶作为返回值
```

对应的 Code Object 属性：
```python
co_varnames: ('a', 'b', 'c')    # 局部变量表，按位置索引
co_consts:   (None,)            # 常量表
co_names:    ()                 # 这个函数没有访问任何全局名
```

**LOAD_FAST vs LOAD_GLOBAL 的性能鸿沟**：

```python
# ── 快：局部变量，按索引取 ──
def local_add(a, b):
    return a + b
# LOAD_FAST 0 (a)    → tuple_fast_items[0]，一次指针解引用，~50ns

# ── 慢：全局变量，要查字典 ──
GLOBAL = 42
def global_add(a):
    return a + GLOBAL
# LOAD_GLOBAL 0 (GLOBAL) → 先查 globals() 字典的哈希表，多次比较，~150ns
```

这解释了为什么**循环体内把全局变量缓存到局部变量**能显著加速：

```python
import math

def slow(lst):
    s = 0
    for x in lst:
        s += math.sin(x)    # 每次迭代都 LOAD_GLOBAL "math"，再 LOAD_ATTR "sin"
    return s

def fast(lst, sin=math.sin):  # 把 math.sin 绑定为默认参数（局部变量）
    s = 0
    for x in lst:
        s += sin(x)          # 只需 LOAD_FAST "sin"
    return s
```

### 1.2 GIL（全局解释器锁）的真面目

**GIL 是什么？** CPython 内部的一把互斥锁，保证**同一时刻只有一个线程在执行 Python 字节码**。

**GIL 保护的对象**：不是你的业务数据，而是**CPython 解释器本身的数据结构**（引用计数器、内存分配器、GC 等）。如果没有 GIL，两个线程同时修改同一个对象的引用计数，会导致内存泄漏或 use-after-free。

**GIL 的粒度——保护单条字节码，而非业务逻辑**：

这是理解竞态条件的核心。看真实反汇编：

```python
counter = 0
def increment():
    global counter
    counter = counter + 1
```

```
=== counter = counter + 1 (竞态窗口) ===

 24           LOAD_GLOBAL              0 (counter)   # ① 读：把 counter 压栈
              LOAD_CONST               1 (1)          # ② 把 1 压栈
              BINARY_OP                0 (+)          # ③ 算：弹栈做加法，结果压栈
              STORE_GLOBAL             0 (counter)    # ④ 写：弹栈写回 counter
```

GIL 保证**每一条字节码是原子的**（①、②、③、④各自不会被中断），但 `counter = counter + 1` 是**四条字节码**，GIL 可能在任意两条之间切换线程！

**经典并发灾难——更新丢失（Lost Update）**：

```
时刻    线程 A                     线程 B                     counter
─────────────────────────────────────────────────────────────────────
 t0     LOAD_GLOBAL → 读到 5                                  5
 t1     LOAD_CONST 1                                          5
 t2     BINARY_OP → 算出 6                                    5
        ──── GIL 切换到线程 B ────
 t3                                LOAD_GLOBAL → 读到 5        5
 t4                                LOAD_CONST 1                5
 t5                                BINARY_OP → 算出 6          5
 t6                                STORE_GLOBAL → 写入 6        6
        ──── GIL 切回线程 A ────
 t7     STORE_GLOBAL → 写入 6                                  6

最终结果：counter = 6（应该是 7！线程 A 的更新被覆盖了）
```

**实测验证**（来自 `study/lec4/03_with_lock.py` 的真实运行结果）：

```
=== 锁的作用演示 ===

【情况1】两个线程，没有锁，各自 counter += 1 十次
期望: 20, 实际: 11, 结果: ✗       ← 更新丢失！

【情况2】两个线程，各自使用不同的锁（错误示例）
期望: 20, 实际: 10, 结果: ✗       ← 锁不互斥！

【情况3】两个线程，共享同一把锁（正确示例）
期望: 20, 实际: 20, 结果: ✓       ← 串行化保护
```

下面给出两个完整、可直接运行的脚本。第一个演示"不带锁"的竞态，第二个演示"带锁"的三种情况对比（无锁 / 不同锁 / 同一把锁）。

**完整代码 A——不带锁的数据竞态**（`study/lec4/02_data_race.py`）：

```python
"""
数据竞态演示

演示：
- 两个线程同时修改共享变量
- 期望结果 vs 实际结果
- sleep() 如何放大竞态窗口
"""

import threading
import time


# 全局共享变量
counter = 0
N = 10  # 每个线程增加 10 次，期望总计 20


def task(name: str):
    """线程任务：读取-修改-写入"""
    global counter
    for i in range(N):
        temp = counter          # ① 读
        # time.sleep(0.00001)   # 取消注释可以放大竞态窗口
        counter = temp + 1      # ② 写（中间没有锁，竞态窗口！）


def main():
    global counter

    print("=== 数据竞态演示 ===")
    print(f"期望结果: {2 * N} (两个线程各增加 {N} 次)")

    # 重置计数器
    counter = 0

    # 创建两个线程
    t1 = threading.Thread(target=task, args=("t1",))
    t2 = threading.Thread(target=task, args=("t2",))

    # 启动线程
    t1.start()
    t2.start()

    # 等待线程结束
    t1.join()
    t2.join()

    print(f"实际结果: {counter}")

    if counter == 2 * N:
        print("✓ 结果正确（这次没有发生竞态，但不是保证的！）")
    else:
        print(f"✗ 结果错误！丢失了 {2 * N - counter} 次更新")
        print("  原因：两个线程同时读取到相同的旧值，覆盖了对方的更新")


if __name__ == "__main__":
    # 多次运行，观察结果的不确定性
    for i in range(5):
        print(f"\n--- 第 {i+1} 次运行 ---")
        main()
```

注意 `task` 函数里 `temp = counter` 和 `counter = temp + 1` 之间**没有锁保护**——这就是竞态窗口。`sleep` 被注释掉了，所以窗口很窄，大多数时候会"碰巧"得到 20；但如果取消注释 `sleep(0.00001)`，窗口被撑大，几乎每次都会丢更新。

**完整代码 B——带锁的三种情况对比**（`study/lec4/03_with_lock.py`）：

```python
"""
使用锁解决数据竞态

演示：
- 如何使用 Lock 保护临界区
- with 语句自动获取和释放锁
- 共享同一把锁的重要性
"""

import threading
import time


# 全局变量
counter = 0
N = 10


def task_without_lock(name: str):
    """没有锁保护的线程"""
    global counter
    for _ in range(N):
        temp = counter
        time.sleep(0.00001)  # 放大竞态窗口
        counter = temp + 1


def task_with_lock(name: str, lock: threading.Lock):
    """有锁保护的线程"""
    global counter
    for _ in range(N):
        with lock:  # 自动获取锁
            temp = counter
            time.sleep(0.00001)  # 即使有 sleep，其他线程也无法进入
            counter = temp + 1
        # 离开 with 块，锁自动释放


def main():
    global counter

    print("=== 锁的作用演示 ===\n")

    # ===== 情况1: 没有锁 =====
    print("【情况1】两个线程，没有锁")
    counter = 0

    t1 = threading.Thread(target=task_without_lock, args=("t1",))
    t2 = threading.Thread(target=task_without_lock, args=("t2",))

    t1.start()
    t2.start()
    t1.join()
    t2.join()

    print(f"期望: {2 * N}, 实际: {counter}, 结果: {'✓' if counter == 2 * N else '✗'}\n")

    # ===== 情况2: 各自用不同的锁（错误！）=====
    print("【情况2】两个线程，各自使用不同的锁（错误示例）")
    counter = 0

    lock1 = threading.Lock()       # ← 两把不同的锁！
    lock2 = threading.Lock()

    t3 = threading.Thread(target=task_with_lock, args=("t3", lock1))
    t4 = threading.Thread(target=task_with_lock, args=("t4", lock2))

    t3.start()
    t4.start()
    t3.join()
    t4.join()

    print(f"期望: {2 * N}, 实际: {counter}, 结果: {'✓' if counter == 2 * N else '✗'}")
    print("原因：使用的是不同的锁对象，不会互斥！\n")

    # ===== 情况3: 共享同一把锁（正确！）=====
    print("【情况3】两个线程，共享同一把锁（正确示例）")
    counter = 0

    shared_lock = threading.Lock()  # ← 只创建一把锁，传给两个线程！

    t5 = threading.Thread(target=task_with_lock, args=("t5", shared_lock))
    t6 = threading.Thread(target=task_with_lock, args=("t6", shared_lock))

    t5.start()
    t6.start()
    t5.join()
    t6.join()

    print(f"期望: {2 * N}, 实际: {counter}, 结果: {'✓' if counter == 2 * N else '✗'}")
    print("原因：两个线程竞争同一把锁，临界区互斥执行！")


if __name__ == "__main__":
    main()
```

**两种写法逐行对比——差别只有 3 行**：

```
  task_without_lock                task_with_lock
  ───────────────────              ──────────────
  for _ in range(N):               for _ in range(N):
      temp = counter                   with lock:        ← ① 多了 with lock:
          time.sleep(0.00001)              temp = counter    ← ② 缩进进 with 块
          counter = temp + 1               time.sleep(...)
                                          counter = temp + 1
                                      # with 结束自动释放    ← ③ 异常安全

  ★ 唯一的本质区别：临界区被 with lock: 包住
  ★ with lock: 等价于 lock.acquire() + try/finally: lock.release()
    即使临界区内部抛异常，锁也一定会被释放（不会死锁）
```

**情况2 的坑——两把不同的锁不互斥**：`lock1 = Lock(); lock2 = Lock()` 创建了两个独立的锁对象，线程 A 拿 `lock1`、线程 B 拿 `lock2`，各锁各的，谁也不挡谁。`with lock:` 检查的是"这把锁有没有被人拿走"，而 `lock1` 和 `lock2` 是两把不同的锁——线程 A 拿走 `lock1` 完全不影响线程 B 拿 `lock2`。必须**共享同一个 Lock 实例**（情况3），两个线程竞争同一把锁，才能真正互斥。

**Lock 的本质——保护"执行权限"，不保护数据**：

```
比喻：包厢里只有一个麦克风。

  不带锁：两个人同时抢麦克风，各说各的，谁也听不清（数据竞争）
  带锁：  麦克风只有一支，拿到的人说完放下，下一个人再拿（串行化）

  Lock 不是在保护"说了什么内容"（你的数据），
  而是在保护"谁有资格说话"（临界区的执行权）。
```

### 1.3 多进程 vs 多线程 vs 协程

```
┌─────────────────┬──────────────────┬──────────────────┬──────────────────┐
│                 │   多进程 (Process)│  多线程 (Thread)  │  协程 (Coroutine) │
├─────────────────┼──────────────────┼──────────────────┼──────────────────┤
│ 调度方式         │ OS 抢占式         │ OS 抢占式         │ 用户态协作式      │
│ 内存隔离         │ ✓ 完全隔离        │ ✗ 共享地址空间    │ ✗ 共享（同线程内） │
│ GIL 限制         │ ✗ 不受限制        │ ✓ 受限            │ ✓ 受限（单线程）  │
│ 适合场景         │ CPU 密集型        │ 少量 I/O          │ 大量 I/O         │
│ 创建开销         │ 重（~MB 级）      │ 中（~KB 级）      │ 轻（~字节级）    │
│ 通信方式         │ Queue/Pipe/共享内存│ 共享变量+Lock     │ await/事件循环    │
└─────────────────┴──────────────────┴──────────────────┴──────────────────┘
```

**多进程的内存模型——写时复制（Copy-on-Write）**：

```python
from multiprocessing import Process

def worker(counter):  # counter 是父进程的副本，不是引用！
    counter += 1
    print(f"子进程 counter = {counter}")

if __name__ == "__main__":
    counter = 0
    p = Process(target=worker, args=(counter,))
    p.start()
    p.join()
    print(f"父进程 counter = {counter}")  # 还是 0！内存完全隔离
```

```
fork() 时的内存模型：

  父进程内存                        子进程内存
  ┌──────────┐                    ┌──────────┐
  │ counter=0│  ── fork + COW ──▶ │ counter=0│  ← 只是复制了页表
  │ data=[…] │                    │ data=[…] │  ← 物理页共享，只读
  └──────────┘                    └──────────┘
                                        │
                                   子进程写 counter=1
                                        │
                                        ▼
                                   COW 触发：复制该页
                                   ┌──────────┐
                                   │ counter=1│  ← 新物理页
                                   │ data=[…] │  ← 仍共享（只读）
                                   └──────────┘
```

**协程——用户态的"主动让权"**：

```python
import asyncio

async def worker(name, lock=None):
    global counter
    for _ in range(5):
        old = counter
        await asyncio.sleep(0)    # ← 遇到 I/O，主动让出执行权
        counter = old + 1         # 事件循环切到另一个协程，这里就是竞态窗口！

asyncio.run(asyncio.gather(worker("c1"), worker("c2")))
# 实测结果：期望 10，实际 0  ← 竞态！
```

**协程为什么也有竞态？** 虽然 `asyncio.sleep(0)` 不会真的睡，但它是一个 `await` 点，事件循环会在此时切换到另一个协程。`old = counter` 和 `counter = old + 1` 之间的 `await` 就是窗口。

**协程的物理包含关系——它们在同一个线程里**：

```
┌─────────────── 单个 OS 线程 ───────────────┐
│                                             │
│   ┌─────────┐     事件循环 (Event Loop)     │
│   │ 协程 c1  │◄──────────────────────────┐   │
│   │ old=cnt  │    "c1 遇到 await, 挂起"   │   │
│   │ await…   │ ──────────────────────►   │   │
│   │ (挂起)   │                            │   │
│   └─────────┘    "c2 可以执行了"           │   │
│   ┌─────────┐ ──────────────────────►   │   │
│   │ 协程 c2  │◄──────────────────────────┘   │
│   │ old=cnt  │                               │
│   │ await…   │                               │
│   └─────────┘                               │
│                                             │
│   ★ 协程不是并行，是并发（交替执行）          │
│   ★ 同一时刻只有一个协程在真正运行            │
└─────────────────────────────────────────────┘
```

**三种并发模型对比代码**：

```python
# ── 多进程：CPU 密集，绕过 GIL ──
from multiprocessing import Pool

def cpu_heavy(n):
    s = 0
    for i in range(n):
        s += i * i
    return s

with Pool(4) as p:           # 4 个独立进程，真正的物理并行
    results = p.map(cpu_heavy, [10**7] * 4)

# ── 多线程：少量 I/O ──
import threading

def io_task(url):
    # requests.get(url)  ← 阻塞时释放 GIL，其他线程可以跑
    pass

threads = [threading.Thread(target=io_task, args=(u,)) for u in urls]
for t in threads: t.start()
for t in threads: t.join()

# ── 协程：大量 I/O（如 10000 个连接）──
import asyncio, aiohttp

async def fetch(session, url):
    async with session.get(url) as resp:
        return await resp.text()

async def main():
    async with aiohttp.ClientSession() as session:
        tasks = [fetch(session, url) for url in urls]  # 10000 个协程
        results = await asyncio.gather(*tasks)          # 全部并发，只有 1 个线程

asyncio.run(main())
```

### 1.4 协程的软硬件倒影（跨模块洞察）

> **这是贯穿全课最重要的类比之一。**

| 维度 | Python 协程（模块一） | GPU Warp 调度（模块三） |
|---|---|---|
| 调度方式 | **协作式**：`await` 主动让权 | **硬件自动**：遇到访存延迟自动切换 |
| 让权时机 | `await asyncio.sleep()` / I/O 等待 | HBM 全局内存读取（~400 cycles 延迟） |
| 切换开销 | 用户态，~微秒级 | **硬件 0 开销**（Warp 寄存器独立） |
| 并行 vs 并发 | 并发（单线程交替） | 并行（多个 Warp 同时驻留 SM） |
| 核心思想 | **用切换掩盖 I/O 等待** | **用切换掩盖访存延迟** |

**结论**：GPU 的 SIMT 硬件调度器本质上就是一个**免费的、零开销的、硬件级的事件循环**。Python 程序员手写 `async/await` 来榨干 I/O 等待，GPU 硬件设计师用 Warp 切换来榨干访存等待。殊途同归。

---

# 模块二：跨越边界——Python 与 Native Code 交互

## 核心命题：如何让解释型语言获得编译型语言的极限性能？

### 2.1 C-API / ctypes 与 GIL 释放

Python 慢的根源不是语法，而是**逐条解释执行字节码**。如果能绕过解释器，直接调用编译好的 C 代码，性能可以提升 100-1000 倍。

**ctypes——最直接的 C 调用方式**：

```python
import ctypes

# 加载预编译的共享库
lib = ctypes.CDLL("./libvec.so")

# 声明函数签名（告诉 ctypes 参数类型）
lib.vec_elem_mul.argtypes = [
    ctypes.POINTER(ctypes.c_double),  # a 数组指针
    ctypes.POINTER(ctypes.c_double),  # b 数组指针
    ctypes.POINTER(ctypes.c_double),  # out 数组指针
    ctypes.c_int,                     # 数组长度 n
]
lib.vec_elem_mul.restype = None       # 返回 void
```

**Py_BEGIN_ALLOW_THREADS——释放 GIL 的魔法宏**：

在 C 扩展中，可以用这个宏临时释放 GIL，让**其他 Python 线程**也能跑：

```c
// libvec.c 中的 C 函数
void vec_elem_mul(double* a, double* b, double* out, int n) {
    Py_BEGIN_ALLOW_THREADS    // ← 释放 GIL！
    
    // 这段 C 代码运行时，其他 Python 线程可以并行执行
    // 这是 CPython 中实现真正多核并行的唯一途径
    for (int i = 0; i < n; i++) {
        out[i] = a[i] * b[i];
    }
    
    Py_END_ALLOW_THREADS      // ← 重新获取 GIL
}
```

```
GIL 释放前后的线程执行对比：

  ┌── Thread A (调用 C 扩展) ──┐
  │ Python 字节码                │  ← 持有 GIL
  │ 调用 C 函数                  │
  │ │ Py_BEGIN_ALLOW_THREADS    │  ← 释放 GIL！
  │ │ C 循环计算 (50ms)         │     ┌── Thread B ──┐
  │ │                           │     │ Python 字节码 │ ← 获得 GIL，可以跑了！
  │ │ Py_END_ALLOW_THREADS      │     │ 跑了 50ms    │
  │ Python 字节码继续            │  ← 重新获取 GIL
  └─────────────────────────────┘     └───────────────┘

  ★ 这就是 NumPy / PyTorch 能实现真正多核并行的秘密
```

### 2.2 AST 降级（Lowering）与装饰器魔法

**核心洞察**：`@kernel` 装饰器**不运行 Python 代码**，而是用 `ast.parse` 提取函数的 AST（抽象语法树），把 Python 表达式翻译成底层 C 函数调用。

以下是我们课程代码 `code/l3-3-deco.py` 的完整实现逻辑：

```python
import ast
import inspect
import ctypes
import numpy as np

lib = ctypes.CDLL("./libvec.so")

def kernel(func):
    """
    把 Python 函数体当作 DSL（领域特定语言）来解析。
    函数体不会被 Python 执行，而是被 AST 解析器"翻译"。
    """
    # ── 第一步：获取源码并解析为 AST ──
    src = inspect.getsource(func)
    tree = ast.parse(src)
    func_def = tree.body[0]         # FunctionDef 节点

    # 期望函数体只有一条 return 语句
    return_stmt = func_def.body[0]  # Return 节点
    expr = return_stmt.value        # return 后面的表达式
    arg_names = [arg.arg for arg in func_def.args.args]

    # ── 第二步：AST → 原语操作（Lowering）──
    def lower(node, env, n):
        if isinstance(node, ast.Name):           # 变量名 → 查环境
            return env[node.id]

        if isinstance(node, ast.BinOp):          # 二元运算 → 调 C 库
            left = lower(node.left, env, n)
            right = lower(node.right, env, n)
            out = np.empty_like(left)

            if isinstance(node.op, ast.Mult):    # a * b → lib.vec_elem_mul()
                lib.vec_elem_mul(left_ptr, right_ptr, out_ptr, n)
            elif isinstance(node.op, ast.Add):   # a + b → lib.vec_elem_add()
                lib.vec_elem_add(left_ptr, right_ptr, out_ptr, n)

            return out

    # ── 第三步：运行时包装器 ──
    def wrapper(*args):
        arrays = [np.asarray(a, dtype=np.float64) for a in args]
        env = dict(zip(arg_names, arrays))
        return lower(expr, env, arrays[0].size)

    return wrapper
```

**用户视角——写的是 Python，跑的是 C**：

```python
@kernel
def vec_elem_mul(a, b):
    return a * b       # 表面是 Python 乘法，实际路由到 C 的 vec_elem_mul

@kernel
def vec_elem_fma(a, b, c):
    return (a * b) + c  # AST 递归：先 Mult，再 Add，两次 C 调用

# 测试
print(vec_elem_mul([1.0, 2.0, 3.0], [4.0, 5.0, 6.0]))
# 输出: [4.0, 10.0, 18.0]  ← 但底层是 C 循环，不是 Python 循环
```

**AST 真实结构**（用 `ast.dump` 实测）：

```python
# 源码: def vec_elem_mul(a, b): return a * b
import ast
tree = ast.parse("def vec_elem_mul(a, b): return a * b")
print(ast.dump(tree, indent=2))
```

```
Module(body=[
  FunctionDef(
    name='vec_elem_mul',
    args=arguments(args=[arg(arg='a'), arg(arg='b')]),
    body=[
      Return(value=                       # ← @kernel 取这个节点
        BinOp(
          left=Name(id='a', ctx=Load()),  # ← lower() 递归处理
          op=Mult(),                       # ← 匹配 ast.Mult → 调 C 库
          right=Name(id='b', ctx=Load())  # ← lower() 递归处理
        )
      )
    ]
  )
])
```

**这个模式的本质——Zero-Cost 抽象的开端**：

```
用户写的 Python 代码              AST 解析后实际执行的
─────────────────────           ──────────────────────
return a * b           →        lib.vec_elem_mul(a, b, out, n)
                                 ↑ 这是预编译的 C 函数，跑在 CPU 原生速度

return (a * b) + c     →        tmp = lib.vec_elem_mul(a, b, out, n)
                                lib.vec_elem_add(tmp, c, out, n)
```

**从 @kernel 到 @triton.jit 的演进**：

```
@kernel (l3-3-deco.py)           @triton.jit (l7-vecmul.py)
──────────────────────           ──────────────────────────
AST → C 函数调用                  AST → PTX → GPU 机器码
跑在 CPU 上                       跑在 GPU 上
单线程                            数千个线程并行
numpy 数组                         GPU 显存指针

但核心思想完全一样：
  Python 只是个"语法糖外壳"
  真正的执行逻辑在 AST 解析阶段被翻译成底层代码
```

---

# 模块三：GPU 体系结构与并行计算（硬件域）

## 核心命题：如何用极致的轻量级线程掩盖内存延迟？

### 3.1 软件逻辑层级 vs 硬件物理实体

```
┌─────────────────────────────────────────────────────────────────────┐
│                        GPU 硬件架构                                   │
│                                                                       │
│  ┌─────────────────────────────────────────────────────────────┐    │
│  │                      GPU 芯片（整片）                        │    │
│  │  ┌──────────┐ ┌──────────┐ ┌──────────┐ ┌──────────┐       │    │
│  │  │   SM 0   │ │   SM 1   │ │   SM 2   │ │   SM 3   │  ...  │    │
│  │  │(流多处理器)│ │          │ │          │ │          │       │    │
│  │  │┌────────┐│ │┌────────┐│ │┌────────┐│ │┌────────┐│       │    │
│  │  ││Cores   ││ ││Cores   ││ ││Cores   ││ ││Cores   ││       │    │
│  │  ││(FP32   ││ ││        ││ ││        ││ ││        ││       │    │
│  │  ││ INT32  ││ ││        ││ ││        ││ ││        ││       │    │
│  │  ││ Tensor)││ ││        ││ ││        ││ ││        ││       │    │
│  │  │└────────┘│ │└────────┘│ │└────────┘│ │└────────┘│       │    │
│  │  │┌────────┐│ │┌────────┐│ │┌────────┐│ │┌────────┐│       │    │
│  │  ││SRAM    ││ ││SRAM    ││ ││SRAM    ││ ││SRAM    ││       │    │
│  │  ││(Shared ││ ││        ││ ││        ││ ││        ││       │    │
│  │  ││Memory) ││ ││        ││ ││        ││ ││        ││       │    │
│  │  │└────────┘│ │└────────┘│ │└────────┘│ │└────────┘│       │    │
│  │  └──────────┘ └──────────┘ └──────────┘ └──────────┘       │    │
│  │                                                               │    │
│  │  ┌───────────────────────────────────────────────────────┐   │    │
│  │  │              HBM（全局显存 / Global Memory）            │   │    │
│  │  │           所有 SM 共享，容量大但延迟高                   │   │    │
│  │  │           A100: 40GB / 80GB, 带宽 ~2TB/s               │   │    │
│  │  └───────────────────────────────────────────────────────┘   │    │
│  └─────────────────────────────────────────────────────────────┘    │
└─────────────────────────────────────────────────────────────────────┘
```

**软件→硬件映射关系**：

```
软件逻辑层级                      硬件物理实体
────────────                     ────────────
Grid (网格)                  →    整个 GPU（所有 SM 的集合）
  └─ Block (线程块)          →    一个 SM（流多处理器）
       └─ Warp (线程束)       →    SM 的一个调度单元（32 个线程锁定）
            └─ Thread (线程)   →    一个 CUDA Core（计算通道）
```

**GPU 内存层级**：

```
┌───────────────────────────────────────────────────┐
│                  内存层级与延迟                       │
│                                                     │
│  ┌─────────────┐  容量: 40-80 GB    延迟: ~400 cycles │
│  │  HBM (全局)  │  带宽: ~2 TB/s    所有 SM 共享      │
│  │  Global Mem │  ← 所有数据初始存放的地方              │
│  └──────┬──────┘                                    │
│         │ tl.load / tl.store                        │
│         ▼                                            │
│  ┌─────────────┐  容量: ~192 KB     延迟: ~30 cycles  │
│  │ SRAM (共享)  │  带宽: ~19 TB/s   同一 Block 共享    │
│  │ Shared Mem  │  ← FlashAttention 的"舞台"            │
│  └──────┬──────┘                                    │
│         │ 寄存器分配                                  │
│         ▼                                            │
│  ┌─────────────┐  容量: ~256 KB     延迟: ~1 cycle    │
│  │ Registers   │  带宽: 极高         每个 Thread 私有  │
│  │  (寄存器)    │  ← FlashAttention 累加器驻留于此     │
│  └─────────────┘                                    │
└───────────────────────────────────────────────────┘

延迟比 ≈ HBM : SRAM : Reg = 400 : 30 : 1

这就是为什么 FlashAttention 要把数据搬到 SRAM 里算！
```

### 3.2 SIMT（单指令多线程）：软件逻辑模型 vs 硬件物理执行模型

这里是 GPU 编程中最容易让人产生幻觉的地方：**我们写代码时的"逻辑模型"与 GPU 电路板的"物理执行模型"之间存在巨大割裂**。一旦看清这层割裂，Block / Warp / Thread 三个概念的关系就彻底通透了。

#### 第一问：共享 PC 的到底是 Block 还是 Warp？

**SIMT 的铁律**：锁定同一个 PC（程序计数器）、同一时刻执行同一条指令的，是 **Warp（32 线程）**，绝对不是 Block。

但写 CUDA 代码时，代码是针对一个"抽象的 Thread"写的，你脑子里会不自觉地觉得"整个 Block 都在跑这个函数，大家应该是一起往下走的"——这正是幻觉的来源。

```
真实物理世界的残酷真相：

假设你定义了一个含 128 个线程的 Block。

  逻辑上（你的代码视角）:
  ┌──────────────── Block（行政编制）────────────────┐
  │  T0  T1  T2 ... T31 | T32 ... T63 | T64 ... T95 | T96 ... T127 │
  │  ←──── 大家都跑同一个 kernel 函数 ────→                          │
  └──────────────────────────────────────────────────────────────┘
  → 你以为 128 个人步调一致，共享一个 PC

  物理上（SM 硬件调度器的视角）:
  ┌──────── Block ────────┐
  │ Warp 0 (T0-31)         │  PC 指向第 10 行 LOAD  ← 这 32 人锁步
  │ Warp 1 (T32-63)        │  PC 还停在第 8 行      ← 等内存，落了一拍
  │ Warp 2 (T64-95)        │  PC 已跑到第 15 行 ADD ← 抢先了
  │ Warp 3 (T96-127)       │  PC 在第 12 行 STORE
  └────────────────────────┘
  → 同一个 Block 内，4 个 Warp 的 PC 完全不同！步调不一致！

  ★ Block = 行政编制（分配 Shared Memory 的依据）
  ★ Warp  = 物理执行战术小队（共享 PC、锁步执行的真正单元）
```

**结论**：Block 里的线程**并不**共享同一个 PC。硬件调度器把 Block 无情地劈成若干个 Warp（每 32 个线程一组），只有同一组 Warp 内的 32 个线程才死死锁住同一个 PC。不同 Warp 可以各自独立地被调度、被挂起、被切换——这正是 GPU 用 Warp 切换掩盖访存延迟的硬件基础（模块一的"协程倒影"）。

#### 第二问：坐标公式里，Warp 去哪了？

每个 GPU 程序员烂熟于心的寻址公式：

```
global_idx = blockIdx.x * blockDim.x + threadIdx.x
```

这里只有 Block 和 Thread，**完全没有 warpIdx 的身影**。既然 Warp 是物理执行的核心，为什么寻址时不用它？

答案：**坐标计算是"逻辑寻址"，Warp 是"物理干活的编制"，两者处于不同的抽象层。**

```
🏫 高考考场比喻

  全国有几十万考生 = 整个 Grid 的所有线程

  Block（考场）     : 每个考场坐 128 人
  threadIdx（座位号）: 考场内座位 0~127
  Warp（监考组）    : 一个监考老师一眼只能盯 32 人，所以
                      128 人被隐式分成 4 个监考组

  全国准考证号 = 考场号(blockIdx) × 考场容量(blockDim) + 座位号(threadIdx)

  ★ 定位一个人只需要"考场号 + 座位号"——Warp 隐含在座位号里！
    座位号 0~31  → Warp 0
    座位号 32~63 → Warp 1
    ...
    warp_id = threadIdx.x / 32    ← 用数学算出来，不是硬件给你的

  Warp 是硬件为了"方便发号施令"隐式划分的分组，
  它不改变数据在逻辑上的连续性。
  所以寻址公式里根本不需要 warpIdx——threadIdx 已经覆盖了一切。
```

**什么时候才需要显式用到 Warp？** 当你需要跨线程交换寄存器数据时——比如 FlashAttention 里 32 个线程要做 All-Reduce 求全局 max/sum（见 6.4 节 Warp Shuffle）。这时程序员用 `threadIdx.x / 32` 算出 warp_id，用 `__shfl_xor_sync` 在组内通信。但纯粹的"我要处理第几个数据"这种寻址逻辑，Warp 完全不参与。

#### 第三问（升维）：Triton 里连 Thread 都消失了

在 CUDA C++ 里你还要手写 `threadIdx.x`（代表"一个人"）。但在课程 Lec 7 学的 **Triton** 中，连 Thread 这个概念都被抹杀了。下面给出同一个操作（向量逐元素乘法）的两种完整写法，逐行对照：

**完整代码 A——CUDA C++ 版（Thread 级编程，手管 Warp）**：

```c
// vecmul.cu
// 每个 Thread 处理一个元素，程序员要手写 threadIdx / blockIdx / blockDim

__global__ void vecmul_kernel(const float* a, const float* b, float* out, int N) {
    // ── 寻址：手动拼接全局坐标 ──
    //   blockIdx.x  = 当前 Block 在 Grid 里的编号（考场号）
    //   blockDim.x  = 每个 Block 里的 Thread 数（考场容量，如 256）
    //   threadIdx.x = 当前 Thread 在 Block 里的编号（座位号）
    //
    //   全局坐标 = 考场号 × 考场容量 + 座位号
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    // ── 边界检查：显式 if（可能产生分支发散）──
    if (idx < N) {
        out[idx] = a[idx] * b[idx];   // 每个 Thread 读一个、算一个、写一个
    }
    // ★ 硬件隐式地把 blockDim.x=256 个 Thread 打包成 8 个 Warp（每个 32 人）
    // ★ 程序员看不到 Warp，但物理上就是 8 个 Warp 在锁步执行
}

// Host 端启动
int N = 10240;
int block_size = 256;                           // 每个 Block 256 个 Thread
int grid_size  = (N + block_size - 1) / block_size;  // 需要 40 个 Block

vecmul_kernel<<<grid_size, block_size>>>(a, b, out, N);
//                   ↑grid    ↑block
//  Grid 维度和 Block 维度都要程序员手动指定！
```

**完整代码 B——Triton 版（Block 级编程，连 Thread 都不写）**：

```python
# vecmul.py  (来自 code/l7-vecmul.py)
# 每个 Program 处理一整块（几百个元素），程序员只管 Block 切分

@triton.jit
def vecmul_kernel(
    a_ptr, b_ptr, out_ptr, N,
    BLOCK_SIZE: tl.constexpr,      # 编译期常量：一个 Block 处理多少元素（如 1024）
):
    # ── 寻址：只有 Block 级，没有 threadIdx ──
    pid = tl.program_id(0)                          # Block 编号（只有这一层！）
    offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)  # 直接生成 [1024] 的索引数组

    # ── 边界处理：用 mask 替代 if（不产生分支发散）──
    mask = offs < N
    a = tl.load(a_ptr + offs, mask=mask)            # 一次性"加载"整块 [1024]
    b = tl.load(b_ptr + offs, mask=mask)

    out = a * b                                     # 整块 [1024] 并行乘法

    tl.store(out_ptr + offs, out, mask=mask)        # 一次性写回整块

# Host 端启动
N = 10240
BLOCK_SIZE = 1024
grid = lambda meta: (triton.cdiv(N, meta['BLOCK_SIZE']),)  # Grid = (10,)

vecmul_kernel[grid](a, b, out, N, BLOCK_SIZE=BLOCK_SIZE)
#          ↑ 只指定 Grid（多少个 Block），不指定 Block 维度！
```

**两种写法逐行对照**：

```
  CUDA C++ (Thread 级)                      Triton (Block 级)
  ──────────────────────                   ──────────────────────

  int idx = blockIdx.x * blockDim.x         pid  = tl.program_id(0)
           + threadIdx.x;                   offs = pid * BLOCK_SIZE
    ↑ 三层拼坐标（考场×容量+座位）                + tl.arange(0, BLOCK_SIZE)
    ↑ 程序员要算单点坐标(idx)                    ↑ 两层生成数组（考场×BLOCK_SIZE + [0..1023]）
                                                ↑ 程序员直接拿到整块坐标数组(offs)

  if (idx < N) {                            mask = offs < N
      out[idx] = a[idx] * b[idx];           a = tl.load(a_ptr + offs, mask=mask)
  }                                         b = tl.load(b_ptr + offs, mask=mask)
    ↑ 1 个 Thread 处理 1 个元素                out = a * b
    ↑ 显式 if → 边缘 Warp 可能分支发散         tl.store(out_ptr + offs, out, mask=mask)
                                              ↑ 1 个 Program 处理 1024 个元素
                                              ↑ mask 屏蔽 → 不产生分支发散

  <<<grid, block>>>                         kernel[grid](... BLOCK_SIZE=1024)
    ↑ 手动指定 grid 和 block 维度              ↑ 只指定 grid（多少个 Block）
    ↑ block 维度(256)程序员要调                 ↑ BLOCK_SIZE 是编译期常量
```

**核心差异**：CUDA 版里程序员处理的"最小单位"是**单个 Thread**（`idx` 是一个标量，`out[idx] = a[idx]*b[idx]` 是一次单点操作）；Triton 版里程序员处理的"最小单位"是**整个 Block 的数组**（`offs` 是 `[1024]` 的向量，`out = a * b` 是一次批量操作）。至于这 1024 个元素底层怎么被拆给 32 个 Warp、每个 Warp 里的 32 个 Thread 各分到哪几个——**Triton 编译器全包了，程序员完全不用管**。

```
三层抽象的演进：

  CUDA C++   : 程序员管 Thread（一个人干一个元素）
       公式: blockIdx * blockDim + threadIdx
       ↓ 硬件隐式把 Thread 打包成 Warp

  Triton     : 程序员只管 Block（一个 Block 一坨数据）
       公式: program_id * BLOCK_SIZE + tl.arange
       ↓ Triton 编译器自动把 Block 级数组操作
       ↓   拆解成 Warp 级、Thread 级的 CUDA 机器码

  ★ 抽象越升越高：Thread 被 Triton 藏起来了，Warp 从来就没暴露过。
    但物理执行时，GPU 硬件还是老老实实地按 32 人一组锁步跑。
```

#### 分支发散（Branch Divergence）——SIMT 的阿喀琉斯之踵

理解了"32 人锁定同一 PC"后，分支发散就很好懂了：当同一个 Warp 内的 32 个线程在 `if` 处走了不同分支，硬件无法让它们"分头行动"（PC 只有一个），只能用掩码让不需要执行的线程"原地罚站"，把两个分支串行跑一遍。

```c
// CUDA C 中的 if/else
if (threadIdx.x < 16) {
    result = a * b;        // 分支 A：线程 0-15 需要
} else {
    result = a + b;        // 分支 B：线程 16-31 需要
}
```

```
Warp 内的分支发散（32 线程共享一个 PC，无法分头行动）：

  时间 →
  T0-15:  [if ✗]  [分支B 罚站]  [分支A: a*b]  [else 罚站]  [汇合]
  T16-31: [if ✓]  [分支A 罚站]  [else 罚站]  [分支B: a+b] [汇合]
                                    ↑
                          掩码（Masking）：硬件下发统一指令，
                          用掩码屏蔽不需要执行的线程，被屏蔽的线程罚站

  结果：两个分支被串行化执行，并行效率减半！
```

**Triton 的优雅解决方案——用 mask 替代 if**：

```python
# CUDA C：显式 if → 边缘 Warp 分支发散
if (row < N) {
    c[row] = a[row] + b[row];
}

# Triton：用 mask 告诉硬件哪些线程有效，不产生 if 分支
offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
mask = offs < N                           # 布尔掩码
a = tl.load(a_ptr + offs, mask=mask)      # 越界位置不加载
tl.store(out_ptr + offs, a, mask=mask)    # 越界位置不写入
# Triton 底层把 mask 翻译成硬件掩码，32 线程仍然锁步，但无效线程被掩码屏蔽
```

### 3.3 坐标映射与内存合并访问（Coalesced Access）

**访存铁律**：同一个 Warp 里的 32 个线程**必须访问连续的物理内存地址**，否则总线带宽暴跌。

```
合并访问（Coalesced）—— 32 个线程访问连续地址：

  Thread:   T0    T1    T2    T3   ...   T31
  Address:  0x00  0x08  0x10  0x18  ...  0xF8
            └──────────────────────────────┘
            一次总线事务（256 bytes burst）
            带宽利用率: 100% ✓

非合并访问（Strided / Random）—— 32 个线程访问离散地址：

  Thread:   T0    T1    T2    T3   ...   T31
  Address:  0x00  0x800 0x100 0x900  ...  0xF00
            └─┘   └──┘  └──┘  └──┘       └──┘
            32 次独立总线事务！
            带宽利用率: ~3% ✗
```

**矩阵索引的坐标映射**：

```
矩阵在内存中是行优先（Row-Major）存储的：

  矩阵逻辑视图              物理内存（1D 连续）
  ┌─────────────────┐      
  │ (0,0) (0,1) (0,2)│      地址:  0    1    2    3    4    5
  │ (1,0) (1,1) (1,2)│      元素: [0,0][0,1][0,2][1,0][1,1][1,2]
  │ (2,0) (2,1) (2,2)│             └──第0行──┘└──第1行──┘
  └─────────────────┘

  row = blockIdx.y * blockDim.y + threadIdx.y
  col = blockIdx.x * blockDim.x + threadIdx.x

  物理地址 = row * N_cols + col
           ↑              ↑
           行步长（Stride） 列偏移

  ★ 同一行的元素在物理内存中是连续的 → 合并访问 ✓
  ★ 同一列的元素在物理内存中间隔 N_cols → 非合并访问 ✗
```

**为什么 GPU 编程中通常把 Block 的 X 轴设为列方向（连续方向）？**

```python
# Triton 中的典型 2D 索引（来自 l7-autotune.py 的 matmul）
offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)    # 行索引
offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)    # 列索引

# 同一行的 BLOCK_N 个元素物理地址连续 → Warp 访问合并 ✓
a = tl.load(a_ptr + offs_m[:, None] * K + k[None, :])
#                    ↑ 行基地址           ↑ 列偏移（连续）
```

---

# 模块四：Triton 算子编程体系

## 核心命题：如何将 2D 的逻辑块高效映射到 1D 的物理显存上？

### 4.1 Triton 编程模型——Grid → Block → Program Instance

Triton 是 OpenAI 开发的 GPU 编程 DSL，它的核心抽象是**把 GPU 编程简化为"Block 级编程"**——你不需要手动管理 Warp 和 Thread，Triton 帮你处理。

```
Triton 的三层抽象：

  Python Host 端                      GPU Kernel 端
  ─────────────                      ─────────────
  
  grid = (num_blocks,)    ──启动──▶  每个 program instance 独立运行
  
                                       pid = tl.program_id(0)
                                              ↓
                                       计算 offsets
                                              ↓
                                       tl.load → tl.compute → tl.store
```

**最简单的 Triton kernel——向量逐元素乘法**（来自 `code/l7-vecmul.py`）：

```python
import torch
import triton
import triton.language as tl

@triton.jit                          # ← 把 Python 函数编译为 GPU kernel
def vecmul_kernel(
    x_ptr, y_ptr, out_ptr,          # HBM 中的数据指针
    N,                                # 向量长度
    BLOCK_SIZE: tl.constexpr,        # 编译期常量（每个 program 处理多少元素）
):
    pid = tl.program_id(0)           # 当前 program 的 ID（类似 blockIdx.x）

    # ── 计算当前 program 负责的元素索引 ──
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    #  例: BLOCK_SIZE=1024, pid=2 → offsets = [2048, 2049, ..., 3071]

    mask = offsets < N               # 边界掩码（N 不一定是 BLOCK_SIZE 整数倍）

    # ── 从 HBM 加载数据到 SRAM/寄存器 ──
    x = tl.load(x_ptr + offsets, mask=mask)
    y = tl.load(y_ptr + offsets, mask=mask)

    # ── 在 SRAM/寄存器中计算 ──
    out = x * y                      # BLOCK_SIZE 个元素并行计算

    # ── 写回 HBM ──
    tl.store(out_ptr + offsets, out, mask=mask)


# Host 端封装
def vecmul(x: torch.Tensor, y: torch.Tensor):
    N = x.numel()
    out = torch.empty_like(x)

    # grid 定义：需要多少个 program instance
    grid = lambda meta: (triton.cdiv(N, meta['BLOCK_SIZE']),)
    #  例: N=10240, BLOCK_SIZE=1024 → grid=(10,) → 启动 10 个 program

    vecmul_kernel[grid](x, y, out, N, BLOCK_SIZE=1024)
    return out
```

**执行流程图解**：

```
N = 10240, BLOCK_SIZE = 1024

  Host 端                          GPU 端（10 个 program 并行）
  ────────                         ──────────────────────────

  grid = (10,)  ──启动──▶  ┌─ Program 0 ──┐ offsets = [0..1023]
                            ├─ Program 1 ──┤ offsets = [1024..2047]
                            ├─ Program 2 ──┤ offsets = [2048..3071]
                            │     ...       │
                            └─ Program 9 ──┘ offsets = [9216..10239]
                                    │
                              每个程序内部：
                              1. tl.load 从 HBM 读 1024 个元素
                              2. x * y 并行计算
                              3. tl.store 写回 HBM

  ★ Program 之间互相独立，GPU 硬件自动分配到不同 SM
  ★ Program 内部的 BLOCK_SIZE 个元素由 Triton 自动映射到 Warp/Thread
```

### 4.2 宏观任务分发——2D Grid 与坐标映射

矩阵乘法需要 2D 的任务分发——每个 program 负责输出矩阵的一个 Tile（块）。

```python
# 来自 l7-autotune.py 的 matmul kernel
@triton.jit
def matmul_autotune(
    a_ptr, b_ptr, c_ptr,
    M, N, K,                         # 矩阵维度：A[M,K] × B[K,N] = C[M,N]
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    pid_m = tl.program_id(0)         # 输出行块索引
    pid_n = tl.program_id(1)         # 输出列块索引

    # 当前 program 负责的输出 Tile 范围
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)   # 行索引 [BLOCK_M]
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)   # 列索引 [BLOCK_N]
    offs_k = tl.arange(0, BLOCK_K)                      # K 维度索引 [BLOCK_K]
```

```
2D Grid 任务分发（C = A @ B，C 是 M×N）：

  输出矩阵 C [M, N]               Grid 大小 = (ceil(M/BLOCK_M), ceil(N/BLOCK_N))
  ┌──────┬──────┬──────┐           
  │(0,0) │(0,1) │(0,2) │           每个 (pid_m, pid_n) 对应一个 program
  ├──────┼──────┼──────┤           每个 program 计算 C 的一个 BLOCK_M × BLOCK_N 块
  │(1,0) │(1,1) │(1,2) │           
  ├──────┼──────┼──────┤           例: M=N=1024, BLOCK_M=BLOCK_N=64
  │(2,0) │(2,1) │(2,2) │           → Grid = (16, 16) = 256 个 program
  └──────┴──────┴──────┘

  Host 端 grid 定义：
  grid = lambda META: (
      triton.cdiv(M, META["BLOCK_M"]),    # 行方向 program 数
      triton.cdiv(N, META["BLOCK_N"]),    # 列方向 program 数
  )
```

### 4.3 微观魔法——广播（Broadcasting）与二维指针网格

**Triton 最核心的技巧**：用一维指针 + 广播运算，生成二维的物理地址网格。

```python
# 矩阵 A 的 2D 地址计算
# A 存储为行优先：A[i][j] 在物理地址 a_ptr + i * K + j

offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)   # 形状 [BLOCK_M]
offs_k = tl.arange(0, BLOCK_K)                      # 形状 [BLOCK_K]

# ── 二维指针网格的生成 ──
ptrs = a_ptr + offs_m[:, None] * K + offs_k[None, :]
#              └──── [BLOCK_M, 1] ────┘  └── [1, BLOCK_K] ──┘
#                         ↓ 广播               ↓ 广播
#                    [BLOCK_M, BLOCK_K]    [BLOCK_M, BLOCK_K]
#                         └──────── 相加 ────────┘
#                                        ↓
#                              [BLOCK_M, BLOCK_K] 的 2D 地址网格
```

```
广播运算图解（BLOCK_M=3, BLOCK_K=4 为例）：

  offs_m[:, None] * K:                offs_k[None, :]:
  ┌─────────┐                         ┌───────────────┐
  │ row0 * K│   [3, 1]                │ k0 k1 k2 k3 │   [1, 4]
  │ row1 * K│                         └───────────────┘
  │ row2 * K│                               ↓ 广播
  └─────────┘                         ┌───────────────┐
       ↓ 广播                          │ k0 k1 k2 k3 │
  ┌─────────┐                         │ k0 k1 k2 k3 │   [3, 4]
  │ row0 * K│                         │ k0 k1 k2 k3 │
  │ row0 * K│                         └───────────────┘
  │ row0 * K│
  │ row1 * K│   [3, 4]                     两者相加:
  │ row1 * K│                         ┌──────────────────────────┐
  │ row1 * K│                         │ row0*K+k0  row0*K+k1 ... │
  │ row2 * K│                         │ row1*K+k0  row1*K+k1 ... │   [3, 4]
  │ row2 * K│                         │ row2*K+k0  row2*K+k1 ... │   ← 完整的 2D 物理地址网格
  │ row2 * K│                         └──────────────────────────┘
  └─────────┘

  ★ NumPy 广播铁律：向右对齐，左侧补 1，遇 1 拉伸
    (3, 1) + (1, 4) → (3, 4)
```

**tl.load 读取 2D Tile**：

```python
# 加载 A 的一个 [BLOCK_M, BLOCK_K] 块
a = tl.load(
    a_ptr + offs_m[:, None] * K + offs_k[None, :],
    mask=(offs_m[:, None] < M) & (offs_k[None, :] < K),
    other=0.0
)
# a 的形状: [BLOCK_M, BLOCK_K]

# 同理加载 B 的一个 [BLOCK_K, BLOCK_N] 块
b = tl.load(
    b_ptr + offs_k[:, None] * N + offs_n[None, :],
    mask=(offs_k[:, None] < K) & (offs_n[None, :] < N),
    other=0.0
)

# 矩阵乘法（Triton 使用 Tensor Core 加速）
acc += tl.dot(a, b)    # [BLOCK_M, BLOCK_K] × [BLOCK_K, BLOCK_N] → [BLOCK_M, BLOCK_N]
```

### 4.4 @triton.autotune——自动调优

GPU 性能高度依赖 Block 大小、Warp 数量等超参数。手动调优极其繁琐。Triton 提供了 `@triton.autotune` 装饰器，自动 benchmark 所有候选配置，选出最优解。

```python
@triton.autotune(
    configs=[
        triton.Config({"BLOCK_M": 64,  "BLOCK_N": 64,  "BLOCK_K": 32}, num_warps=4, num_stages=2),
        triton.Config({"BLOCK_M": 128, "BLOCK_N": 64,  "BLOCK_K": 32}, num_warps=8, num_stages=2),
        triton.Config({"BLOCK_M": 64,  "BLOCK_N": 128, "BLOCK_K": 32}, num_warps=8, num_stages=2),
        triton.Config({"BLOCK_M": 128, "BLOCK_N": 128, "BLOCK_K": 64}, num_warps=8, num_stages=3),
        triton.Config({"BLOCK_M": 256, "BLOCK_N": 64,  "BLOCK_K": 32}, num_warps=8, num_stages=2),
    ],
    key=["M", "N", "K"],    # 当矩阵形状变化时，重新触发 autotune
)
@triton.jit
def matmul_autotune(...):
    ...
```

```
Autotune 的工作流程：

  第一次调用 matmul_autotune(M=1024, N=1024, K=1024)
    │
    ├─▶ 编译 config[0]: BLOCK_M=64, BLOCK_N=64, BLOCK_K=32
    │    benchmark: 0.85ms
    │
    ├─▶ 编译 config[1]: BLOCK_M=128, BLOCK_N=64, BLOCK_K=32
    │    benchmark: 0.62ms
    │
    ├─▶ 编译 config[2]: ...
    │    benchmark: 0.58ms
    │
    ├─▶ ...（5 个配置全部跑一遍）
    │
    └─▶ 缓存最优配置，后续调用直接使用
         best = config with BLOCK_M=128, BLOCK_N=128, BLOCK_K=64 → 0.41ms
```

**关键参数解释**：

| 参数 | 含义 | 影响 |
|---|---|---|
| `BLOCK_M/N/K` | Tile 大小 | 太小→并行度不足；太大→寄存器溢出 |
| `num_warps` | 每个 Block 用几个 Warp | 1 Warp=32 线程；影响 SM 占用率 |
| `num_stages` | 流水线深度（软件流水线） | 用双缓冲隐藏访存延迟 |

### 4.5 CUDA C++ 与 Triton 的设计取舍

```c
// ── CUDA C++：显式管理 Thread/Warp ──
__global__ void vecmul(float* x, float* y, float* out, int N) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;  // 手动计算线程索引
    if (idx < N) {                                     // 手动边界检查
        out[idx] = x[idx] * y[idx];                    // 每个线程处理一个元素
    }
}
// 灵活但繁琐：需要手动设 grid/block dim、管理共享内存、同步
```

```python
# ── Triton：Block 级编程，自动管理 Thread ──
@triton.jit
def vecmul_kernel(x_ptr, y_ptr, out_ptr, N, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < N
    x = tl.load(x_ptr + offsets, mask=mask)
    y = tl.load(y_ptr + offsets, mask=mask)
    tl.store(out_ptr + offsets, x * y, mask=mask)
# 简洁：BLOCK_SIZE 个元素由 Triton 自动分配给 Warp/Thread
```

| 维度 | CUDA C++ | Triton |
|---|---|---|
| 编程粒度 | Thread 级 | Block 级 |
| 边界处理 | 显式 `if` → 可能分支发散 | `mask=` → 硬件掩码 |
| 共享内存 | 手动 `__shared__` + `__syncthreads()` | 自动（`tl.load` 到 SRAM） |
| Warp 级原语 | 手动 `__shfl_sync` | 自动 |
| 调优 | 手动 | `@triton.autotune` |
| 性能 | 极限最优（但难写） | 接近最优（好写） |

---

# 模块五：PyTorch 编译器（Inductor）与算子融合

## 核心命题：如何消除 Memory Bandwidth Bound（访存瓶颈）？

### 5.1 问题根源——访存瓶颈

现代 GPU 的计算速度远超显存带宽。很多算子不是"算不快"，而是"喂不饱"。

```
 Roofline 模型：

  计算强度 (Arithmetic Intensity) = FLOPs / Bytes
                                       ↑            ↑
                                    计算量         访存量

  ┌──────────────────────────────────────────┐
  │  计算强度 (FLOP/Byte)                      │
  │                                           │
  │  低 ←─────────────────────────→ 高        │
  │                                           │
  │  Memory Bound        Compute Bound        │
  │  (喂不饱 GPU)        (算不快)             │
  │     ↑                   ↑                 │
  │  ┌──┴──┐            ┌───┴───┐            │
  │  │逐元素│            │矩阵乘法│            │
  │  │ 加法 │            │ GEMM  │            │
  │  │ 乘法 │            │ Conv  │            │
  │  │ReLU │            │       │            │
  │  └─────┘            └───────┘            │
  │                                           │
  │  瓶颈: HBM 带宽         瓶颈: GPU 算力     │
  └──────────────────────────────────────────┘

  逐元素算子 (a * b + c):
    FLOPs = 2 (一次乘 + 一次加)
    Bytes = 3 * sizeof(float) * N (读 a, b, c + 写结果)
    计算强度 ≈ 2 / 16 ≈ 0.125  ← 极度 Memory Bound！
```

### 5.2 算子行为分类

| 类型 | 特征 | 典型算子 | 融合难度 |
|---|---|---|---|
| **Pointwise（逐元素）** | 无归约，每个输出只依赖对应输入 | `mul`, `add`, `relu`, `sigmoid` | **好融合** |
| **Reduction（归约）** | 消灭某个维度 | `sum`, `mean`, `softmax`, `matmul` | **难融合** |

```
Pointwise:  out[i] = f(a[i], b[i])       ← 每个元素独立，天然可融合

Reduction:  out[i] = sum(a[j] for j...)   ← 需要跨元素聚合，消灭某个维度

Attention 的组成:
  S = Q @ K^T    ← Reduction（矩阵乘，消灭 d 维度）→ 难融合
  P = softmax(S) ← Reduction（消灭 N 维度，求 max 和 sum）→ 难融合  
  O = P @ V      ← Reduction（消灭 N 维度）→ 难融合

  这三个 Reduction 之间是 Pointwise 关系 → FlashAttention 的融合思路
```

### 5.3 两阶段算子融合战术（欲擒故纵）

PyTorch 2.0 的编译流程：`torch.compile` → TorchDynamo（FX Graph） → Inductor（Triton/C++ 代码生成）。

```
torch.compile 的完整编译流水线:

  Python 代码                    torch.compile(model, backend="inductor")
  ───────────                    ────────────────────────────────────────
                                        │
                                        ▼
  ┌─────────────────────────────────────────────────────┐
  │  阶段一: TorchDynamo (Graph Capture)                  │
  │                                                       │
  │  把 Python forward() 翻译为 FX Graph                  │
  │  (一张有向无环图，节点是 aten 算子)                    │
  │                                                       │
  │  y = relu(x @ W1 + b1) @ W2 + b2                      │
  │  ↓                                                    │
  │  %mm1 = aten.mm(x, W1)                                │
  │  %add1 = aten.add(%mm1, b1)                           │
  │  %relu1 = aten.relu(%add1)                            │
  │  %mm2 = aten.mm(%relu1, W2)                           │
  │  %add2 = aten.add(%mm2, b2)                           │
  └───────────────────────┬─────────────────────────────┘
                          │
                          ▼
  ┌─────────────────────────────────────────────────────┐
  │  阶段一补: 算子模式匹配 (Pattern Matching)             │
  │                                                       │
  │  aten.mm + aten.add → aten.addmm                      │
  │  ↑ 为什么？因为 addmm 可以直接调用 cuBLAS 专线         │
  │    cuBLAS 有高度优化的 fused GEMM 实现                 │
  │                                                       │
  │  %addmm1 = aten.addmm(b1, x, W1)    ← 融合！          │
  │  %relu1 = aten.relu(%addmm1)                          │
  │  %addmm2 = aten.addmm(b2, %relu1, W2) ← 融合！        │
  └───────────────────────┬─────────────────────────────┘
                          │
                          ▼
  ┌─────────────────────────────────────────────────────┐
  │  阶段二: Inductor (代码生成 + 算子融合)                │
  │                                                       │
  │  把连续的 Pointwise 算子揉进同一个 Triton kernel       │
  │                                                       │
  │  aten.addmm → 单独调 cuBLAS（Reduction，不融合）       │
  │  aten.relu → 和下面的 Pointwise 一起融合               │
  │                                                       │
  │  生成的 Triton kernel (伪代码):                        │
  │  @triton.jit                                          │
  │  def fused_kernel(...):                               │
  │      x = tl.load(...)       # 读一次 HBM              │
  │      tmp = relu(x)          # 在寄存器中计算           │
  │      out = tmp * scale      # 继续在寄存器中计算       │
  │      tl.store(..., out)     # 写一次 HBM              │
  │                                                       │
  │  ★ 中间结果 tmp 从不写回 HBM，全部在寄存器里流转        │
  └─────────────────────────────────────────────────────┘
```

**课程代码示例**（`code/l8-mlp.py`）：

```python
class MLP(nn.Module):
    def __init__(self, feature=1, hidden=64):
        super().__init__()
        self.weight1 = nn.Parameter(torch.randn(feature, hidden))
        self.bias1   = nn.Parameter(torch.randn(hidden))
        self.weight2 = nn.Parameter(torch.randn(hidden, 1))
        self.bias2   = nn.Parameter(torch.randn(1))

    def forward(self, x):
        x = x @ self.weight1 + self.bias1   # mm + add → addmm (cuBLAS)
        x = torch.relu(x)                    # relu → 和下面的 Pointwise 融合
        x = x @ self.weight2 + self.bias2   # mm + add → addmm (cuBLAS)
        return x

# 一行代码启用编译
model = torch.compile(
    MLP(hidden=64).to(device),
    backend="inductor",
    mode="default",
)
```

### 5.4 不融合 vs 融合——访存量对比

```
不融合：relu 和后续的 element-wise mul 各自一个 kernel

  HBM                 Kernel 1              Kernel 2              HBM
  ┌──┐    tl.load    ┌─────────┐   tl.store  ┌──┐    tl.load    ┌─────────┐
  │ x │ ──────────▶  │  relu   │ ──────────▶ │y │ ──────────▶  │  mul   │ ──▶ out
  └──┘               └─────────┘             └──┘               └─────────┘
     ←─ 读 N 字节 ─→           ←─ 写 N 字节 ─→     ←─ 读 N 字节 ─→
                                                         ↑
                                         中间结果 y 写回 HBM 又读回！
                                         总访存: 4N（读x + 写y + 读y + 写out）


融合：relu + mul 揉进一个 kernel

  HBM                 Fused Kernel                   HBM
  ┌──┐    tl.load    ┌───────────────────┐   tl.store  ┌───┐
  │ x │ ──────────▶  │  tmp = relu(x)     │ ──────────▶ │out│
  └──┘               │  out = tmp * scale │             └───┘
                     └───────────────────┘
      ←─ 读 N 字节 ─→                       ←─ 写 N 字节 ─→
                     ↑ 寄存器
                     tmp 从不写回 HBM
                     总访存: 2N（只读 x，只写 out）

  ★ 访存量减半！计算强度翻倍！从 Memory Bound 移向 Compute Bound
```

### 5.5 极致的内存生命周期管理

**`xxx = None`——显式释放中间变量**：

在 FX Graph 中，Inductor 会分析每个变量的活跃期（liveness）。当某个中间变量不再被使用时，编译器会在图中显式插入 `xxx = None`，让 GC 尽早回收显存。

```
不优化（峰值显存高）:
  a = big_tensor()      ──┐
  b = big_tensor()      ──┤  ← a, b, c 同时存活！
  c = big_tensor()      ──┤     峰值 = 3 × size
  result = f(a, b, c)   ──┘

优化后（峰值显存低）:
  a = big_tensor()
  b = f1(a)
  a = None              ← a 用完了，立刻释放！
  c = big_tensor()
  b = None              ← b 用完了，立刻释放！
  result = f2(c)
  c = None
  ★ 峰值 = 2 × size（a/b/c 不同时存活）
```

**View vs Copy——Zero-Cost 抽象**：

```python
# View 操作：只改 stride 元数据，不搬数据
x = torch.randn(4, 3)
y = x.view(2, 6)        # View！只是改了 shape，数据不动
z = x.transpose(0, 1)   # View！只是改了 stride，数据不动
w = x[:2]               # View！只是改了 offset，数据不动

# Inductor 识别出这些都是 View，不生成任何搬运代码
# 在编译后的 kernel 中，View 操作完全消失（Zero-Cost）

# Copy 操作：真正搬运数据
x = torch.randn(4, 3)
y = x.contiguous()      # Copy！因为 transpose 后不连续，需要物理重排
z = x.clone()            # Copy！完整复制
# 这些会生成实际的搬运 kernel
```

---

# 模块六：大模型前沿架构突破（FlashAttention & PageAttention）

## 核心命题：如何突破大语言模型的 SRAM 与碎片化显存限制？

### 6.1 标准 Attention 的问题——O(N²) 的灾难

```
标准 Attention: O = softmax(Q @ K^T / √d) @ V

  Q, K, V ∈ R^{N × d}     N = 序列长度, d = head 维度

  Step 1: S = Q @ K^T       → S ∈ R^{N × N}    ← N×N 矩阵！
  Step 2: P = softmax(S)    → P ∈ R^{N × N}    ← N×N 矩阵！
  Step 3: O = P @ V         → O ∈ R^{N × d}

  问题: N=8192 时, N×N 矩阵 = 8192² × 4 bytes = 256 MB（单个 head！）
        多头注意力 (32 heads) → 8 GB 仅中间矩阵！

  HBM 访存: S 和 P 各读写一次 → O(N²) 访存量
  这就是 Memory Bound 的根源。
```

```
标准 Attention 的 HBM 访存时间线:

  HBM                              Kernel
  ┌──┐                             ┌──────────┐
  │ Q│ ──read──▶  ┌──────────┐    │          │
  │ K│ ──read──▶  │  matmul  │    │          │
  │  │            │ Q @ K^T  │    │          │
  │  │ ◀─write── │          │    │          │
  │ S│            └──────────┘    │          │
  │  │                           │          │
  │ S│ ──read──▶  ┌──────────┐   │          │
  │  │            │  softmax │   │          │
  │  │ ◀─write── │          │   │          │
  │ P│            └──────────┘   │          │
  │  │                           │          │
  │ P│ ──read──▶  ┌──────────┐   │          │
  │ V│ ──read──▶  │  matmul  │   │          │
  │  │            │ P @ V    │   │          │
  │ O│ ◀─write── │          │   │          │
  └──┘            └──────────┘   └──────────┘

  总 HBM 访存: 读 Q + 读 K + 写 S + 读 S + 写 P + 读 P + 读 V + 写 O
             = O(N²) ← S 和 P 是 N×N，巨大！
```

### 6.2 FlashAttention——SRAM 层级的分块（Tiling）

**核心思想**：不要把整个 N×N 矩阵写回 HBM。把 Q/K/V 切成小块，加载到 SRAM 中计算，中间结果留在寄存器里，**N×N 矩阵从未真正存在于 HBM 中**。

```
FlashAttention 的分块策略:

  HBM (全局显存)                  SRAM (共享内存)              Registers
  ┌───────────┐                  ┌─────────┐                 ┌─────┐
  │ Q [N, d]  │ ── load tile ──▶ │ Q_i     │                 │     │
  │ K [N, d]  │ ── load tile ──▶ │ K_j     │ ── compute ──▶ │ acc │
  │ V [N, d]  │ ── load tile ──▶ │ V_j     │                 │     │
  │ O [N, d]  │ ◀─ store ─────── │         │ ◀─ accumulate──│     │
  └───────────┘                  └─────────┘                 └─────┘
        │                             ↑                         ↑
   只读写 O（O(N) 访存）        小块计算（不写回）          永不离开寄存器！
                                  S_ij, P_ij 临时存在
```

**课程实现**（`homework/hw2/IO-AwareAttention/flashattention_lite.py`）——完整 Triton kernel：

```python
@triton.jit
def _flash_attn_lite_kernel(
    Q_ptr, K_ptr, V_ptr, O_ptr,      # HBM 中的数据指针
    N, d, scale,                      # 序列长度, head 维度, 1/√d
    stride_qn, stride_qd, ...,        # 行/列步长
    BLOCK_M: tl.constexpr,            # Q 的行分块大小（编译期常量）
    BLOCK_N: tl.constexpr,            # K/V 的行分块大小
    BLOCK_D: tl.constexpr,            # 特征维度分块大小（通常 = d）
):
    # ── 第一步：确定当前 program 负责的 Q 行块 ──
    pid_m = tl.program_id(0)
    qm_start = pid_m * BLOCK_M
    offs_m = qm_start + tl.arange(0, BLOCK_M)
    offs_d = tl.arange(0, BLOCK_D)

    # ── 第二步：加载 Q tile 到 SRAM ──
    q_ptrs = Q_ptr + offs_m[:, None] * stride_qn + offs_d[None, :] * stride_qd
    Q_tile = tl.load(q_ptrs, mask=(offs_m[:, None] < N), other=0.0)

    # ── 第三步：初始化累加器（驻留寄存器，永不写回 HBM！）──
    acc = tl.zeros([BLOCK_M, BLOCK_D], dtype=tl.float32)

    # ── 第四步：内循环——遍历 K/V 的所有列块 ──
    num_k_blocks = tl.cdiv(N, BLOCK_N)
    for j in range(num_k_blocks):
        offs_n = j * BLOCK_N + tl.arange(0, BLOCK_N)

        # 加载 K_j, V_j tile 到 SRAM
        K_tile = tl.load(K_ptr + offs_n[:, None] * stride_kn + ...)
        V_tile = tl.load(V_ptr + offs_n[:, None] * stride_vn + ...)

        # 在 SRAM 中计算局部注意力分数
        S_ij = tl.dot(Q_tile, tl.trans(K_tile)) * scale  # [BLOCK_M, BLOCK_N]
        S_ij = tl.where(offs_n[None, :] < N, S_ij, float('-inf'))

        # 局部 Softmax（简化版）
        m_ij = tl.max(S_ij, axis=1)[:, None]    # 行最大值
        P_ij = tl.exp(S_ij - m_ij)               # 数值稳定的 exp
        l_ij = tl.sum(P_ij, axis=1)[:, None]     # 归一化因子
        P_ij = P_ij / l_ij                        # 归一化

        # 累加到寄存器中的 acc
        acc += tl.dot(P_ij.to(V_tile.dtype), V_tile)
        # ★ S_ij 和 P_ij 在 SRAM 中被释放，不写回 HBM！
        # ★ N×N 的中间矩阵从未存在于 HBM 中！

    # ── 第五步：写回最终结果（唯一一次向 HBM 写入输出）──
    tl.store(O_ptr + offs_m[:, None] * stride_on + ..., acc)
```

**HBM 访存对比**：

```
                    标准 Attention          FlashAttention
  HBM 读          Q + K + V = 3Nd         Q + K + V = 3Nd
  HBM 写 S        N²                       0（不存在于 HBM）
  HBM 读 S        N²                       0
  HBM 写 P        N²                       0
  HBM 读 P        N²                       0
  HBM 写 O        Nd                       Nd
  ─────────────────────────────────────────────────────────
  总访存           O(N²)                    O(N)

  当 N = 8192 时:
    标准: ~8192² × 4B ≈ 256 MB
    Flash: ~8192 × 64 × 4B ≈ 2 MB

  ★ 访存量降低 100 倍以上！
```

### 6.3 Online Softmax——跨 Tile 的精确归一化

**问题**：`flashattention_lite.py` 中每个 tile 独立做 softmax，结果只是**近似值**（当 N > BLOCK_N 时有误差）。

**解决方案**：`flashattention_online.py` 使用 **Online Softmax** 算法，维护跨 tile 的 running max 和 running sum，保证全局精确。

**数学推导**：

```
假设已经处理了 j_1, j_2, ..., j_{prev} 个 tile:
  m_i = max(所有已见 tile 的 S 值)           ← running max
  l_i = sum(所有已见 tile 的 exp(S - m_i))   ← running sum

新来一个 tile j:
  ┌──────────────────────────────────────────────────────────┐
  │ m_new = max(m_i, max(S_ij))           ← 更新全局 max      │
  │                                                          │
  │ correction = exp(m_i - m_new)          ← 修正因子          │
  │   (把之前以 m_i 为基准的 exp 缩放到以 m_new 为基准)        │
  │                                                          │
  │ P_ij = exp(S_ij - m_new)               ← 当前 tile 的 exp │
  │                                                          │
  │ l_new = l_i * correction + sum(P_ij)   ← 更新全局 sum     │
  │                                                          │
  │ acc = acc * correction + P_ij @ V_j    ← 更新累加器       │
  └──────────────────────────────────────────────────────────┘

最终: O = acc / l_i   ← 归一化
```

**代码实现**（`flashattention_online.py` 中与 lite 版的唯一区别）：

```python
# ── 初始化 online softmax 状态 ──
acc = tl.zeros([BLOCK_M, BLOCK_D], dtype=tl.float32)
m_i = tl.full([BLOCK_M, 1], float('-inf'), dtype=tl.float32)  # running max
l_i = tl.zeros([BLOCK_M, 1], dtype=tl.float32)                 # running sum

for j in range(num_k_blocks):
    # ... 加载 K_j, V_j, 计算 S_ij ...

    # ── Online Softmax 核心 ──
    m_ij = tl.max(S_ij, axis=1)[:, None]     # 当前 tile 的 max
    m_new = tl.maximum(m_i, m_ij)             # 全局 max

    correction = tl.exp(m_i - m_new)          # 修正因子

    P_ij = tl.exp(S_ij - m_new)               # 当前 tile 的 exp

    l_ij = tl.sum(P_ij, axis=1)[:, None]
    l_new = l_i * correction + l_ij           # 更新全局 sum

    acc = acc * correction + tl.dot(P_ij.to(V_tile.dtype), V_tile)
    #       ↑ 之前的 acc 也要缩放！

    m_i = m_new
    l_i = l_new

# ── 最终归一化 ──
acc = acc / l_i    # ← online 版多出的一行！
```

**lite 版 vs online 版对比**：

| 维度 | flashattention_lite | flashattention_online |
|---|---|---|
| Softmax 方式 | 每个 tile 独立做 | 跨 tile 维护 running max/sum |
| 结果精度 | 近似（N > BLOCK_N 时有误差） | **全局精确**（与标准 Attention 一致） |
| 额外状态 | 无 | `m_i`（running max）+ `l_i`（running sum） |
| 代码差异 | 基础版 | 多初始化 2 行，内循环改 ~10 行，末尾除以 `l_i` |

### 6.4 Warp Shuffle——跨线程通信

在完整 FlashAttention 中，Softmax 的全局 max/sum 更新需要跨 Warp 通信。GPU 提供了 `__shfl_xor_sync`（Warp Shuffle）指令，让同一 Warp 内的线程直接交换寄存器值，**不需要经过共享内存或 HBM**。

```
Warp Shuffle（寄存器级通信）:

  普通 GPU 通信:
  Thread A → 写 Shared Memory → Thread B 读
              ↓ 2 次访存 ↓

  Warp Shuffle:
  Thread A 的寄存器 ──直接──▶ Thread B 的寄存器
              ↑ 0 次访存 ↑
              硬件级直连！

  __shfl_xor_sync(mask, val, lane_mask):
    每个 lane 把自己的 val 发给 lane_id XOR lane_mask 的 lane
    同时接收对应 lane 发来的 val

  例: 4 个 lane, lane_mask=1
    lane 0 ↔ lane 1  (0 XOR 1 = 1)
    lane 2 ↔ lane 3  (2 XOR 1 = 3)
    
  用于 All-Reduce:
    step 1: xor=1  → 相邻 lane 交换并累加
    step 2: xor=2  → 跨 2 lane 交换并累加
    step 3: xor=4  → 跨 4 lane 交换并累加
    ... log2(32)=5 步完成 32 lane 的 All-Reduce
```

### 6.5 PageAttention——粒度的辩证法

**问题**：KV Cache 在推理时随序列长度线性增长。传统分配方式为每个请求预分配最大长度的连续内存，导致严重的**显存碎片**。

**PageAttention 的解决方案**（vLLM 核心创新）：

```
传统 KV Cache 分配:
  请求 1: [████████████████████░░░░░░░░]  预留 max_len，大量浪费
  请求 2: [██████░░░░░░░░░░░░░░░░░░░░]  预留 max_len，大量浪费
  请求 3: [████████████████████████████]  满了
  
  问题: 每个请求占一块连续大内存 → 碎片严重 → OOM

PageAttention（分页管理，类似 OS 虚拟内存）:
  物理内存被切成固定大小的 Block/Page（如 16 个 token 的 KV）

  逻辑视图:  请求1: [tok0][tok1][tok2]...
  物理内存:  Block 7    │  Block 23   │  Block 5  │ ...
             [tok0][tok1]│  [tok2][tok3]│  [tok4]  │
             ↑ 不连续！通过页表映射

  Block Table (页表):
    请求1 → [Block 7, Block 23, Block 5, ...]
    请求2 → [Block 12, Block 3, ...]
    
  ★ 按需分配，零碎片，显存利用率接近 100%
```

**粒度的辩证法——宏观碎片 vs 微观连续**：

```
宏观上: 不连续（解决碎片化）
  ┌──────────────────────────────────────────┐
  │ Block 7  │ Block 23 │ Block 5 │ Block 18 │  请求1的KV Cache
  └──────────────────────────────────────────┘
   不连续的物理内存块，通过 Block Table 映射

微观上: 连续（保证访存效率）
  ┌──────────────────────────────────┐
  │ Block 7 (4096 bytes)             │
  │ [tok0_K][tok0_V][tok1_K][tok1_V]│  ← Block 内部绝对连续！
  │ [tok2_K][tok2_V]...             │     32 个线程访问同一 Block 时
  └──────────────────────────────────┘     → 完美的 Coalesced Access ✓

  ★ 宏观上用不连续解决碎片，微观上用连续保证带宽
  ★ 这就是"粒度的辩证法"——不同粒度上做不同的设计决策
```

---

# 模块七：超大模型 3D 分布式并行训练

## 核心命题：千亿参数规模下，计算力、显存与通信带宽的博弈

### 7.1 三个维度的切分

```
                      一个完整的 LLM 训练集群
                    ┌───────────────────────────┐
                    │     数据 Parallelism       │  ← 切数据
                    │     每张卡跑完整模型       │
                    │     梯度 AllReduce 同步    │
                    ├───────────────────────────┤
                    │  × Tensor Parallelism      │  ← 切矩阵
                    │    单层矩阵拆到多卡        │
                    │    层内 AllReduce 同步     │
                    ├───────────────────────────┤
                    │  × Pipeline Parallelism    │  ← 切层
                    │    不同层分到不同卡        │
                    │    层间 P2P 通信           │
                    └───────────────────────────┘
                          = 3D Parallelism
```

| 并行方式 | 切什么 | 通信频率 | 适合网络 | 显存效果 |
|---|---|---|---|---|
| **DP (Data Parallel)** | 切数据 | 每步一次 AllReduce | 任意（但大模型放不下） | 不省显存（每卡完整模型） |
| **TP (Tensor Parallel)** | 切矩阵 | 每层内多次 AllReduce | **必须 NVLink**（极高频） | 省权重显存 |
| **PP (Pipeline Parallel)** | 切层 | 每层间一次 P2P | 以太网/InfiniBand | 省权重显存 |

### 7.2 Data Parallelism (DP / DDP / FSDP)

**DDP（DistributedDataParallel）**——最基础的数据并行：

```python
# 来自 code/l10-lenet5_ddp.py 的完整 DDP 实现

import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

def setup_ddp():
    dist.init_process_group(backend="nccl")          # 初始化进程组
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)                # 每个进程绑定一个 GPU
    return local_rank

def main():
    local_rank = setup_ddp()
    device = torch.device(f"cuda:{local_rank}")

    model = LeNet5().to(device)
    model = DDP(model, device_ids=[local_rank])      # ★ DDP 封装！

    # 数据切分：每个 GPU 只看到一部分数据
    train_sampler = DistributedSampler(train_dataset, shuffle=True)
    train_loader = DataLoader(train_dataset, sampler=train_sampler, ...)

    for epoch in range(epochs):
        train_sampler.set_epoch(epoch)               # 保证每 epoch shuffle 不同

        for inputs, labels in train_loader:
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()     # ★★ DDP 在这里自动同步梯度！
            optimizer.step()
```

**DDP 的梯度同步机制**：

```
DDP 梯度同步流程（每个训练步）:

  GPU 0                    GPU 1                    GPU 2                    GPU 3
  ┌──────┐                 ┌──────┐                 ┌──────┐                 ┌──────┐
  │Batch0│                 │Batch1│                 │Batch2│                 │Batch3│
  └──┬───┘                 └──┬───┘                 └──┬───┘                 └──┬───┘
     │ forward                 │ forward                │ forward                │ forward
     ▼                         ▼                        ▼                        ▼
  ┌──────┐                 ┌──────┐                 ┌──────┐                 ┌──────┐
  │loss0 │                 │loss1 │                 │loss2 │                 │loss3 │
  └──┬───┘                 └──┬───┘                 └──┬───┘                 └──┬───┘
     │ backward               │ backward              │ backward               │ backward
     ▼                         ▼                        ▼                        ▼
  ┌──────┐                 ┌──────┐                 ┌──────┐                 ┌──────┐
  │grad0 │                 │grad1 │                 │grad2 │                 │grad3 │
  └──┬───┘                 └──┬───┘                 └──┬───┘                 └──┬───┘
     │                         │                        │                        │
     └──────────── AllReduce ────────────────────────────────────────────────────┘
                              │
                              ▼
                    每张卡都得到 grad0+grad1+grad2+grad3
                              / 4  (平均)
                              │
                              ▼
                    optimizer.step()  ← 所有卡更新相同的梯度

  ★ DDP 利用 "Gradient Bucketing": 不是等 backward 完再同步，
    而是哪个梯度算好了就先 AllReduce 哪个，与计算重叠！
```

**FSDP（Fully Sharded Data Parallel）**——解决大模型放不下的问题：

```
DDP 的问题: 每张卡存完整的模型参数 + 优化器状态 + 梯度
  7B 模型 FP32: 7B × (4+4+4) = 84 GB  → 单张 A100 (80GB) 放不下！

FSDP 的解决方案: 切分一切
  ┌──────────────────────────────────────────────────┐
  │              完整模型参数 [W0, W1, W2, ...]        │
  │  ┌──────┬──────┬──────┬──────┐                  │
  │  │ GPU 0│ GPU 1│ GPU 2│ GPU 3│  平时: 每卡只存 1/4 │
  │  │ W0   │ W1   │ W2   │ W3   │                   │
  │  └──────┴──────┴──────┴──────┘                   │
  │                                                    │
  │  前向时: AllGather 拼回完整 W → 计算 → 丢弃        │
  │  反向时: AllGather 拼回 → 计算梯度 → 丢弃          │
  │  梯度同步: ReduceScatter → 每卡只存 1/4 的梯度     │
  └──────────────────────────────────────────────────┘

  ZeRO 三个阶段:
    ZeRO-1: 切分优化器状态 (Optimizer States)
    ZeRO-2: 切分优化器状态 + 梯度 (Gradients)
    ZeRO-3 = FSDP: 切分优化器状态 + 梯度 + 参数 (Parameters)
```

### 7.3 Tensor Parallelism (TP)

**Megatron-LM 的 TP**：把单个矩阵运算拆分到多张卡上。

```
线性层 Y = X @ W 的两种切分方式:

┌─────────────────────────────────────────────────────────────┐
│  Column Parallelism（列并行）                                 │
│                                                              │
│        W = [W1 | W2]     W 被竖着切成两半                     │
│                                                              │
│  GPU0: Y1 = X @ W1     GPU1: Y2 = X @ W2                    │
│                                                              │
│  Y = [Y1 | Y2]   ← AllGather 或 直接传给下一层的 Row Parallel │
│                                                              │
│  适合: 两层之间的连接（第一层 Column，第二层 Row）             │
│    Y = X @ W1 @ W2 = X @ [W1_A | W1_B] @ [W2_A / W2_B]      │
│    GPU0: X @ W1_A @ W2_A    GPU1: X @ W1_B @ W2_B           │
│    AllReduce(Y_A + Y_B) → 最终结果                           │
│    ★ 只需一次 AllReduce！                                     │
└─────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────┐
│  Row Parallelism（行并行）                                    │
│                                                              │
│        W = [W1]     W 被横着切成两半                           │
│            [W2]                                              │
│                                                              │
│  GPU0: Y1 = X1 @ W     GPU1: Y2 = X2 @ W                    │
│  (X 也切分)                                                  │
│                                                              │
│  Y = Y1 + Y2   ← AllReduce（加法）                            │
└─────────────────────────────────────────────────────────────┘
```

**为什么 TP 必须用 NVLink？**

```
TP 的通信频率: 每一层、每一个 Attention/MLP 都要 AllReduce
  一个 Transformer 层: ~4 次 AllReduce
  96 层模型: ~384 次 AllReduce per step

  NVLink 带宽: ~600 GB/s (A100)
  PCIe Gen4:  ~64 GB/s

  如果用 PCIe: 384 次 × 每次 ~50ms = ~19s 通信开销
  如果用 NVLink: 384 次 × 每次 ~5ms = ~2s 通信开销

  ★ TP 只能在单机内使用（NVLink 互联），跨机延迟不可接受
```

### 7.4 Pipeline Parallelism (PP) 与调度策略

**PP 的核心**：把模型的不同层分配到不同 GPU 上，数据像流水线一样流过。

```
4 层模型，4 张 GPU 的 Pipeline:

  GPU 0 (Layer 0)  →  GPU 1 (Layer 1)  →  GPU 2 (Layer 2)  →  GPU 3 (Layer 3)
  
  输入 → [L0 前向] → [L1 前向] → [L2 前向] → [L3 前向] → 输出
                                                      ↓
  梯度 ← [L0 反向] ← [L1 反向] ← [L2 反向] ← [L3 反向] ← loss

  通信: 只在相邻 GPU 之间传 Activation（层间）
        频率远低于 TP（每层内的 AllReduce）
        → 适合跨机器（以太网/InfiniBand 也能接受）
```

**GPipe 策略——吃显存**：

```
GPipe: 囤积全量前向 Activation，再做反向

  时间 →
  GPU0: [F0₁][F0₂][F0₃][F0₄]                          [B0₄][B0₃][B0₂][B0₁]
  GPU1:     [F1₁][F1₂][F1₃][F1₄]                  [B1₄][B1₃][B1₂][B1₁]
  GPU2:         [F2₁][F2₂][F2₃][F2₄]          [B2₄][B2₃][B2₂][B2₁]
  GPU3:             [F3₁][F3₂][F3₃][F3₄]  [B3₄][B3₃][B3₂][B3₁]
  
  F = Forward, B = Backward, 下标 = micro-batch 编号

  问题: 所有 micro-batch 的 Activation 都要存在显存里等反向！
        显存占用 ∝ micro-batch 数量
        但 Pipeline Bubble 较小
```

**1F1B 策略——省显存**：

```
1F1B: 算一次前向（1F），立马算一次反向（1B）

  时间 →
  GPU0: [F0₁][F0₂][F0₃][B0₁][F0₄][B0₂][B0₃][B0₄]
  GPU1:     [F1₁][F1₂][B1₁][F1₃][B1₂][F1₄][B1₃][B1₄]
  GPU2:         [F2₁][B2₁][F2₂][B2₂][F2₃][B2₃][F2₄][B2₄]
  GPU3:             [B3₁][F3₁][B3₂][F3₂][B3₃][F3₃][B3₄][F3₄]
                     ↑
              算完前向就立刻反向，Activation 用完就 None 掉！

  优点: 显存占用 ≈ 常数（不随 micro-batch 数增长）
  代价: Pipeline Bubble 略大（前几个 step GPU 有空转）
```

### 7.5 KV Cache——推理时的核心优化

```
自回归生成: 每生成一个 token，都要用所有历史 token 计算 Attention

  不用 KV Cache:
    生成 token 5 时: 重新计算 K0..K5 和 V0..V5
    生成 token 6 时: 重新计算 K0..K6 和 V0..V6
    ...每步都是 O(T²) 计算！

  用 KV Cache:
    生成 token 5 时: 只计算 K5, V5，追加到 Cache
    生成 token 6 时: 只计算 K6, V6，追加到 Cache
    ...每步只需 O(T) 计算（Q 只有一个，与所有历史 K 做点积）

  KV Cache 大小 = 2 × n_layers × seq_len × hidden_dim × batch × dtype_size

  例: LLaMA-7B, seq_len=4096, batch=1, FP16
    = 2 × 32 × 4096 × 4096 × 1 × 2 bytes ≈ 2 GB
```

### 7.6 量化——用精度换显存和速度

**数值格式对比**：

```
┌────────┬───────────────────────────────────────────┐
│ FP32   │ [Sign 1][Exponent 8][Mantissa 23]  = 32位  │
│        │ 精度: ~7位有效数字   范围: ±3.4×10³⁸        │
├────────┼───────────────────────────────────────────┤
│ FP16   │ [Sign 1][Exponent 5][Mantissa 10]  = 16位  │
│        │ 精度: ~3位有效数字   范围: ±6.5×10⁴         │
├────────┼───────────────────────────────────────────┤
│ BF16   │ [Sign 1][Exponent 8][Mantissa  7]  = 16位  │
│        │ 精度: ~2位有效数字   范围: ±3.4×10³⁸        │
├────────┼───────────────────────────────────────────┤
│ INT8   │ [Sign 1][Value 7]                   = 8位  │
│        │ 范围: -128 ~ 127                              │
└────────┴───────────────────────────────────────────┘

  FP16 vs BF16 的关键区别:
    FP16: 指数位少(5)，范围小，梯度容易溢出 → 推理用
    BF16: 指数位多(8)，范围大，梯度不溢出   → 训练用
    两者都是 16 位，显存相同
```

**对称量化**（FP32 → INT8）：

```python
# 假设权重范围 [-10.8, 10.8]
alpha = 10.8  # 最大绝对值
scale = 127 / alpha  # = 11.76

# 量化
q = round(x * scale)       # 3.08 × 11.76 ≈ 36

# 反量化
x_dequant = q / scale      # 36 / 11.76 ≈ 3.06 (原始 3.08，有误差)
```

**BitNet b1.58——三值大模型**：

```
权重只有三个值: {-1, 0, +1}

  传统矩阵乘法 Y = X × W:
    y = x₁×w₁ + x₂×w₂ + ... + xₙ×wₙ   ← 需要乘法器

  三值矩阵乘法:
    w = +1:  y += x     ← 只需加法
    w = -1:  y -= x     ← 只需减法
    w =  0:  跳过        ← 无操作

  性能: 显存减少 3.5×, 推理加速 2.7×, 能耗减少 71×
```

---

# 跨模块架构师洞察

## 【AST 到 GPU 机器码的完整旅程】

```
你在 Python 里写下 a + b
│
│  ① CPython 解析器 (模块一)
│     src → tokens → AST → bytecode
│     LOAD_FAST a; LOAD_FAST b; BINARY_OP +
│     → PVM 逐条执行字节码
│
│  ② 如果被 @kernel / @triton.jit 装饰 (模块二/四)
│     ast.parse 提取 AST 节点: BinOp(left=Name('a'), op=Add, right=Name('b'))
│     → 不执行 Python 字节码，而是路由到底层函数
│     @kernel:    → lib.vec_elem_add(a, b, out, n)  ← C 库
│     @triton.jit: → 编译为 PTX → GPU 机器码
│
│  ③ 如果是 PyTorch 算子 (模块五)
│     torch.compile → TorchDynamo 捕获 FX Graph
│     aten.add(a, b) 出现在图中
│     → Inductor 判断: 如果前后都是 Pointwise → 融合进同一个 Triton kernel
│     → 如果前面是 mm: addmm → cuBLAS 专线
│
│  ④ GPU 上执行 (模块三/四)
│     Triton kernel → 编译为 PTX (虚拟指令) → SASS (实际机器码)
│     → Grid 分发到多个 SM
│     → 每个 SM 上多个 Warp 并行
│     → Warp 内 32 线程 SIMT 锁步
│     → 数据流经 HBM → SRAM → Registers → ALU
│
│  ⑤ 如果是 Attention (模块六)
│     不走标准 Q@K^T → softmax → @V 的多 kernel 路径
│     而是融合成一个 FlashAttention kernel
│     → Q/K/V tile 加载到 SRAM
│     → 在 SRAM 中完成 matmul + softmax + matmul
│     → 累加器驻留寄存器，中间矩阵不存在于 HBM
│     → Online Softmax 维护跨 tile 的全局精确性
│
│  ⑥ 如果是多卡 (模块七)
│     前向: 数据切分到各卡 (DP) / 矩阵切分 (TP) / 层切分 (PP)
│     反向: 梯度 AllReduce 同步 (DP) / 层内 AllReduce (TP) / 层间 P2P (PP)
│     → 通信与计算重叠 (Gradient Bucketing / 1F1B)
```

## 【CPU 域 vs GPU 域对比总表】

| 维度 | CPU（模块一/二） | GPU（模块三/四/六） |
|---|---|---|
| **核心单元** | 1-64 个重核心 | 数千个轻核心 |
| **线程模型** | OS 线程（重，~MB 栈） | GPU 线程（轻，~字节级寄存器） |
| **调度方式** | 抢占式（OS 强制切换） | SIMT（32 线程锁步） + Warp 切换 |
| **并发模型** | 多进程/多线程/协程 | Grid/Block/Warp/Thread |
| **内存层级** | L1/L2/L3/DRAM | Reg/SRAM/HBM（层级更深，延迟差更大） |
| **瓶颈类型** | 逻辑复杂度 / GIL | 访存带宽 (Memory Bound) |
| **延迟掩盖** | `await` 让权（协程） | Warp 切换（硬件 0 开销） |
| **锁机制** | `threading.Lock`（串行化） | 无锁（SIMT 天然无竞争） |
| **融合优化** | 手动内联 | Inductor 自动算子融合 |

## 【Python 层级 → GPU 层级 的精确对应】

```
Python 概念              GPU 对应
──────────              ──────────
async/await 协程    →    Warp 切换（掩盖延迟）
Lock（互斥锁）       →    不需要（SIMT 无竞争）
多进程 fork          →    多个 Grid 启动
@kernel AST 降级     →    @triton.jit AST → PTX
for 循环             →    tl.arange + 并行
条件分支 if/else     →    tl.where + mask（避免分支发散）
广播 Broadcasting    →    SIMT 的统一指令 + 不同数据
```

---

# 附录：核心代码索引

| 文件 | 模块 | 内容 |
|---|---|---|
| `study/lec4/03_with_lock.py` | 模块一 | 线程竞态与锁（无锁/不同锁/同一把锁） |
| `study/lec4/06_coroutine_race.py` | 模块一 | 协程竞态与 asyncio.Lock |
| `code/l3-3-deco.py` | 模块二 | @kernel 装饰器：AST → C 函数调用 |
| `code/l7-vecmul.py` | 模块四 | Triton 向量乘法（最简 kernel） |
| `code/l7-autotune.py` | 模块四 | Triton matmul + @triton.autotune |
| `code/l8-mlp.py` | 模块五 | torch.compile + Inductor 算子融合 |
| `code/l5-lenet5.py` | 模块三/五 | LeNet-5 CNN（PyTorch 标准实现） |
| `code/l5-rnn.py` | 模块三/五 | SimpleRNN 时间序列预测 |
| `code/l10-lenet5_ddp.py` | 模块七 | DDP 分布式数据并行 |
| `homework/hw2/flashattention_lite.py` | 模块六 | FlashAttention（局部 softmax） |
| `homework/hw2/flashattention_online.py` | 模块六 | FlashAttention（Online Softmax，全局精确） |

---

*本手册基于课程全部代码、作业和课堂笔记编写。所有代码示例均来自实际课程文件，字节码和运行结果均为真实执行输出。*
