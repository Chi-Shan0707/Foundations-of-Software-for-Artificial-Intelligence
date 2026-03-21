# L3: Python 与 Native 交互

> 本节课核心：理解 Python 性能瓶颈，以及如何通过调用 Native 代码加速计算。

---

## 目录

1. [Python 性能问题的根本原因](#1-python-性能问题的根本原因)
2. [Python 与 Native 交互方式](#2-python-与-native-交互方式)
3. [内存布局对性能的影响](#3-内存布局对性能的影响)
4. [ctypes: 调用 C 动态库](#4-ctypes-调用-c-动态库)
5. [Numba: JIT 编译](#5-numba-jit-编译)
6. [装饰器与 DSL](#6-装饰器与-dsl)
7. [工具链总结](#7-工具链总结)

---

## 1. Python 性能问题的根本原因

### 1.1 虚拟机解释执行

```
Python 源代码
    ↓
字节码 (Bytecode)
    ↓
Python 虚拟机 (解释器循环)
    ↓
CPU 执行
```

**问题**：每一条字节码都需要解释器处理，包含：
- 类型检查
- 引用计数管理
- 动态分发
- 虚拟机调度开销

### 1.2 Python 对象的内存开销

```python
# Python float 对象
import sys
sys.getsizeof(1.0)  # 24 bytes

# C double
sizeof(double)  # 8 bytes
```

**Python float 结构**：
```
PyObject_HEAD (16 bytes)
  - ob_refcnt: 引用计数
  - ob_type: 类型指针
━━━━━━━━━━━━━━━━━━━━━━
double value (8 bytes)
━━━━━━━━━━━━━━━━━━━━━━
Total: 24 bytes
```

### 1.3 性能对比

| 实现方式 | 执行路径 | 相对性能 |
|---------|---------|---------|
| 纯 Python | 字节码 → 解释器 → CPU | 1x (基准) |
| NumPy | Python 调用 → C 内核 → CPU | ~100x |
| ctypes | Python → FFI → C 代码 → CPU | ~100x |
| Numba JIT | 编译成机器码 → CPU | ~100x |

---

## 2. Python 与 Native 交互方式

### 2.1 交互方式对比

```
┌─────────────────────────────────────────────────────────────────────┐
│                         Python 与 Native 交互                        │
├─────────────────────────────────────────────────────────────────────┤
│                                                                     │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐  ┌───────────┐ │
│  │   ctypes    │  │  C Extension│  │    Numba    │  │   Cython  │ │
│  │   (FFI)      │  │  (深度集成)  │  │   (JIT)     │  │  (AOT)     │ │
│  └─────────────┘  └─────────────┘  └─────────────┘  └───────────┘ │
│         │                │                │              │         │
│         ▼                ▼                ▼              ▼         │
│  ┌─────────────────────────────────────────────────────────────┐   │
│  │                    Native Code (C/C++/机器码)               │   │
│  └─────────────────────────────────────────────────────────────┘   │
│                                                                     │
└─────────────────────────────────────────────────────────────────────┘
```

### 2.2 特点对比

| 方式 | 优点 | 缺点 | 适用场景 |
|------|------|------|---------|
| **ctypes** | 无需编译、简单 | 有调用开销、类型转换 | 快速原型、复用现有库 |
| **C Extension** | 零开销、深度集成 | 需要编译、复杂 | 核心算法、高性能需求 |
| **Numba** | Python 语法、透明 | 首次编译慢 | 数值计算、循环密集 |
| **Cython** | 接近 C 性能 | 需要额外语法 | 大型项目、已有代码 |

---

## 3. 内存布局对性能的影响

### 3.1 Python List vs NumPy Array

```
Python List (离散内存)
┌─────┐   ┌─────┐   ┌─────┐   ┌─────┐
│ 1.0 │ → │ 2.0 │ → │ 3.0 │ → │ 4.0 │
└─────┘   └─────┘   └─────┘   └─────┘
  24B      24B        24B        24B
  (分散在堆中，通过指针连接)

NumPy Array (连续内存)
┌──────────────────────────────────────┐
│ 1.0 │ 2.0 │ 3.0 │ 4.0 │ ... │       │
└──────────────────────────────────────┘
  8B    8B    8B    8B
  (连续内存，缓存友好)
```

### 3.2 性能影响

| 特性 | Python List | NumPy Array |
|------|------------|-------------|
| 内存连续性 | ❌ 离散 | ✅ 连续 |
| 缓存命中率 | 低 | 高 |
| SIMD 向量化 | ❌ 不支持 | ✅ 支持 |
| 内存开销 | 3x | 1x |

---

## 4. ctypes: 调用 C 动态库

### 4.1 编译 C 代码为动态库

```bash
# 编译 C 代码
gcc -O3 -shared -fPIC -o libdot.so l3-1-bench_dot.c

# 选项说明
# -O3:     最高优化级别
# -shared: 生成共享库
# -fPIC:   生成位置无关代码
```

### 4.2 Python 加载和调用

```python
from ctypes import CDLL, c_double, POINTER
import numpy as np

# 1. 加载动态库（操作系统把机器码载入内存）
lib = CDLL("./libdot.so")

# 2. 配置函数签名
lib.dot_c.restype = c_double
lib.dot_c.argtypes = [
    POINTER(c_double),  # a 指针
    POINTER(c_double),  # b 指针
    c_int,              # n 长度
]

# 3. 准备数据
a = np.array([1.0, 2.0, 3.0], dtype=np.float64)
b = np.array([4.0, 5.0, 6.0], dtype=np.float64)

# 4. 获取内存指针
a_ptr = a.ctypes.data_as(POINTER(c_double))
b_ptr = b.ctypes.data_as(POINTER(c_double))

# 5. 调用 C 函数
result = lib.dot_c(a_ptr, b_ptr, c_double(len(a)))
```

### 4.3 执行层级

```
┌─────────────────────────────────────────────────────────────┐
│  Python 进程                                                 │
│  ┌─────────────┐      ┌──────────────────────────────┐      │
│  │ Python 代码  │      │  libdot.so (机器码)           │      │
│  │             │ ───► │  ┌────────────────────────┐   │      │
│  │ lib.dot_... │      │  │ dot_c:                 │   │      │
│  │             │      │  │   MOV ...              │   │      │
│  └─────────────┘      │  │   MUL ...              │   │      │
│                       │  └────────────────────────┘   │      │
│                       └──────────────────────────────┘      │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
                         CPU 直接执行
```

---

## 5. Numba: JIT 编译

### 5.1 装饰器 `@njit` 的工作原理

```python
from numba import njit

@njit
def dot_numba(a, b):
    s = 0.0
    for i in range(len(a)):
        s += a[i] * b[i]
    return s
```

**等价于**：
```python
def dot_numba(a, b):
    s = 0.0
    for i in range(len(a)):
        s += a[i] * b[i]
    return s
dot_numba = njit(dot_numba)
```

### 5.2 JIT 编译流程

```
┌─────────────────────────────────────────────────────────────────┐
│  定义时：@njit 装饰器执行                                         │
├─────────────────────────────────────────────────────────────────┤
│  创建 JITCompiledFunction 对象                                   │
│  - 保存原始 Python 函数的 AST                                     │
│  - 编译缓存 = {}（空）                                            │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│  第一次调用：dot_numba(a_np, b_np)                               │
├─────────────────────────────────────────────────────────────────┤
│  1. 类型推断：根据实际参数推断类型                                │
│     a_np: ndarray[float64, 1d]                                   │
│     b_np: ndarray[float64, 1d]                                   │
│     s:    float64                                                │
├─────────────────────────────────────────────────────────────────┤
│  2. 生成 LLVM IR                                                 │
├─────────────────────────────────────────────────────────────────┤
│  3. LLVM 优化（向量化、循环展开等）                               │
├─────────────────────────────────────────────────────────────────┤
│  4. 生成机器码                                                    │
├─────────────────────────────────────────────────────────────────┤
│  5. 存入缓存                                                      │
├─────────────────────────────────────────────────────────────────┤
│  6. 执行机器码                                                    │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│  后续调用：直接从缓存取机器码执行                                  │
└─────────────────────────────────────────────────────────────────┘
```

### 5.3 Numba 组件

```
Python 源代码
    ↓
┌─────────┐  ┌───────┐  ┌─────────────────┐
│ Parser  │→ │  AST  │→ │ Type Inference  │
└─────────┘  └───────┘  └─────────────────┘
                            ↓
              ┌───────────────────────┐
              │     Numba IR          │
              └───────────────────────┘
                            ↓
              ┌───────────────────────┐
              │     LLVM IR           │
              └───────────────────────┘
                            ↓
              ┌─────────┐  ┌───────┐  ┌───────────┐
              │  LLVM   │→ │ Opt   │→ │ Code Gen  │
              └─────────┘  └───────┘  └───────────┘
                            ↓
              ┌───────────────────────┐
              │   Native Code (机器码) │
              └───────────────────────┘
```

### 5.4 验证编译结果

```python
from numba import njit
import numpy as np

@njit
def dot_numba(a, b):
    s = 0.0
    for i in range(len(a)):
        s += a[i] * b[i]
    return s

a = np.array([1.0, 2.0, 3.0])
b = np.array([4.0, 5.0, 6.0])
result = dot_numba(a, b)  # 触发编译

# 查看编译签名
print(dot_numba.signatures)
# [(float64[:], float64[:])]

# 查看类型信息
print(dot_numba.inspect_types())

# 查看 LLVM IR
print(dot_numba.inspect_llvm())

# 查看汇编代码
print(dot_numba.inspect_asm())
```

---

## 6. 装饰器与 DSL

### 6.1 自定义装饰器示例

```python
# l3-3-deco.py

def kernel(func):
    """
    将函数体作为嵌入式 DSL 处理
    函数体被解析，而不是执行
    """
    # 解析 Python AST
    src = inspect.getsource(func)
    tree = ast.parse(src)
    func_def = tree.body[0]

    # 期望: return <expr>
    return_stmt = func_def.body[0]
    expr = return_stmt.value
    arg_names = [arg.arg for arg in func_def.args.args]

    # Lowering: AST → primitive operations
    def lower(node, env, n):
        if isinstance(node, ast.Name):
            return env[node.id]

        if isinstance(node, ast.BinOp):
            left = lower(node.left, env, n)
            right = lower(node.right, env, n)
            out = np.empty_like(left)

            if isinstance(node.op, ast.Mult):
                lib.vec_elem_mul(...)
            elif isinstance(node.op, ast.Add):
                lib.vec_elem_add(...)

            return out

    # 运行时包装器
    def wrapper(*args):
        arrays = [np.asarray(a, dtype=np.float64) for a in args]
        n = arrays[0].size
        env = dict(zip(arg_names, arrays))
        result = lower(expr, env, n)
        return result

    return wrapper
```

### 6.2 使用装饰器

```python
@kernel
def vec_elem_mul(a, b):
    return a * b  # 被解析为 AST，调用 C 函数

@kernel
def vec_elem_fma(a, b, c):
    return (a * b) + c  # 融合乘加
```

### 6.3 自举 (Bootstrapping)

```
自举：最开始的版本必须用更低层级实现

汇编
  ↓
C (基础版)
  ↓
Python (解释器)
  ↓
装饰器/JIT (实现加速)
  ↓
更高级的 DSL
```

---

## 7. 工具链总结

### 7.1 编译命令速查

```bash
# C/C++ 编译
gcc -O0 -shared -fPIC -o libdot.so dot.c      # 无优化
gcc -O3 -shared -fPIC -o libdot.so dot.c      # 最高优化
g++ -std=c++17 -O3 -shared -fPIC -o lib.so dot.cpp

# 查看生成的内容
file libdot.so         # 文件类型
nm libdot.so          # 符号表
objdump -d libdot.so  # 反汇编
readelf -S libdot.so  # Section 信息

# 调试
gdb -ex "break main" -ex "run" ./program
```

### 7.2 Python 基准测试

```python
import time

def benchmark(func, *args, repeat=10):
    times = []
    for _ in range(repeat):
        start = time.perf_counter()
        func(*args)
        end = time.perf_counter()
        times.append(end - start)
    return np.mean(times), np.std(times)

# 使用
t_mean, t_std = benchmark(dot_product, a, b)
print(f"Mean: {t_mean:.4f} ± {t_std:.4f} s")
```

### 7.3 Conda 环境管理

```bash
# 创建虚拟环境
conda create -n lec3 python=3.10 numpy numba

# 激活环境
conda activate lec3

# 安装包
conda install numpy
pip install numba

# 导出环境
conda env export > environment.yml
```

---

## 8. 文件清单

| 文件 | 说明 |
|------|------|
| `dot_product.cpp` | C++ 点积实现（含命令行注释） |
| `bench.py` | Python 基准测试框架 |
| `l3-1-bench_dot.c` | C 点积实现 |
| `l3-1-bench_dot.py` | Python ctypes 调用示例 |
| `l3-2-bench_numba.py` | Numba JIT 示例 |
| `l3-3-deco.py` | 自定义装饰器 DSL 示例 |
| `l3-3-vec.c` | 向量操作 C 实现 |

---

## 9. 核心要点总结

1. **Python 慢的根本原因**：虚拟机解释执行，每条字节码都有开销
2. **加速的核心思路**：绕过解释器，直接执行机器码
3. **内存布局很关键**：连续内存、缓存友好、SIMD 支持
4. **ctypes vs Numba**：
   - ctypes：调用已有的 C 库，需要编译
   - Numba：Python 语法自动 JIT，透明编译
5. **装饰器的本质**：函数转换，可以用来构建 DSL

---

## 10. 扩展阅读

- [Python C API 文档](https://docs.python.org/3/c-api/index.html)
- [Numba 官方文档](https://numba.pydata.org/)
- [ctypes 文档](https://docs.python.org/3/library/ctypes.html)
- [LLVM IR 参考](https://llvm.org/docs/LangRef.html)

---

## 附录 A: 编程语言对比

### A.1 执行模型对比

| 语言 | 虚拟机 | 编译器 | 解释器 | 执行方式 |
|------|--------|--------|--------|---------|
| **C** | ❌ | ✅ | ❌ | AOT 编译 → 机器码 |
| **C++** | ❌ | ✅ | ❌ | AOT 编译 → 机器码 |
| **Go** | ❌ | ✅ | ❌ | AOT 编译 → 机器码 |
| **Rust** | ❌ | ✅ | ❌ | AOT 编译 → 机器码 |
| **Java** | ✅ JVM | ✅ | ✅ | 源码 → 字节码 → JVM |
| **C#** | ✅ CLR | ✅ | ✅ | 源码 → IL → CLR |
| **Python** | ✅ Python VM | ❌ | ✅ | 源码 → 字节码 → VM |
| **JavaScript** | ✅ V8/SpiderMonkey | ✅ JIT | ✅ | 源码 → 字节码 → 机器码(JIT) |
| **Lua** | ✅ Lua VM | ❌ | ✅ | 源码 → 字节码 → VM |
| **Ruby** | ✅ YARV | ❌ | ✅ | 源码 → 字节码 → VM |
| **PHP** | ✅ Zend VM | ❌ | ✅ | 源码 → 字节码 → VM |

### A.2 编译期与链接期

| 语言 | 编译期 | 链接期 | 运行时 |
|------|--------|--------|--------|
| **C/C++** | ✅ 源码 → 目标文件 | ✅ 目标文件 → 可执行文件/动态库 | ❌ 直接执行 |
| **Go** | ✅ 源码 → 目标文件 | ✅ 链接所有依赖到可执行文件 | ❌ 直接执行 |
| **Rust** | ✅ 源码 → rmeta/rlib | ✅ 链接 → 二进制 | ❌ 直接执行 |
| **Java** | ✅ 源码 → .class | ❌ (类加载时链接) | ✅ JVM 动态链接 |
| **C#** | ✅ 源码 → .il | ❌ (CLR 加载时链接) | ✅ CLR 动态链接 |
| **Python** | ✅ 源码 → .pyc | ❌ | ✅ 导入时链接 |
| **JavaScript** | ❌ | ❌ | ✅ JIT 编译时链接 |

### A.3 中间文件后缀

| 语言 | 源文件 | 中间文件 | 目标文件 | 动态库 | 静态库 |
|------|--------|----------|----------|--------|--------|
| **C** | `.c` | `.i` (预处理) | `.o` / `.obj` | `.so` / `.dll` | `.a` / `.lib` |
| **C++** | `.cpp` / `.cc` | `.ii` | `.o` / `.obj` | `.so` / `.dll` | `.a` / `.lib` |
| **Go** | `.go` | - | - | `.so` | `.a` |
| **Rust** | `.rs` | - | `.rlib` / `.rmeta` | `.so` | `.rlib` |
| **Java** | `.java` | - | `.class` | `.jar` | - |
| **C#** | `.cs` | - | `.il` | `.dll` | - |
| **Python** | `.py` | - | `.pyc` (字节码) | `.pyd` / `.so` | - |
| **JavaScript** | `.js` | - | - | - | - |

### A.4 编译流程详解

#### C/C++ 编译流程

```
┌─────────────────────────────────────────────────────────────────┐
│                         C/C++ 编译流程                           │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  main.c                    main.h                               │
│    │                          │                                 │
│    ▼                          ▼                                 │
│  ┌─────────────────────────────────────────────────────────┐   │
│  │  预处理器 (Preprocessor)                                 │   │
│  │  gcc -E main.c -o main.i                               │   │
│  └─────────────────────────────────────────────────────────┘   │
│    │                                                           │
│    ▼ main.i                                                    │
│  ┌─────────────────────────────────────────────────────────┐   │
│  │  编译器 (Compiler)                                       │   │
│  │  gcc -S main.i -o main.s  或  gcc -c main.c -o main.o   │   │
│  │  输出: main.s (汇编)  →  main.o (目标文件)               │   │
│  └─────────────────────────────────────────────────────────┘   │
│    │                                                           │
│    ▼ main.o                                                    │
│  ┌─────────────────────────────────────────────────────────┐   │
│  │  链接器 (Linker)                                         │   │
│  │  gcc main.o -o main  或  gcc -shared main.o -o lib.so   │   │
│  │  输出: main (可执行文件)  或  lib.so (动态库)            │   │
│  └─────────────────────────────────────────────────────────┘   │
│    │                                                           │
│    ▼ main / lib.so                                            │
│  ┌─────────────────────────────────────────────────────────┐   │
│  │  加载器 (Loader) - 运行时                                │   │
│  │  操作系统将程序加载到内存，跳转到入口点                   │   │
│  └─────────────────────────────────────────────────────────┘   │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

#### Python 编译流程

```
┌─────────────────────────────────────────────────────────────────┐
│                         Python 编译流程                         │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  main.py                                                        │
│    │                                                            │
│    ▼                                                            │
│  ┌─────────────────────────────────────────────────────────┐   │
│  │  Python 解析器 (Parser)                                  │   │
│  │  将源码解析为 AST (抽象语法树)                           │   │
│  └─────────────────────────────────────────────────────────┘   │
│    │                                                            │
│    ▼ AST                                                        │
│  ┌─────────────────────────────────────────────────────────┐   │
│  │  字节码编译器 (Bytecode Compiler)                       │   │
│  │  将 AST 编译为字节码                                    │   │
│  └─────────────────────────────────────────────────────────┘   │
│    │                                                            │
│    ▼ main.pyc                                                  │
│  ┌─────────────────────────────────────────────────────────┐   │
│  │  Python 虚拟机 (Python VM)                              │   │
│  │  解释执行字节码                                         │   │
│  └─────────────────────────────────────────────────────────┘   │
│    │                                                            │
│    ▼                                                            │
│  CPU 指令                                                      │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

#### Java 编译流程

```
┌─────────────────────────────────────────────────────────────────┐
│                         Java 编译流程                           │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  Main.java                                                      │
│    │                                                            │
│    ▼                                                            │
│  ┌─────────────────────────────────────────────────────────┐   │
│  │  javac (Java 编译器)                                    │   │
│  │  将源码编译为字节码                                    │   │
│  └─────────────────────────────────────────────────────────┘   │
│    │                                                            │
│    ▼ Main.class                                                │
│  ┌─────────────────────────────────────────────────────────┐   │
│  │  JVM 类加载器 (Class Loader)                            │   │
│  │  加载 .class 文件到内存                                 │   │
│  └─────────────────────────────────────────────────────────┘   │
│    │                                                            │
│    ▼                                                            │
│  ┌─────────────────────────────────────────────────────────┐   │
│  │  JVM 解释器 / JIT 编译器                                │   │
│  │  解释执行字节码 或 JIT 编译为机器码                     │   │
│  └─────────────────────────────────────────────────────────┘   │
│    │                                                            │
│    ▼                                                            │
│  CPU 指令                                                      │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

### A.5 命令速查

#### C/C++

```bash
# 预处理
gcc -E main.c -o main.i

# 生成汇编
gcc -S main.c -o main.s

# 编译为目标文件
gcc -c main.c -o main.o

# 链接为可执行文件
gcc main.o -o main

# 编译动态库
gcc -shared -fPIC -o libfoo.so foo.c

# 编译静态库
gcc -c foo.c -o foo.o
ar rcs libfoo.a foo.o

# 查看依赖
ldd ./main
nm ./libfoo.so
objdump -d ./main
```

#### Python

```bash
# 直接运行
python main.py

# 查看字节码
python -m py_compile main.py  # 生成 main.pyc
python -m dis main.py          # 反汇编字节码

# 冻结为可执行文件
pyinstaller --onefile main.py
```

#### Java

```bash
# 编译
javac Main.java

# 运行
java Main

# 查看字节码
javap -c Main.class

# 打包为 JAR
jar cvf main.jar Main.class
```

#### Go

```bash
# 编译
go build main.go

# 运行
go run main.go

# 交叉编译
GOOS=linux GOARCH=amd64 go build main.go

# 查看汇编
go tool compile -S main.go
```

### A.6 性能对比参考

| 语言 | 相对性能 | 启动速度 | 内存占用 | 适用场景 |
|------|---------|---------|---------|---------|
| **C/C++** | 100x | 快 | 低 | 系统编程、游戏、嵌入式 |
| **Rust** | ~100x | 快 | 低 | 系统编程、WebAssembly |
| **Go** | ~50x | 快 | 中 | 云服务、微服务 |
| **Java** | ~30x | 中 | 中 | 企业应用、大数据 |
| **C#** | ~30x | 中 | 中 | Windows 应用、游戏 |
| **Python** | 1x | 慢 | 高 | 脚本、AI/ML、原型 |
| **JavaScript** | ~10x | 快 | 中 | Web 前端/后端 |
| **Lua** | ~5x | 快 | 低 | 游戏脚本、嵌入式 |
