# Lambda 演算入门：通过 Python 语法分析理解计算的本质

> 本报告通过 Python 的 `ast` 和 `dis` 模块，探索 Lambda 演算的核心概念，并展示如何通过装饰器将 Lambda 演算 lowering 到 Native 代码。

---

## 1. 引言

Lambda 演算（Lambda Calculus）是由数学家**阿隆佐·邱奇**（Alonzo Church）于 1930 年代提出的形式系统。它是最早的**通用计算模型**之一，与图灵机具有等价的计算能力。

**核心思想**：一切计算都可以通过**函数定义**和**函数应用**来表示。

Lambda 演算的极简语法：
```
<表达式> ::= <变量>           -- 变量引用
           | λ<变量>. <表达式> -- 函数抽象
           | <表达式> <表达式> -- 函数应用
```

---

## 2. Lambda 演算的 Python 实现

### 2.1 函数即一切

```python
IDENTITY = lambda x: x    # λx.x
```

从 AST 视角看：
```python
Module(Assign(targets=[Name('IDENTITY')],
              value=Lambda(args=[x], body=Name(x))))
```

### 2.2 Church 编码：数与布尔

在 Lambda 演算中，可以用函数表示数据和逻辑：

```python
# Church Numerals
ZERO  = lambda f: lambda x: x         # λf.λx.x
ONE   = lambda f: lambda x: f(x)      # λf.λx.f x
TWO   = lambda f: lambda x: f(f(x))   # λf.λx.f (f x)

# Church Booleans
TRUE  = lambda t: lambda f: t         # 选择第一个参数
FALSE = lambda t: lambda f: f         # 选择第二个参数

# IF 条件：IF(p)(a)(b) = if p then a else b
IF    = lambda p: lambda a: lambda b: p(a)(b)

# Church Arithmetic
ADD   = lambda m: lambda n: lambda f: lambda x: m(f)(n(f)(x))

# 转换函数
to_int = lambda n: n(lambda k: k + 1)(0)
```

**AST 视角**：以 `TRUE` 为例

```python
TRUE = lambda t: lambda f: t    # λt.λf.t
```

从 AST 看，这是一个嵌套的 Lambda 表达式：
```
Lambda(
  args=[arg='t'],
  body=Lambda(
    args=[arg='f'],
    body=Name('t')    # 返回第一个参数 t
  )
)
```

**字节码视角**：`TRUE` 的执行

```text
--- TRUE = lambda t: lambda f: t ---
  1  LOAD_CONST (<code object <lambda>>)  # 加载内层 lambda
     MAKE_FUNCTION                         # 创建函数
     RETURN_VALUE

--- 内层 lambda (接收 t) ---
  1  LOAD_FAST (t)                         # 加载闭包变量 t
     BUILD_TUPLE 1                         # 构建闭包元组
     LOAD_CONST (<code object <lambda>>)   # 加载最内层 lambda
     MAKE_FUNCTION                         # 创建带闭包的函数
     SET_FUNCTION_ATTRIBUTE (closure)      # 绑定闭包

--- 最内层 lambda (接收 f，返回 t) ---
  1  LOAD_DEREF (t)                        # 从闭包中取 t
     RETURN_VALUE
```

**关键观察**：
- `TRUE` 和 `FALSE` 的 AST 结构完全相同，只是返回的变量不同
- `LOAD_DEREF` 表示从闭包中捕获外部变量，这是 Lambda 演算的核心机制

### 2.3 IF 条件的 AST 分析

```python
IF = lambda p: lambda a: lambda b: p(a)(b)
```

AST 结构：
```
Lambda(args=[p], body=
  Lambda(args=[a], body=
    Lambda(args=[b], body=
      Call(                          # p(a)(b)
        func=Call(
          func=Name('p'),
          args=[Name('a')]
        ),
        args=[Name('b')]
      )
    )
  )
)
```

字节码核心部分（最内层）：
```text
  LOAD_DEREF (p)      # 加载谓词 p
  CALL
  LOAD_DEREF (a)      # 加载 then 分支 a
  CALL                # p(a)
  LOAD_FAST (b)       # 加载 else 分支 b
  CALL                # p(a)(b)
  RETURN_VALUE
```

**这体现了 Lambda 演算中"控制流即函数应用"的思想**：`IF` 不是语法结构，而是一个接受三个参数的函数。

### 2.4 ADD 的字节码深入分析

```python
ADD = lambda m: lambda n: lambda f: lambda x: m(f)(n(f)(x))
```

让我们追踪 `ADD(ONE)(TWO)` 的完整执行流程：

**第一步：ADD 本身的字节码**

```text
ADD = lambda m: lambda n: lambda f: lambda x: m(f)(n(f)(x))

--- 外层 lambda (接收 m) ---
  LOAD_FAST (m)
  BUILD_TUPLE 1         # 将 m 加入闭包
  LOAD_CONST (<lambda>) # 加载下一层 lambda
  MAKE_FUNCTION
  SET_FUNCTION_ATTRIBUTE (closure)
  RETURN_VALUE          # 返回等待 n 的函数
```

**第二步：ADD(ONE) 返回等待 n 的函数**

```text
--- 第二层 lambda (接收 n) ---
  COPY_FREE_VARS 1      # 复制 m 到闭包
  LOAD_FAST (m)
  LOAD_FAST (n)
  BUILD_TUPLE 2         # 将 m, n 都加入闭包
  LOAD_CONST (<lambda>) # 加载下一层 lambda
  MAKE_FUNCTION
  SET_FUNCTION_ATTRIBUTE (closure)
  RETURN_VALUE          # 返回等待 f 的函数
```

**第三步：ADD(ONE)(TWO) 返回等待 f 的函数**

```text
--- 第三层 lambda (接收 f) ---
  COPY_FREE_VARS 2      # 复制 m, n 到闭包
  LOAD_FAST (f)
  LOAD_FAST (m)
  LOAD_FAST (n)
  BUILD_TUPLE 3         # 将 m, n, f 都加入闭包
  LOAD_CONST (<lambda>) # 加载最内层 lambda
  MAKE_FUNCTION
  SET_FUNCTION_ATTRIBUTE (closure)
  RETURN_VALUE          # 返回等待 x 的函数
```

**第四步：最终执行（当传入 f 和 x 时）**

```text
--- 最内层 lambda (接收 x，执行计算) ---
  COPY_FREE_VARS 3      # 复制 m, n, f 到闭包

  # 计算 m(f)
  LOAD_DEREF (m)
  CALL, LOAD_DEREF (f), CALL

  # 计算 n(f)(x)
  LOAD_DEREF (n), CALL, LOAD_DEREF (f), CALL, LOAD_FAST (x), CALL

  # m(f)(n(f)(x))  -- 前面的结果作为函数调用后面的结果
  CALL
  RETURN_VALUE
```

**性能分析**：每次 `CALL` 都是函数调用开销，`ADD(ONE)(TWO)` 需要多次 `CALL` 才能完成。这解释了为什么 Lambda 递归比普通递归慢得多。

### 2.5 柯里化的本质

从上述字节码分析可以看出，**柯里化**（Currying）的代价：

1. **多层闭包**：每个参数都创建一层新的闭包，需要 `BUILD_TUPLE` 和 `SET_FUNCTION_ATTRIBUTE`
2. **变量捕获**：使用 `LOAD_DEREF` 从闭包中取值，比直接访问局部变量慢
3. **多次函数调用**：`m(f)(n(f)(x))` 需要 5 次 `CALL` 指令

尽管如此，这种**所有函数都是单参数**的统一性使得 Lambda 演算具有极简的理论美感。

---

## 3. Lambda 演算的性能思考

啊好的，既然我们已经了解了其基本运作方式了，我们思考一下，它好不好用呢？

从理论上看，Lambda 演算非常优雅——用纯函数表示一切。但实际运行起来，这种"函数套函数"的方式效率如何？我们来做个简单的实验：用 Fibonacci 数列来对比普通递归和 Lambda 递归（Church 数）的性能差异。

### 3.1 Python 中的性能对比

```python
# 普通递归
def fib_normal(n):
    if n <= 1:
        return n
    return fib_normal(n - 1) + fib_normal(n - 2)

# Lambda 递归（Church 数）
def fib_lambda(n):
    if n <= 1:
        return to_church(n)
    return add(fib_lambda(n - 1))(fib_lambda(n - 2))
```

运行结果：

| n | 普通递归耗时 | Lambda 递归耗时 | 慢 |
|---|:-----------:|:--------------:|:--:|
| 10 | 7μs | 72μs | **10x** |
| 12 | 13μs | 146μs | **11x** |
| 14 | 34μs | 516μs | **15x** |
| 16 | 90μs | 1431μs | **16x** |
| 18 | 185μs | 3641μs | **20x** |

结论：在 Python 中，Lambda 递归比普通递归慢 **10-20 倍**，且差距随 n 增大而扩大。

### 3.2 跨语言对比：C++ 的表现

那么，如果是编译型语言如 C++，情况会有所不同吗？我们用类似的方式实现 C++ 版本：

```cpp
// C++ 普通递归
long long fib_normal(long long n) {
    if (n <= 1) return n;
    return fib_normal(n - 1) + fib_normal(n - 2);
}

// C++ Lambda 递归（模仿 Church 数）
using Church = function<function<long long(long long)>(function<long long(long long)>)>;
Church fib_lambda(long long n) {
    if (n <= 1) return to_church(n);
    return add(fib_lambda(n - 1), fib_lambda(n - 2));
}
```

C++ 运行结果：

| n | C++ 普通递归 | C++ Lambda 递归 | 慢 |
|---|:-----------:|:--------------:|:--:|
| 10 | ~0μs | 105μs | **>100x** |
| 12 | 1μs | 157μs | **157x** |
| 14 | 1μs | 457μs | **457x** |
| 16 | 2μs | 1293μs | **647x** |
| 18 | 5μs | 3764μs | **753x** |

### 3.3 四种方式汇总对比

将四种方式放在一起：

| n | Python 普通 | Python Lambda | C++ 普通 | C++ Lambda |
|---|:-----------:|:-------------:|:-------:|:----------:|
| 10 | 7μs | 72μs | ~0μs | 105μs |
| 14 | 34μs | 516μs | 1μs | 457μs |
| 18 | 185μs | 3641μs | 5μs | 3764μs |

**关键发现：**

1. **Lambda 递归确实慢**：无论 Python 还是 C++，Lambda 递归都比普通递归慢得多
2. **语言差异**：C++ 普通递归比 Python 快 10-40 倍，但 C++ Lambda 与 Python Lambda 速度相当
3. **C++ 中差异更极端**：C++ Lambda 比 C++ 普通慢 **150-750 倍**，远超 Python 的 10-20 倍

### 3.4 为什么 C++ 中这种差异更加明显？

这背后有几个原因：

1. **编译器优化的极限**：C++ 的普通递归可以被编译器极度优化（内联、尾调用优化等），执行速度接近机器极限；而 `std::function` 的虚函数调用开销无法被完全优化掉

2. **类型系统的代价**：C++ 是静态类型语言，用 `std::function` 包装 lambda 会引入类型擦除（type erasure），每次调用都需要通过虚函数表分发

3. **Python 的"先天劣势"**：Python 的函数调用本身就慢（解释执行、动态分发），所以 lambda 和普通函数的相对差距较小；C++ 普通函数调用太快，lambda 的相对开销就被放大了

4. **内存布局**：C++ 的 `std::function` 需要在堆上分配闭包对象，而 Python 的函数对象本来就是堆分配的，相对差距不明显

**结论**：Lambda 演算理论上很美，但在实际工程中，性能代价是显著的。这也解释了为什么我们需要编译器优化——让程序员写优雅的 Lambda 代码，执行高效的 Native 代码。

---

## 4. 装饰器即编译器：Lowering 到 Native 代码

### 3.1 Python 性能问题

Lambda 演算在 Python 中的实现涉及多层 lambda 嵌套调用，每次调用都有：
- 闭包创建开销
- 虚拟机分发开销
- 类型检查和引用计数

<!-- **核心思想**：~~如果我们能在 AST 层面识别 Lambda 演算的模式，直接 lowering 到优化过的 C 实现，就能获得性能提升。~~ 但是我写不出来 -->

### 3.2 @kernel 装饰器（只保留加法）

```python
def kernel(func):
    """将函数体作为嵌入式 DSL 处理"""
    # -------- Parse Python AST --------
    src = inspect.getsource(func)
    tree = ast.parse(src)
    func_def = tree.body[0]

    # Expect: return <expr>
    expr = func_def.body[0].value
    arg_names = [arg.arg for arg in func_def.args.args]

    # -------- Lowering: AST -> native AST nodes --------
    op_codes = {'ADD': 1}

    def lower(node, env):
        if isinstance(node, ast.Name):
            if node.id in env:
                return lib.church_ast_int(env[node.id])
            return lib.church_ast_op(op_codes[node.id])

        if isinstance(node, ast.Call):
            # 通用函数应用：f(x)
            fn_node = lower(node.func, env)
            arg_node = lower(node.args[0], env)
            return lib.church_ast_call(fn_node, arg_node)

    # -------- Runtime wrapper --------
    def wrapper(*args):
        env = dict(zip(arg_names, args))
        root = lower(expr, env)
        try:
            return lib.church_ast_eval(root)
        finally:
            lib.church_ast_free(root)

    return wrapper

@kernel
def add(m, n):
    return ADD(m)(n)
```

### 3.3 C 实现

```c
typedef enum { NODE_INT, NODE_OP, NODE_CALL } NodeKind;

typedef struct Node {
    int kind;
    int value;
    struct Node* fn;
    struct Node* arg;
} Node;

// 构造应用树：ADD(m)(n) 会变成 CALL(CALL(OP_ADD, m), n)
void* church_ast_call(void* fn, void* arg);
int church_ast_eval(void* root);     // 仅匹配 ADD(m)(n) 并计算 m+n

```
**关键洞察**：只实现最核心链路也足够展示 lowering：
Python 负责构造应用树，C 负责识别 `ADD(m)(n)` 的应用结构并返回结果。

也就是说，Python 侧仍然负责 `ast.parse` 与语法树遍历；C 侧负责执行一个简化版 Lambda 应用树（`OP` + `CALL`），保留了“函数应用即计算”的形态。

---

## 5. 总结

本报告通过 Lambda 演算展示了计算的抽象层次：

1. **Lambda 演算**：数学上的极简形式系统，用函数表示一切
2. **Python 实现**：通过 lambda 嵌套模拟函数应用
3. **AST 分析**：把 `ADD(m)(n)` 看成语法树上的应用节点
4. **装饰器编译器**：将应用树 lowering 到 Native，并完成加法计算

**核心洞察**：计算的本质是将高层次的抽象逐步 lowering 到物理机器可执行的指令。装饰器给了我们在 Python 中介入这个过程的能力，实现了"用户写优雅的 Lambda 代码，执行高效的 Native 代码"。

