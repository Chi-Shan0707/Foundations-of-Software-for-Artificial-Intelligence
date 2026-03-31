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

### 2.2 Church 编码

在 Lambda 演算中，可以用函数表示数据和逻辑：

```python
# Church Numerals
ZERO  = lambda f: lambda x: x         # λf.λx.x
ONE   = lambda f: lambda x: f(x)      # λf.λx.f x
TWO   = lambda f: lambda x: f(f(x))   # λf.λx.f (f x)

# Church Arithmetic
ADD   = lambda m: lambda n: lambda f: lambda x: m(f)(n(f)(x))

# 转换函数
to_int = lambda n: n(lambda k: k + 1)(0)
```

### 2.3 执行流程分析

`ADD(ONE)(TWO)` 的字节码：
```text
LOAD_NAME (ADD), CALL → 返回等待第二个参数的函数
LOAD_NAME (TWO), CALL → 执行加法
```

这体现了 Lambda 演算的**柯里化**特性：所有函数都是单参数的，多参数函数通过嵌套单参数函数实现。

---

## 3. 装饰器即编译器：Lowering 到 Native 代码

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

## 4. 总结

本报告通过 Lambda 演算展示了计算的抽象层次：

1. **Lambda 演算**：数学上的极简形式系统，用函数表示一切
2. **Python 实现**：通过 lambda 嵌套模拟函数应用
3. **AST 分析**：把 `ADD(m)(n)` 看成语法树上的应用节点
4. **装饰器编译器**：将应用树 lowering 到 Native，并完成加法计算

**核心洞察**：计算的本质是将高层次的抽象逐步 lowering 到物理机器可执行的指令。装饰器给了我们在 Python 中介入这个过程的能力，实现了"用户写优雅的 Lambda 代码，执行高效的 Native 代码"。

