# Lambda 演算入门：从 Python AST/字节码出发

> 使用 Python 作为理解 Lambda 演算的桥梁

---

## 动机与切入点

### 为什么选择这个角度？

1. **Lambda 演算的抽象性**：纯数学形式对初学者难以理解
2. **Python 的实用性**：`lambda` 关键字、闭包、高阶函数等特性与 lambda 演算有天然联系
3. **AST/字节码的可观测性**：通过 `ast`、`dis` 模块，可以"看见"函数的内部结构
4. **循序渐进**：从具体（Python）到抽象（Lambda 演算），符合认知规律

### 核心问题

> 如何用有限的语法（只有函数定义和函数调用）表示所有可计算量？

---

## 第一部分：Python 中的 Lambda — 可观测的形式

### 1.1 从 AST 看函数结构

**目标**：通过 `ast` 模块理解 Python 函数的语法结构

```python
import ast

# 简单的 identity 函数
IDENTITY = lambda x: x

# 查看 AST
print(ast.dump(ast.parse("IDENTITY = lambda x: x"), indent=2))
```

**AST 结构分析**：
```
Lambda(
  args=arguments(args=[arg(arg='x')]),
  body=Name(id='x', ctx=Load())
)
```

**关键观察**：
- `Lambda` 节点 = λ 抽象（λx. body）
- `args` = 参数列表
- `body` = 函数体

### 1.2 嵌套 Lambda 的 AST

**Church Booleans 的 AST**：

```python
# TRUE = λa. λb. a
TRUE = lambda a: lambda b: a
```

**AST 嵌套结构**：
```
Lambda(args=[a], body=
  Lambda(args=[b], body=Name(a))
)
```

**对应关系**：
- Python: `lambda a: lambda b: a`
- Lambda 演算: `λa. λb. a` (K I 组合子的变体)

### 1.3 从字节码看执行机制

**目标**：通过 `dis` 模块理解闭包的运行时行为

```python
import dis

# TRUE 的字节码
TRUE = lambda a: lambda b: a
dis.dis(TRUE)
```

**关键字节码指令**：
- `MAKE_FUNCTION`: 创建函数对象
- `BUILD_TUPLE` + `SET_FUNCTION_ATTRIBUTE(closure)`: 捕获自由变量
- `LOAD_DEREF`: 访问闭包变量

**核心发现**：
- 嵌套 lambda 产生**闭包（Closure）**
- 内层函数"记住"了外层的变量 `a`
- 这就是 lambda 演算中**自由变量**的运行时实现

---

## 第二部分：Lambda 演算核心概念 — Python 实现

### 2.1 Church 编码：用函数表示数据

**核心思想**：数据只是函数的另一种形式

#### Church Booleans

```python
# TRUE = λa. λb. a  (选择第一个参数)
TRUE = lambda a: lambda b: a

# FALSE = λa. λb. b  (选择第二个参数)
FALSE = lambda a: lambda b: b
```

**AST 解读**：
```
TRUE: Λ(a). Λ(b). a
FALSE: Λ(a). Λ(b). b
```
（Λ 表示 Python Lambda，区别于数学 λ）

**验证**：
```python
# IF = λp. λx. λy. p x y
IF = lambda p: lambda x: lambda y: p(x)(y)

IF(TRUE)("yes")("no")   # → "yes"
IF(FALSE)("yes")("no")  # → "no"
```

#### Church Numerals

**思想**：数字 n = "将函数 f 应用 n 次"

```python
ZERO  = lambda f: lambda x: x        # f 不应用
ONE   = lambda f: lambda x: f(x)     # f 应用 1 次
TWO   = lambda f: lambda x: f(f(x))  # f 应用 2 次
THREE = lambda f: lambda x: f(f(f(x)))  # f 应用 3 次
```

**AST 结构分析**：
```python
# TWO 的 AST
Lambda(args=[f], body=
  Lambda(args=[x], body=
    Call(func=Name(f), args=[
      Call(func=Name(f), args=[Name(x)])
    ])
  )
)
```

**关键字节码**：
- `LOAD_DEREF`: 访问闭包变量 `f`
- `CALL`: 函数调用（应用）

### 2.2 Lambda 演算基础概念

#### 2.2.1 语法（Syntax）

Lambda 演算只有三种构造：

```
<表达式> ::= <变量>
           | λ<变量>. <表达式>      -- 抽象 (Abstraction)
           | <表达式> <表达式>      -- 应用 (Application)
```

| 构造 | Lambda 演算 | Python | AST 节点 | 说明 |
|------|------------|--------|----------|------|
| **变量** | `x`, `y`, `z` | `x`, `y`, `z` | `Name(id='x')` | 值的引用 |
| **抽象** | `λx. M` | `lambda x: M` | `Lambda(args=[x], body=M)` | 函数定义 |
| **应用** | `M N` | `M(N)` | `Call(func=M, args=[N])` | 函数调用 |

**示例**：

```python
# Lambda 演算: λx. λy. x y
# Python:     lambda x: lambda y: x(y)

# AST 结构:
Lambda(args=[x], body=
  Lambda(args=[y], body=
    Call(func=Name(x), args=[Name(y)])
  )
)
```

#### 2.2.2 自由变量与约束变量

**定义**：
- **约束变量（Bound Variable）**：被某个 λ 抽象"捕获"的变量
- **自由变量（Free Variable）**：未被任何 λ 抽象捕获的变量

**形式化定义**：

```
FV(x) = {x}                                    -- 变量的自由变量集是其本身
FV(λx. M) = FV(M) \ {x}                        -- λx. M 的自由变量是 M 的自由变量去掉 x
FV(M N) = FV(M) ∪ FV(N)                        -- 应用的自由变量是两边的并集
```

**示例对比**：

| Lambda 演算 | 自由变量 | 约束变量 | Python 对应 |
|------------|---------|---------|------------|
| `λx. x` | ∅ | {x} | `lambda x: x` |
| `λx. y` | {y} | {x} | `lambda x: y` |
| `λx. λy. x y` | ∅ | {x, y} | `lambda x: lambda y: x(y)` |
| `(λx. x) y` | {y} | {x} | `(lambda x: x)(y)` |
| `λx. y x` | {y} | {x} | `lambda x: y(x)` |

**字节码观察**：
```python
import dis

# 自由变量示例
def make_adder(y):
    return lambda x: x + y  # y 是自由变量

dis.dis(make_adder)
# 关键指令: LOAD_DEREF (加载闭包变量 y)
```

#### 2.2.3 转换规则

##### α-转换（Alpha Conversion / Alpha Renaming）

**规则**：`λx. M ≡ λy. M[y/x]`（当 y 不在 M 中自由出现）

> 直观理解：参数名可以任意改名，不影响语义

**示例**：

```python
# 以下表达式 α-等价:
λx. x           ≡   λy. y
λx. λy. x y     ≡   λa. λb. a b
lambda x: x     ==  lambda y: y  # Python 中语义相同
```

**AST 对应**：

```python
import ast

# α-转换前
tree1 = ast.parse("lambda x: x")
# Lambda(args=[arg='x'], body=Name('x'))

# α-转换后
tree2 = ast.parse("lambda y: y")
# Lambda(args=[arg='y'], body=Name('y'))

# 结构相同，只是参数名不同
```

##### β-规约（Beta Reduction）

**规则**：`(λx. M) N →β M[x := N]`

> 直观理解：函数调用时，用参数值替换函数体中的参数

**归约过程**：

```
(λx. x) y
→β y                    [用 y 替换 x]

(λx. λy. x y) (λz. z)
→β λy. (λz. z) y        [用 (λz. z) 替换 x]
→β λy. y                [继续归约]

(λx. x x) (λx. x)
→β (λx. x) (λx. x)      [用 (λx. x) 替换第一个 x]
→β (λx. x)              [继续归约]
→β (λx. x)              [无法继续，到正规序]
```

**Python 中的 β-规约**：

```python
# 源代码
result = (lambda x: x * 2)(5)  # β-规约: x 被 5 替换

# 字节码视角
dis.dis(lambda x: x * 2)
# LOAD_FAST (加载 x)
# LOAD_CONST (加载 2)
# BINARY_MULTIPLY
# RETURN_VALUE

# 调用时 (5)
# x 的值被绑定到 5，然后执行函数体
```

**归约策略对比**：

| 策略 | 描述 | 优点 | 缺点 |
|------|------|------|------|
| **Normal Order** | 最左最外优先 | 总能找到正规序（若存在） | 可能产生冗余计算 |
| **Applicative Order** | 最左最内优先 | 效率更高，减少中间项 | 可能不终止 |
| **Lazy Evaluation** | 只在需要时规约 | 惰性求值，处理无限数据 | 实现复杂 |

**示例对比**：

```
# 表达式: (λx. λy. y) ((λz. z z) (λz. z z))

Normal Order (先归约外层):
→β λy. y                                    [外层先规约，内层不计算]
✅ 终止

Applicative Order (先归约内层):
→β (λx. λy. y) ((λz. z z) (λz. z z))
   内层 (λz. z z) (λz. z z) 无限展开...
❌ 不终止
```

##### η-转换（Eta Conversion）

**规则**：`λx. f x ≡η f`（当 x 不在 f 中自由出现）

> 直观理解：如果函数只是"转发"参数到另一个函数，则等价于直接使用那个函数

**示例**：

```python
# η-等价
λx. f x         ≡η   f
lambda x: f(x)  ==  f           # Python 中语义等价（在大多数情况）
```

**AST 对应**：

```python
# η-冗余形式
ast.dump(ast.parse("lambda x: f(x)"))
# Lambda(args=[x], body=Call(func=Name(f), args=[Name(x)]))

# η-简化形式
ast.dump(ast.parse("f"))
# Name(id='f')
```

**字节码优化**：

```python
import dis

# η-冗余
dis.dis(lambda x: f(x))
# LOAD_FAST (f)
# LOAD_FAST (x)
# CALL
# RETURN

# η-简化（Python 不会自动优化，但等价）
dis.dis(lambda: f)  # 注意这里签名不同
```

##### 扩展：其他重要概念

**Church-Rosser 定理**：

> 如果一个表达式可以规约到两个不同的正规形式，那么这两个正规形式一定是 α-等价的。
> 即：规约的顺序不影响最终结果（如果能终止）。

**正规形式（Normal Form）**：

> 无法再进行 β-规约的表达式

```
x                    -- 正规形式（变量）
λx. x                -- 正规形式
(λx. x) y            -- 不是正规形式，可规约到 y
```

**组合子（Combinators）**：

> 没有自由变量的 Lambda 表达式

| 组合子 | Lambda 演算 | Python | 作用 |
|--------|------------|--------|------|
| **I** | `λx. x` | `lambda x: x` | 恒等 |
| **K** | `λx. λy. x` | `lambda x: lambda y: x` | 常量函数（第一参数） |
| **K** (KI) | `λx. λy. y` | `lambda x: lambda y: y` | 常量函数（第二参数） |
| **S** | `λx. λy. λz. x z (y z)` | `lambda x: lambda y: lambda z: x(z)(y(z))` | 代换 |
| **Y** | `λf. (λx. f (x x)) (λx. f (x x))` | 见后文 | 不动点 |

**SKI 组合子完备性**：

> 任何 Lambda 表达式都可以只用 S、K、I 三个组合子表示

```
λx. x         ≡ I
λx. y         ≡ K y
λx. x x       ≡ S I I (K I)
```

**β-归约示例**（通过 AST 追踪）：

```python
# 表达式: (λf. λx. f(f(x))) (λk. k+1) (0)
TWO = lambda f: lambda x: f(f(x))
inc = lambda k: k + 1

# β-归约步骤:
# 1. 替换 f → inc
#    lambda x: inc(inc(x))
# 2. 应用到 x = 0
#    inc(inc(0))
# 3. 计算
#    inc(1) → 2
```

---

## 第三部分：高级概念 — 通过 Python 理解

### 3.1 Y 组合子（不动点）

**问题**：Lambda 演算如何实现递归（没有命名的递归函数）？

**Y 组合子**：`Y = λf. (λx. f (x x)) (λx. f (x x))`

```python
# Python 实现（需要一点技巧处理惰性求值）
Y = lambda f: (lambda x: x(x))(lambda x: f(lambda *args: x(x)(*args)))

# 使用 Y 实现阶乘
factorial = Y(lambda f: lambda n: 1 if n == 0 else n * f(n-1))
```

**AST 分析**：
- `x(x)`: 自应用（Self-application）
- Lambda 演算的"自指"能力

### 3.2 作用域与变量捕获

**对比 Python 和 Lambda 演算的作用域规则**：

```python
# Python: 词法作用域（闭包）
def make_adder(n):
    return lambda x: x + n

add_5 = make_adder(5)
add_5(3)  # → 8
```

**AST 中的 `LOAD_DEREF`**：
- 字节码通过 `LOAD_DEREF` 访问外层变量
- 这就是 Lambda 演算中**自由变量**的机制

---

## 第四部分：从 Python 到纯 Lambda 演算

### 4.1 语法对比表

| 概念 | Lambda 演算 | Python | AST 节点 |
|------|------------|--------|----------|
| 变量 | `x` | `x` | `Name(id='x')` |
| 抽象 | `λx. M` | `lambda x: M` | `Lambda(args=[x], body=M)` |
| 应用 | `M N` | `M(N)` | `Call(func=M, args=[N])` |

### 4.2 Python Lambda 的限制

1. **单表达式**：`lambda x: x + 1` ✓，`lambda x: if x: ...` ✗
2. **无多行语句**：不能包含 `return`、`赋值`等
3. **这些限制恰好让 Python lambda 更接近纯 Lambda 演算！**

### 4.3 实现一个微型 Lambda 演算解释器

```python
import ast
from dataclasses import dataclass
from typing import Dict, Optional

@dataclass
class Env:
    vars: Dict[str, 'Expr']
    parent: Optional['Env'] = None

    def get(self, name: str) -> 'Expr':
        if name in self.vars:
            return self.vars[name]
        if self.parent:
            return self.parent.get(name)
        raise NameError(f"Undefined: {name}")

@dataclass
class Var:
    name: str

@dataclass
class Lam:
    param: str
    body: Expr

@dataclass
class App:
    func: Expr
    arg: Expr

Expr = Var | Lam | App

def eval_expr(expr: Expr, env: Env = None) -> Expr:
    env = env or Env({})

    if isinstance(expr, Var):
        return env.get(expr.name)

    if isinstance(expr, Lam):
        # 创建闭包环境
        return Closure(expr, env)

    if isinstance(expr, App):
        func = eval_expr(expr.func, env)
        arg = eval_expr(expr.arg, env)
        if isinstance(func, Closure):
            # β-归约
            new_env = Env({func.lam.param: arg}, func.env)
            return eval_expr(func.lam.body, new_env)

@dataclass
class Closure:
    lam: Lam
    env: Env

def from_python_ast(node: ast.AST) -> Expr:
    """将 Python Lambda AST 转换为我们的 Lambda 表达式"""
    if isinstance(node, ast.Name):
        return Var(node.id)
    if isinstance(node, ast.Lambda):
        body = from_python_ast(node.body)
        return Lam(node.args.args[0].arg, body)
    if isinstance(node, ast.Call):
        func = from_python_ast(node.func)
        arg = from_python_ast(node.args[0]) if node.args else None
        return App(func, arg)
    raise ValueError(f"Unsupported: {node}")
```

---

## 第五部分：总结与反思

### 5.1 学习路径总结

```
Python 源代码
    ↓ [ast 模块]
AST (抽象语法树)
    ↓ [观察结构]
Lambda 演算语法 (λx. M)
    ↓ [理解语义]
Lambda 演算语义 (α/β/η)
    ↓ [实现]
微型 Lambda 演算解释器
```

### 5.2 核心收获

1. **Lambda 演算不是抽象的数学**，它的每个概念都有具体的编程实现
2. **Python 的闭包** = Lambda 演算的自由变量捕获
3. **高阶函数** = 函数作为一等公民
4. **AST 和字节码**是理解语言实现的有力工具

### 5.3 扩展方向

- [ ] 实现 Curry 化（Currying）
- [ ] 实现类型推断（简单类型 Lambda 演算）
- [ ] 对比其他语言的 Lambda（JavaScript、Rust、Haskell）
- [ ] 研究 Lambda 演算与范畴论的关系

---

## 参考资源

- [Python AST 文档](https://docs.python.org/3/library/ast.html)
- [Python dis 文档](https://docs.python.org/3/library/dis.html)
- [Lambda Calculus - Stanford Encyclopedia of Philosophy](https://plato.stanford.edu/entries/lambda-calculus/)
- [Structure and Interpretation of Computer Programs](https://mitpress.mit.edu/books/structure-and-interpretation-computer-programs)

---

## 附录：完整代码示例

见 `lambda_calculus_intro.py`、`ast_analysis.py`、`bytecode_analysis.py`
