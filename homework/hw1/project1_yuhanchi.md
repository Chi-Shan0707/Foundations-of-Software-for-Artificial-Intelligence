# Lambda 演算入门：通过 Python 语法分析理解计算的本质

> 本报告通过 Python 的 `ast` 和 `dis` 模块，以可视化的方式探索 Lambda 演算的核心概念。<br>
> Python 的 lambda 并不等于完整的 lambda 演算语言。
  但它是初学者理解 lambda 演算的实用桥梁。<br>
---
## 1. 引言

Lambda 演算（Lambda Calculus）是由数学家**阿隆佐·邱奇**（Alonzo Church）于 1930 年代提出的形式系统。它是最早的**通用计算模型**之一，与图灵机具有等价的计算能力，但采用了完全不同的表达方式。

**核心思想**：一切计算都可以通过**函数定义**和**函数应用**来表示。

Lambda 演算的魅力在于其极简性——整个系统只需要三种语法构造：

```
<表达式> ::= <变量>           -- 变量引用
           | λ<变量>. <表达式> -- 函数抽象（定义）
           | <表达式> <表达式> -- 函数应用（调用）
```
## 2. 

### 2.1 函数即一切

```python
IDENTITY = lambda x: x    # Identity function: λx.x
```
如何理解这个构造？ 
```python
Module(
   body=[
      Assign(
         targets=[
            Name(id='IDENTITY', ctx=Store())],
         value=Lambda(
            args=arguments(
               args=[
                  arg(arg='x')]),
            body=Name(id='x', ctx=Load())))])
```
从 AST 视角看，这是一个 `Assign` 节点：`targets` 中的 `Name(id='IDENTITY')` 表示左值变量名，`value` 是 `Lambda` 节点；该 `Lambda` 的参数列表由 `arg(arg='x')` 给出，而其 `body` 为 `Name(id='x')`，即函数体直接返回参数 `x`，因此该定义对应恒等函数。

### 2.2 Church

顺序，分支，循环是一个程序很关键的三种结构。顺序很自然，所以我们你我们这会儿来认识分支。<br>
```python
TRUE = lambda a: (lambda b: a)                  # TRUE  = λa.λb.a   (choose first)
FALSE = lambda a: (lambda b: b)                 # FALSE = λa.λb.b   (choose second)
IF = lambda p: (lambda x: (lambda y: p(x)(y)))  # IF    = λp.λx.λy. p x y  (condition p is selector itself)
```
有的同学可能会很奇怪，我们一直都认为`true==1`和`false==0`，可我们显然并没有出现数。<br>
```python
Module(
   body=[
      Assign(
         targets=[
            Name(id='TRUE', ctx=Store())],
         value=Lambda(
            args=arguments(
               args=[
                  arg(arg='a')]),
            body=Lambda(
               args=arguments(
                  args=[
                     arg(arg='b')]),
               body=Name(id='a', ctx=Load())))),
      Assign(
         targets=[
            Name(id='FALSE', ctx=Store())],
         value=Lambda(
            args=arguments(
               args=[
                  arg(arg='a')]),
            body=Lambda(
               args=arguments(
                  args=[
                     arg(arg='b')]),
               body=Name(id='b', ctx=Load())))),
      Assign(
         targets=[
            Name(id='IF', ctx=Store())],
         value=Lambda(
            args=arguments(
               args=[
                  arg(arg='p')]),
            body=Lambda(
               args=arguments(
                  args=[
                     arg(arg='x')]),
               body=Lambda(
                  args=arguments(
                     args=[
                        arg(arg='y')]),
                  body=Call(
                     func=Call(
                        func=Name(id='p', ctx=Load()),
                        args=[
                           Name(id='x', ctx=Load())]),
                     args=[
                        Name(id='y', ctx=Load())])))))])
```
从 AST 结构可见，`TRUE` 与 `FALSE` 都是二层 `Lambda`（柯里化）并分别对应不同的 `body` 绑定策略：
- `TRUE` 的内层 `body` 是 `Name(id='a')`，表示在接收 `a`、`b` 后返回第一个参数。
- `FALSE` 的内层 `body` 是 `Name(id='b')`，表示在接收 `a`、`b` 后返回第二个参数。
- `IF` 的参数由 `arg(arg='p')`、`arg(arg='x')`、`arg(arg='y')` 逐层引入，其最内层 `body` 是嵌套 `Call`：`Call(Call(Name('p'), [Name('x')]), [Name('y')])`，即语义上的 `p(x)(y)` (p(x)为一个函数，这个函数接受参数y)。

因此，`IF` 并不依赖内建布尔字面量，而是把“条件”编码为一个可调用选择器：当 `p` 绑定 `TRUE` 时归约到 `x`，当 `p` 绑定 `FALSE` 时归约到 `y`。而我们也不是通过值而是通过选择定义了`TRUE`和`FALSE`。<br>
但我们并不是无法定义“值”。
```python
ZERO  = lambda f: (lambda x: x)         # 0 = λf.λx.x        (apply f zero times to x)
ONE   = lambda f: (lambda x: f(x))      # 1 = λf.λx.f x      (apply f once to x)
TWO   = lambda f: (lambda x: f(f(x)))   # 2 = λf.λx.f (f x)  (apply f twice to x)
SUCC  = lambda n: (lambda f: (lambda x: f(n(f)(x))))      # SUCC n = λf.λx. f (n f x) （successor: n -> n+1 (one more application of f) ）
ADD   = lambda m: (lambda n: (lambda f: (lambda x: m(f)(n(f)(x)))))# ADD m n = λf.λx. m f (n f x) （addition: apply n times, then m times; total m+n ）
```

有了刚刚的铺垫，我们似乎已经能理解了一部分柯里化，现在让我们来仔细研究一下这个是怎么运作的。
首先要明确，我们刚刚定义的都是【逻辑上】成立（首先是作为 **lambda 演算语义模型** 来理解；Python 只是把这些语义编码成可执行对象。）。现在我们需要【显式】表现出来，则需要
```python
to_bool = lambda b: b(True)(False)
to_int = lambda n: n(lambda k: k + 1)(0)
```
这个的意义我们可以结合dis来看


```text
33           LOAD_CONST ... (<code object <lambda> ... line 33>)
             MAKE_FUNCTION
             STORE_NAME               2 (TRUE)

34           LOAD_CONST ... (<code object <lambda> ... line 34>)
             MAKE_FUNCTION
             STORE_NAME               3 (FALSE)

38           LOAD_CONST ... (<code object <lambda> ... line 38>)
             MAKE_FUNCTION
             STORE_NAME               4 (IF)
```

这段说明了一个关键事实：在 Python 字节码层面，`TRUE/FALSE/IF` 首先被构造成函数对象（`MAKE_FUNCTION`），然后绑定到名字（`STORE_NAME`）。这与我们在 AST 中看到的 `Assign(Name(...), Lambda(...))` 一致。

再看 `IF` 的内层调用组织：

```text
Disassembly of <code object <lambda> ... line 38>:
...
LOAD_DEREF               1 (p)
PUSH_NULL
LOAD_DEREF               2 (x)
CALL                     1
PUSH_NULL
LOAD_FAST                0 (y)
CALL                     1
RETURN_VALUE
```

这对应语义 `p(x)(y)`：先调用 `p(x)` 得到一个函数，再把 `y` 作为参数调用该函数。也就是说，布尔值在这里不是“数值真假”，而是“选择器函数”。

因此可以把结论写得更严格一些：

- AST 证明结构上它们是嵌套 `Lambda` / `Call`；
- dis 证明执行上它们被组织为“先创建函数、再按柯里化顺序调用”；
- 二者共同支撑“这些表达式主要应按逻辑语义理解，再由 Python 运行时实现”的说法。

进一步地，用 `ADD(ONE)(TWO)`（柯里化写法）可以更直接观察“加法即函数复合次数叠加”：

```text
9            LOAD_NAME                4 (ADD)
             PUSH_NULL
             LOAD_NAME                1 (ONE)
             CALL                     1
             PUSH_NULL
             LOAD_NAME                2 (TWO)
             CALL                     1
             STORE_NAME               6 (THREE)

10           LOAD_NAME                5 (to_int)
             PUSH_NULL
             LOAD_NAME                6 (THREE)
             CALL                     1
             STORE_NAME               7 (THREE_INT)
```

这段字节码对应的执行序列是：

- 先执行 `ADD(ONE)`，得到“等待第二个 Church numeral 的函数”；
- 再把 `TWO` 传入，得到 `THREE`；
- 最后通过 `to_int(THREE)` 显式投影到 Python 整数（结果为 `3`）。
