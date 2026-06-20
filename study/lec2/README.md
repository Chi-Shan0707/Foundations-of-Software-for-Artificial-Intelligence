# L2: Python 程序的解析和运行

> Python 程序从源代码到运行结果的全过程：词法分析、语法分析与 AST、编译为字节码，以及 Python 虚拟机的执行机制。

---

## 目录

1. [本章学习目标](#1-本章学习目标)
2. [编程语言与程序设计](#2-编程语言与程序设计)
3. [Python 语言与 AI 生态](#3-python-语言与-ai-生态)
4. [Python、CPython 与 Jupyter Notebook 的关系](#4-pythoncpython-与-jupyter-notebook-的关系)
5. [Python 程序的总体执行流程](#5-python-程序的总体执行流程)
6. [语法解析](#6-语法解析)
7. [词法分析](#7-词法分析)
8. [句法分析与抽象语法树](#8-句法分析与抽象语法树)
9. [编译](#9-编译)
10. [字节码](#10-字节码)
11. [Python 虚拟机（PVM）](#11-python-虚拟机pvm)
12. [例子：`relu` 程序如何执行](#12-例子relu-程序如何执行)
13. [补充理解：Python 为什么显得"动态"](#13-补充理解python-为什么显得动态)
14. [常见误区](#14-常见误区)
15. [动手实验建议](#15-动手实验建议)
16. [本章小结](#16-本章小结)

---

## 1. 本章学习目标

* 理解 Python 程序从**源代码**到**运行结果**的大致流程。

* 理解 **AST（抽象语法树）**、**字节码（bytecode）** 与 **Python 虚拟机（PVM）** 的基本作用。

* 能够借助工具查看 Python 代码在不同阶段的表示形式。

* 理解 **Python 语言**、**CPython 解释器** 与 **.ipynb 笔记本**三者的关系和区别。

---

## 2. 编程语言与程序设计

### 2.1 编程语言的概念

* **编程语言**是人类描述计算过程、数据和控制逻辑的形式化工具。

* 从抽象层次看，编程语言大致可分为：

  * **低级语言**：如机器语言、汇编语言，更接近硬件。
  * **高级语言**：如 Python、Java、C++，更接近人的思维方式。

### 2.2 编译型语言 vs 解释型语言

| 维度 | 编译型 | 解释型 |
| ---- | ------ | ------ |
| 执行方式 | 先整体编译，再运行 | 通常由解释器执行 |
| 典型代表 | C, C++, Rust | Python, JavaScript, Ruby |
| 优点 | 运行速度快 | 调试方便、跨平台性强 |
| 缺点 | 编译流程较重 | 运行时开销相对更高 |

* **更准确的说法**：Python 常被称为"解释型语言"，但在 CPython 中，源码并不是直接按字符逐行执行，而是通常先经历**解析**与**编译为字节码**，再由虚拟机执行。

---

## 3. Python 语言与 AI 生态

* Python 语法简洁、可读性强，非常适合教学、快速原型开发和科研实验。

* 在人工智能和数据科学中，Python 之所以重要，很大程度上是因为它拥有丰富生态：如 NumPy、Pandas、Matplotlib、Scikit-learn、PyTorch 等。

---

## 4. Python、CPython 与 Jupyter Notebook 的关系

在正式讲解执行流程之前，需要先厘清三个经常被混淆的概念：

```text
Python  = 一门语言（Language）          — 定义语法规则
CPython = 这门语言的解释器实现（Interpreter） — 负责解析、编译、执行
.ipynb  = Jupyter Notebook 的文件格式  — 交互式笔记本，不是解释器
```

### Python 是"语言"

Python 指的是一套语言规范——语法、变量、函数、类、模块、异常机制等。类似于"中文/英文"这种语言规则。

```python
x = 1
print(x + 2)
```

这些语法规则属于 **Python 语言本身**，与具体用什么程序来执行它无关。

### CPython 是"执行 Python 的程序"

CPython 是最主流的 Python 实现。你在终端输入：

```bash
python main.py
```

大概率启动的就是 CPython。它的内部流程正是下一节要讲的：

```text
Python 源代码 → 词法分析 / 语法分析 → AST → 编译成字节码 → Python 虚拟机执行
```

可以类比：

```text
C 语言 ≠ GCC 编译器
Python 语言 ≠ CPython 解释器
```

除了 CPython，还有 PyPy、Jython、IronPython、MicroPython 等实现，但主流科研和工程（NumPy、PyTorch、Jupyter）中默认就是 CPython。

### `.ipynb` 是 Notebook 文件，不是解释器

`.ipynb` 是一个 JSON 文件，里面保存的是：

```text
代码 cell、Markdown cell、运行输出、图片、metadata、kernel 信息
```

当你在 Jupyter Notebook 中点"运行"时，实际发生的过程是：

```text
浏览器中的 Jupyter 前端
    ↓ 发送代码
Jupyter Python kernel（ipykernel）
    ↓ 调用
CPython 解释器
    ↓ 解析 → AST → 字节码 → 虚拟机执行
运行结果
    ↑ 返回
Jupyter 前端显示在 notebook 中
```

所以 `.ipynb` 更像一个"交互式实验记录本"，真正的执行者仍然是 CPython。

### `.py` 与 `.ipynb` 的关键区别

| 维度 | `.py` | `.ipynb` |
|:--|:--|:--|
| 本质 | 普通 Python 源文件 | JSON 格式的笔记本文件 |
| 执行方式 | 从上到下顺序执行 | 按 cell 逐格执行，可乱序 |
| 状态一致性 | 文件顺序 = 执行顺序 | 当前内存状态可能与文件顺序不一致 |

> **Notebook 的常见坑**：如果你先运行了 cell 2（`x + 1`），再运行 cell 1（`x = 10`），不会报错——但如果你重启 kernel 后只运行 cell 2，就会因为 `x` 未定义而报错。这是 notebook 的"状态与顺序不一致"问题，`.py` 文件不存在这个问题。

**一句话总结**：

```text
你写的是 Python，通常由 CPython 执行，在 Jupyter 里保存成 .ipynb。
```

---

## 5. Python 程序的总体执行流程

Python 程序的运行过程可以概括为三步：

```text
源代码
    ↓
解析
    ↓
AST（抽象语法树）
    ↓
编译
    ↓
字节码
    ↓
Python 虚拟机执行
    ↓
运行结果
```

* **解析（Parsing）**：把源代码转换成 AST。
* **编译（Compilation）**：把 AST 转换成字节码。
* **执行（Execution）**：Python 虚拟机逐条执行字节码。

### 5.1 常用观察命令

```bash
# 查看 AST
python -m ast toy.py

# 查看 token 序列
python -m tokenize toy.py

# 查看字节码
python -m dis toy.py

# 编译为 .pyc
python -m compileall toy.py
```

这些命令帮助我们从"源码视角"切换到"解释器视角"。

---

## 6. 语法解析

语法解析通常分成两步：

1. **词法分析（Lexical Analysis）**
2. **句法分析（Syntax Analysis）**

---

## 7. 词法分析

### 7.1 词法分析的目标

* 词法分析把源代码的**字符流（character stream）**切分成一系列**词元（tokens）**。

* 这些 token 带有语法意义，例如：关键字、标识符、数字、括号、逗号、缩进等。

### 7.2 Python 词法分析的特殊点：缩进

Python 不是靠 `{}` 表示代码块，而是靠**缩进**表示代码块层次。

因此词法分析器除了生成普通 token，还要显式生成：

* `INDENT`：代码块开始
* `DEDENT`：代码块结束

### 7.3 例子

源代码：

```python
def relu(x):
    return max(0.0, x)

print(relu(100))
```

对应的 token 序列可抽象写成：

```text
ENCODING(utf-8)
NAME(def) NAME(relu) ( NAME(x) ) : NEWLINE
INDENT
NAME(return) NAME(max) ( NUMBER(0.0), NAME(x) ) NEWLINE
NL
DEDENT
NAME(print) ( NAME(relu) ( NUMBER(100) ) ) NEWLINE
ENDMARKER
```

这里要注意：

* `def` 和 `return` 在该讲义的展示里，词法阶段仍显示成 `NAME(def)`、`NAME(return)`。
* `NL` 与 `NEWLINE` 不完全相同：

  * `NEWLINE` 表示逻辑行结束。
  * `NL` 常用于空行或括号内部换行。

### 7.4 正则表达式与词法模式

词法分析通常用正则表达式描述 token 模式。例如：

```text
NAME   → [A-Za-z_][A-Za-z_0-9]*
NUMBER → [0-9]+(.[0-9]+)?
```

* `NAME` 表示标识符。
* `NUMBER` 表示整数或浮点数常量。

### 7.5 注释与字符串

* Python 注释通常以 `#` 开始，到行末结束。
* Python 没有专门的"多行注释"语法。
* 三引号 `'''...'''` 或 `"""..."""` 在词法和语法层面本质上仍然是**字符串**，不是注释。

---

## 8. 句法分析与抽象语法树

### 8.1 句法分析的目标

句法分析在 token 序列的基础上判断程序是否**结构合法**，并构造出程序的层次化表示。

* 它关注"结构是否符合语法规则"。
* 它**不直接处理**变量是否已定义、类型是否匹配等语义问题。

### 8.2 上下文无关文法

许多编程语言的语法可以用**上下文无关文法（CFG）**描述。一个 CFG 常写作：

```text
G = (V, Σ, P, S)
```

其中：

* `V`：非终结符集合
* `Σ`：终结符集合
* `P`：产生式集合
* `S`：开始符号

### 8.3 讲义中的简化语法

```text
Module → StmtList
StmtList → Stmt StmtList | ϵ
Stmt → SimpleStmt | CompoundStmt
CompoundStmt → FunctionDef
FunctionDef → def Identifier ( ParamList ) : NEWLINE Block
Block → INDENT StmtList DEDENT
SimpleStmt → ReturnStmt NEWLINE | ExprStmt NEWLINE
ReturnStmt → return Expr
ExprStmt → Expr
```

这说明：

* 一个模块由若干语句组成；
* 语句分为简单语句与复合语句；
* 函数定义是复合语句的一种；
* 缩进块在语法层面非常重要。

### 8.4 AST 的作用

AST（抽象语法树）会**丢弃不重要的表面语法细节**，只保留结构信息。例如括号、分号、部分换行等细节，通常不会以原样保留。

对于上面的 `relu` 例子，AST 的核心结构大致是：

```text
Module
├── FunctionDef(relu)
│   ├── arguments(x)
│   └── Return
│       └── Call(max, [0.0, x])
└── Expr
    └── Call(print, [Call(relu, [100])])
```

一句话理解：**AST 是"程序长什么样"的树状表示，不是"程序怎么执行"的全过程。**

---

## 9. 编译

### 9.1 从 AST 到字节码

Python 在完成语法解析后，不会直接解释 AST，而是进入编译阶段。抽象上可以写成：

```text
AST → Code Object → Bytecode
```

* **代码对象（Code Object）**是对一段可执行代码的封装。
* 它除了包含字节码指令，还包含常量表、符号表、局部变量信息等元数据。

### 9.2 为什么会有多个代码对象

以 `relu` 例子为例，编译阶段会产生两个代码对象：

1. **模块级代码对象**：描述顶层语句。
2. **函数 `relu` 的代码对象**：描述函数体内部逻辑。

---

## 10. 字节码

### 10.1 字节码是什么

* 字节码是与具体硬件平台无关的中间表示。
* 它比源代码更接近执行，但仍然不是机器码。
* 它通常可以缓存为 `.pyc` 文件。

### 10.2 常见字节码指令

* `LOAD_CONST`：把常量加载到栈上。
* `LOAD_NAME` / `LOAD_GLOBAL`：把名字对应的对象加载到栈上。
* `STORE_NAME`：把栈顶对象绑定到某个名字。
* `MAKE_FUNCTION`：创建函数对象。
* `CALL_FUNCTION`：调用函数。
* `LOAD_FAST`：加载局部变量。
* `POP_TOP`：弹出并丢弃栈顶值。
* `RETURN_VALUE`：返回栈顶值。

> 补充说明：不同 Python 版本的具体 opcode 名字和细节可能有所变化，但"先编译为某种中间表示，再由虚拟机执行"的核心思想不变。

### 10.3 代码对象中的几张重要表

每个代码对象通常维护三类重要信息：

* **常量表（constants table）**：保存数值、字符串、内部代码对象等。
* **符号表（names table）**：保存模块或函数使用到的名字。
* **局部变量表（local variables table）**：保存参数和局部变量。

---

## 11. Python 虚拟机（PVM）

### 11.1 PVM 是什么

Python 字节码不是由操作系统直接执行，而是由 **Python 虚拟机（PVM）** 执行。

* PVM 提供统一执行环境。
* 它采用**栈式架构（stack-based architecture）**。
* 运行时会维护**操作数栈**、**调用栈**以及当前执行状态。

### 11.2 执行帧

当 PVM 执行某个代码对象时，会创建一个**执行帧（frame）**。

一个 frame 一般包括：

* 对代码对象的引用
* 指令计数器
* 操作数栈
* 局部运行时状态

---

## 12. 例子：`relu` 程序如何执行

### 12.1 模块级代码对象执行过程

对于：

```python
def relu(x):
    return max(0.0, x)

print(relu(100))
```

模块级的大致逻辑是：

1. 把 `relu` 的函数体代码对象压栈。
2. 把函数名 `"relu"` 压栈。
3. `MAKE_FUNCTION` 创建函数对象。
4. `STORE_NAME` 把它绑定到名字 `relu`。
5. 加载 `print`。
6. 加载 `relu`。
7. 加载常量 `100`。
8. 调用 `relu(100)`。
9. 再调用 `print(...)`。
10. 丢弃 `print` 的返回值 `None`。
11. 返回模块级 `None`。

### 12.2 `relu` 函数体的执行过程

函数内部的大致逻辑是：

1. 加载全局名字 `max`。
2. 加载常量 `0.0`。
3. 加载参数 `x`。
4. 调用 `max(0.0, x)`。
5. 返回结果。

---

## 13. 补充理解：Python 为什么显得"动态"

这一部分不完全是讲义主线，但和"程序如何运行"密切相关。

### 13.1 动态类型

* Python 变量本身不固定携带某种静态类型标签；变量是在运行时绑定到对象上的。

```python
x = 5
x = "five"
```

* 上面是合法的，因为 `x` 只是先后绑定到了不同对象。

### 13.2 强类型

* Python 也是**强类型语言**：不同类型之间的操作通常不能随意混用。

```python
# "5" + 5   # TypeError
int("5") + 5
```

### 13.3 一切皆对象

在 Python 中，函数、类、模块、整数、字符串通常都可以视为对象。

```python
def say_hello():
    print("Hello")

print(type(say_hello))
print(say_hello.__name__)
```

这也是 Python 具有高度灵活性的原因之一。

---

## 14. 常见误区

### 14.1 "Python 是解释型语言" ≠ "源码直接逐字符执行"

更准确的流程是：

```text
源代码 → 解析 → AST → 编译 → 字节码 → 虚拟机执行
```

### 14.2 AST 不是程序运行时的全部状态

AST 只描述结构；运行时还涉及名字绑定、栈、frame、函数对象、返回值等。

### 14.3 三引号不等于"真正的多行注释"

三引号本质上还是字符串字面量。

### 14.4 Python ≠ CPython，.ipynb ≠ Python 解释器

- **Python** 是语言规范，**CPython** 是最常用的解释器实现（类比 C 语言 ≠ GCC）。
- **`.ipynb`** 是 Jupyter 的笔记本文件格式，代码最终仍由 CPython 执行，.ipynb 本身不是解释器。

---

## 15. 动手实验建议

### 15.1 观察 AST

```bash
python -m ast toy.py
```

### 15.2 观察 token

```bash
python -m tokenize toy.py
```

### 15.3 观察字节码

```bash
python -m dis toy.py
```

### 15.4 编译为 `.pyc`

```bash
python -m compileall toy.py
```

### 15.5 讲义练习建议

* 写一个正则表达式，匹配 Python 的单引号、双引号、三引号字符串。
* 对上节课的神经网络程序分析 token、AST、bytecode。
* 实现一个小工具，通过 `ast`、`dis`、`tokenize` 模块查看源码的不同表示。

---

## 16. 本章小结

1. Python 程序运行的主线是：**源代码 → AST → 字节码 → PVM 执行**。
2. **词法分析**把字符流切分成 token；**句法分析**把 token 组织成 AST。
3. Python 的词法分析具有鲜明特色：它需要处理 **INDENT / DEDENT**。
4. 编译阶段会生成**代码对象**和**字节码**，代码对象中还包含常量表、符号表和局部变量表。
5. Python 虚拟机采用**栈式执行模型**，通过 frame、操作数栈和调用机制完成程序运行。

---

## 附录：代码文件索引

| 文件 | 说明 |
|------|------|
| `code/l2-relu.py` | `relu` 示例程序，贯穿全讲的解析、编译与执行分析 |
| `code/l2-regex.py` | 词法模式正则表达式示例（NAME、NUMBER 匹配） |
| `code/l2-pytool.py` | 基于 `ast`、`dis`、`tokenize` 模块的源码多阶段查看工具 |

---
