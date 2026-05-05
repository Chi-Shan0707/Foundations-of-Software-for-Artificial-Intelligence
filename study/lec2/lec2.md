下面是一版可直接使用的双语 Markdown note，我在你现有版本基础上补充了“词法分析—句法分析—编译—虚拟机执行”这条主线，并加了一些更适合复习的细节。

# Lecture 2: Python程序的解析和运行 / Parsing and Execution of Python Programs

## 0. 本章学习目标 / Learning Objectives

* 理解 Python 程序从**源代码**到**运行结果**的大致流程。
  Understand the overall pipeline from **source code** to **runtime result**.

* 理解 **AST（抽象语法树）**、**字节码（bytecode）** 与 **Python 虚拟机（PVM）** 的基本作用。
  Understand the roles of the **AST (Abstract Syntax Tree)**, **bytecode**, and the **Python Virtual Machine (PVM)**.

* 能够借助工具查看 Python 代码在不同阶段的表示形式。
  Be able to inspect Python code at different stages using built-in tools.

* 理解 **Python 语言**、**CPython 解释器** 与 **.ipynb 笔记本**三者的关系和区别。
  Understand the relationship and distinction between **Python (language)**, **CPython (interpreter)**, and **.ipynb (Jupyter Notebook)**.

---

## 1. 编程语言与程序设计 / Programming Languages and Program Design

### 1.1 编程语言的概念 / What is a Programming Language?

* **编程语言**是人类描述计算过程、数据和控制逻辑的形式化工具。
  A **programming language** is a formal tool for describing computation, data, and control flow.

* 从抽象层次看，编程语言大致可分为：
  From the perspective of abstraction, programming languages can be roughly divided into:

  * **低级语言（Low-level languages）**：如机器语言、汇编语言，更接近硬件。
    **Low-level languages** such as machine code and assembly are closer to hardware.
  * **高级语言（High-level languages）**：如 Python、Java、C++，更接近人的思维方式。
    **High-level languages** such as Python, Java, and C++ are closer to human reasoning.

### 1.2 编译型语言 vs 解释型语言 / Compiled vs Interpreted Languages

| 维度 / Aspect        | 编译型 / Compiled                      | 解释型 / Interpreted                             |
| ------------------ | ----------------------------------- | --------------------------------------------- |
| 执行方式 / Execution   | 先整体编译，再运行 / Compile first, then run | 通常由解释器执行 / Usually executed by an interpreter |
| 典型代表 / Examples    | C, C++, Rust                        | Python, JavaScript, Ruby                      |
| 优点 / Advantages    | 运行速度快 / Fast execution              | 调试方便、跨平台性强 / Easy debugging, good portability |
| 缺点 / Disadvantages | 编译流程较重 / Heavier build process      | 运行时开销相对更高 / Higher runtime overhead           |

* **更准确的说法**：Python 常被称为“解释型语言”，但在 CPython 中，源码并不是直接按字符逐行执行，而是通常先经历**解析**与**编译为字节码**，再由虚拟机执行。
  **A more accurate statement**: Python is often called an “interpreted language,” but in CPython, source code is usually first **parsed** and **compiled into bytecode**, and then executed by the virtual machine.

---

## 2. Python语言与AI生态 / Python and the AI Ecosystem

* Python 语法简洁、可读性强，非常适合教学、快速原型开发和科研实验。
  Python has simple, readable syntax, making it ideal for teaching, rapid prototyping, and research.

* 在人工智能和数据科学中，Python 之所以重要，很大程度上是因为它拥有丰富生态：如 NumPy、Pandas、Matplotlib、Scikit-learn、PyTorch 等。
  Python is central to AI and data science largely because of its rich ecosystem: NumPy, Pandas, Matplotlib, Scikit-learn, PyTorch, and more. 

---

## 2.1 Python、CPython 与 Jupyter Notebook 的关系 / Python, CPython, and Jupyter Notebook

在正式讲解执行流程之前，需要先厘清三个经常被混淆的概念：

Before diving into the execution pipeline, let's clarify three frequently confused concepts:

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

| 维度 / Aspect | `.py` | `.ipynb` |
|:--|:--|:--|
| 本质 / Nature | 普通 Python 源文件 | JSON 格式的笔记本文件 |
| 执行方式 / Execution | 从上到下顺序执行 | 按 cell 逐格执行，可乱序 |
| 状态一致性 / State | 文件顺序 = 执行顺序 | 当前内存状态可能与文件顺序不一致 |

> **Notebook 的常见坑**：如果你先运行了 cell 2（`x + 1`），再运行 cell 1（`x = 10`），不会报错——但如果你重启 kernel 后只运行 cell 2，就会因为 `x` 未定义而报错。这是 notebook 的"状态与顺序不一致"问题，`.py` 文件不存在这个问题。

**一句话总结**：

```text
你写的是 Python，通常由 CPython 执行，在 Jupyter 里保存成 .ipynb。
```

---

## 3. Python程序的总体执行流程 / Overall Execution Pipeline of a Python Program

Python 程序的运行过程可以概括为三步：
The execution of a Python program can be summarized in three stages:

```text
源代码 Source Code
    ↓
解析 Parsing
    ↓
AST (Abstract Syntax Tree)
    ↓
编译 Compilation
    ↓
字节码 Bytecode
    ↓
Python 虚拟机执行 Execution by PVM
    ↓
运行结果 Runtime Result
```

* **解析（Parsing）**：把源代码转换成 AST。
  **Parsing** transforms source code into an AST.
* **编译（Compilation）**：把 AST 转换成字节码。
  **Compilation** turns the AST into bytecode.
* **执行（Execution）**：Python 虚拟机逐条执行字节码。
  **Execution** means the Python virtual machine executes the bytecode instruction by instruction. 

### 3.1 常用观察命令 / Useful Inspection Commands

```bash
# 查看 AST / Inspect the AST
python -m ast toy.py

# 查看 token 序列 / Inspect the token stream
python -m tokenize toy.py

# 查看字节码 / Inspect the bytecode
python -m dis toy.py

# 编译为 .pyc / Compile into .pyc files
python -m compileall toy.py
```

这些命令帮助我们从“源码视角”切换到“解释器视角”。
These commands help us move from the “source-level view” to the “interpreter-level view.” 

---

## 4. 语法解析 / Parsing

语法解析通常分成两步：
Parsing is usually divided into two steps:

1. **词法分析（Lexical Analysis）**
2. **句法分析（Syntax Analysis）** 

---

## 5. 词法分析 / Lexical Analysis

### 5.1 词法分析的目标 / Goal of Lexical Analysis

* 词法分析把源代码的**字符流（character stream）**切分成一系列**词元（tokens）**。
  Lexical analysis splits the source code’s **character stream** into a sequence of **tokens**.

* 这些 token 带有语法意义，例如：关键字、标识符、数字、括号、逗号、缩进等。
  These tokens carry syntactic meaning, such as keywords, identifiers, numbers, parentheses, commas, and indentation. 

### 5.2 Python词法分析的特殊点：缩进 / Python’s Special Feature: Indentation

Python 不是靠 `{}` 表示代码块，而是靠**缩进**表示代码块层次。
Python does not use `{}` to mark blocks; it uses **indentation**.

因此词法分析器除了生成普通 token，还要显式生成：

* `INDENT`：代码块开始
  `INDENT`: beginning of a block
* `DEDENT`：代码块结束
  `DEDENT`: end of a block 

### 5.3 例子 / Example

源代码：
Source code:

```python
def relu(x):
    return max(0.0, x)

print(relu(100))
```

对应的 token 序列可抽象写成：
The token sequence can be abstractly written as:

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
A few details are worth noticing:

* `def` 和 `return` 在该讲义的展示里，词法阶段仍显示成 `NAME(def)`、`NAME(return)`。
  In the lecture’s presentation, `def` and `return` still appear as `NAME(def)` and `NAME(return)` at the lexical stage.
* `NL` 与 `NEWLINE` 不完全相同：
  `NL` and `NEWLINE` are not the same:

  * `NEWLINE` 表示逻辑行结束。
    `NEWLINE` marks the end of a logical line.
  * `NL` 常用于空行或括号内部换行。
    `NL` is often used for blank lines or line breaks inside parentheses. 

### 5.4 正则表达式与词法模式 / Regular Expressions and Lexical Patterns

词法分析通常用正则表达式描述 token 模式。
Lexical patterns are usually described using regular expressions.

例如：
For example:

```text
NAME   → [A-Za-z_][A-Za-z_0-9]*
NUMBER → [0-9]+(.[0-9]+)?
```

* `NAME` 表示标识符。
  `NAME` represents an identifier.
* `NUMBER` 表示整数或浮点数常量。
  `NUMBER` represents an integer or floating-point literal. 

### 5.5 注释与字符串 / Comments and Strings

* Python 注释通常以 `#` 开始，到行末结束。
  Python comments usually start with `#` and continue to the end of the line.
* Python 没有专门的“多行注释”语法。
  Python has no dedicated syntax for “multi-line comments.”
* 三引号 `'''...'''` 或 `"""..."""` 在词法和语法层面本质上仍然是**字符串**，不是注释。
  Triple-quoted text is still a **string** at the lexical and syntactic levels, not a comment. 

---

## 6. 句法分析与抽象语法树 / Syntax Analysis and the AST

### 6.1 句法分析的目标 / Goal of Syntax Analysis

句法分析在 token 序列的基础上判断程序是否**结构合法**，并构造出程序的层次化表示。
Syntax analysis checks whether the token sequence is **structurally valid** and constructs a hierarchical representation of the program.

* 它关注“结构是否符合语法规则”。
  It focuses on whether the structure obeys grammar rules.
* 它**不直接处理**变量是否已定义、类型是否匹配等语义问题。
  It does **not directly handle** semantic issues such as undefined variables or type mismatches. 

### 6.2 上下文无关文法 / Context-Free Grammar (CFG)

许多编程语言的语法可以用**上下文无关文法（CFG）**描述。
The syntax of many programming languages can be described using a **context-free grammar (CFG)**.

一个 CFG 常写作：
A CFG is often written as:

```text
G = (V, Σ, P, S)
```

其中：
where:

* `V`：非终结符集合 / set of nonterminals
* `Σ`：终结符集合 / set of terminals
* `P`：产生式集合 / set of productions
* `S`：开始符号 / start symbol 

### 6.3 讲义中的简化语法 / Simplified Grammar in the Lecture

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
This shows that:

* 一个模块由若干语句组成；
  a module consists of a sequence of statements;
* 语句分为简单语句与复合语句；
  statements can be simple or compound;
* 函数定义是复合语句的一种；
  function definition is a kind of compound statement;
* 缩进块在语法层面非常重要。
  indentation blocks are syntactically crucial. 

### 6.4 AST的作用 / The Role of the AST

AST（抽象语法树）会**丢弃不重要的表面语法细节**，只保留结构信息。
The AST **discards unimportant surface syntax details** and keeps the structural information.

例如括号、分号、部分换行等细节，通常不会以原样保留。
For example, parentheses, semicolons, and some line-break details are usually not preserved literally.

对于上面的 `relu` 例子，AST 的核心结构大致是：
For the `relu` example above, the core AST structure is roughly:

```text
Module
├── FunctionDef(relu)
│   ├── arguments(x)
│   └── Return
│       └── Call(max, [0.0, x])
└── Expr
    └── Call(print, [Call(relu, [100])])
```

一句话理解：**AST 是“程序长什么样”的树状表示，不是“程序怎么执行”的全过程。**
In one sentence: **the AST is a tree showing what the program looks like structurally, not the full story of how it executes.** 

---

## 7. 编译 / Compilation

### 7.1 从AST到字节码 / From AST to Bytecode

Python 在完成语法解析后，不会直接解释 AST，而是进入编译阶段。
After parsing, Python does not execute the AST directly; it enters the compilation stage.

抽象上可以写成：
Abstractly, the process is:

```text
AST → Code Object → Bytecode
```

* **代码对象（Code Object）**是对一段可执行代码的封装。
  A **code object** is a packaged representation of executable code.
* 它除了包含字节码指令，还包含常量表、符号表、局部变量信息等元数据。
  Besides bytecode instructions, it also contains metadata such as the constants table, names table, and local variables. 

### 7.2 为什么会有多个代码对象 / Why There Can Be Multiple Code Objects

以 `relu` 例子为例，编译阶段会产生两个代码对象：
In the `relu` example, compilation produces two code objects:

1. **模块级代码对象**：描述顶层语句。
   The **module-level code object**, which describes top-level statements.
2. **函数 `relu` 的代码对象**：描述函数体内部逻辑。
   The **code object for `relu`**, which describes the function body. 

---

## 8. 字节码 / Bytecode

### 8.1 字节码是什么 / What Is Bytecode?

* 字节码是与具体硬件平台无关的中间表示。
  Bytecode is a hardware-independent intermediate representation.
* 它比源代码更接近执行，但仍然不是机器码。
  It is closer to execution than source code, but it is still not machine code.
* 它通常可以缓存为 `.pyc` 文件。
  It can often be cached in `.pyc` files. 

### 8.2 常见字节码指令 / Common Bytecode Instructions

* `LOAD_CONST`：把常量加载到栈上。
  `LOAD_CONST`: load a constant onto the stack.
* `LOAD_NAME` / `LOAD_GLOBAL`：把名字对应的对象加载到栈上。
  `LOAD_NAME` / `LOAD_GLOBAL`: load an object referenced by a name onto the stack.
* `STORE_NAME`：把栈顶对象绑定到某个名字。
  `STORE_NAME`: bind the top-of-stack object to a name.
* `MAKE_FUNCTION`：创建函数对象。
  `MAKE_FUNCTION`: create a function object.
* `CALL_FUNCTION`：调用函数。
  `CALL_FUNCTION`: call a function.
* `LOAD_FAST`：加载局部变量。
  `LOAD_FAST`: load a local variable.
* `POP_TOP`：弹出并丢弃栈顶值。
  `POP_TOP`: discard the top value on the stack.
* `RETURN_VALUE`：返回栈顶值。
  `RETURN_VALUE`: return the top value on the stack. 

> 补充说明 / Extra note: 不同 Python 版本的具体 opcode 名字和细节可能有所变化，但“先编译为某种中间表示，再由虚拟机执行”的核心思想不变。
> The exact opcodes may differ across Python versions, but the central idea remains the same: compile to an intermediate representation first, then execute it in a virtual machine.

### 8.3 代码对象中的几张重要表 / Important Tables Inside a Code Object

每个代码对象通常维护三类重要信息：
Each code object typically maintains three important kinds of information:

* **常量表（constants table）**：保存数值、字符串、内部代码对象等。
  **Constants table**: stores numbers, strings, inner code objects, etc.
* **符号表（names table）**：保存模块或函数使用到的名字。
  **Names table**: stores names used by the module or function.
* **局部变量表（local variables table）**：保存参数和局部变量。
  **Local variables table**: stores parameters and local variables. 

---

## 9. Python虚拟机（PVM） / Python Virtual Machine (PVM)

### 9.1 PVM是什么 / What Is the PVM?

Python 字节码不是由操作系统直接执行，而是由 **Python 虚拟机（PVM）** 执行。
Python bytecode is not executed directly by the operating system; it is executed by the **Python Virtual Machine (PVM)**.

* PVM 提供统一执行环境。
  The PVM provides a unified execution environment.
* 它采用**栈式架构（stack-based architecture）**。
  It uses a **stack-based architecture**.
* 运行时会维护**操作数栈**、**调用栈**以及当前执行状态。
  At runtime, it maintains an **operand stack**, a **call stack**, and the current execution state. 

### 9.2 执行帧 / Execution Frames

当 PVM 执行某个代码对象时，会创建一个**执行帧（frame）**。
When the PVM executes a code object, it creates an **execution frame**.

一个 frame 一般包括：
A frame generally includes:

* 对代码对象的引用 / a reference to the code object
* 指令计数器 / a program counter
* 操作数栈 / an operand stack
* 局部运行时状态 / local runtime state 

---

## 10. 例子：`relu` 程序如何执行 / Example: How the `relu` Program Executes

### 10.1 模块级代码对象执行过程 / Execution of the Module-Level Code Object

对于：

```python
def relu(x):
    return max(0.0, x)

print(relu(100))
```

模块级的大致逻辑是：
At the module level, the rough logic is:

1. 把 `relu` 的函数体代码对象压栈。
   Push the code object of `relu` onto the stack.
2. 把函数名 `"relu"` 压栈。
   Push the function name `"relu"` onto the stack.
3. `MAKE_FUNCTION` 创建函数对象。
   `MAKE_FUNCTION` creates the function object.
4. `STORE_NAME` 把它绑定到名字 `relu`。
   `STORE_NAME` binds it to the name `relu`.
5. 加载 `print`。
   Load `print`.
6. 加载 `relu`。
   Load `relu`.
7. 加载常量 `100`。
   Load the constant `100`.
8. 调用 `relu(100)`。
   Call `relu(100)`.
9. 再调用 `print(...)`。
   Then call `print(...)`.
10. 丢弃 `print` 的返回值 `None`。
    Discard `print`’s return value `None`.
11. 返回模块级 `None`。
    Return module-level `None`. 

### 10.2 `relu` 函数体的执行过程 / Execution of the Function Body

函数内部的大致逻辑是：
Inside the function, the rough logic is:

1. 加载全局名字 `max`。
   Load the global name `max`.
2. 加载常量 `0.0`。
   Load the constant `0.0`.
3. 加载参数 `x`。
   Load the argument `x`.
4. 调用 `max(0.0, x)`。
   Call `max(0.0, x)`.
5. 返回结果。
   Return the result. 

---

## 11. 补充理解：Python为什么显得“动态” / Extra Perspective: Why Python Feels “Dynamic”

这一部分不完全是讲义主线，但和“程序如何运行”密切相关。
This part is not the central line of the lecture, but it is closely related to runtime behavior.

### 11.1 动态类型 / Dynamic Typing

* Python 变量本身不固定携带某种静态类型标签；变量是在运行时绑定到对象上的。
  Python variables are not permanently tied to a static type; they are bound to objects at runtime.

```python
x = 5
x = "five"
```

* 上面是合法的，因为 `x` 只是先后绑定到了不同对象。
  This is valid because `x` is simply rebound to different objects.

### 11.2 强类型 / Strong Typing

* Python 也是**强类型语言**：不同类型之间的操作通常不能随意混用。
  Python is also a **strongly typed language**: operations across different types are usually not mixed implicitly.

```python
# "5" + 5   # TypeError
int("5") + 5
```

### 11.3 一切皆对象 / Everything Is an Object

在 Python 中，函数、类、模块、整数、字符串通常都可以视为对象。
In Python, functions, classes, modules, integers, and strings can all be treated as objects.

```python
def say_hello():
    print("Hello")

print(type(say_hello))
print(say_hello.__name__)
```

这也是 Python 具有高度灵活性的原因之一。
This is one reason Python is so flexible.

---

## 12. 常见误区 / Common Misconceptions

### 12.1 “Python 是解释型语言” ≠ “源码直接逐字符执行”

“Python is interpreted” ≠ “the source code is executed directly character by character”

更准确的流程是：
A more accurate pipeline is:

```text
源代码 → 解析 → AST → 编译 → 字节码 → 虚拟机执行
Source code → Parsing → AST → Compilation → Bytecode → Execution by VM
```

### 12.2 AST 不是程序运行时的全部状态

The AST is not the full runtime state of the program

AST 只描述结构；运行时还涉及名字绑定、栈、frame、函数对象、返回值等。
The AST only describes structure; runtime also involves name binding, stacks, frames, function objects, return values, and more.

### 12.3 三引号不等于”真正的多行注释”

Triple quotes do not mean “true multi-line comments”

三引号本质上还是字符串字面量。
Triple quotes are still string literals in essence.

### 12.4 Python ≠ CPython，.ipynb ≠ Python 解释器

Python ≠ CPython, .ipynb ≠ Python interpreter

- **Python** 是语言规范，**CPython** 是最常用的解释器实现（类比 C 语言 ≠ GCC）。
- **`.ipynb`** 是 Jupyter 的笔记本文件格式，代码最终仍由 CPython 执行，.ipynb 本身不是解释器。

---

## 13. 动手实验建议 / Suggested Hands-On Experiments

### 13.1 观察 AST / Inspect the AST

```bash
python -m ast toy.py
```

### 13.2 观察 token / Inspect tokens

```bash
python -m tokenize toy.py
```

### 13.3 观察字节码 / Inspect bytecode

```bash
python -m dis toy.py
```

### 13.4 编译为 `.pyc` / Compile to `.pyc`

```bash
python -m compileall toy.py
```

### 13.5 讲义练习建议 / Exercises from the Lecture

* 写一个正则表达式，匹配 Python 的单引号、双引号、三引号字符串。
  Write a regex to match Python single-quoted, double-quoted, and triple-quoted strings.
* 对上节课的神经网络程序分析 token、AST、bytecode。
  Analyze the tokens, AST, and bytecode of the neural-network program from the previous lecture.
* 实现一个小工具，通过 `ast`、`dis`、`tokenize` 模块查看源码的不同表示。
  Implement a small tool using `ast`, `dis`, and `tokenize` to inspect different representations of source code. 

---

## 14. 本章小结 / Chapter Summary

1. Python 程序运行的主线是：**源代码 → AST → 字节码 → PVM执行**。
   The main pipeline of Python execution is: **source code → AST → bytecode → execution by the PVM**.

2. **词法分析**把字符流切分成 token；**句法分析**把 token 组织成 AST。
   **Lexical analysis** splits the character stream into tokens; **syntax analysis** organizes tokens into an AST.

3. Python 的词法分析具有鲜明特色：它需要处理 **INDENT / DEDENT**。
   Python’s lexical analysis has a special feature: it must handle **INDENT / DEDENT**.

4. 编译阶段会生成**代码对象**和**字节码**，代码对象中还包含常量表、符号表和局部变量表。
   The compilation stage produces **code objects** and **bytecode**; code objects also contain constants tables, names tables, and local-variable tables.

5. Python 虚拟机采用**栈式执行模型**，通过 frame、操作数栈和调用机制完成程序运行。
   The Python virtual machine uses a **stack-based execution model**, relying on frames, operand stacks, and function calls. 

