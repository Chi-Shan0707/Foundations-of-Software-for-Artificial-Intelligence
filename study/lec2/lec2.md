# Lecture 2: Python程序的解析和运行

## 1. 编程语言与程序设计

### 1.1 编程语言的概念

- **编程语言**是用于与计算机沟通、表达计算逻辑的工具
- 分为**低级语言**（机器语言、汇编语言）和**高级语言**（Python、Java、C++等）

### 1.2 编译型语言 vs 解释型语言

| 特性 | 编译型语言 | 解释型语言 |
|------|-----------|-----------|
| 执行方式 | 一次性编译为机器码 | 逐行解释执行 |
| 典型代表 | C、C++、Go | Python、JavaScript、Ruby |
| 优点 | 执行速度快 | 跨平台、调试方便 |
| 缺点 | 平台依赖、编译时间长 | 执行速度相对较慢 |

### 1.3 Python的执行方式

```python
# 交互式执行
python -i  # 进入交互模式

# 文件执行
python helloworld.py
python add.py
```

## 2. Python程序的执行过程

### 2.1 代码执行流程

```
源代码 (.py) → 词法分析 → 语法分析 → 字节码编译 → 虚拟机执行
```

1. **词法分析 (Lexical Analysis)**: 将源代码分解为Token（标识符、关键字、运算符等）
2. **语法分析 (Syntax Analysis)**: 检查语法规则，生成抽象语法树(AST)
3. **字节码编译 (Bytecode Compilation)**: 将AST编译为Python字节码(.pyc文件)
4. **虚拟机执行 (PVM - Python Virtual Machine)**: 逐条执行字节码指令

### 2.2 Python解释器

- **CPython**: Python的官方解释器，使用C语言实现
- 解释器本身也是程序，将Python代码转换为机器可执行的指令

### 2.3 示例：Python的动态特性

```python
# 动态类型：变量类型在运行时确定
a = 10        # 整数
a = "hello"   # 字符串，合法！

# 动态执行：代码可以在运行时生成和执行
code = "print('Hello from dynamic code')"
exec(code)

# 动态导入
module = __import__('math')
print(module.sqrt(4))  # 2.0
```

## 3. Python的面向对象模型

### 3.1 一切皆对象

在Python中，**所有数据都是对象**：

```python
# 基本类型也是对象
a = 10          # int对象
b = 3.14        # float对象
c = "hello"     # str对象
d = [1, 2, 3]   # list对象
e = {"a": 1}    # dict对象

# 函数也是对象
def say_hello():
    print("Hello")

print(type(say_hello))  # <class 'function'>
print(say_hello.__name__)  # say_hello
```

### 3.2 对象的组成

每个Python对象包含：
- **身份 (Identity)**: 对象的内存地址
- **类型 (Type)**: 决定对象可以进行的操作
- **值 (Value)**: 对象存储的数据

```python
a = [1, 2, 3]
print(id(a))    # 对象的内存地址
print(type(a))  # <class 'list'>
print(a)        # [1, 2, 3]
```

### 3.3 动态类型系统

- Python使用**动态类型**：变量在运行时绑定到对象
- **强类型**：不同类型之间的操作需要显式转换

```python
# 动态类型示例
x = 5
x = "five"  # 完全合法

# 强类型示例
# "5" + 5  # TypeError: can only concatenate str (not "int") to str
int("5") + 5  # 10，需要显式转换
```

## 4. 代码实例分析

### 4.1 helloworld.py

```python
print("Hello, World!")
```

执行流程：
1. 解释器读取源代码
2. 创建字符串对象 `"Hello, World!"`
3. 调用 `print` 函数输出

### 4.2 add.py

```python
a = int(input("Please enter the first number: "))
b = int(input("Please enter the second number: "))
print(a + b)
```

执行流程：
1. `input()` 读取用户输入，返回字符串
2. `int()` 将字符串转换为整数对象
3. `+` 操作符执行加法运算
4. `print()` 输出结果

## 5. 本章小结

1. **编程语言**是人与计算机沟通的桥梁
2. **Python是解释型语言**，执行过程：源码 → 词法分析 → 语法分析 → 字节码 → 虚拟机
3. **Python是动态强类型语言**：动态类型 + 强类型检查
4. **一切皆对象**：Python中的所有数据都是对象，包括函数、类等
5. Python的**动态特性**使得代码更灵活，但也需要开发者注意类型安全
