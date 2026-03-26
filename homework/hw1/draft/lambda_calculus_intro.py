"""
Lambda Calculus Intro (Python Edition)
Lambda 演算入门（Python 版）

This file uses Python's lambda to simulate core ideas of pure lambda calculus.
本文件使用 Python 的 lambda 来模拟纯 lambda 演算中的核心思想。

Important note / 重要说明:
- Python lambda is NOT the full lambda calculus language.
  Python 的 lambda 并不等于完整的 lambda 演算语言。
- But it is a practical bridge for beginners.
  但它是初学者理解 lambda 演算的实用桥梁。
"""

# ============================================================
# 1) Function identity / 函数即一切
# ============================================================
# In lambda calculus, every computation is function application.
# 在 lambda 演算中，所有计算都由“函数应用”构成。

# Identity function: λx.x
# 恒等函数：输入什么就返回什么。
IDENTITY = lambda x: x


# ============================================================
# 2) Church booleans / Church 布尔值
# ============================================================
# TRUE  = λa.λb.a   (choose first)
# FALSE = λa.λb.b   (choose second)
# TRUE  选择第一个参数，FALSE 选择第二个参数。

TRUE = lambda a: (lambda b: a)
FALSE = lambda a: (lambda b: b)

# IF = λp.λx.λy. p x y
# IF 的本质：条件 p 本身就是“选择器”。
IF = lambda p: (lambda x: (lambda y: p(x)(y)))


# ============================================================
# 3) Church numerals / Church 数字
# ============================================================
# A Church numeral n means: apply function f exactly n times to x.
# Church 数字 n 的定义：把函数 f 对 x 作用 n 次。
#
# 0 = λf.λx.x
# 1 = λf.λx.f x
# 2 = λf.λx.f (f x)
# ...

ZERO = lambda f: (lambda x: x)
ONE = lambda f: (lambda x: f(x))
TWO = lambda f: (lambda x: f(f(x)))
THREE = lambda f: (lambda x: f(f(f(x))))

# Successor / 后继函数
# SUCC n = λf.λx. f (n f x)
# 含义：先做 n 次，再额外做 1 次。
SUCC = lambda n: (lambda f: (lambda x: f(n(f)(x))))

# Addition / 加法
# ADD m n = λf.λx. m f (n f x)
# 含义：先执行 n 次，再执行 m 次，总共 m+n 次。
ADD = lambda m: (lambda n: (lambda f: (lambda x: m(f)(n(f)(x)))))

# Multiplication / 乘法
# MUL m n = λf. m (n f)
# 含义：把 n 次应用作为一个“整体步骤”，重复 m 轮。
MUL = lambda m: (lambda n: (lambda f: m(n(f))))


# ============================================================
# 4) Converters / 转换器（为了观察结果）
# ============================================================
# to_int: Church numeral -> Python int
# 把 Church 数字转换为普通整数，便于打印。
to_int = lambda n: n(lambda k: k + 1)(0)

# to_bool: Church boolean -> Python bool
# 把 Church 布尔值转换成 Python 布尔值。
to_bool = lambda b: b(True)(False)


# ============================================================
# 5) Demo / 演示
# ============================================================
if __name__ == "__main__":
    print("=== Lambda Calculus Intro Demo / Lambda 演算入门演示 ===")

    # Church boolean demo / Church 布尔值演示
    print("TRUE as Python bool:", to_bool(TRUE))
    print("FALSE as Python bool:", to_bool(FALSE))

    # IF TRUE then "yes" else "no"
    # 如果条件为 TRUE，选择 yes；否则选择 no。
    choose_yes = IF(TRUE)("yes")("no")
    choose_no = IF(FALSE)("yes")("no")
    print("IF(TRUE)(yes)(no):", choose_yes)
    print("IF(FALSE)(yes)(no):", choose_no)

    # Church numeral demo / Church 数字演示
    print("ZERO:", to_int(ZERO))
    print("ONE:", to_int(ONE))
    print("TWO:", to_int(TWO))
    print("THREE:", to_int(THREE))

    four = SUCC(THREE)
    five = ADD(TWO)(THREE)
    six = MUL(TWO)(THREE)

    print("SUCC(THREE):", to_int(four))
    print("ADD(TWO)(THREE):", to_int(five))
    print("MUL(TWO)(THREE):", to_int(six))

    # One-line takeaway / 一句话总结
    # Numbers are behaviors: "how many times to apply a function".
    # 数字本质是行为："把函数应用多少次"。
    print("Takeaway: numerals are repeated function application.")
