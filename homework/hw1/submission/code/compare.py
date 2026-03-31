"""
比较普通递归和Lambda递归计算Fibonacci的性能差异
"""

import time

# ============================================================
# Lambda 演算基本定义
# ============================================================

# 转换Church数到整数
to_int = lambda n: n(lambda k: k + 1)(0)

# 整数转Church数
def to_church(n):
    if n == 0:
        return lambda f: lambda x: x
    return lambda f: lambda x: f(to_church(n - 1)(f)(x))

# 后继 SUCC(n) = n + 1
succ = lambda n: lambda f: lambda x: f(n(f)(x))

# 加法 ADD(m)(n) = m + n
add = lambda m: lambda n: lambda f: lambda x: m(f)(n(f)(x))


# ============================================================
# 两种Fibonacci实现
# ============================================================

def fib_normal(n):
    """普通递归求Fibonacci"""
    if n <= 1:
        return n
    return fib_normal(n - 1) + fib_normal(n - 2)


def fib_lambda(n):
    """Lambda递归求Fibonacci (用Church数)"""
    if n <= 1:
        return to_church(n)
    return add(fib_lambda(n - 1))(fib_lambda(n - 2))


# ============================================================
# 性能比较
# ============================================================

def compare(n):
    print(f"\n计算 fib({n}):")
    print("-" * 30)

    # 普通递归
    start = time.perf_counter()
    result1 = fib_normal(n)
    time1 = time.perf_counter() - start
    print(f"普通递归: {result1}, 耗时 {time1:.6f} 秒")

    # Lambda递归
    start = time.perf_counter()
    church_result = fib_lambda(n)
    result2 = to_int(church_result)
    time2 = time.perf_counter() - start
    print(f"Lambda递归: {result2}, 耗时 {time2:.6f} 秒")

    # 验证正确性
    assert result1 == result2, "结果不一致！"

    # 计算倍数
    if time1 > 0:
        ratio = time2 / time1
        print(f"\nLambda递归慢了 {ratio:.1f} 倍")


if __name__ == "__main__":
    print("=" * 40)
    print("Fibonacci性能对比")
    print("=" * 40)

    # 测试不同规模
    for n in [10, 12, 14, 16, 18]:
        compare(n)

    print("\n" + "=" * 40)
    print("结论: Lambda递归随着n增大，性能下降极其明显")
    print("原因: Church数的计算需要大量函数调用和lambda嵌套")
