"""
Performance Comparison: Normal Addition vs Lambda Recursion (Church Numerals)
比较正常加法和Lambda形式的递归到底差几倍
"""

import time
from lambda_decorator import ZERO, ONE, SUCC, ADD, to_int


def normal_add(a, b):
    """Normal Python addition"""
    return a + b


def church_add(m, n):
    """Church numeral addition using lambda calculus"""
    return ADD(m)(n)


def church_to_int(n):
    """Convert Church numeral to int"""
    return n(lambda k: k + 1)(0)


def int_to_church(n):
    """Convert int to Church numeral"""
    if n == 0:
        return ZERO
    return SUCC(int_to_church(n - 1))


def test_performance():
    """Run performance comparison"""
    print("=" * 60)
    print("Performance Comparison: Normal vs Lambda Recursion")
    print("=" * 60)

    # Test different scales
    test_sizes = [10, 50, 100, 200]

    for size in test_sizes:
        print(f"\n--- Testing with numbers up to {size} ---")

        # 1. Normal Addition
        start = time.perf_counter()
        normal_results = []
        for i in range(size):
            for j in range(size):
                normal_results.append(normal_add(i, j))
        normal_time = time.perf_counter() - start

        # 2. Church Addition (convert to church, add, convert back)
        start = time.perf_counter()
        church_results = []
        for i in range(size):
            church_i = int_to_church(i)
            for j in range(size):
                church_j = int_to_church(j)
                result_church = church_add(church_i, church_j)
                church_results.append(church_to_int(result_church))
        church_time = time.perf_counter() - start

        # Calculate slowdown
        slowdown = church_time / normal_time if normal_time > 0 else float('inf')

        print(f"Normal Addition:     {normal_time:.6f} seconds")
        print(f"Church Addition:     {church_time:.6f} seconds")
        print(f"Slowdown:            {slowdown:.1f}x slower")
        print(f"Operations:          {size * size:,} additions")

    # Verify correctness
    print("\n" + "=" * 60)
    print("Correctness Verification")
    print("=" * 60)

    # Church numerals: 0-10
    church_nums = [int_to_church(i) for i in range(11)]
    int_nums = list(range(11))

    # Test various additions
    test_cases = [
        (0, 0), (1, 0), (0, 1), (1, 1),
        (2, 3), (5, 5), (10, 0), (7, 8)
    ]

    print("\nTesting Church numerals:")
    for i in range(11):
        assert church_to_int(church_nums[i]) == i, f"Failed: {i}"
    print("✓ Church numeral conversion (0-10) works correctly")

    print("\nTesting Church addition:")
    for a, b in test_cases:
        church_a = int_to_church(a)
        church_b = int_to_church(b)
        result = church_to_int(church_add(church_a, church_b))
        expected = normal_add(a, b)
        assert result == expected, f"Failed: {a} + {b} = {result}, expected {expected}"
        print(f"✓ {a} + {b} = {result}")

    print("\n" + "=" * 60)
    print("Summary")
    print("=" * 60)
    print("Church numerals use pure lambda recursion for arithmetic.")
    print("This is theoretically interesting but practically slow!")
    print("The slowdown comes from:")
    print("  1. Function call overhead (each number is nested lambdas)")
    print("  2. Recursion depth (computing n requires n function calls)")
    print("  3. Conversion between int and Church representation")


def test_recursive_fibonacci():
    """Compare normal vs Church-style recursion for Fibonacci"""
    print("\n" + "=" * 60)
    print("Bonus: Fibonacci Comparison")
    print("=" * 60)

    def normal_fib(n):
        if n <= 1:
            return n
        return normal_fib(n - 1) + normal_fib(n - 2)

    # Church Y combinator for recursion
    Y = lambda f: (lambda x: x(x))(lambda x: f(lambda *args: x(x)(*args)))

    def church_fib_factory():
        # Fibonacci using Church numerals would be extremely complex
        # Here we just show the Y combinator pattern with normal ints
        def fib_rec(fib):
            return lambda n: n if n <= 1 else fib(n - 1) + fib(n - 2)
        return Y(fib_rec)

    church_fib = church_fib_factory()

    for n in [10, 15, 20]:
        # Normal
        start = time.perf_counter()
        normal_result = normal_fib(n)
        normal_time = time.perf_counter() - start

        # Y combinator version
        start = time.perf_counter()
        y_result = church_fib(n)
        y_time = time.perf_counter() - start

        print(f"\nfib({n}):")
        print(f"  Normal:   {normal_result} in {normal_time:.6f}s")
        print(f"  Y-Combinator: {y_result} in {y_time:.6f}s")
        print(f"  Slowdown: {y_time / normal_time:.1f}x")


if __name__ == "__main__":
    test_performance()
    # test_recursive_fibonacci()  # Uncomment for Y-combinator comparison
