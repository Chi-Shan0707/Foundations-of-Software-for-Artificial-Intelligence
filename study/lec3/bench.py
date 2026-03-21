import random
import time
import numpy as np
from ctypes import CDLL, c_double, POINTER

# ============================================================================
# Data preparation
# ============================================================================
N = 5_000_000  # 向量长度
SEED = 42
random.seed(SEED)

# Python list：元素为 Python float 对象（双精度浮点数）
a_list = [random.random() for _ in range(N)]
b_list = [random.random() for _ in range(N)]

# NumPy array：连续内存中的 double 类型（双精度浮点数）
a_np = np.array(a_list, dtype=np.float64)
b_np = np.array(b_list, dtype=np.float64)


# ============================================================================
# Dot product implementations
# ============================================================================

def dot_python(a, b):
    """纯 Python 实现：解释器逐元素执行"""
    return sum(x * y for x, y in zip(a, b))


def dot_numpy(a, b):
    """NumPy 实现：Python 层函数调用，底层 C 实现"""
    return np.dot(a, b)


# 加载 C 扩展库
def load_c_library():
    """加载编译好的 C 语言动态链接库"""
    try:
        # 尝试加载编译好的 C 库
        lib = CDll("./dot_product.so")  # Linux/macOS
        # lib = CDll("./dot_product.dll")  # Windows

        # 配置函数签名：返回 double，参数是两个 double* 指针和长度
        lib.dot_product_c.restype = c_double
        lib.dot_product_c.argtypes = [POINTER(c_double), POINTER(c_double), c_double]

        def dot_c(a, b):
            """C 实现：全部在本地机器码中执行"""
            # 确保 a, b 是 C 连续的 NumPy 数组
            a_contiguous = np.ascontiguousarray(a, dtype=np.float64)
            b_contiguous = np.ascontiguousarray(b, dtype=np.float64)

            # 获取数据指针
            a_ptr = a_contiguous.ctypes.data_as(POINTER(c_double))
            b_ptr = b_contiguous.ctypes.data_as(POINTER(c_double))

            # 调用 C 函数
            return lib.dot_product_c(a_ptr, b_ptr, c_double(len(a)))

        return dot_c
    except Exception as e:
        print(f"Warning: Could not load C library: {e}")
        print("Skipping C benchmark...")
        return None


# 加载 C 函数
dot_c = load_c_library()


# ============================================================================
# Benchmark helper
# ============================================================================
def benchmark(func, *args):
    """测量函数执行时间"""
    times = []
    start = time.perf_counter()
    result = func(*args)
    end = time.perf_counter()
    return end - start, result


# ============================================================================
# Run benchmarks
# ============================================================================
print("=" * 60)
print(f"Benchmarking dot product with N={N:,} elements")
print("=" * 60)

# 纯 Python：解释器逐元素执行
t_py, result_py = benchmark(dot_python, a_list, b_list)
print(f"Dot via Python: {t_py:.4f} s (result: {result_py:.2f})")

# NumPy：Python 层函数调用，底层 C 实现
t_np, result_np = benchmark(dot_numpy, a_np, b_np)
print(f"Dot via NumPy : {t_np:.4f} s (result: {result_np:.2f})")

# C 实现：全部在本地机器码中执行
if dot_c is not None:
    t_c, result_c = benchmark(dot_c, a_np, b_np)
    print(f"Dot via C     : {t_c:.4f} s (result: {result_c:.2f})")

print("=" * 60)
print("Speedup factors:")
print(f"  NumPy vs Python: {t_py / t_np:.2f}x faster")
if dot_c is not None:
    print(f"  C vs Python     : {t_py / t_c:.2f}x faster")
    print(f"  C vs NumPy      : {t_np / t_c:.2f}x faster")
