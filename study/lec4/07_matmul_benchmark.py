"""
矩阵乘法性能测试

演示：
- Python 纯实现 vs C + OpenMP vs NumPy
- 编译 C 代码为动态库
- 通过 ctypes 调用 C 函数

使用前需要先编译 C 代码：
    cd ../../code
    gcc -O3 -fopenmp -shared -fPIC -o libmatmul.so l4-matmul.c
"""

import ctypes
import numpy as np
import time
import os


def matmul_python(A, B):
    """纯 Python 实现的矩阵乘法"""
    M, K = A.shape
    K, N = B.shape
    C = [[0.0] * N for _ in range(M)]

    for i in range(M):
        for j in range(N):
            s = 0.0
            for k in range(K):
                s += A[i][k] * B[k][j]
            C[i][j] = s

    return C


def benchmark():
    # 测试矩阵大小
    M, K, N = 256, 256, 256
    BS = 64  # 块大小

    print(f"=== 矩阵乘法性能测试 ({M}×{K} × {K}×{N}) ===\n")

    # 准备数据
    np.random.seed(42)
    A_np = np.random.rand(M, K).astype(np.float64)
    B_np = np.random.rand(K, N).astype(np.float64)
    A_py = A_np.tolist()
    B_py = B_np.tolist()

    # ===== 1. 纯 Python =====
    print("【1】纯 Python 实现...")
    start = time.time()
    C_py = matmul_python(A_py, B_py)
    t_python = time.time() - start
    print(f"耗时: {t_python:.3f}s\n")

    # ===== 2. C + OpenMP =====
    lib_path = "../../code/libmatmul.so"
    if not os.path.exists(lib_path):
        print(f"【2】C + OpenMP: 未找到 {lib_path}")
        print("请先编译: cd ../../code && gcc -O3 -fopenmp -shared -fPIC -o libmatmul.so l4-matmul.c\n")
        t_native = None
    else:
        print("【2】C + OpenMP 实现...")
        lib = ctypes.CDLL(lib_path)

        # 配置函数签名
        lib.matmul_blocked.argtypes = [
            ctypes.POINTER(ctypes.c_double),
            ctypes.POINTER(ctypes.c_double),
            ctypes.POINTER(ctypes.c_double),
            ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int,
        ]

        C_native = np.zeros((M, N), dtype=np.float64)

        start = time.time()
        lib.matmul_blocked(
            A_np.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
            B_np.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
            C_native.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
            M, K, N, BS,
        )
        t_native = time.time() - start
        print(f"耗时: {t_native:.3f}s\n")

    # ===== 3. NumPy (BLAS) =====
    print("【3】NumPy (BLAS) 实现...")
    start = time.time()
    C_ref = A_np @ B_np
    t_numpy = time.time() - start
    print(f"耗时: {t_numpy:.3f}s\n")

    # ===== 性能对比 =====
    print("=== 性能对比 ===")
    print(f"纯 Python:        {t_python:.3f}s  (1.0x)")
    if t_native:
        print(f"C + OpenMP:       {t_native:.3f}s  ({t_python/t_native:.1f}x 加速)")
    print(f"NumPy (BLAS):     {t_numpy:.3f}s  ({t_python/t_numpy:.1f}x 加速)")

    # ===== 正确性验证 =====
    if t_native:
        C_py_np = np.array(C_py)
        error_py = np.max(np.abs(C_py_np - C_ref))
        error_native = np.max(np.abs(C_native - C_ref))

        print("\n=== 正确性验证 ===")
        print(f"Python 最大误差:  {error_py:.2e}")
        print(f"C 实现最大误差:  {error_native:.2e}")
        print(f"Python 正确:     {'✓' if error_py < 1e-9 else '✗'}")
        print(f"C 实现正确:      {'✓' if error_native < 1e-9 else '✗'}")


if __name__ == "__main__":
    benchmark()
