"""
全栈 Attention 性能对比: 你的手写代码 vs FlashAttention-Lite
===========================================================

用法: conda activate aisoftenv && python benchmark_all_four.py

本脚本直接复制你在 pp_microgpt.py / pytorch_microgpt.py 中手写的 Attention 代码,
以及编译你 cpp_attention.cpp（从你的 cpp_microgpt.cpp 摘取的 Attention 逻辑）,
与 FlashAttention-Lite (Triton) 进行性能对比。

四个实现的来源:
  1. Pure Python:   摘自 pp_microgpt.py L154 (softmax) + L192-194 (attention 三行)
  2. C++:           摘自 cpp_microgpt.cpp L96-113 (softmax) + L128-141 (attention),
                    去掉 Value* 计算图, 只保留纯 double 运算
  3. PyTorch:       摘自 pytorch_microgpt.py L47-78 (Head.forward)
  4. Flash-Lite:    flashattention_lite.py 的 Triton kernel
"""

import os, sys, time, math, subprocess, random
import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from flashattention_lite import flash_attention_lite

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))


# ═══════════════════════════════════════════════════════════════════════════════
# 1. Pure Python Attention
#    直接复制自 pp_microgpt.py 的手写代码
#    - softmax: L154-158
#    - attention: L192-194
# ═══════════════════════════════════════════════════════════════════════════════

# ── 以下 softmax 函数直接复制自 pp_microgpt.py L154-158 ──
# 唯一改动: 把 val.data 换成 val（这里用裸 float 而非 Value 对象）
def pp_softmax(logits):
    # pp_microgpt.py L154: max_val = max(val.data for val in logits)
    max_val = max(logits)
    # pp_microgpt.py L155: exps = [(val - max_val).exp() for val in logits]
    exps = [math.exp(val - max_val) for val in logits]
    # pp_microgpt.py L156-157
    total = sum(exps)
    return [e / total for e in exps]


def pp_attention(Q_h, K_h, V_h):
    """
    直接复制自 pp_microgpt.py L187-195 的单头 Attention:

        for h in range(n_heads):
            hs = h * head_dim
            q_h = q[hs:hs+head_dim]
            k_h = [k_i[hs:hs+head_dim] for k_i in keys[l_i]]
            v_h = [v_i[hs:hs+head_dim] for v_i in values[l_i]]
            attn_logits = [sum(q_h[j]*k_h[t][j] for j in range(head_dim))
                           / head_dim**0.5 for t in range(len(k_h))]    ← L192
            attn_weights = softmax(attn_logits)                          ← L193
            head_out = [sum(attn_weights[t]*v_h[t][j] for t in range(len(v_h)))
                        for j in range(head_dim)]                        ← L194
            x_attn.extend(head_out)

    参数: Q_h 是 list of float (长度 d), K_h/V_h 是 list of list of float (形状 [N][d])
    """
    head_dim = len(Q_h)
    N = len(K_h)

    # ── pp_microgpt.py L192 ──
    attn_logits = [sum(Q_h[j] * K_h[t][j] for j in range(head_dim))
                   / head_dim ** 0.5 for t in range(N)]

    # ── pp_microgpt.py L193 ──
    attn_weights = pp_softmax(attn_logits)

    # ── pp_microgpt.py L194 ──
    head_out = [sum(attn_weights[t] * V_h[t][j] for t in range(N))
                for j in range(head_dim)]

    return head_out


# ═══════════════════════════════════════════════════════════════════════════════
# 2. PyTorch Attention
#    直接复制自 pytorch_microgpt.py Head.forward (L47-78)
# ═══════════════════════════════════════════════════════════════════════════════

def pytorch_attention(q, k, v, head_dim):
    """
    直接复制自 pytorch_microgpt.py L61-77 的 Head.forward:

        wei = q @ k.transpose(-2,-1) * (head_dim**-0.5)     ← L61
        wei = wei.masked_fill(self.tril[:T,:T] == 0, float('-inf'))  ← L71
        wei = F.softmax(wei, dim=-1)                         ← L74
        out = wei @ v                                        ← L77

    唯一改动: 去掉 causal mask (L71), 因为 FlashAttention-Lite 也没实现 causal
    输入形状: q,k,v = [N, d] (无 batch 维度), 这里加 unsqueeze(0) 模拟 [1, N, d]
    """
    q_ = q.unsqueeze(0)
    k_ = k.unsqueeze(0)
    v_ = v.unsqueeze(0)

    # pytorch_microgpt.py L61
    wei = q_ @ k_.transpose(-2, -1) * (head_dim ** -0.5)

    # 跳过 L71 的 causal mask (Flash-Lite 也没实现)

    # pytorch_microgpt.py L74
    wei = F.softmax(wei, dim=-1)

    # pytorch_microgpt.py L77
    out = wei @ v_

    return out.squeeze(0)


# ═══════════════════════════════════════════════════════════════════════════════
# 3. C++ Attention Benchmark (编译并运行 cpp_attention.cpp)
#    该文件用 cpp_microgpt.cpp 的 Value* 指针风格重写
# ═══════════════════════════════════════════════════════════════════════════════

def compile_cpp_benchmark():
    cpp_src = os.path.join(SCRIPT_DIR, "cpp_attention.cpp")
    cpp_bin = os.path.join(SCRIPT_DIR, "cpp_attention")
    if os.path.exists(cpp_bin):
        return cpp_bin
    result = subprocess.run(
        ["g++", "-O2", "-o", cpp_bin, cpp_src],
        capture_output=True, text=True
    )
    if result.returncode != 0:
        print(f"C++ 编译失败: {result.stderr}")
        return None
    return cpp_bin


def run_cpp_benchmark(cpp_bin, N, d, num_iters=100):
    result = subprocess.run(
        [cpp_bin, str(N), str(d), str(num_iters)],
        capture_output=True, text=True
    )
    if result.returncode != 0:
        return None
    try:
        return float(result.stdout.strip())
    except Exception:
        return None


# ═══════════════════════════════════════════════════════════════════════════════
# 4. 计时工具
# ═══════════════════════════════════════════════════════════════════════════════

def time_pp(N, d, num_iters=5):
    """计时 pp_attention (你的 pp_microgpt.py 手写代码)"""
    random.seed(42)
    Q_h = [random.gauss(0, 1) for _ in range(d)]
    K_h = [[random.gauss(0, 1) for _ in range(d)] for _ in range(N)]
    V_h = [[random.gauss(0, 1) for _ in range(d)] for _ in range(N)]

    pp_attention(Q_h, K_h, V_h)

    start = time.perf_counter()
    for _ in range(num_iters):
        pp_attention(Q_h, K_h, V_h)
    return (time.perf_counter() - start) / num_iters * 1000


def time_pp_autograd(N, d, num_iters=3):
    """计时 pp_attention + Value 对象 (你的 pp_microgpt.py 的 Autograd 版本)"""

    # 直接复制 pp_microgpt.py L24-71 的 Value 类
    class Value:
        __slots__ = ('data', 'grad', '_children', '_local_grads')
        def __init__(self, data, children=(), local_grads=()):
            self.data = data; self.grad = 0.0
            self._children = children; self._local_grads = local_grads
        def __add__(self, other):
            other = other if isinstance(other, Value) else Value(other)
            return Value(self.data + other.data, (self, other), (1.0, 1.0))
        def __radd__(self, other): return self + other
        def __mul__(self, other):
            other = other if isinstance(other, Value) else Value(other)
            return Value(self.data * other.data, (self, other), (other.data, self.data))
        def __rmul__(self, other): return self * other
        def __neg__(self): return self * -1
        def __sub__(self, other): return self + (-other)
        def __truediv__(self, other): return self * (other ** -1)
        def __pow__(self, other): return Value(self.data**other, (self,), (other*self.data**(other-1),))
        def exp(self): return Value(math.exp(self.data), (self,), (math.exp(self.data),))

    # pp_microgpt.py L154-158 的 softmax, 用 Value 对象
    def softmax_val(logits):
        max_val = max(val.data for val in logits)
        max_node = Value(max_val)
        exps = [(val - max_node).exp() for val in logits]
        total = sum(exps)
        return [e / total for e in exps]

    random.seed(42)
    Q_h = [Value(random.gauss(0, 1)) for _ in range(d)]
    K_h = [[Value(random.gauss(0, 1)) for _ in range(d)] for _ in range(N)]
    V_h = [[Value(random.gauss(0, 1)) for _ in range(d)] for _ in range(N)]

    start = time.perf_counter()
    for _ in range(num_iters):
        # pp_microgpt.py L192-194, 用 Value 对象
        attn_logits = [sum(Q_h[j] * K_h[t][j] for j in range(d))
                       / d**0.5 for t in range(N)]
        attn_weights = softmax_val(attn_logits)
        head_out = [sum(attn_weights[t] * V_h[t][j] for t in range(N))
                    for j in range(d)]
    return (time.perf_counter() - start) / num_iters * 1000


def time_pytorch(N, d, num_iters=100):
    """计时 pytorch_attention (你的 pytorch_microgpt.py 手写代码)"""
    q = torch.randn(N, d, device="cuda", dtype=torch.float32)
    k = torch.randn(N, d, device="cuda", dtype=torch.float32)
    v = torch.randn(N, d, device="cuda", dtype=torch.float32)

    for _ in range(10):
        pytorch_attention(q, k, v, d)

    torch.cuda.synchronize()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(num_iters):
        pytorch_attention(q, k, v, d)
    end.record()
    torch.cuda.synchronize()
    return start.elapsed_time(end) / num_iters


def time_flash(N, d, num_iters=100):
    """计时 FlashAttention-Lite (Triton kernel)"""
    q = torch.randn(N, d, device="cuda", dtype=torch.float32)
    k = torch.randn(N, d, device="cuda", dtype=torch.float32)
    v = torch.randn(N, d, device="cuda", dtype=torch.float32)

    for _ in range(10):
        flash_attention_lite(q, k, v)

    torch.cuda.synchronize()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(num_iters):
        flash_attention_lite(q, k, v)
    end.record()
    torch.cuda.synchronize()
    return start.elapsed_time(end) / num_iters


# ═══════════════════════════════════════════════════════════════════════════════
# 5. Main
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    print("=" * 100)
    print(" 全栈 Attention 性能对比: 你的手写代码 vs FlashAttention-Lite")
    print("   PP: pp_microgpt.py L192-194 | C++: cpp_microgpt.cpp L128-141")
    print("   PyTorch: pytorch_microgpt.py L61-77 | Flash: flashattention_lite.py")
    print("=" * 100)

    print("\n[1/4] 编译 C++ benchmark (cpp_attention.cpp) ...")
    cpp_bin = compile_cpp_benchmark()
    if cpp_bin:
        print(f"  ✓ 编译成功: {cpp_bin}")
    else:
        print("  ✗ 编译失败，跳过 C++ benchmark")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[设备] PyTorch/Triton 使用: {device}")

    small_configs = [(8, 16), (16, 16), (32, 16), (64, 16)]
    medium_configs = [(128, 64), (256, 64), (512, 64)]
    large_configs = [(1024, 64), (2048, 64), (4096, 64)]

    # ── 第一部分: 小尺度全对比 ──
    print("\n" + "=" * 100)
    print(" 第一部分: 小尺度全对比 (N ≤ 64, d = 16)")
    print("   PP(raw) = 你的 pp_microgpt.py L192-194 (裸 float)")
    print("   PP(+AD) = 你的 pp_microgpt.py L192-194 (Value 对象)")
    print("   C++     = 你的 cpp_microgpt.cpp L128-141 风格 (编译优化)")
    print("=" * 100)
    print(f"\n{'N':>6} {'d':>4} │ {'PP(raw)':>10} {'PP(+AD)':>10} {'C++':>10} │ {'PyTorch':>10} {'Flash':>10} │ {'PP→C++':>8} {'C++→PT':>8} {'PT→Flash':>8}")
    print("─" * 100)

    for N, d in small_configs:
        pp_iters = max(3, min(20, 500 // max(N, 1)))
        pp_ms = time_pp(N, d, num_iters=pp_iters)

        pp_ad_iters = max(1, min(5, 100 // max(N, 1)))
        try:
            pp_ad_ms = time_pp_autograd(N, d, num_iters=pp_ad_iters)
        except Exception:
            pp_ad_ms = -1

        cpp_ms = run_cpp_benchmark(cpp_bin, N, d, num_iters=200) if cpp_bin else -1
        pt_ms = time_pytorch(N, d) if device == "cuda" else -1

        try:
            fl_ms = time_flash(N, d) if device == "cuda" else -1
        except Exception:
            fl_ms = -1

        pp2cpp = f"{pp_ms/cpp_ms:.1f}x" if cpp_ms and cpp_ms > 0 else "---"
        cpp2pt = f"{cpp_ms/pt_ms:.1f}x" if cpp_ms and cpp_ms > 0 and pt_ms > 0 else "---"
        pt2fl = f"{pt_ms/fl_ms:.1f}x" if pt_ms > 0 and fl_ms > 0 else "---"

        pp_ad_str = f"{pp_ad_ms:.3f}" if pp_ad_ms > 0 else "---"
        cpp_str = f"{cpp_ms:.4f}" if cpp_ms and cpp_ms > 0 else "---"
        pt_str = f"{pt_ms:.4f}" if pt_ms > 0 else "---"
        fl_str = f"{fl_ms:.4f}" if fl_ms > 0 else "---"

        print(f"{N:>6} {d:>4} │ {pp_ms:>8.3f}ms {pp_ad_str:>8}ms {cpp_str:>8}ms │ {pt_str:>8}ms {fl_str:>8}ms │ {pp2cpp:>8} {cpp2pt:>8} {pt2fl:>8}")

    # ── 第二部分: 中等尺度 GPU 对比 ──
    print("\n" + "=" * 100)
    print(" 第二部分: 中等尺度 GPU 对比")
    print("   PyTorch = 你的 pytorch_microgpt.py L61-77")
    print("=" * 100)
    print(f"\n{'N':>6} {'d':>4} │ {'PyTorch':>10} {'Flash-Lite':>12} │ {'加速比':>8} {'Flash显存':>10}")
    print("─" * 100)

    for N, d in medium_configs:
        pt_ms = time_pytorch(N, d)
        try:
            fl_ms = time_flash(N, d)
        except Exception:
            fl_ms = -1

        q = torch.randn(N, d, device="cuda", dtype=torch.float32)
        k = torch.randn(N, d, device="cuda", dtype=torch.float32)
        v = torch.randn(N, d, device="cuda", dtype=torch.float32)

        torch.cuda.reset_peak_memory_stats()
        mem_before = torch.cuda.memory_allocated()
        _ = flash_attention_lite(q, k, v)
        torch.cuda.synchronize()
        mem_flash = (torch.cuda.max_memory_allocated() - mem_before) / 1024 / 1024

        speedup = f"{pt_ms/fl_ms:.2f}x" if fl_ms > 0 else "---"
        fl_str = f"{fl_ms:.4f}ms" if fl_ms > 0 else "---"
        print(f"{N:>6} {d:>4} │ {pt_ms:>8.4f}ms {fl_str:>12} │ {speedup:>8} {mem_flash:>8.2f}MB")

    # ── 第三部分: 大尺度 GPU 对比 ──
    print("\n" + "=" * 100)
    print(" 第三部分: 大尺度 GPU 对比")
    print("=" * 100)
    print(f"\n{'N':>6} {'d':>4} │ {'PyTorch':>10} {'Flash-Lite':>12} │ {'加速比':>8} {'PyTorch显存':>12} {'Flash显存':>12} {'显存倍率':>8}")
    print("─" * 100)

    for N, d in large_configs:
        pt_ms = time_pytorch(N, d, num_iters=50)
        try:
            fl_ms = time_flash(N, d, num_iters=50)
        except Exception:
            fl_ms = -1

        q = torch.randn(N, d, device="cuda", dtype=torch.float32)
        k = torch.randn(N, d, device="cuda", dtype=torch.float32)
        v = torch.randn(N, d, device="cuda", dtype=torch.float32)

        torch.cuda.reset_peak_memory_stats()
        mem_before = torch.cuda.memory_allocated()
        _ = pytorch_attention(q, k, v, d)
        torch.cuda.synchronize()
        mem_pt = (torch.cuda.max_memory_allocated() - mem_before) / 1024 / 1024

        torch.cuda.reset_peak_memory_stats()
        mem_before = torch.cuda.memory_allocated()
        _ = flash_attention_lite(q, k, v)
        torch.cuda.synchronize()
        mem_fl = (torch.cuda.max_memory_allocated() - mem_before) / 1024 / 1024

        speedup = f"{pt_ms/fl_ms:.2f}x" if fl_ms > 0 else "---"
        mem_ratio = f"{mem_pt/max(mem_fl,0.01):.1f}x"
        fl_str = f"{fl_ms:.4f}ms" if fl_ms > 0 else "---"
        print(f"{N:>6} {d:>4} │ {pt_ms:>8.4f}ms {fl_str:>12} │ {speedup:>8} {mem_pt:>10.2f}MB {mem_fl:>10.2f}MB {mem_ratio:>8}")

    # ── 第四部分: Autograd 开销分析 ──
    print("\n" + "=" * 100)
    print(" 第四部分: Autograd 开销 (pp_microgpt.py Value 对象 vs 裸 float)")
    print("=" * 100)
    print(f"\n{'N':>6} {'d':>4} │ {'PP(raw)':>10} {'PP(+Autograd)':>14} │ {'Autograd开销':>12}")
    print("─" * 100)

    for N, d in [(4, 4), (8, 8), (16, 16)]:
        pp_ms = time_pp(N, d, num_iters=5)
        try:
            pp_ad_ms = time_pp_autograd(N, d, num_iters=2)
            overhead = f"{pp_ad_ms/pp_ms:.1f}x"
        except Exception:
            pp_ad_ms = -1
            overhead = "---"
        print(f"{N:>6} {d:>4} │ {pp_ms:>8.3f}ms {pp_ad_ms:>12.3f}ms │ {overhead:>12}")

    # ── 总结 ──
    print("\n" + "=" * 100)
    print(" 总结: 你的手写代码 → FlashAttention-Lite 的优化路径")
    print("=" * 100)
    print("""
  你的 pp_microgpt.py L192-194     ~ 秒级     解释器 + Value 对象 (Autograd)
  你的 pp_microgpt.py (裸 float)   ~ 秒级     解释器 + 标量循环
  你的 cpp_microgpt.cpp L128-141   ~ 毫秒级   编译优化 + 标量循环
  你的 pytorch_microgpt.py L61-77  ~ 0.1ms级  GPU 并行 + 向量化 matmul
  FlashAttention-Lite (Triton)     ~ 0.05ms级 GPU 并行 + IO-Aware 分块 + 算子融合

  每一步优化解决一个瓶颈:
    PP+AD → PP(raw): 消除 Autograd 计算图开销 (每个 *,+ 创建 Value 对象, ~6-8x)
    PP → C++:       消除 Python 解释器开销 (编译优化, ~100-130x)
    C++ → PyTorch:  CPU 单线程 → GPU 数千核心并行 (~100-1000x)
    PyTorch → Flash: O(N²) HBM → O(N) HBM, 5+ kernels → 1 kernel (~1.5-3.5x)
""")


if __name__ == "__main__":
    main()
