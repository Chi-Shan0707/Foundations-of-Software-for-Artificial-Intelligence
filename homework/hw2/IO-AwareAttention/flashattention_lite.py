"""
FlashAttention-Lite: 基于 Triton 的 IO-Aware 分块注意力机制
============================================================

本文件实现了一个教学简化版的 FlashAttention（称为 FlashAttention-Lite），
核心目标是演示如何通过 GPU 内存层级感知（IO-Aware）的分块策略，
将标准 Attention 的 O(N^2) HBM 访存量降至 O(N)。

课程对应:
  - L5 (PyTorch): 标准注意力机制的向量化和并行化
  - L6 (GPU编程): GPU 内存层级（HBM vs SRAM），Roofline 模型
  - L7 (算子开发): Triton 编程模型，Tile 抽象，Program Grid
  - L8 (算子融合): 将 matmul + softmax + matmul 融合为单一 kernel

数学原理:
  标准 Attention: O = softmax(Q @ K^T / sqrt(d)) @ V
  - Q, K, V ∈ R^{N × d}, N 为序列长度, d 为 head 维度
  - 中间矩阵 S = Q @ K^T ∈ R^{N × N}, P = softmax(S) ∈ R^{N × N}
  - 问题: N 很大时, N×N 矩阵占用显存 O(N^2)，且 HBM 读写成为瓶颈

  FlashAttention 的核心思想:
  - 将 Q 按行分块为 BLOCK_M 行一组，K/V 按行分块为 BLOCK_N 行一组
  - 每次只加载一小块 Q_i, K_j, V_j 到 SRAM
  - 在 SRAM 中完成 S_ij = Q_i @ K_j^T 的计算和局部 softmax
  - 将结果累加到寄存器中的 acc（永不写回 HBM）
  - 最终只将 O_i 写回 HBM

  HBM 访存量对比:
  - 标准实现: O(N^2) —— 必须读写完整的 S, P 矩阵
  - FlashAttention-Lite: O(N) —— 只读写 Q, K, V, O

简化说明（与完整 FlashAttention 的区别）:
  1. Softmax: 本实现采用每个 tile 内独立计算局部 softmax（近似），
     而非完整 FlashAttention 的 online softmax（全局精确）
  2. 未实现 backward pass
  3. 未实现 causal mask（可作 Bonus 扩展）
  4. 未做 FlashAttention-2 的 Split-K / Warp 级优化

"""

import torch
import triton
import triton.language as tl
import math


# ═══════════════════════════════════════════════════════════════════════════════
# Triton Kernel: FlashAttention-Lite 的核心计算
# ═══════════════════════════════════════════════════════════════════════════════

@triton.jit
def _flash_attn_lite_kernel(
    Q_ptr,        # Q 矩阵的 HBM 起始地址，形状 [N, d]
    K_ptr,        # K 矩阵的 HBM 起始地址，形状 [N, d]
    V_ptr,        # V 矩阵的 HBM 起始地址，形状 [N, d]
    O_ptr,        # 输出 O 矩阵的 HBM 起始地址，形状 [N, d]
    N,            # 序列长度
    d,            # head 维度（特征维度）
    scale,        # 注意力缩放因子 1/sqrt(d)，在 Python 端预计算
    stride_qn,    # Q 的行步长（字节偏移），即 Q[i+1, :] - Q[i, :] 的偏移量
    stride_qd,    # Q 的列步长
    stride_kn,    # K 的行步长
    stride_kd,    # K 的列步长
    stride_vn,    # V 的行步长
    stride_vd,    # V 的列步长
    stride_on,    # O 的行步长
    stride_od,    # O 的列步长
    BLOCK_M: tl.constexpr,  # Q 的行分块大小（编译期常量，Triton 会据此生成专用代码）
    BLOCK_N: tl.constexpr,  # K/V 的行分块大小（编译期常量）
    BLOCK_D: tl.constexpr,  # 特征维度分块大小（通常等于 d）
):
    """
    FlashAttention-Lite 的 Triton Kernel

    调度模型（L7 概念）:
      - Grid 大小 = (ceil(N / BLOCK_M),)
      - 每个 program instance（即一个 GPU block）负责计算 O 的一组行 O[i*BM : (i+1)*BM]
      - Kernel 内部通过循环遍历 K/V 的所有列块

    内存层级映射（L6 概念）:
      - HBM (全局内存): Q, K, V, O 的存储位置
      - SRAM (共享内存/寄存器): 通过 tl.load 加载的 tile 数据
      - 寄存器: acc 累加器，始终驻留，永不写回 HBM
    """

# ── 第一步: 确定当前 program instance 负责的行块索引 ──

    # pid_m = program ID，标识当前 block 处理 Q 的哪 BLOCK_M 行
    # 相当于blockIdx.x 在 CUDA 中的作用
    pid_m = tl.program_id(0)

    # 计算当前 block 负责的 Q 行范围: [qm_start, qm_start + BLOCK_M)
    qm_start = pid_m * BLOCK_M

# ── 第二步: 构建 Q tile 的行列索引 ──

    # offsets: 全局偏移量，用于计算从 HBM 加载数据的地址


    # 整理计算我们的block在全局矩阵的索引范围

    # offs_m: 当前 block 负责的行索引，形状 [BLOCK_M]
    #   例: BLOCK_M=64, pid_m=2 → offs_m = [128, 129, ..., 191]
    offs_m = qm_start + tl.arange(0, BLOCK_M)

    # offs_d: 特征维度的列索引，形状 [BLOCK_D]
    #   例: d=64, BLOCK_D=64 → offs_d = [0, 1, ..., 63]
    offs_d = tl.arange(0, BLOCK_D)

# ── 第三步: 从 HBM 加载 Q tile 到 SRAM ──

    # Q_tile 形状: [BLOCK_M, BLOCK_D]
    # 指针计算: Q_ptr + 行偏移 * stride_qn + 列偏移 * stride_qd
    # mask 确保: 行不越界（当 N 不是 BLOCK_M 整数倍时，最后一行块可能不完整）
    q_ptrs = Q_ptr + offs_m[:, None] * stride_qn + offs_d[None, :] * stride_qd
    q_mask = (offs_m[:, None] < N) & (offs_d[None, :] < d)
    Q_tile = tl.load(q_ptrs, mask=q_mask, other=0.0)

    # ── 第四步: 初始化累加器（驻留在寄存器中）──
    # acc: 累加输出，形状 [BLOCK_M, BLOCK_D]
    # 关键: 这个累加器在整个内循环中一直驻留在 SRAM/寄存器，永不写回 HBM！
    acc = tl.zeros([BLOCK_M, BLOCK_D], dtype=tl.float32)

# ── 第五步: 内循环——遍历 K/V 的所有列块 ──

    # num_k_blocks: K/V 需要分多少块
    #   例: N=1024, BLOCK_N=64 → num_k_blocks = 16
    # 每次迭代加载 K/V 的一块，计算局部注意力，累加到 acc


    # 固定Q， 才有后面的事情

    num_k_blocks = tl.cdiv(N, BLOCK_N)

    for j in range(num_k_blocks):

        # ── 5a: 计算当前 K/V 块的行范围 ──
        kn_start = j * BLOCK_N
        offs_n = kn_start + tl.arange(0, BLOCK_N)



        # ── 5b: 从 HBM 加载 K_j tile 到 SRAM ──
        # K_tile 形状: [BLOCK_N, BLOCK_D]

        k_ptrs = K_ptr + offs_n[:, None] * stride_kn + offs_d[None, :] * stride_kd
        k_mask = (offs_n[:, None] < N) & (offs_d[None, :] < d)
        K_tile = tl.load(k_ptrs, mask=k_mask, other=0.0)

        # ── 5c: 从 HBM 加载 V_j tile 到 SRAM ──
        # V_tile 形状: [BLOCK_N, BLOCK_D]
        v_ptrs = V_ptr + offs_n[:, None] * stride_vn + offs_d[None, :] * stride_vd
        v_mask = (offs_n[:, None] < N) & (offs_d[None, :] < d)
        V_tile = tl.load(v_ptrs, mask=v_mask, other=0.0)

        # ── 5d: 在 SRAM 中计算局部注意力分数 S_ij = Q_i @ K_j^T / sqrt(d) ──
        # S_ij 形状: [BLOCK_M, BLOCK_N]
        # 这一步是两个小矩阵的乘法，完全在 SRAM 中完成
        # 数学: S[i,j] = sum_d Q[i,d] * K[j,d] / sqrt(d)
        # 注意: scale 从 Python 端传入（= 1/sqrt(d)），避免 Triton JIT 类型问题
        S_ij = tl.dot(Q_tile, tl.trans(K_tile)) * scale

        # ── 5d-fix: 屏蔽越界 K/V 行 ──
        # 当 N 不是 BLOCK_N 整数倍时，最后一个 K/V tile 会有越界行
        # 这些行被 tl.load 填充为 0，导致 S_ij 中对应列为 0
        # 但 softmax 中 exp(0)=1 会污染归一化，所以必须把这些位置设为 -inf
        S_ij = tl.where(offs_n[None, :] < N, S_ij, float('-inf'))

        # ── 5e: 局部 Softmax（简化版）──
        # 在每个 tile 内独立计算 softmax
        # 注意: 完整 FlashAttention 使用 online softmax 维护跨 tile 的 running max/sum
        # 这里为了教学简洁，每个 tile 独立做 softmax（近似值）

        # 计算每行的最大值（数值稳定性：减去 max 防止 exp 溢出）
        # m_ij 形状: [BLOCK_M, 1]（broadcast）
        # 原本是每行取一个最大值，相当于得到一个  一维数组，  后面保留第一维度，  None表示增加第二个维度,  这样就可以和 S_ij 做广播运算
        m_ij = tl.max(S_ij, axis=1)[:, None]

        # 减去最大值后取指数
        # P_ij 形状: [BLOCK_M, BLOCK_N]
        # P，S，m 都是局部 tile 内的矩阵，完全驻留在 SRAM 中，永不写回 HBM，且维度相同，所以可以这么做
        P_ij = tl.exp(S_ij - m_ij)

        # 计算每行的归一化因子（sum of exp）
        # l_ij 形状: [BLOCK_M, 1]
        l_ij = tl.sum(P_ij, axis=1)[:, None]

        # 归一化得到局部注意力权重
        # P_ij 形状: [BLOCK_M, BLOCK_N]
        P_ij = P_ij / l_ij

        # ── 5f: 局部输出累加 O_ij = P_ij @ V_j ──
        # 累加到寄存器中的 acc
        # acc += P_ij @ V_j
        # P_ij: [BLOCK_M, BLOCK_N], V_tile: [BLOCK_N, BLOCK_D]
        # 结果: [BLOCK_M, BLOCK_D]
        acc += tl.dot(P_ij.to(V_tile.dtype), V_tile)

        # ── 循环结束，S_ij 和 P_ij 在 SRAM 中被释放，不写回 HBM ──
        # 这就是 FlashAttention 的核心: N×N 的中间矩阵从未真正存在于 HBM 中

    # ── 第六步: 将最终结果从 SRAM 写回 HBM ──
    # 这是整个 kernel 中唯一一次向 HBM 写入输出
    # acc 形状: [BLOCK_M, BLOCK_D] → 写到 O[qm_start:qm_start+BLOCK_M, :]

    # 小o，是当前分块得到的outputs, 形状 [BLOCK_M, BLOCK_D],
    # 大O，不然，是整个输出矩阵，形状 [N, d]
    o_ptrs = O_ptr + offs_m[:, None] * stride_on + offs_d[None, :] * stride_od
    o_mask = (offs_m[:, None] < N) & (offs_d[None, :] < d)

    tl.store(o_ptrs, acc.to(O_ptr.dtype.element_ty), mask=o_mask)


# ═══════════════════════════════════════════════════════════════════════════════
# Python 接口: 封装 Triton kernel 的调用
# ═══════════════════════════════════════════════════════════════════════════════

def flash_attention_lite(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor,
                         block_m: int = 64, block_n: int = 64) -> torch.Tensor:
    """
    FlashAttention-Lite 的 Python 接口

    参数:
        q: Query 矩阵, 形状 [N, d], 必须在 CUDA 设备上
        k: Key 矩阵, 形状 [N, d], 必须在 CUDA 设备上
        v: Value 矩阵, 形状 [N, d], 必须在 CUDA 设备上
        block_m: Q 的行分块大小（默认 64）
        block_n: K/V 的行分块大小（默认 64）

    返回:
        o: 输出矩阵, 形状 [N, d], 与输入相同的 dtype 和 device

    实现说明:
        1. 确保 BLOCK_D 等于 head_dim d（特征维度不分块）
        2. 分配输出张量 O
        3. 构建 1D Grid，大小 = ceil(N / BLOCK_M)
        4. 启动 kernel

    对应课程概念:
        - L7 的 Grid/Block 调度: 每个 program instance = 一个 GPU block
        - L8 的算子融合: matmul + softmax + matmul 在同一个 kernel 中完成
    """
    assert q.is_cuda and k.is_cuda and v.is_cuda, "输入张量必须在 CUDA 设备上"
    assert q.shape == k.shape == v.shape, "Q, K, V 形状必须相同"
    N, d = q.shape
    assert d <= 128, f"head_dim={d} 过大，本简化实现建议 d <= 128"

    # 分配输出张量，形状与输入相同
    o = torch.empty_like(q)

    # 特征维度分块大小直接等于 d
    BLOCK_D = triton.next_power_of_2(d)

    # 预计算注意力缩放因子，传入 kernel（避免 Triton JIT 内 float(tensor) 问题）
    scale = 1.0 / math.sqrt(d)

    # 计算 Grid 大小: 每个 program instance 负责 BLOCK_M 行
    # ceil(N / BLOCK_M) 个 program instance
    grid = (triton.cdiv(N, block_m),)

    # 启动 Triton kernel
    # 对应 L7: Triton 的 kernel launch 语法
    # grid 指定了并行启动的 program instance 数量
    _flash_attn_lite_kernel[grid](
        q, k, v, o,
        N, d, scale,
        q.stride(0), q.stride(1),
        k.stride(0), k.stride(1),
        v.stride(0), v.stride(1),
        o.stride(0), o.stride(1),
        BLOCK_M=block_m,
        BLOCK_N=block_n,
        BLOCK_D=BLOCK_D,
    )

    return o


# ═══════════════════════════════════════════════════════════════════════════════
# PyTorch 标准注意力实现（Baseline）
# ═══════════════════════════════════════════════════════════════════════════════

def naive_attention_pytorch(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    """
    PyTorch 标准注意力实现（Baseline，用于对比）

    这是最直接的 Attention 实现:
      S = Q @ K^T / sqrt(d)     # [N, d] @ [d, N] → [N, N]
      P = softmax(S, dim=-1)     # [N, N]
      O = P @ V                  # [N, N] @ [N, d] → [N, d]

    问题:
      1. 显存: 中间矩阵 S, P 都是 N×N，峰值显存 O(N^2)
      2. HBM 访存: S 和 P 都必须写回 HBM 再读回来（至少 16N^2 bytes）
      3. Kernel 数量: matmul + softmax + matmul = 多个 kernel 启动
      4. 计算强度低: Arithmetic Intensity ≈ d/8，严重 Memory Bound

    对比 FlashAttention-Lite:
      - HBM 访存: O(N^2) → O(N)
      - 峰值显存: O(N^2) → O(N)
      - Kernel 数量: 多个 → 1 个（算子融合）
      - 计算强度: O(1) → O(N)，从 Memory Bound 移向 Compute Bound
    """
    d = q.shape[-1]
    scale = 1.0 / math.sqrt(d)
    # S = Q @ K^T / sqrt(d)  — 显式构造 N×N 矩阵，写入 HBM
    s = torch.matmul(q, k.transpose(-2, -1)) * scale
    # P = softmax(S) — 在 N×N 矩阵上做 softmax，再次读写 HBM
    p = torch.softmax(s, dim=-1)
    # O = P @ V — 再次读取 N×N 的 P 矩阵
    o = torch.matmul(p, v)
    return o
