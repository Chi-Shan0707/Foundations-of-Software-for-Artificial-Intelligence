"""
FlashAttention with Online Softmax: 全局精确的 IO-Aware 分块注意力
================================================================

与 flashattention_lite.py 的区别 ONLY 在内循环的 softmax 部分:

  flashattention_lite.py (局部 softmax):
    每个 tile 独立做 softmax → 累加 → 结果是近似的 (N > BLOCK_N 时有误差)

  本文件 (online softmax):
    维护 running max (m_i) 和 running sum (l_i)
    每处理一个新 tile, 用修正因子更新之前的累加器
    → 结果与标准 Attention 全局精确一致

改动量: 初始化多 2 行 (m_i, l_i), 内循环改 ~10 行, 末尾加 1 行 (acc /= l_i)
其余代码与 flashattention_lite.py 完全相同。
"""

import torch
import triton
import triton.language as tl
import math


@triton.jit
def _flash_attn_online_kernel(
    Q_ptr, K_ptr, V_ptr, O_ptr,
    N, d, scale,
    stride_qn, stride_qd,
    stride_kn, stride_kd,
    stride_vn, stride_vd,
    stride_on, stride_od,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    pid_m = tl.program_id(0)
    qm_start = pid_m * BLOCK_M
    offs_m = qm_start + tl.arange(0, BLOCK_M)
    offs_d = tl.arange(0, BLOCK_D)

    # 加载 Q tile 到 SRAM
    q_ptrs = Q_ptr + offs_m[:, None] * stride_qn + offs_d[None, :] * stride_qd
    q_mask = (offs_m[:, None] < N) & (offs_d[None, :] < d)
    Q_tile = tl.load(q_ptrs, mask=q_mask, other=0.0)

    # ── 初始化累加器 + online softmax 状态 ──
    # acc: 加权输出累加器 [BLOCK_M, BLOCK_D]
    # m_i: 每行的 running max [BLOCK_M, 1]，初始 -inf（任何数都大于 -inf）
    # l_i: 每行的 running sum-of-exp [BLOCK_M, 1]，初始 0
    acc = tl.zeros([BLOCK_M, BLOCK_D], dtype=tl.float32)
    m_i = tl.full([BLOCK_M, 1], float('-inf'), dtype=tl.float32)
    l_i = tl.zeros([BLOCK_M, 1], dtype=tl.float32)

    num_k_blocks = tl.cdiv(N, BLOCK_N)

    for j in range(num_k_blocks):
        kn_start = j * BLOCK_N
        offs_n = kn_start + tl.arange(0, BLOCK_N)

        # 加载 K_j, V_j tile 到 SRAM
        k_ptrs = K_ptr + offs_n[:, None] * stride_kn + offs_d[None, :] * stride_kd
        k_mask = (offs_n[:, None] < N) & (offs_d[None, :] < d)
        K_tile = tl.load(k_ptrs, mask=k_mask, other=0.0)

        v_ptrs = V_ptr + offs_n[:, None] * stride_vn + offs_d[None, :] * stride_vd
        v_mask = (offs_n[:, None] < N) & (offs_d[None, :] < d)
        V_tile = tl.load(v_ptrs, mask=v_mask, other=0.0)

        # S_ij = Q_i @ K_j^T * scale
        S_ij = tl.dot(Q_tile, tl.trans(K_tile)) * scale
        S_ij = tl.where(offs_n[None, :] < N, S_ij, float('-inf'))

        # ── Online Softmax: 跨 tile 维护全局精确的 softmax ──
        #
        # 数学推导:
        #   假设之前已经看了 j_1, j_2, ..., j_{prev} 个 tile
        #   m_i = max(所有已见 tile 的 S 值)  ← running max
        #   l_i = sum(所有已见 tile 的 exp(S - m_i))  ← running sum
        #   acc = sum(所有已见 tile 的 exp(S - m_i) / l_i * V)
        #
        #   新来一个 tile j:
        #   m_new = max(m_i, max(S_ij))          ← 更新全局 max
        #   修正因子 = exp(m_i - m_new)           ← 之前的 exp 值要缩放
        #   P_ij = exp(S_ij - m_new)              ← 当前 tile 的 exp
        #   l_new = l_i * exp(m_i - m_new) + sum(P_ij)  ← 更新全局 sum
        #   acc  = acc * exp(m_i - m_new) + P_ij @ V_j  ← 更新累加器
        #
        #   关键: exp(m_i - m_new) 把之前的贡献从 "以 m_i 为基准" 
        #         修正为 "以 m_new 为基准"，保证全局 softmax 精确
        #
        # 与 lite 版的区别:
        #   lite: m_ij = max(S_ij)          ← 仅当前 tile 的 max (局部)
        #   online: m_new = max(m_i, max(S_ij))  ← 所有 tile 的 max (全局)

        # 当前 tile 的行最大值
        m_ij = tl.max(S_ij, axis=1)[:, None]

        # 更新全局 running max
        m_new = tl.maximum(m_i, m_ij)

        # 修正因子: 将之前的 acc 和 l_i 从旧 max 缩放到新 max
        correction = tl.exp(m_i - m_new)

        # 当前 tile 的 exp(S_ij - m_new)
        P_ij = tl.exp(S_ij - m_new)

        # 更新 running sum: l_new = l_old * correction + sum(P_ij)
        l_ij = tl.sum(P_ij, axis=1)[:, None]
        l_new = l_i * correction + l_ij

        # 更新累加器: acc = acc * correction + P_ij @ V_j
        acc = acc * correction + tl.dot(P_ij.to(V_tile.dtype), V_tile)

        # 更新 running 状态
        m_i = m_new
        l_i = l_new

    # ── 最终归一化: acc /= l_i ──
    # online softmax 的最后一步: 之前 acc 累加的是 "未归一化的加权和"
    # 除以 l_i 得到真正的 O = softmax(QK^T/√d) @ V
    acc = acc / l_i

    # 写回 HBM
    o_ptrs = O_ptr + offs_m[:, None] * stride_on + offs_d[None, :] * stride_od
    o_mask = (offs_m[:, None] < N) & (offs_d[None, :] < d)
    tl.store(o_ptrs, acc.to(O_ptr.dtype.element_ty), mask=o_mask)


def flash_attention_online(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor,
                           block_m: int = 64, block_n: int = 64) -> torch.Tensor:
    """
    FlashAttention with Online Softmax 的 Python 接口
    接口与 flash_attention_lite 完全相同，但结果全局精确。
    """
    assert q.is_cuda and k.is_cuda and v.is_cuda, "输入张量必须在 CUDA 设备上"
    assert q.shape == k.shape == v.shape, "Q, K, V 形状必须相同"
    N, d = q.shape
    assert d <= 128, f"head_dim={d} 过大，建议 d <= 128"

    o = torch.empty_like(q)
    BLOCK_D = triton.next_power_of_2(d)
    scale = 1.0 / math.sqrt(d)
    grid = (triton.cdiv(N, block_m),)

    _flash_attn_online_kernel[grid](
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
