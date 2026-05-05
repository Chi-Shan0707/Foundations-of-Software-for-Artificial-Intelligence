# AOT ID: ['0_backward']
from ctypes import c_void_p, c_long, c_int
import torch
import math
import random
import os
import tempfile
from math import inf, nan
from cmath import nanj
from torch._inductor.hooks import run_intermediate_hooks
from torch._inductor.utils import maybe_profile
from torch._inductor.codegen.memory_planning import _align as align
from torch import device, empty_strided
from torch._inductor.async_compile import AsyncCompile
from torch._inductor.select_algorithm import extern_kernels
import triton
import triton.language as tl
from torch._inductor.runtime.triton_heuristics import start_graph, end_graph
from torch._C import _cuda_getCurrentRawStream as get_raw_stream

aten = torch.ops.aten
inductor_ops = torch.ops.inductor
_quantized = torch.ops._quantized
assert_size_stride = torch._C._dynamo.guards.assert_size_stride
assert_alignment = torch._C._dynamo.guards.assert_alignment
empty_strided_cpu = torch._C._dynamo.guards._empty_strided_cpu
empty_strided_cpu_pinned = torch._C._dynamo.guards._empty_strided_cpu_pinned
empty_strided_cuda = torch._C._dynamo.guards._empty_strided_cuda
empty_strided_xpu = torch._C._dynamo.guards._empty_strided_xpu
empty_strided_mtia = torch._C._dynamo.guards._empty_strided_mtia
reinterpret_tensor = torch._C._dynamo.guards._reinterpret_tensor
alloc_from_pool = torch.ops.inductor._alloc_from_pool
async_compile = AsyncCompile()
empty_strided_p2p = torch._C._distributed_c10d._SymmetricMemory.empty_strided_p2p


# kernel path: /tmp/torchinductor_chishan/vm/cvm2q5b2lqelp5z4six6awbkfpjgmfg7w533krimpyqz5udvrlql.py
# Topologically Sorted Source Nodes: [sum_1], Original ATen: [aten.sum]
# Source node to ATen node mapping:
#   sum_1 => sum_1
# Graph fragment:
#   %tangents_1 : Tensor "f32[128, 1][1, 1]cuda:0" = PlaceHolder[target=tangents_1]
#   %sum_1 : Tensor "f32[1, 1][1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sum.dim_IntList](args = (%tangents_1, [0], True), kwargs = {})
#   return %sum_1
triton_per_fused_sum_0 = async_compile.triton('triton_per_fused_sum_0', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.persistent_reduction(
    size_hints={'x': 1, 'r0_': 128},
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'out_ptr0': '*fp32', 'xnumel': 'constexpr', 'r0_numel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=26, cc=120, major=12, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, max_threads_per_block=1024, warp_size=32), 'constants': {'xnumel': 1}, 'native_matmul': False, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]]}], 'enable_fp_fusion': True},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_per_fused_sum_0', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': None, 'atomic_add_found': False, 'num_load': 1, 'num_store': 1, 'num_reduction': 1, 'backend_hash': '3D121FA2EE34AE10E62572A15F25384E1A4682049D5A62D7F25E2FC5672C2859', 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'deterministic': False, 'force_filter_reduction_configs': False, 'are_deterministic_algorithms_enabled': False, 'tiling_scores': {'r0_': 512}}
)
@triton.jit
def triton_per_fused_sum_0(in_ptr0, out_ptr0, xnumel, r0_numel, XBLOCK : tl.constexpr):
    xnumel = 1
    r0_numel = 128
    R0_BLOCK: tl.constexpr = 128
    rnumel = r0_numel
    RBLOCK: tl.constexpr = R0_BLOCK
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = tl.full([XBLOCK], True, tl.int1)[:, None]
    r0_index = tl.arange(0, R0_BLOCK)[None, :]
    r0_offset = 0
    r0_mask = tl.full([R0_BLOCK], True, tl.int1)[None, :]
    roffset = r0_offset
    rindex = r0_index
    r0_0 = r0_index
    tmp0 = tl.load(in_ptr0 + (r0_0), None)
    tmp1 = tl.broadcast_to(tmp0, [XBLOCK, R0_BLOCK])
    tmp3 = tl.sum(tmp1, 1)[:, None].to(tl.float32)
    tl.store(out_ptr0 + (tl.full([1, 1], 0, tl.int32).broadcast_to(XBLOCK, 1)), tmp3, None)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_chishan/uw/cuw5wak3cpjceusr5x2rgghgqgv546mi7r4dtuk72d3dqfvwpt7y.py
# Topologically Sorted Source Nodes: [le, scalar_tensor, where, sum_2], Original ATen: [aten.threshold_backward, aten.sum]
# Source node to ATen node mapping:
#   le => le
#   scalar_tensor => full_default
#   sum_2 => sum_2
#   where => where
# Graph fragment:
#   %relu : Tensor "f32[128, 64][64, 1]cuda:0" = PlaceHolder[target=relu]
#   %mm_3 : Tensor "f32[128, 64][64, 1]cuda:0" = PlaceHolder[target=mm_3]
#   %where : Tensor "f32[128, 64][64, 1]cuda:0" = PlaceHolder[target=where]
#   %le : Tensor "b8[128, 64][64, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.le.Scalar](args = (%relu, 0), kwargs = {})
#   %full_default : Tensor "f32[][]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.full.default](args = ([], 0.0), kwargs = {dtype: torch.float32, layout: torch.strided, device: cuda:0, pin_memory: False})
#   %where : Tensor "f32[128, 64][64, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.where.self](args = (%le, %full_default, %mm_3), kwargs = {})
#   %sum_2 : Tensor "f32[1, 64][64, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sum.dim_IntList](args = (%where, [0], True), kwargs = {})
#   return %where,%sum_2
triton_red_fused_sum_threshold_backward_1 = async_compile.triton('triton_red_fused_sum_threshold_backward_1', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.reduction(
    size_hints={'x': 64, 'r0_': 128},
    reduction_hint=ReductionHint.OUTER,
    filename=__file__,
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'in_ptr0': '*fp32', 'out_ptr0': '*fp32', 'xnumel': 'i32', 'r0_numel': 'i32', 'XBLOCK': 'constexpr', 'R0_BLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=26, cc=120, major=12, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, max_threads_per_block=1024, warp_size=32), 'constants': {}, 'native_matmul': False, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]]}], 'enable_fp_fusion': True},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_red_fused_sum_threshold_backward_1', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': True, 'no_x_dim': False, 'atomic_add_found': False, 'num_load': 2, 'num_store': 2, 'num_reduction': 1, 'backend_hash': '3D121FA2EE34AE10E62572A15F25384E1A4682049D5A62D7F25E2FC5672C2859', 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'deterministic': False, 'force_filter_reduction_configs': False, 'are_deterministic_algorithms_enabled': False, 'tiling_scores': {'x': 131584, 'r0_': 0}}
)
@triton.jit
def triton_red_fused_sum_threshold_backward_1(in_out_ptr0, in_ptr0, out_ptr0, xnumel, r0_numel, XBLOCK : tl.constexpr, R0_BLOCK : tl.constexpr):
    xnumel = 64
    r0_numel = 128
    rnumel = r0_numel
    RBLOCK: tl.constexpr = R0_BLOCK
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    r0_base = tl.arange(0, R0_BLOCK)[None, :]
    rbase = r0_base
    x0 = xindex
    _tmp6 = tl.full([XBLOCK, R0_BLOCK], 0, tl.float32)
    for r0_offset in range(0, r0_numel, R0_BLOCK):
        r0_index = r0_offset + r0_base
        r0_mask = r0_index < r0_numel
        roffset = r0_offset
        rindex = r0_index
        r0_1 = r0_index
        tmp0 = tl.load(in_out_ptr0 + (x0 + 64*r0_1), r0_mask & xmask, eviction_policy='evict_first', other=0.0)
        tmp3 = tl.load(in_ptr0 + (x0 + 64*r0_1), r0_mask & xmask, eviction_policy='evict_first', other=0.0)
        tmp1 = 0.0
        tmp2 = tmp0 <= tmp1
        tmp4 = tl.where(tmp2, tmp1, tmp3)
        tmp5 = tl.broadcast_to(tmp4, [XBLOCK, R0_BLOCK])
        tmp7 = _tmp6 + tmp5
        _tmp6 = tl.where(r0_mask & xmask, tmp7, _tmp6)
        tl.store(in_out_ptr0 + (x0 + 64*r0_1), tmp4, r0_mask & xmask)
    tmp6 = tl.sum(_tmp6, 1)[:, None]
    tl.store(out_ptr0 + (x0), tmp6, xmask)
''', device_str='cuda')


async_compile.wait(globals())
del async_compile

class Runner:
    def __init__(self, partitions):
        self.partitions = partitions

    def recursively_apply_fns(self, fns):
        new_callables = []
        for fn, c in zip(fns, self.partitions):
            new_callables.append(fn(c))
        self.partitions = new_callables

    def call(self, args):
        primals_4, relu, permute_2, tangents_1 = args
        args.clear()
        assert_size_stride(primals_4, (64, 1), (1, 1))
        assert_size_stride(relu, (128, 64), (64, 1))
        assert_size_stride(permute_2, (1, 128), (1, 1))
        assert_size_stride(tangents_1, (128, 1), (1, 1))
        with torch.cuda._DeviceGuard(0):
            torch.cuda.set_device(0)
            buf0 = empty_strided_cuda((1, 1), (1, 1), torch.float32)
            # Topologically Sorted Source Nodes: [sum_1], Original ATen: [aten.sum]
            # [Provenance debug handles] triton_per_fused_sum_0:1
            stream0 = get_raw_stream(0)
            triton_per_fused_sum_0.run(tangents_1, buf0, 1, 128, stream=stream0)
            buf1 = empty_strided_cuda((64, 1), (1, 1), torch.float32)
            # Topologically Sorted Source Nodes: [permute, mm_2], Original ATen: [aten.t, aten.mm]
            # [Provenance debug handles] extern_kernels.mm:2
            extern_kernels.mm(reinterpret_tensor(relu, (64, 128), (1, 64), 0), tangents_1, out=buf1)
            buf2 = empty_strided_cuda((128, 64), (64, 1), torch.float32)
            # Topologically Sorted Source Nodes: [permute_1, mm_3], Original ATen: [aten.t, aten.mm]
            # [Provenance debug handles] extern_kernels.mm:3
            extern_kernels.mm(tangents_1, reinterpret_tensor(primals_4, (1, 64), (1, 1), 0), out=buf2)
            del primals_4
            del tangents_1
            buf3 = relu; del relu  # reuse
            buf4 = empty_strided_cuda((1, 64), (64, 1), torch.float32)
            # Topologically Sorted Source Nodes: [le, scalar_tensor, where, sum_2], Original ATen: [aten.threshold_backward, aten.sum]
            # [Provenance debug handles] triton_red_fused_sum_threshold_backward_1:4
            stream0 = get_raw_stream(0)
            triton_red_fused_sum_threshold_backward_1.run(buf3, buf2, buf4, 64, 128, stream=stream0)
            del buf2
            buf5 = empty_strided_cuda((1, 64), (64, 1), torch.float32)
            # Topologically Sorted Source Nodes: [mm_4], Original ATen: [aten.mm]
            # [Provenance debug handles] extern_kernels.mm:5
            extern_kernels.mm(permute_2, buf3, out=buf5)
            del buf3
            del permute_2
        return (buf5, None, reinterpret_tensor(buf4, (64, ), (1, ), 0), buf1, reinterpret_tensor(buf0, (1, ), (1, ), 0), )

runner = Runner(partitions=[])
call = runner.call
recursively_apply_fns = runner.recursively_apply_fns


def benchmark_compiled_module(times=10, repeat=10):
    from torch._dynamo.testing import rand_strided
    from torch._inductor.utils import print_performance
    primals_4 = rand_strided((64, 1), (1, 1), device='cuda:0', dtype=torch.float32)
    relu = rand_strided((128, 64), (64, 1), device='cuda:0', dtype=torch.float32)
    permute_2 = rand_strided((1, 128), (1, 1), device='cuda:0', dtype=torch.float32)
    tangents_1 = rand_strided((128, 1), (1, 1), device='cuda:0', dtype=torch.float32)
    fn = lambda: call([primals_4, relu, permute_2, tangents_1])
    return print_performance(fn, times=times, repeat=repeat)


if __name__ == "__main__":
    from torch._inductor.wrapper_benchmark import compiled_module_main
    compiled_module_main('None', benchmark_compiled_module)
