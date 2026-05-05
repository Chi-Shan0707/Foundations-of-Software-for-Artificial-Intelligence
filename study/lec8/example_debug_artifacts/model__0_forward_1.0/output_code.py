# AOT ID: ['0_forward']
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


# kernel path: /tmp/torchinductor_chishan/gj/cgjcrzwnpiqgtal2ucheot4j6ap7y2x5gy4l5c5t7jbr53p4m2um.py
# Topologically Sorted Source Nodes: [x, x_1], Original ATen: [aten.add, aten.relu]
# Source node to ATen node mapping:
#   x => add
#   x_1 => relu
# Graph fragment:
#   %mm : Tensor "f32[128, 64][64, 1]cuda:0" = PlaceHolder[target=mm]
#   %primals_3 : Tensor "f32[64][1]cuda:0" = PlaceHolder[target=primals_3]
#   %add : Tensor "f32[128, 64][64, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mm, %primals_3), kwargs = {})
#   %relu : Tensor "f32[128, 64][64, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.relu.default](args = (%add,), kwargs = {})
#   return %relu
triton_poi_fused_add_relu_0 = async_compile.triton('triton_poi_fused_add_relu_0', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 8192}, 
    filename=__file__,
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'in_ptr0': '*fp32', 'xnumel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=26, cc=120, major=12, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, max_threads_per_block=1024, warp_size=32), 'constants': {}, 'native_matmul': False, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]]}], 'enable_fp_fusion': True},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_relu_0', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': False, 'no_x_dim': False, 'atomic_add_found': False, 'num_load': 2, 'num_store': 1, 'num_reduction': 0, 'backend_hash': '3D121FA2EE34AE10E62572A15F25384E1A4682049D5A62D7F25E2FC5672C2859', 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'deterministic': False, 'force_filter_reduction_configs': False, 'are_deterministic_algorithms_enabled': False, 'tiling_scores': {'x': 98560}},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_add_relu_0(in_out_ptr0, in_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 8192
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)[:]
    x2 = xindex
    x0 = (xindex % 64)
    tmp0 = tl.load(in_out_ptr0 + (x2), None)
    tmp1 = tl.load(in_ptr0 + (x0), None, eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    tmp3 = tl.full([1], 0, tl.int32)
    tmp4 = triton_helpers.maximum(tmp3, tmp2)
    tl.store(in_out_ptr0 + (x2), tmp4, None)
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
        primals_1, primals_2, primals_3, primals_4, primals_5 = args
        args.clear()
        assert_size_stride(primals_1, (1, 64), (64, 1))
        assert_size_stride(primals_2, (128, 1), (1, 1))
        assert_size_stride(primals_3, (64, ), (1, ))
        assert_size_stride(primals_4, (64, 1), (1, 1))
        assert_size_stride(primals_5, (1, ), (1, ))
        with torch.cuda._DeviceGuard(0):
            torch.cuda.set_device(0)
            buf0 = empty_strided_cuda((128, 64), (64, 1), torch.float32)
            # Topologically Sorted Source Nodes: [matmul], Original ATen: [aten.mm]
            # [Provenance debug handles] extern_kernels.mm:1
            extern_kernels.mm(primals_2, primals_1, out=buf0)
            del primals_1
            buf1 = buf0; del buf0  # reuse
            # Topologically Sorted Source Nodes: [x, x_1], Original ATen: [aten.add, aten.relu]
            # [Provenance debug handles] triton_poi_fused_add_relu_0:2
            stream0 = get_raw_stream(0)
            triton_poi_fused_add_relu_0.run(buf1, primals_3, 8192, stream=stream0)
            del primals_3
            buf3 = empty_strided_cuda((128, 1), (1, 1), torch.float32)
            # Topologically Sorted Source Nodes: [], Original ATen: [aten.addmm]
            # [Provenance debug handles] extern_kernels.addmm:3
            extern_kernels.addmm(primals_5, buf1, primals_4, alpha=1, beta=1, out=buf3)
            del primals_5
        return (buf3, primals_4, buf1, reinterpret_tensor(primals_2, (1, 128), (1, 1), 0), )

runner = Runner(partitions=[])
call = runner.call
recursively_apply_fns = runner.recursively_apply_fns


def benchmark_compiled_module(times=10, repeat=10):
    from torch._dynamo.testing import rand_strided
    from torch._inductor.utils import print_performance
    primals_1 = rand_strided((1, 64), (64, 1), device='cuda:0', dtype=torch.float32)
    primals_2 = rand_strided((128, 1), (1, 1), device='cuda:0', dtype=torch.float32)
    primals_3 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_4 = rand_strided((64, 1), (1, 1), device='cuda:0', dtype=torch.float32)
    primals_5 = rand_strided((1, ), (1, ), device='cuda:0', dtype=torch.float32)
    fn = lambda: call([primals_1, primals_2, primals_3, primals_4, primals_5])
    return print_performance(fn, times=times, repeat=repeat)


if __name__ == "__main__":
    from torch._inductor.wrapper_benchmark import compiled_module_main
    compiled_module_main('None', benchmark_compiled_module)
