
import os
os.environ['TORCH_COMPILE_DEBUG'] = '1'
os.environ['TORCHINDUCTOR_CACHE_DIR'] = '/tmp/torchinductor_chishan'
os.environ['TRITON_CACHE_DIR'] = '/tmp/torchinductor_chishan/triton/0'

import torch
from torch import tensor, device
import torch.fx as fx
from torch._dynamo.testing import rand_strided
from math import inf
import torch._inductor.inductor_prims



import torch._dynamo.config
import torch._inductor.config
import torch._functorch.config
import torch.fx.experimental._config

torch._inductor.config.deterministic = False
torch._inductor.config.triton.store_cubin = False
torch._inductor.config.trace.enabled = False
torch._inductor.config.trace.save_real_tensors = False
torch._functorch.config.functionalize_rng_ops = False
torch._functorch.config.debug_partitioner = False
torch._functorch.config.fake_tensor_allow_unsafe_data_ptr_access = True
torch._functorch.config.unlift_effect_tokens = False
torch._functorch.config.selective_decompose = False



isolate_fails_code_str = None





if "__compile_source__" in globals():
    import inspect as __after_aot_inspect
    import linecache as __after_aot_linecache
    __after_aot_filename = __after_aot_inspect.currentframe().f_code.co_filename
    __after_aot_linecache.cache[__after_aot_filename] = (
        len(__compile_source__),
        None,
        __compile_source__.splitlines(True),
        __after_aot_filename,
    )
# torch version: 2.10.0+cu128
# torch cuda version: 12.8
# torch git version: 449b1768410104d3ed79d3bcfe4ba1d65c7f22c0


# CUDA Info: 
# nvcc: NVIDIA (R) Cuda compiler driver 
# Copyright (c) 2005-2023 NVIDIA Corporation 
# Built on Fri_Jan__6_16:45:21_PST_2023 
# Cuda compilation tools, release 12.0, V12.0.140 
# Build cuda_12.0.r12.0/compiler.32267302_0 

# GPU Hardware Info: 
# NVIDIA GeForce RTX 5060 Laptop GPU : 1 


from torch.nn import *
class Repro(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()

    
    
    def forward(self, primals_4, relu, permute_2, tangents_1):
        sum_1 = torch.ops.aten.sum.dim_IntList(tangents_1, [0], True)
        view = torch.ops.aten.view.default(sum_1, [1]);  sum_1 = None
        permute = torch.ops.aten.permute.default(relu, [1, 0])
        mm_2 = torch.ops.aten.mm.default(permute, tangents_1);  permute = None
        permute_1 = torch.ops.aten.permute.default(primals_4, [1, 0]);  primals_4 = None
        mm_3 = torch.ops.aten.mm.default(tangents_1, permute_1);  tangents_1 = permute_1 = None
        le = torch.ops.aten.le.Scalar(relu, 0);  relu = None
        full_default = torch.ops.aten.full.default([], 0.0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        where = torch.ops.aten.where.self(le, full_default, mm_3);  le = full_default = mm_3 = None
        sum_2 = torch.ops.aten.sum.dim_IntList(where, [0], True)
        view_1 = torch.ops.aten.view.default(sum_2, [64]);  sum_2 = None
        mm_4 = torch.ops.aten.mm.default(permute_2, where);  permute_2 = where = None
        return (mm_4, None, view_1, mm_2, view)
        
def load_args(reader):
    buf0 = reader.storage(None, 256, device=device(type='cuda', index=0))
    reader.tensor(buf0, (64, 1), is_leaf=True)  # primals_4
    buf1 = reader.storage(None, 32768, device=device(type='cuda', index=0))
    reader.tensor(buf1, (128, 64), is_leaf=True)  # relu
    buf2 = reader.storage(None, 512, device=device(type='cuda', index=0))
    reader.tensor(buf2, (1, 128), (1, 1), is_leaf=True)  # permute_2
    buf3 = reader.storage(None, 512, device=device(type='cuda', index=0))
    reader.tensor(buf3, (128, 1), is_leaf=True)  # tangents_1
load_args._version = 0
mod = Repro()
if __name__ == '__main__':
    from torch._dynamo.repro.after_aot import run_repro
    with torch.no_grad():
        run_repro(mod, load_args, accuracy=False, command='run', save_dir=None, tracing_mode='real', check_str=None)
        # To run it separately, do 
        # mod, args = run_repro(mod, load_args, accuracy=False, command='get_args', save_dir=None, tracing_mode='real', check_str=None)
        # mod(*args)