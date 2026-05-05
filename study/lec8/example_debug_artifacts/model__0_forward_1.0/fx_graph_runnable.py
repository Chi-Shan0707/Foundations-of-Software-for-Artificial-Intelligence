
import os
os.environ['TORCH_COMPILE_DEBUG'] = '1'
os.environ['TORCHINDUCTOR_CACHE_DIR'] = '/tmp/torchinductor_chishan'

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

torch._inductor.config.trace.enabled = False
torch._inductor.config.trace.save_real_tensors = False
torch._functorch.config.functionalize_rng_ops = False
torch._functorch.config.debug_partitioner = True
torch._functorch.config.fake_tensor_allow_unsafe_data_ptr_access = True
torch._functorch.config.unlift_effect_tokens = True
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

    
    
    def forward(self, primals_1, primals_2, primals_3, primals_4, primals_5):
        mm = torch.ops.aten.mm.default(primals_2, primals_1);  primals_1 = None
        add = torch.ops.aten.add.Tensor(mm, primals_3);  mm = primals_3 = None
        relu = torch.ops.aten.relu.default(add);  add = None
        mm_1 = torch.ops.aten.mm.default(relu, primals_4)
        add_1 = torch.ops.aten.add.Tensor(mm_1, primals_5);  mm_1 = primals_5 = None
        permute_2 = torch.ops.aten.permute.default(primals_2, [1, 0]);  primals_2 = None
        return (add_1, primals_4, relu, permute_2)
        
def load_args(reader):
    buf0 = reader.storage(None, 256, device=device(type='cuda', index=0))
    reader.tensor(buf0, (1, 64), is_leaf=True)  # primals_1
    buf1 = reader.storage(None, 512, device=device(type='cuda', index=0))
    reader.tensor(buf1, (128, 1), is_leaf=True)  # primals_2
    buf2 = reader.storage(None, 256, device=device(type='cuda', index=0))
    reader.tensor(buf2, (64,), is_leaf=True)  # primals_3
    buf3 = reader.storage(None, 256, device=device(type='cuda', index=0))
    reader.tensor(buf3, (64, 1), is_leaf=True)  # primals_4
    buf4 = reader.storage(None, 4, device=device(type='cuda', index=0))
    reader.tensor(buf4, (1,), is_leaf=True)  # primals_5
load_args._version = 0
mod = Repro()
if __name__ == '__main__':
    from torch._dynamo.repro.after_aot import run_repro
    with torch.no_grad():
        run_repro(mod, load_args, accuracy=False, command='run', save_dir=None, tracing_mode='real', check_str=None)
        # To run it separately, do 
        # mod, args = run_repro(mod, load_args, accuracy=False, command='get_args', save_dir=None, tracing_mode='real', check_str=None)
        # mod(*args)