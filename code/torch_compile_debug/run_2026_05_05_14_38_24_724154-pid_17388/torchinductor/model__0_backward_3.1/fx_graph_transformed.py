class GraphModule(torch.nn.Module):
    def forward(self, primals_4: "f32[64, 1]", relu: "f32[128, 64]", permute_2: "f32[1, 128]", tangents_1: "f32[128, 1]"):
        # File: /mnt/d/FudanUniversity/Fdu1/Foundations of Software for Artificial Intelligence/code/l8-mlp.py:23 in forward, code: x = x @ self.weight2 + self.bias2   # (B, 1)
        sum_1: "f32[1, 1]" = torch.ops.aten.sum.dim_IntList(tangents_1, [0], True)
        view: "f32[1]" = torch.ops.aten.reshape.default(sum_1, [1]);  sum_1 = None
        permute: "f32[64, 128]" = torch.ops.aten.permute.default(relu, [1, 0])
        mm_2: "f32[64, 1]" = torch.ops.aten.mm.default(permute, tangents_1);  permute = None
        permute_1: "f32[1, 64]" = torch.ops.aten.permute.default(primals_4, [1, 0]);  primals_4 = None
        mm_3: "f32[128, 64]" = torch.ops.aten.mm.default(tangents_1, permute_1);  tangents_1 = permute_1 = None
        
        # File: /mnt/d/FudanUniversity/Fdu1/Foundations of Software for Artificial Intelligence/code/l8-mlp.py:22 in forward, code: x = torch.relu(x)
        le: "b8[128, 64]" = torch.ops.aten.le.Scalar(relu, 0);  relu = None
        full_default: "f32[]" = torch.ops.aten.full.default([], 0.0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        where: "f32[128, 64]" = torch.ops.aten.where.self(le, full_default, mm_3);  le = full_default = mm_3 = None
        
        # File: /mnt/d/FudanUniversity/Fdu1/Foundations of Software for Artificial Intelligence/code/l8-mlp.py:21 in forward, code: x = x @ self.weight1 + self.bias1   # (B, H)
        sum_2: "f32[1, 64]" = torch.ops.aten.sum.dim_IntList(where, [0], True)
        view_1: "f32[64]" = torch.ops.aten.reshape.default(sum_2, [64]);  sum_2 = None
        mm_4: "f32[1, 64]" = torch.ops.aten.mm.default(permute_2, where);  permute_2 = where = None
        return (mm_4, None, view_1, mm_2, view)
        