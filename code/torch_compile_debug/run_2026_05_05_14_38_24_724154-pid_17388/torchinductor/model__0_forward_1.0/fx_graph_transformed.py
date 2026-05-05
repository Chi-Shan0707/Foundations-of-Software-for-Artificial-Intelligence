class GraphModule(torch.nn.Module):
    def forward(self, primals_1: "f32[1, 64]", primals_2: "f32[128, 1]", primals_3: "f32[64]", primals_4: "f32[64, 1]", primals_5: "f32[1]"):
        # File: /mnt/d/FudanUniversity/Fdu1/Foundations of Software for Artificial Intelligence/code/l8-mlp.py:21 in forward, code: x = x @ self.weight1 + self.bias1   # (B, H)
        mm: "f32[128, 64]" = torch.ops.aten.mm.default(primals_2, primals_1);  primals_1 = None
        add: "f32[128, 64]" = torch.ops.aten.add.Tensor(mm, primals_3);  mm = primals_3 = None
        
        # File: /mnt/d/FudanUniversity/Fdu1/Foundations of Software for Artificial Intelligence/code/l8-mlp.py:22 in forward, code: x = torch.relu(x)
        relu: "f32[128, 64]" = torch.ops.aten.relu.default(add);  add = None
        
        # No stacktrace found for following nodes
        addmm_default: "f32[128, 1]" = torch.ops.aten.addmm.default(primals_5, relu, primals_4);  primals_5 = None
        
        # File: /mnt/d/FudanUniversity/Fdu1/Foundations of Software for Artificial Intelligence/code/l8-mlp.py:21 in forward, code: x = x @ self.weight1 + self.bias1   # (B, H)
        permute_2: "f32[1, 128]" = torch.ops.aten.permute.default(primals_2, [1, 0]);  primals_2 = None
        return (addmm_default, primals_4, relu, permute_2)
        