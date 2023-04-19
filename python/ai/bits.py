import torch
import bitsandbytes as bnb

# 创建一个普通的PyTorch张量
torch_tensor = torch.randn(8, 16)

# 使用bitsandbytes库将PyTorch张量转换为优化张量
bnb_tensor = bnb.Float16CompressedTensor(torch_tensor)

# 执行张量操作
result = bnb_tensor * 2

# 将优化张量转换回普通的PyTorch张量
result_torch_tensor = result.to_normal_tensor()

print("Original Tensor:\n", torch_tensor)
print("BitsAndBytes Tensor:\n", bnb_tensor)
print("Result (multiplied by 2):\n", result_torch_tensor)
