# 导入torch和torch.nn.utils.prune模块
import torch
import torch.nn.utils.prune as prune

# 定义一个简单的线性层
linear = torch.nn.Linear(10, 5)

# 打印线性层的权重矩阵
print(linear.weight)

# 使用L1Norm剪枝方法，对线性层的权重矩阵进行40%的剪枝
prune.l1_unstructured(linear, name="weight", amount=0.4)

# 打印剪枝后的权重矩阵，可以看到有40%的权重被设置为0
print(linear.weight)

# 打印线性层的权重掩码，可以看到被剪枝的权重对应的掩码为0，未被剪枝的权重对应的掩码为1
print(linear.weight_mask)
