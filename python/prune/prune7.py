import torch
import torch.nn as nn
import torch.nn.utils.prune as prune

# 定义一个简单的神经网络
class SimpleNN(nn.Module):
    def __init__(self):
        super(SimpleNN, self).__init__()
        self.fc1 = nn.Linear(10, 5)
        self.fc2 = nn.Linear(5, 3)

    def forward(self, x):
        x = self.fc1(x)
        x = self.fc2(x)
        return x

# 实例化网络
model = SimpleNN()

# 使用 L1Unstructured 对第一个全连接层进行剪枝
# 剪枝前，查看权重
print("Before pruning:")
print(model.fc1.weight)

# 应用 L1Unstructured 剪枝方法，保留 50% 的权重
prune.l1_unstructured(model.fc1,name='weight', amount=0.5)

# 剪枝后，查看权重
print("After pruning:")
print(model.fc1.weight)
