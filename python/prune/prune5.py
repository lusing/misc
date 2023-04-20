import torch
import torch.nn as nn
import torch.nn.utils.prune as prune

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(784, 256)
        self.fc2 = nn.Linear(256, 10)

    def forward(self, x):
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = nn.functional.relu(x)
        x = self.fc2(x)
        return x

net = Net()

# 打印剪枝前的参数数量
num_params_before = net.fc1.weight.numel()
print(f"Number of parameters before pruning: {num_params_before}")

# 定义要剪枝的层和剪枝比例
module_to_prune = net.fc1
pruning_perc = 50

# 进行剪枝
prune.ln_structured(module_to_prune, name='weight', amount=pruning_perc, n=2, dim=0)

# 打印剪枝后的参数数量
num_params_after = net.fc1.weight.numel()
print(f"Number of parameters after pruning: {num_params_after}")

# 保存剪枝后的模型参数
#torch.save(net.state_dict(), 'pruned_model.pth')
