import torch
import fbgemm_pack 
from torch.quantization import QuantStub, DeQuantStub

# 定义一个简单的全连接网络
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.quant = QuantStub()
        self.fc1 = nn.Linear(4, 2)
        self.fc2 = nn.Linear(2, 4) 
        self.dequant = DeQuantStub()

    def forward(self, x):
        x = self.quant(x)
        x = self.fc1(x)
        x = torch.relu(x)
        x = self.fc2(x)
        x = self.dequant(x)
        return x

# 量化全连接层参数   
model = Net()
model.fc1.weight = torch.nn.Parameter(torch.tensor([[1., 2.], [3., 4.]]), 
                                      requires_grad=False)
model.fc1.bias = torch.nn.Parameter(torch.tensor([0., 0.]), requires_grad=False)
model.fc2.weight = torch.nn.Parameter(torch.tensor([[1., 3.], [5., 7.], 
                                                    [9., 11.], [13., 15.]]), 
                                      requires_grad=False) 
model.fc2.bias = torch.nn.Parameter(torch.tensor([1., 1., 1., 1.]), 
                                    requires_grad=False)

# 量化输入                       
input = torch.tensor([[1., 2., 3., 4.]])  

# 使用fbgemm进行量化矩阵乘法
packed_fc1 = fbgemm_pack.LinearQuantized(model.fc1)    
output = packed_fc1(input) 

print(output)
