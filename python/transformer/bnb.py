# 导入相关库
import torch
import bitsandbytes as bnb

# 创建一个随机的输入矩阵
x = torch.randn(1000, 1000).cuda()

# 创建一个8位的线性层
linear = bnb.nn.Linear8bitLt(1000, 1000)

# 通过线性层得到输出
y = linear(x)

# 打印输出的形状
print(y.shape)
