import torch
import torch.quantization as quant

# 定义一个简单的模型
class Net(torch.nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.fc1 = torch.nn.Linear(784, 512)
        self.fc2 = torch.nn.Linear(512, 10)
        self.relu = torch.nn.ReLU()

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x

# 创建一个模型实例
model = Net()

# 创建一个数据集
data = torch.randn(1, 784)

# 量化模型
quantized_model = quant.quantize_dynamic(model, {torch.nn.Linear}, dtype=torch.qint8)

# 使用量化模型进行前向传递
quantized_output = quantized_model(data)

# 解量化输出
dequantized_output = quantized_output.dequantize()

# 打印输出
print("Quantized output:", quantized_output)
print("Dequantized output:", dequantized_output)
