import torch
import torchvision
from torch.quantization import QuantStub, DeQuantStub
from torchvision.models import resnet18

# 定义一个简单的量化模型
class QuantizedModel(torch.nn.Module):
    def __init__(self, model):
        super(QuantizedModel, self).__init__()
        self.quant = QuantStub()
        self.model = model
        self.dequant = DeQuantStub()

    def forward(self, x):
        x = self.quant(x)
        x = self.model(x)
        x = self.dequant(x)
        return x

# 加载预训练模型
model = resnet18(pretrained=True)

# 设置模型为评估模式
model.eval()

# 定义一个输入张量
input_tensor = torch.rand(1, 3, 224, 224)

# 定义量化配置
quantization_config = torch.quantization.get_default_qconfig("fbgemm")

# 准备量化模型
quantized_model = QuantizedModel(model)
quantized_model.qconfig = quantization_config

# 准备模型和输入数据
torch.quantization.prepare(quantized_model, inplace=True)
quantized_model(input_tensor)
torch.quantization.convert(quantized_model, inplace=True)

# 运行量化模型
quantized_output = quantized_model(input_tensor)

# 打印量化输出
print("Quantized output:")
print(quantized_output)

# 解量化输出
dequantized_output = quantized_model.dequant(quantized_output)

# 打印解量化输出
print("Dequantized output:")
print(dequantized_output)
