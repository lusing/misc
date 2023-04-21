import torch
# x = torch.rand(2, 3, dtype=torch.float32)

x = torch.tensor([[0.8779, 0.2919, 0.6965],
        [0.8018, 0.2809, 0.0910]])
print(x)

observer = torch.quantization.MinMaxObserver(quant_min=0,quant_max=15)
observer(x)

scale, zero_point = observer.calculate_qparams()
print(scale, zero_point)

xq = torch.quantize_per_tensor(x, scale=scale, zero_point=zero_point, dtype=torch.quint8)
print(xq)

# 看整数的表示：
print(xq.int_repr())

# 解量化
xd = xq.dequantize()
print(xd)
