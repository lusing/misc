# 导入必要的库
import numpy as np
import torch
import torch.nn as nn

# 定义自注意力模型类
class SelfAttention(nn.Module):
    def __init__(self, input_dim, output_dim, num_heads):
        super(SelfAttention, self).__init__()
        # 参数检查
        assert output_dim % num_heads == 0, "output_dim must be divisible by num_heads"
        # 定义线性变换层
        self.W_q = nn.Linear(input_dim, output_dim)
        self.W_k = nn.Linear(input_dim, output_dim)
        self.W_v = nn.Linear(input_dim, output_dim)
        # 定义输出层
        self.W_o = nn.Linear(output_dim, output_dim)
        # 定义头数和头部维度
        self.num_heads = num_heads
        self.head_dim = output_dim // num_heads

    def forward(self, x):
        # x: (batch_size, seq_len, input_dim)
        # 计算Q,K,V
        Q = self.W_q(x) # (batch_size, seq_len, output_dim)
        K = self.W_k(x) # (batch_size, seq_len, output_dim)
        V = self.W_v(x) # (batch_size, seq_len, output_dim)
        # 将Q,K,V分割成多个头部
        Q = Q.view(Q.shape[0], Q.shape[1], self.num_heads, self.head_dim) # (batch_size, seq_len, num_heads, head_dim)
        K = K.view(K.shape[0], K.shape[1], self.num_heads, self.head_dim) # (batch_size, seq_len, num_heads, head_dim)
        V = V.view(V.shape[0], V.shape[1], self.num_heads, self.head_dim) # (batch_size, seq_len, num_heads, head_dim)
        # 调整维度顺序，便于计算注意力权重
        Q = Q.permute(0, 2, 1, 3) # (batch_size, num_heads, seq_len, head_dim)
        K = K.permute(0, 2, 3, 1) # (batch_size, num_heads, head_dim, seq_len)
        V = V.permute(0, 2, 1, 3) # (batch_size, num_heads, seq_len, head_dim)
        # 计算注意力权重，使用缩放点积注意力
        A = torch.matmul(Q, K) / np.sqrt(self.head_dim) # (batch_size, num_heads, seq_len_q ,seq_len_k)
        A = torch.softmax(A ,dim=-1) # (batch_size,num_heads ,seq_len_q ,seq_len_k)
        # 计算注意力输出
        O = torch.matmul(A ,V) # (batch_size,num_heads ,seq_len_q ,head_dim)
        # 调整维度顺序，便于拼接头部
        O = O.permute(0 ,2 ,1 ,3) # (batch_size ,seq_len_q ,num_heads ,head_dim)
        # 拼接头部，得到最终输出
        O = O.reshape(O.shape[0] ,O.shape[1] ,-1) # (batch_size ,seq_len_q ,output_dim)
        O = self.W_o(O) # (batch_size ,seq_len_q ,output_dim)

        return O

# 测试代码
input_dim = 8
output_dim = 16
num_heads = 8
seq_len = 3
batch_size = 2

x = torch.randn(batch_size ,seq_len ,input_dim)

model = SelfAttention(input_dim ,output_dim ,num_heads)

y = model(x)

print(y.shape) # torch.Size([2 ,3 ,8])
