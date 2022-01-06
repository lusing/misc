# 如何使用机器学习自动修复bug: loss的计算

前面我们通过《如何使用机器学习自动修复bug: 上手指南》和《如何使用机器学习自动修复bug: 数据处理和模型搭建》了解了使用机器学习方法去自动修复bug的操作和原理。
了解了模型之后，有的同学还是有疑问，训练的loss是如何计算出来的呢？

从代码中可以看到，调用模型计算需要处理的返回值只有一个，就是loss:

```
for step in bar:
    batch = next(train_dataloader)
    batch = tuple(t.to(device) for t in batch)
    source_ids,source_mask,target_ids,target_mask = batch
    loss,_,_ = model(source_ids=source_ids,source_mask=source_mask,target_ids=target_ids,target_mask=target_mask)
```

进入正题之前，我们先补充一点点PyTorch向量的基本操作的知识。

## 一点点PyTorch向量操作基础知识

### 全1矩阵

通过torch.ones我们可以生成值全为1的矩阵：

```python
import torch as t

a1 = t.ones(2048, 2048)
print(a1)
```

### 对角矩阵

通过torch.tril可以生成对角矩阵。

比如我们要把上面的全1矩阵转换成对角矩阵：

```python
import torch as t

a1 = t.ones(2048, 2048)
print(a1)
a2 = t.tril(a1)
print(a2)
```

输出如下：
```
tensor([[1., 1., 1.,  ..., 1., 1., 1.],
        [1., 1., 1.,  ..., 1., 1., 1.],
        [1., 1., 1.,  ..., 1., 1., 1.],
        ...,
        [1., 1., 1.,  ..., 1., 1., 1.],
        [1., 1., 1.,  ..., 1., 1., 1.],
        [1., 1., 1.,  ..., 1., 1., 1.]])
tensor([[1., 0., 0.,  ..., 0., 0., 0.],
        [1., 1., 0.,  ..., 0., 0., 0.],
        [1., 1., 1.,  ..., 0., 0., 0.],
        ...,
        [1., 1., 1.,  ..., 1., 0., 0.],
        [1., 1., 1.,  ..., 1., 1., 0.],
        [1., 1., 1.,  ..., 1., 1., 1.]])
```

### 向量的permute

由浅入深，我们先从permute说起。

permute是重新排队矩阵的顺序。

我们先从平面矩阵的permute来理解。

我们先搞个2行3列的矩阵：

```python
x1 = t.tensor([[0,1,2],[3,4,5]])
```

然后我们将其变成3行2列的：

```python
x2 = t.permute(x1,(1,0))
print(x2)
```

输出的结果是0，1，2从行变成列，3，4，5从行变成列。于是第一行是0，3，第二行是1，4，第三行是2，5：

```
tensor([[0, 3],
        [1, 4],
        [2, 5]])
```

permute并不见得要变形状，即使形状相同，但是取的顺序不同，也同样获得不一样的结果。
我们看个最简单的三维向量的例子,默认的顺序是(0,1,2),然后我们将其变成(1,2,0):

```python
x3 = t.tensor([[[1,2],[3,4]],[[5,6],[7,8]]])
print(x3.size())
print(x3)

x4 = t.permute(x3,(1,2,0))
print(x4.size())
print(x4)
```

输出如下：
```
torch.Size([2, 2, 2])
tensor([[[1, 2],
         [3, 4]],

        [[5, 6],
         [7, 8]]])
torch.Size([2, 2, 2])
tensor([[[1, 5],
         [2, 6]],

        [[3, 7],
         [4, 8]]])
```

我们再看转成(1,0,2)的情况：
```python
x5 = t.permute(x3,(1,0,2))
print(x5)
```

输出结果如下：

```
tensor([[[1, 2],
         [5, 6]],

        [[3, 4],
         [7, 8]]])
```

如果感觉难以理解的话，我们用不同的三维坐标来再来理解一下。
刚才举二维的例子主要是想说明，即使是形状不变，但是permute的结果是不同的。

我们使用arange先生成一个一维向量，然后将其转成(2,3,4)的形状：
```
x10 = t.arange(0,24)
print(x10)
x11 = x10.view(2,3,4)
print(x11)
print(x11.size())
```

输出结果如下：
```
tensor([ 0,  1,  2,  3,  4,  5,  6,  7,  8,  9, 10, 11, 12, 13, 14, 15, 16, 17,
        18, 19, 20, 21, 22, 23])
tensor([[[ 0,  1,  2,  3],
         [ 4,  5,  6,  7],
         [ 8,  9, 10, 11]],

        [[12, 13, 14, 15],
         [16, 17, 18, 19],
         [20, 21, 22, 23]]])
torch.Size([2, 3, 4])
```

我们将其permute成(1,2,0),也就是（3,4,2）这样的顺序。我们先取1，也就是y轴，那么就是说0的下一个数1是在竖着的方向排。
当（0，1，2，3）排满之后，下一个方向是2，也就是Z轴，所以（4，5，6，7）排在（0，1，2，3）的下面一层。
（8，9，10，11）再排到（4，5，6，7）的下一层。
排满之后，才回到最上面一层的X轴方向，0的右边排12，以此类推。

```
x12 = t.permute(x11,(1,2,0))
print(x12)
print(x12.size())
```

输出结果为：
```
tensor([[[ 0, 12],
         [ 1, 13],
         [ 2, 14],
         [ 3, 15]],

        [[ 4, 16],
         [ 5, 17],
         [ 6国, 18],
         [ 7, 19]],

        [[ 8, 20],
         [ 9, 21],
         [10, 22],
         [11, 23]]])
torch.Size([3, 4, 2])
```

如果还不理解的话，我们再倒过来试试，就是从(2,3,4)变成(4,3,2)，先2，再1，最后0.
所以1是在0的下面一层，2在1的下面一层，3在2的下面一层。
4回到最上一层，排到0的Y方向。

```
x13 = t.permute(x11,(2,1,0))
print(x13)
print(x13.size())
```

输出结果为：
```
tensor([[[ 0, 12],
         [ 4, 16],
         [ 8, 20]],

        [[ 1, 13],
         [ 5, 17],
         [ 9, 21]],

        [[ 2, 14],
         [ 6, 18],
         [10, 22]],

        [[ 3, 15],
         [ 7, 19],
         [11, 23]]])
torch.Size([4, 3, 2])
```

## CodeBERT模型的返回值

之前我们都是用命令行传参数来使用huggingface Transformer库，现在我们换成手写一段：

```python
from transformers import RobertaConfig, RobertaModel, RobertaTokenizer
import torch

tokenizer = RobertaTokenizer.from_pretrained('microsoft/codebert-base')
model = RobertaModel.from_pretrained('microsoft/codebert-base')

inputs = tokenizer("console.log();", return_tensors="pt")
outputs = model(**inputs)

print(outputs[0])
print(outputs[0].shape())

last_hidden_states = outputs.last_hidden_state
print(last_hidden_states)
```

CodeBERT在transformer库中的名字是“microsoft/codebert-base”，它是Roberta的改进版，所以使用RobertaModel和RobertaTokenizer。
预训练模型的输出是一个BaseModelOutputWithPoolingAndCrossAttentions的结构，其主要数据是last_hidden_state和pooler_output。

其中，last_hidden_state是一个[6,1,768]形状的向量。因为它是BaseModelOutputWithPoolingAndCrossAttentions的第0项，所以它也是outputs[0]。

下面我们回到CodeBERT模型部分，输出的last_hidden_state经过[1,0,2]形状的permute，从[6,1,768]变成[1,6,768]。
contiguous()的目的是为这个向量分配一块连结的内存区域。

```
def forward(self, source_ids=None,source_mask=None,target_ids=None,target_mask=None,args=None):   
    outputs = self.encoder(source_ids, attention_mask=source_mask)
    encoder_output = outputs[0].permute([1,0,2]).contiguous()
```

后面用到了开头我们讲到的对角阵的知识：
```python
attn_mask=-1e4 *(1-self.bias[:input_ids.shape[1],:input_ids.shape[1]])
```

input_ids.shape[1]是最大长度，在命令行指定的，我们之前给的是512.

乘以-1e4之后的值为：
```
tensor([[    -0., -10000., -10000.,  ..., -10000., -10000., -10000.],
        [    -0.,     -0., -10000.,  ..., -10000., -10000., -10000.],
        [    -0.,     -0.,     -0.,  ..., -10000., -10000., -10000.],
        ...,
        [    -0.,     -0.,     -0.,  ...,     -0., -10000., -10000.],
        [    -0.,     -0.,     -0.,  ...,     -0.,     -0., -10000.],
        [    -0.,     -0.,     -0.,  ...,     -0.,     -0.,     -0.]],
```

## 解码器处理部分

对源数据，在我们的情况下是有问题的代码，进行处理之后，我们还要对目标数据，也就是代码的修复，进行处理。最后同样permute一下：

```python
tgt_embeddings = self.encoder.embeddings(target_ids).permute([1,0,2]).contiguous()
```
