# 2023年的深度学习入门指南(23) - ChatGLM2

在《在你的电脑上运行大模型》这一节，我们曾经介绍过ChatGLM模型，它是当时最好的中文大模型之一。现在，它又更新到了第二代，即ChatGLM2。

当时，我们的技术储备还不足，只能让它运行起来，还不敢讲解它的原理和代码。

现在，经过LLaMA 2和百川的代码的狂轰滥炸，大家已经适应了看代码的节奏了。现在，是时候来看看ChatGLM2的原理和代码了。

## 运行ChatGLM2

首先我们还是将ChatGLM2的代码运行起来。在大于13GB显存的机器上，ChatGLM2都可以顺利运行起来。比如我是在一个15G的T4上运行的。

第一步还是将安装相关的库：

```bash
pip install protobuf
pip install transformers==4.30.2
pip install cpm_kernels
pip install torch>=2.0
pip install gradio
pip install mdtex2html
pip install sentencepiece
pip install accelerate
pip install sse-starlette
pip install streamlit>=1.24.0
```

第二步就可以用Transformers的标准接口来调用ChatGLM2了：

```python
from transformers import AutoTokenizer, AutoModel
tokenizer = AutoTokenizer.from_pretrained("THUDM/chatglm2-6b", trust_remote_code=True)
model = AutoModel.from_pretrained("THUDM/chatglm2-6b", trust_remote_code=True, device='cuda')
model = model.eval()
response, history = model.chat(tokenizer, "生成scala语言的快速排序", history=[])
print(response)
```

输出如下：
```scala
def quickSort(arr: Int[]): Int[] = {
  val pivot = arr(arr.length / 2)
  val left = 0
  val right = arr.length - 1
  while (left <= right) {
    while (arr(left) < pivot) {
      left = left + 1
    }
    arr(left) = pivot
    while (arr(right) > pivot) {
      right = right - 1
    }
    arr(right) = pivot
    left = left + 1
    right = right - 1
  }
  return arr
}
```

如果在更小显存的显卡上运行，我们可以使用4位量化后的结果：

```python
from transformers import AutoTokenizer, AutoModel
tokenizer = AutoTokenizer.from_pretrained("THUDM/chatglm2-6b", trust_remote_code=True)
model = AutoModel.from_pretrained("THUDM/chatglm2-6b-int4",trust_remote_code=True).cuda()
model = model.eval()
response, history = model.chat(tokenizer, "生成Kotlin语言编写的快速排序", history=[])
print(response)

```

这是我在3060上运行的结果：

```kotlin                                         
fun quickSort(arr: IntArray): IntArray {    
    val left = 0                            
    val right = arr.size - 1                
    val quicksortFactor = arr.size / 2      
                                            
    while (left < right) {                  
        quicksortFactor--.let {             
            let x = left                    
            let y = right                   
            let temp = arr[x]               
                                            
            if (temp < arr[y]) {            
                x++                         
            } else {                        
                y--                         
            }                               
                                            
            if (x == y) {                   
                break                       
            }                               
                                            
            quicksortFactor++.let {         
                arr[x] = arr[y]             
                    arr[y] = temp           
            }                               
        }                                   
    }                                       
                                            
    return arr                              
}                                           
```

## 量化的CUDA代码解析

之前讲过不少多头注意力的代码实现了，后面也还会讲。在本节中我们讲一个之前没有讲到的内容，量化所用的CUDA代码。LLaMA 2部分没讲是因为它还没有量化部分，而百川是CUDA核代码暂时还没开源。所以我们就先借着GLM的代码来讲一下。

我们先看一下CUDA核部分的Makefile:
```makefile
NVCC=nvcc
OPTIONS=-gencode arch=compute_61,code=sm_61 \
		-gencode arch=compute_62,code=sm_62 \
		-gencode arch=compute_70,code=sm_70 \
		-gencode arch=compute_72,code=sm_72 \
		-gencode arch=compute_75,code=sm_75 \
		-gencode arch=compute_80,code=sm_80 \
		-gencode arch=compute_86,code=sm_86

TARGETS=$(patsubst %.cu, %.fatbin, $(wildcard *.cu))

all: $(TARGETS)

%.fatbin: %.cu
	$(NVCC) -fatbin $^ $(OPTIONS) -o $@

.PHONY : clean, copy
clean:
	rm $(TARGETS)

copy:
	cp $(TARGETS) ../kernels/
```

我们可以看到，这里的代码是支持多个CUDA架构的，包括了6.1、6.2、7.0、7.2、7.5、8.0、8.6。这里的架构是指GPU的架构，比如RTX 3090的架构是8.6，RTX 3060的架构是8.0。
- 6.1和6.2对应的是Pascal架构，比如P100, GTX 1060
- 7.0是Volta架构，比如V100
- 7.5是Turing架构，比如RTX 2080, T4
- 8.0和8.6是Ampere架构，比如A100, RTX 3090

![](https://xulun-mooc.oss-cn-beijing.aliyuncs.com/nvidia.png)

别看已经支持这么多架构了，但是更早的Maxwell和Kepler等更老的架构已经随风而去了。

要支持这么多架构，就需要引入一个新的知识点 - fatbin. 

.fatbin 文件是 CUDA 二进制格式（CUDA Fat Binary Format）的文件。这是 NVIDIA 的 CUDA 平台使用的一种特殊的二进制文件格式。fatbin文件包含了针对多种 GPU 架构和计算能力的代码，可以在多种不同类型的处理器上运行。

在 CUDA 编程中，GPU 代码（通常称为 kernel）经常以类似于内联汇编的方式存储在主机代码中。然而，这种方法在实际应用中存在一些困难，主要是因为不同的 GPU 架构和设备可能需要不同的 GPU 代码版本。CUDA Fat Binary 解决了这个问题，它包含了多个版本的 GPU 代码，每个版本都针对一个特定的 GPU 架构进行优化。

当 CUDA 程序运行时，CUDA 运行时系统会检查正在运行的设备，并从 fat binary 文件中选择最适合该设备的 GPU 代码版本。这样就可以用一个.fatbin文件,使同一个CUDA程序可以在不同算力的GPU上运行。不需要为不同GPU单独编译。

下面我们就来看8位量化的实现，这个代码完全就是一个如何写最简单CUDA代码的例子：

```c
template<typename T>
__device__ void
int8WeightExtractionDevice(const int8_t* weight,
                                const T* scale_list,
                                T* output,
                                const int n,
                                const int k)
{
    for(int i = blockIdx.x * k + threadIdx.x; i < blockIdx.x * k + k; i += blockDim.x){
        output[i] = T(weight[i]) * scale_list[blockIdx.x];
    }
}
```

在GPU线程中：
- 计算当前线程读取的weight索引:blockIdx.x代表块id,_k代表每个块处理k个值,threadIdx.x代表线程id
- 读取weight数组当前索引处的int8值
- 对其缩放:用scale_list中对应块id的缩放因子乘以weight值
- 结果输出到output数组对应索引处

最后，通过blockDim.x线程并行完成weight数组到output数组的整体复制和缩放计算。

如果大家忘了CUDA那一节的内容，我们来复习一下blockIdx，threadIdx和blockDim的概念：
- blockIdx: CUDA将线程组织成块(block),每个块有一个id,称为blockIdx。可以有多个块,通过blockIdx区分不同块
- threadIdx: 每个块里面有多个线程,通过threadIdx区分同一块中的不同线程。线程id从0开始计数
- blockDim: 指明每个块中含有的线程数目

在启动核函数时指定,例如调用核函数时执行<<<32, 128>>>表示有32个块,每个块中有128个线程。

在函数体内部，有一个for循环，请注意，这个循环不是像在单核CPU上那样串行的，而是在CUDA的每个线程上执行的！
循环变量i的初始值是 blockIdx.x * k + threadIdx.x，这是一个常用的模式，用于将数据的不同部分分配给不同的CUDA线程。每轮循环中，i增加 blockDim.x，这表示每个线程处理的数据间隔是一个block的大小。

在for循环中，函数将权重乘以对应的缩放因子，并将结果存储在输出数组中。这里，权重的类型被转换为T，然后乘以对应的scale_list元素。注意scale_list[blockIdx.x]的使用，这表示对于同一个block内的所有线程，它们使用的是同一个缩放因子。

CUDA的核被封装在host的函数里：
```c
extern "C" __global__ void int8WeightExtractionHalf(const int8_t* weight,
                                const half* scale_list,
                                half* output,
                                const int n,
                                const int k){
                                    int8WeightExtractionDevice<half>(weight, scale_list, output, n, k);
                                }

extern "C" __global__ void int8WeightExtractionFloat(const int8_t* weight,
                                const float* scale_list,
                                float* output,
                                const int n,
                                const int k){
                                    int8WeightExtractionDevice<float>(weight, scale_list, output, n, k);
                                }
```

我们来看一下在Python中如何调用这个函数的：

```python
def extract_weight_to_half(weight: torch.Tensor, scale_list: torch.Tensor, source_bit_width: int):
    if source_bit_width == 8:
        func = kernels.int8WeightExtractionHalf
    elif source_bit_width == 4:
        func = kernels.int4WeightExtractionHalf
    else:
        assert False, "Unsupported bit-width"

    with torch.cuda.device(weight.device):
        n, m = weight.size(0), weight.size(1)
        out = torch.empty(n, m * (8 // source_bit_width), dtype=torch.half, device="cuda")
        stream = torch.cuda.current_stream()

        gridDim = (n, 1, 1)
        blockDim = (min(round_up(m, 32), 1024), 1, 1)

        func(
            gridDim,
            blockDim,
            0,
            stream,
            [
                ctypes.c_void_p(weight.data_ptr()),
                ctypes.c_void_p(scale_list.data_ptr()),
                ctypes.c_void_p(out.data_ptr()),
                ctypes.c_int32(n),
                ctypes.c_int32(m),
            ],
        )
        return out
```

我们可以看到，如果是针对8位量化的时候，就会调用 kernels.int8WeightExtractionHalf 函数，这个函数对应的就是我们上面写的那个函数。

下面我们讲解一下是如何划分并行度的。
gridDim这个变量表示 CUDA 内核函数的网格大小。在这段代码中，它被设置为 (n, 1, 1)，其中 n 是权重张量的第一维大小。这意味着网格中有 n 个块，每个块负责处理权重张量的一行。

blockDim是一个三元组，用于指定每个线程块的维度。在这个代码中，blockDim 被设置为 (min(round_up(m, 32), 1024), 1, 1)。这表示每个线程块中的线程数量为 min(round_up(m, 32), 1024)，这个数量是 m（weight 张量的第二维的大小）向上取到最近的32的倍数，但最大不超过1024。这是因为CUDA架构的限制，每个线程块的线程数量不能超过1024。

下面我们再讲一下4位压缩的：
```c
__device__ void
int4WeightCompressionDevice(const int8_t* input,
                                int8_t* output,
                                const int n,
                                const int k)
{
    for(int i = blockIdx.x * k + threadIdx.x; i < blockIdx.x * k + k; i += blockDim.x){
        output[i] = (input[i * 2] << 4) | (input[i * 2 + 1] & 0b00001111);
    }
}
```

int4WeightCompressionDevice里面，对于每个线程，它会计算出它应该处理的元素的索引 i。然后，它会将输入数组中索引为 i * 2 和 i * 2 + 1 的两个元素压缩成一个元素。压缩方法是将第一个元素左移 4 位，然后与第二个元素进行按位或运算。最后，将结果存储在输出数组中索引为 i 的位置。

虽然可以高度并行，但是其实GPU上的代码写起来跟CPU上也并没有太大的不同，不需要学新的语句。

同理，我们看下将4位压缩的权重转换为8位的：

```c
template<typename T>
__device__ void
int4WeightExtractionDevice(const int8_t* weight,
                                const T* scale_list,
                                T* output,
                                const int n,
                                const int k)
{
    for(int i = blockIdx.x * k + threadIdx.x; i < blockIdx.x * k + k; i += blockDim.x){
        int8_t original = weight[i];
        int8_t high = original >> 4;
        int8_t low = original << 4; low = low >> 4;
        output[i * 2] = T(high) * scale_list[blockIdx.x];
        output[i * 2 + 1] = T(low) * scale_list[blockIdx.x];
    }
}
```

有了上面的知识，这里不需要额外讲解了吧？

## 量化层的实现

最后，我们来看看量化如何在神经网络中使用：

```python
解释下面的代码：
import torch

from kernels import extract_weight_to_half


class W8A16Linear(torch.autograd.Function):
    @staticmethod
    def forward(ctx, inp: torch.Tensor, quant_w: torch.Tensor, scale_w: torch.Tensor, weight_bit_width):
        ctx.inp_shape = inp.size()
        ctx.weight_shape = quant_w.size()
        ctx.weight_bit_width = weight_bit_width
        out_features = quant_w.size(0)
        inp = inp.contiguous().view(-1, inp.size(-1))
        weight = extract_weight_to_half(quant_w, scale_w, weight_bit_width)
        output = inp.mm(weight.t())
        ctx.save_for_backward(inp, quant_w, scale_w)
        return output.view(*(ctx.inp_shape[:-1] + (out_features,)))

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        inp, quant_w, scale_w = ctx.saved_tensors
        weight = extract_weight_to_half(quant_w, scale_w, ctx.weight_bit_width)
        grad_output = grad_output.contiguous().view(-1, weight.size(0))
        grad_input = grad_output.mm(weight)
        grad_weight = grad_output.t().mm(inp)
        return grad_input.view(ctx.inp_shape), grad_weight.view(ctx.weight_shape), None
```

forward 方法接受四个参数：inp 是一个输入张量，quant_w 是一个量化后的权重张量，scale_w 是一个权重缩放张量，weight_bit_width 是权重的位宽。这个方法首先保存输入张量和权重张量的形状，然后将输入张量转换为连续的并调整形状。接着，它使用我们刚才讲过的 extract_weight_to_half 函数从量化后的权重和权重缩放中提取出半精度的权重。然后，它使用矩阵乘法计算输出，并将结果调整为正确的形状。最后，它将输入、量化后的权重和权重缩放保存起来，以便在反向传播时使用。

backward 方法接受一个参数：grad_output 是一个梯度输出张量。这个方法首先从上下文中获取保存的输入、量化后的权重和权重缩放，并从中提取出半精度的权重。然后，它将梯度输出转换为连续的并调整形状。接着，它使用矩阵乘法计算输入梯度和权重梯度。最后，它返回调整形状后的输入梯度和权重梯度。

这段代码实现了一个自定义的线性层，它使用半精度的权重进行计算，并支持 PyTorch 的自动求导机制。

这还没有完，为了实现更大规模并行化，量化层还可以进一步的封装：

```python
import torch
from torch.nn.parameter import Parameter

from SwissArmyTransformer.mpu import copy_to_model_parallel_region
from SwissArmyTransformer.mpu import gather_from_model_parallel_region
from SwissArmyTransformer.mpu import reduce_from_model_parallel_region
from SwissArmyTransformer.mpu import scatter_to_model_parallel_region
from SwissArmyTransformer.mpu import ColumnParallelLinear, RowParallelLinear

from .functional import W8A16Linear
from kernels import compress_int4_weight


class QuantizedColumnParallelLinear(ColumnParallelLinear):
    def __init__(self, weight_bit_width: int, weight=None, *args, **kwargs):
        super(QuantizedColumnParallelLinear, self).__init__(*args, **kwargs)
        self.weight_bit_width = weight_bit_width

        shape = self.weight.shape
        del self.weight

        if weight is None:
            self.weight = torch.empty(
                shape[0], shape[1] * weight_bit_width // 8, dtype=torch.int8, device=kwargs["device"]
            )
            self.weight_scale = torch.empty(shape[0], dtype=kwargs["params_dtype"], device=kwargs["device"])
        else:
            self.weight_scale = (weight.abs().max(dim=-1).values / ((2 ** (weight_bit_width - 1)) - 1)).half()
            self.weight = torch.round(weight / self.weight_scale[:, None]).to(torch.int8)
            if weight_bit_width == 4:
                self.weight = compress_int4_weight(self.weight)

        self.weight = Parameter(self.weight.to(kwargs["device"]), requires_grad=False)
        self.weight_scale = Parameter(self.weight_scale.to(kwargs["device"]), requires_grad=False)

    def forward(self, input_):
        # Set up backprop all-reduce.
        input_parallel = copy_to_model_parallel_region(input_)
        # Matrix multiply.
        output_parallel = W8A16Linear.apply(input_parallel, self.weight, self.weight_scale, self.weight_bit_width)
        if self.bias is not None:
            output_parallel = output_parallel + self.bias
        if self.gather_output:
            # All-gather across the partitions.
            output = gather_from_model_parallel_region(output_parallel)
        else:
            output = output_parallel
        return output
```

这段代码定义了一个名为 QuantizedColumnParallelLinear 的类，它继承自 ColumnParallelLinear 类。这个类实现了一个量化的列并行线性层。

`__init__` 方法接受若干个参数，其中 weight_bit_width 是权重的位宽，weight 是一个可选的权重张量。这个方法首先调用父类的构造函数，然后保存权重的位宽。接着，它获取权重的形状并删除权重属性。如果没有提供权重，则创建一个空的权重张量和一个空的权重缩放张量。否则，根据提供的权重计算权重缩放，并将权重量化为整数。如果位宽为 4，则使用 compress_int4_weight 函数对权重进行压缩。最后，将权重和权重缩放转换为 PyTorch 参数并保存。

forward 方法接受一个参数：input_ 是一个输入张量。这个方法首先使用 copy_to_model_parallel_region 函数将输入复制到模型并行区域。然后，使用 W8A16Linear.apply 函数计算输出。如果有偏置，则将偏置加到输出上。如果需要收集输出，则使用 gather_from_model_parallel_region 函数收集输出。否则，直接返回输出。

这段代码实现了一个量化的列并行线性层，它可以在多个 GPU 上并行计算。

## 小结

本节我们借着讲ChatGLM2功能的机会，顺便把从CUDA一直到多GPU并行时要用到的量化方法完整地介绍了一遍。
如果你有哪些功能可以用CUDA设备代码进行加速的，那就毫不犹豫地去实现它吧！算法的加速并不是局限在如何使用别人的框架和现有的功能上的。
