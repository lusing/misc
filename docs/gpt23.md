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

