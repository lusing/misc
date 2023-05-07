# 2023年的深度学习入门指南(12) - 参数高效微调PEFT

大家都知道，大模型的训练需要海量的算力。其实，即使是只对大模型做微调训练，也是需要大量的计算资源的。

有没有用更少的计算资源来进行微调的方法呢？研究者研发出了几种被Hugging Face统称为参数高效微调PEFT(Parameter-Efficient Fine-Tuning)的技术。

这其中常用的几个大家应该已经耳熟能详了，比如广泛应用的LoRA技术(Low Rank Adapters,低秩适配)，Prefix Tuning技术，Prompt Tuning技术等等。

我们先学习如何使用，然后我们再学习其背后的原理。

## 用Huggingface PEFT库进行适配

首先我们先安装相关的库，主要有量化用的bitsandbytes库，低秩适配器loralib库，以及加速库accelerate。
另外，PEFT库和transformers库都用最新的版本。

```
pip install -q bitsandbytes datasets accelerate loralib
pip install -q git+https://github.com/huggingface/transformers.git@main git+https://github.com/huggingface/peft.git
```

我们来尝试训练一个7B左左的模型，我们选用opt-6.7b模型，它以float16的精度存储，大小大约为13GB！如果我们使用bitsandbytes库以8位加载它们，我们需要大约7GB的显存。

但是，这只是加载用的，在实际训练的时候，16G显存都照样不够用。最终的消耗大约在20G左右。

加载大模型仍然使用我们前面学过的AutoModelForCausalLM.from_pretrained()函数，只是我们需要加上load_in_8bit=True参数来调用bitsandbytes库进行8位量化。

```python
import os

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import torch
import torch.nn as nn
import bitsandbytes as bnb
from transformers import AutoTokenizer, AutoConfig, AutoModelForCausalLM

model = AutoModelForCausalLM.from_pretrained(
    "facebook/opt-6.7b",
    load_in_8bit=True,
    device_map="auto",
)

tokenizer = AutoTokenizer.from_pretrained("facebook/opt-6.7b")
```

下面PEFT就正式出场了，我们先针对所有非int8的模块进行预处理以提升精度：

```python
from peft import prepare_model_for_int8_training

model = prepare_model_for_int8_training(model)
```

我们再配置下LoRA的参数，参数的具体含义我们后面结合原理再讲。

```python
from peft import LoraConfig, get_peft_model

config = LoraConfig(
    r=16, lora_alpha=32, target_modules=["q_proj", "v_proj"], lora_dropout=0.05, bias="none", task_type="CAUSAL_LM"
)

model = get_peft_model(model, config)
```

我们选用名人名言数据集作为训练数据：

```python
from datasets import load_dataset

data = load_dataset("Abirate/english_quotes")
data = data.map(lambda samples: tokenizer(samples["quote"]), batched=True)
```

然后就可以开始训练了：

```python
trainer = transformers.Trainer(
    model=model,
    train_dataset=data["train"],
    args=transformers.TrainingArguments(
        per_device_train_batch_size=4,
        gradient_accumulation_steps=4,
        warmup_steps=100,
        max_steps=200,
        learning_rate=2e-4,
        fp16=True,
        logging_steps=1,
        output_dir="outputs",
    ),
    data_collator=transformers.DataCollatorForLanguageModeling(tokenizer, mlm=False),
)
trainer.train()
```

最后，我们做一个推理测试下效果：
    
```python
batch = tokenizer("Two things are infinite: ", return_tensors="pt")

with torch.cuda.amp.autocast():
    output_tokens = model.generate(**batch, max_new_tokens=50)

print("\n\n", tokenizer.decode(output_tokens[0], skip_special_tokens=True))
```

输出的结果如下：
```
 Two things are infinite:  the universe and human stupidity; and I'm not sure about the universe.  -Albert Einstein
I'm not sure about the universe either.
```

基本上，我们除了配置了一个LoRA参数之外什么也没干。

## LoRA的原理

![](https://xulun-mooc.oss-cn-beijing.aliyuncs.com/LowRank.png)

LoRA的思想是将原始的权重矩阵分解为两个低秩矩阵的乘积，这样就可以大大减少参数量。其本质思想还是将复杂的问题拆解为简单的问题的组合。
LoRA通过注入优化后的秩分解矩阵，将预训练模型参数冻结，减少了下游任务的可训练参数数量，使得训练更加高效。并且在使用适应性优化器时，降低了硬件进入门槛。
因为我们不需要计算大多数参数的梯度或维护优化器状态，而是仅优化注入的、远小于原参数量的秩分解矩阵。

光定量地这么讲，大家没有观感，我们以上面训练的例子来看看LoRA的效果。

我们写一个函数来计算模型中的可训练参数数量：

```python
def print_trainable_parameters(model):
    """
    Prints the number of trainable parameters in the model.
    """
    trainable_params = 0
    all_param = 0
    for _, param in model.named_parameters():
        all_param += param.numel()
        if param.requires_grad:
            trainable_params += param.numel()
    print(
        f"trainable params: {trainable_params} || all params: {all_param} || trainable%: {100 * trainable_params / all_param}"
    )
```

运行一下：
```python
print_trainable_parameters(model)
```

输出结果如下：
```
trainable params: 8388608 || all params: 6666862592 || trainable%: 0.12582542214183376
```

我们看到，原始的模型参数有66亿多个，但是我们只训练了838多万个，只占了0.125%。

所以这也就是为什么我们经常看到有6b,7b，还有13b参数的大模型了。因为这个量级的模型，刚好可以在一张40G或者80G的A100显卡上训练。甚至在24G的3090上也能训练。

