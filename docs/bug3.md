# 如何使用机器学习进行代码翻译

前面我们通过《如何使用机器学习自动修复bug: 上手指南》和《如何使用机器学习自动修复bug: 数据处理和模型搭建》了解了使用机器学习方法去自动修复bug的操作和原理。

学会了这个原理之后，我们进行举一反三，将其应用于其它方向。比如两种编程语言的翻译。

翻译的问题我们已经在《代码智能：问题与解法》一文中介绍过了。是实现java语言到C#语言的翻译。

命令是这样的：

```
python run.py \
	--do_train \
	--do_eval \
	--model_type roberta \
	--model_name_or_path microsoft/codebert-base \
	--config_name roberta-base \
	--tokenizer_name roberta-base \
	--train_filename ../data/train.java-cs.txt.java,../data/train.java-cs.txt.cs \
	--dev_filename ../data/valid.java-cs.txt.java,../data/valid.java-cs.txt.cs \
	--output_dir ./out \
	--max_source_length 512 \
	--max_target_length 512 \
	--beam_size 5 \
	--train_batch_size 16 \
	--eval_batch_size 16 \
	--learning_rate 5e-5 \
	--train_steps 100000 \
	--eval_steps 5000
```

## 代码翻译的数据处理

代码翻译的主要过程跟之前的《如何使用机器学习自动修复bug: 数据处理和模型搭建》介绍的基本一致。我们不妨再过一遍。

读取两个文件的仍然是read_examples函数：

```python
def read_examples(filename):
    examples=[]
    assert len(filename.split(','))==2
    src_filename = filename.split(',')[0]
    trg_filename = filename.split(',')[1]
    idx = 0
    with open(src_filename) as f1,open(trg_filename) as f2:
            for line1,line2 in zip(f1,f2):
                examples.append(
                Example(
                        idx = idx,
                        source=line1.strip(),
                        target=line2.strip(),
                        )
                )
                idx+=1
    return examples
```

读取文件之后会存到Example结构之中：

```python
class Example(object):
    def __init__(self,
                 idx,
                 source,
                 target,
                 ):
        self.idx = idx
        self.source = source
        self.target = target
```

接着将源代码进行tokenize处理：

```python
def convert_examples_to_features(examples, tokenizer, args,stage=None):
    features = []
    for example_index, example in enumerate(examples):
        #source
        source_tokens = tokenizer.tokenize(example.source)[:args.max_source_length-2]
        source_tokens =[tokenizer.cls_token]+source_tokens+[tokenizer.sep_token]
        source_ids =  tokenizer.convert_tokens_to_ids(source_tokens)
        source_mask = [1] * (len(source_tokens))
        padding_length = args.max_source_length - len(source_ids)
        source_ids+=[tokenizer.pad_token_id]*padding_length
        source_mask+=[0]*padding_length

        #target
        if stage=="test":
            target_tokens = tokenizer.tokenize("None")
        else:
            target_tokens = tokenizer.tokenize(example.target)[:args.max_target_length-2]
        target_tokens = [tokenizer.cls_token]+target_tokens+[tokenizer.sep_token]            
        target_ids = tokenizer.convert_tokens_to_ids(target_tokens)
        target_mask = [1] *len(target_ids)
        padding_length = args.max_target_length - len(target_ids)
        target_ids+=[tokenizer.pad_token_id]*padding_length
        target_mask+=[0]*padding_length  
...
        features.append(
            InputFeatures(
                 example_index,
                 source_ids,
                 target_ids,
                 source_mask,
                 target_mask,
            )
        )
    return features
```
