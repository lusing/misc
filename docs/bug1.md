# 如何使用机器学习自动修复bug: 原理与细节

上一篇《如何使用机器学习自动修复bug: 上手指南》我们介绍了使用CodeBERT自动修复bug的操作方法。

这一篇我们进一步介绍其中的原理。

## 将源代码转成训练特征

上一节我们学习了训练的命令：
```
cd code
export pretrained_model=microsoft/codebert-base
export output_dir=./output
python run.py \
	--do_train \
	--do_eval \
	--model_type roberta \
	--model_name_or_path $pretrained_model \
	--config_name roberta-base \
	--tokenizer_name roberta-base \
	--train_filename ../data/small/train.buggy-fixed.buggy,../data/small/train.buggy-fixed.fixed \
	--dev_filename ../data/small/valid.buggy-fixed.buggy,../data/small/valid.buggy-fixed.fixed \
	--output_dir $output_dir \
	--max_source_length 256 \
	--max_target_length 256 \
	--beam_size 5 \
	--train_batch_size 16 \
	--eval_batch_size 16 \
	--learning_rate 5e-5 \
	--train_steps 100000 \
	--eval_steps 5000
```

不知道大家有没有疑惑，像这样`--train_filename ../data/small/train.buggy-fixed.buggy,../data/small/train.buggy-fixed.fixed`指定了两个文件作为训练集，模型是如何知道该怎么处理的呢？

答案在代码中：

```python
if args.do_train:
    # Prepare training data loader
    train_examples = read_examples(args.train_filename)
    train_features = convert_examples_to_features(train_examples, tokenizer,args,stage='train')
    all_source_ids = torch.tensor([f.source_ids for f in train_features], dtype=torch.long)
    all_source_mask = torch.tensor([f.source_mask for f in train_features], dtype=torch.long)
    all_target_ids = torch.tensor([f.target_ids for f in train_features], dtype=torch.long)
    all_target_mask = torch.tensor([f.target_mask for f in train_features], dtype=torch.long)    
    train_data = TensorDataset(all_source_ids,all_source_mask,all_target_ids,all_target_mask)
```

首先要从文件中将有bug和修复后的代码读出来：

```python
def read_examples(filename):
    """Read examples from filename."""
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

Example类的作用非常简单，就是索引号、源和目标三个字段的简单组合。

```python
class Example(object):
    """A single training/test example."""
    def __init__(self,
                 idx,
                 source,
                 target,
                 ):
        self.idx = idx
        self.source = source
        self.target = target
```

然后调用convert_examples_to_features函数将代码文本转换成token:

```python
def convert_examples_to_features(examples, tokenizer, args,stage=None):
    features = []
    for example_index, example in enumerate(examples):
        #source
        source_tokens = tokenizer.tokenize(example.source)[:args.max_source_length-2]
        source_tokens =[tokenizer.cls_token]+source_tokens+[tokenizer.sep_token]
```

源代码是这样的：

```
public java.lang.String METHOD_1 ( ) { return new TYPE_1 ( STRING_1 ) . format ( VAR_1 [ ( ( VAR_1 . length ) - 1 ) ] . getTime ( ) ) ; }
```

转换成token之后：

```
['<s>', 'public', '_java', '.', 'lang', '.', 'String', '_M', 'ETHOD', '_', '1', '_(', '_)', '_{', '_return', '_new', '_TYPE', '_', '1', '_(', '_STR', 'ING', '_', '1', '_)', '_.', '_format', '_(', '_V', 'AR', '_', '1', '_[', '_(', '_(', '_V', 'AR', '_', '1', '_.', '_length', '_)', '_-', '_1', '_)', '_]', '_.', '_get', 'Time', '_(', '_)', '_)', '_;', '_}', '</s>']
```

BERT模型中需要的是ID，我们通过convert_tokens_to_ids将token转id:

```python
source_ids =  tokenizer.convert_tokens_to_ids(source_tokens)
source_mask = [1] * (len(source_tokens))
padding_length = args.max_source_length - len(source_ids)
source_ids+=[tokenizer.pad_token_id]*padding_length
source_mask+=[0]*padding_length
```

转换出来的效果是这样的：

```
source_ids: 0 15110 46900 4 32373 4 34222 256 40086 1215 134 36 4839 25522 671 92 44731 1215 134 36 20857 1862 1215 134 4839 479 7390 36 468 2747 1215 134 646 36 36 468 2747 1215 134 479 5933 4839 111 112 4839 27779 479 120 14699 36 4839 4839 25606 35524 2 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1
```

然后我们要生成一个mask, 就是有值的地方是1，没值的全填0：

```
source_mask: 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
```

下面是针对修复的数据。首先，如果是测试集的话，自然就没有数据，需要模型推断出来；如果是训练集，则跟着训练的数据：

```python
#target
if stage=="test":
    target_tokens = tokenizer.tokenize("None")
else:
    target_tokens = tokenizer.tokenize(example.target)[:args.max_target_length-2]
```

后面就是和缺陷代码一样的操作：

```python
target_tokens = [tokenizer.cls_token]+target_tokens+[tokenizer.sep_token]            
target_ids = tokenizer.convert_tokens_to_ids(target_tokens)
target_mask = [1] *len(target_ids)
padding_length = args.max_target_length - len(target_ids)
target_ids+=[tokenizer.pad_token_id]*padding_length
target_mask+=[0]*padding_length   
```

数据准备好了之后，我们再用一个新类封装一下：
```python
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

InputFeatures类也是个简单的封装：
```python
class InputFeatures(object):
    """A single training/test features for a example."""
    def __init__(self,
                 example_id,
                 source_ids,
                 target_ids,
                 source_mask,
                 target_mask,

    ):
        self.example_id = example_id
        self.source_ids = source_ids
        self.target_ids = target_ids
        self.source_mask = source_mask
        self.target_mask = target_mask       
```

下面回到训练的主函数，将 ID 和 MASK 都转换成 PYTORCH 的向量：

```
all_source_ids = torch.tensor([f.source_ids for f in train_features], dtype=torch.long)
all_source_mask = torch.tensor([f.source_mask for f in train_features], dtype=torch.long)
all_target_ids = torch.tensor([f.target_ids for f in train_features], dtype=torch.long)
all_target_mask = torch.tensor([f.target_mask for f in train_features], dtype=torch.long)    
train_data = TensorDataset(all_source_ids,all_source_mask,all_target_ids,all_target_mask)
```

## Seq2Seq模型

```python
def forward(self, source_ids=None,source_mask=None,target_ids=None,target_mask=None,args=None):   
    outputs = self.encoder(source_ids, attention_mask=source_mask)
    encoder_output = outputs[0].permute([1,0,2]).contiguous()
    if target_ids is not None:  
        attn_mask=-1e4 *(1-self.bias[:target_ids.shape[1],:target_ids.shape[1]])
        tgt_embeddings = self.encoder.embeddings(target_ids).permute([1,0,2]).contiguous()
        out = self.decoder(tgt_embeddings,encoder_output,tgt_mask=attn_mask,memory_key_padding_mask=(1-source_mask).bool())
        hidden_states = torch.tanh(self.dense(out)).permute([1,0,2]).contiguous()
        lm_logits = self.lm_head(hidden_states)
        # Shift so that tokens < n predict n
        active_loss = target_mask[..., 1:].ne(0).view(-1) == 1
        shift_logits = lm_logits[..., :-1, :].contiguous()
        shift_labels = target_ids[..., 1:].contiguous()
        # Flatten the tokens
        loss_fct = nn.CrossEntropyLoss(ignore_index=-1)
        loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1))[active_loss],
                        shift_labels.view(-1)[active_loss])

        outputs = loss,loss*active_loss.sum(),active_loss.sum()
        return outputs
```

```python
else:
    #Predict
    preds=[]       
    zero=torch.cuda.LongTensor(1).fill_(0)     
    for i in range(source_ids.shape[0]):
        context=encoder_output[:,i:i+1]
        context_mask=source_mask[i:i+1,:]
        beam = Beam(self.beam_size,self.sos_id,self.eos_id)
        input_ids=beam.getCurrentState()
        context=context.repeat(1, self.beam_size,1)
        context_mask=context_mask.repeat(self.beam_size,1)
        for _ in range(self.max_length):
            if beam.done():
                break
            attn_mask=-1e4 *(1-self.bias[:input_ids.shape[1],:input_ids.shape[1]])
            tgt_embeddings = self.encoder.embeddings(input_ids).permute([1,0,2]).contiguous()
            out = self.decoder(tgt_embeddings,context,tgt_mask=attn_mask,memory_key_padding_mask=(1-context_mask).bool())
            out = torch.tanh(self.dense(out))
            hidden_states=out.permute([1,0,2]).contiguous()[:,-1,:]
            out = self.lsm(self.lm_head(hidden_states)).data
            beam.advance(out)
            input_ids.data.copy_(input_ids.data.index_select(0, beam.getCurrentOrigin()))
            input_ids=torch.cat((input_ids,beam.getCurrentState()),-1)
        hyp= beam.getHyp(beam.getFinal())
        pred=beam.buildTargetTokens(hyp)[:self.beam_size]
        pred=[torch.cat([x.view(-1) for x in p]+[zero]*(self.max_length-len(p))).view(1,-1) for p in pred]
        preds.append(torch.cat(pred,0).unsqueeze(0))

    preds=torch.cat(preds,0)                
    return preds   
```
