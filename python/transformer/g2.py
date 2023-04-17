import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
model = AutoModelForCausalLM.from_pretrained("gpt2")

# 加载预训练模型及对应的分词器
tokenizer = AutoTokenizer.from_pretrained("gpt2", cache_dir='e:/xulun/models/')
tokenizer.pad_token_id = tokenizer.eos_token_id
model = AutoModelForCausalLM.from_pretrained("gpt2", cache_dir='e:/xulun/models/')

# 使用分词器将文本转换为tokens
input_tokens = tokenizer.encode("I have a dream ", return_tensors="pt")

# 使用模型生成文本
output = model.generate(input_tokens, max_length=250,
                        num_return_sequences=1, no_repeat_ngram_size=2)

# 将生成的tokens转换回文本
generated_text = tokenizer.decode(output[0], skip_special_tokens=True)

print(generated_text)
