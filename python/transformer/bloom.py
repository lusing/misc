import torch
from transformers import AutoTokenizer, BloomForCausalLM

tokenizer = AutoTokenizer.from_pretrained("bigscience/bloom-560m")
model = BloomForCausalLM.from_pretrained("bigscience/bloom-560m")

input_tokens = tokenizer("I have a dream ", return_tensors="pt")
#outputs = model(**inputs, labels=inputs["input_ids"])
#loss = outputs.loss
#logits = outputs.logits

# 使用模型生成文本
output = model.generate(input_tokens, max_length=100)

# 将生成的 tokens 转换回文本
generated_text = tokenizer.decode(output[0], skip_special_tokens=True)

print(generated_text)