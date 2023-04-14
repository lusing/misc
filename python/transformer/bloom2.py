import torch
from transformers import AutoTokenizer, BloomForCausalLM

tokenizer = AutoTokenizer.from_pretrained("bigscience/bloom-560m")
model = BloomForCausalLM.from_pretrained("bigscience/bloom-560m",output_hidden_states=True)

inputs = tokenizer("I have a dream ", return_tensors="pt")
outputs = model(**inputs, labels=inputs["input_ids"],)
#print(outputs.logits.shape)

generated_text = tokenizer.decode(outputs.logits, skip_special_tokens=True)

print(generated_text)

# 获取最后一层的隐藏状态
#last_hidden_state = outputs.hidden_states[-1]

# 获得最大概率的token
#token_ids = torch.argmax(last_hidden_state, dim=-1)

# 将生成的token_ids转换回文本
#decoded_tokens = [tokenizer.decode(token_id) for token_id in token_ids.squeeze()]

#print(decoded_tokens)