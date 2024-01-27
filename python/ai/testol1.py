import torch
from transformers import LlamaTokenizer, LlamaForCausalLM

## v2 models
model_path = 'openlm-research/open_llama_3b_v2'
# model_path = 'openlm-research/open_llama_7b_v2'

## v1 models
# model_path = 'openlm-research/open_llama_3b'
# model_path = 'openlm-research/open_llama_7b'
# model_path = 'openlm-research/open_llama_13b'

tokenizer = LlamaTokenizer.from_pretrained(model_path)
model = LlamaForCausalLM.from_pretrained(
    model_path, torch_dtype=torch.float16, device_map='auto',
)

prompt = '上下五千年，英雄万万千\n'
input_ids = tokenizer(prompt, return_tensors="pt").input_ids.to('cuda')

generation_output = model.generate(
    input_ids=input_ids, max_new_tokens=256
)
print(tokenizer.decode(generation_output[0]))
