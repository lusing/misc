from transformers import RobertaConfig, RobertaModel, RobertaTokenizer
import torch

tokenizer = RobertaTokenizer.from_pretrained('microsoft/codebert-base')
model = RobertaModel.from_pretrained('microsoft/codebert-base')

inputs = tokenizer("console.log();", return_tensors="pt")
outputs = model(**inputs)

print(outputs)
print(outputs[0].shape)

encoder_output = outputs[0].permute([1,0,2]).contiguous()
print(encoder_output)

print(encoder_output.shape)

last_hidden_states = outputs.last_hidden_state
print(last_hidden_states)

print(outputs.pooler_output.shape)
