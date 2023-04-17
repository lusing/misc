from transformers import T5Tokenizer, T5ForConditionalGeneration

tokenizer = T5Tokenizer.from_pretrained("google/t5-v1_1-large")
model = T5ForConditionalGeneration.from_pretrained("google/t5-v1_1-large",max_length=250)

str1 = """
Summarize:
We have explored chain-of-thought prompting as a simple and broadly applicable method for enhancing
reasoning in language models. Through experiments on arithmetic, symbolic, and commonsense
reasoning, we find that chain-of-thought reasoning is an emergent property of model scale that allows
sufficiently large language models to perform reasoning tasks that otherwise have flat scaling curves.
Broadening the range of reasoning tasks that language models can perform will hopefully inspire
further work on language-based approaches to reasoning.
"""

input_ids = tokenizer(str1, return_tensors="pt").input_ids
outputs = model.generate(input_ids,max_length=100,num_beams=5)
print(tokenizer.decode(outputs[0], skip_special_tokens=True))
