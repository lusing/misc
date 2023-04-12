import torch
from transformers import GPT2LMHeadModel, GPT2Tokenizer

def generate_text(prompt, model_name="gpt2", max_length=2000):
    # 加载预训练模型及对应的分词器
    tokenizer = GPT2Tokenizer.from_pretrained(model_name)
    model = GPT2LMHeadModel.from_pretrained(model_name)

    # 使用分词器将文本转换为tokens
    input_tokens = tokenizer.encode(prompt, return_tensors="pt")

    # 使用模型生成文本
    output = model.generate(input_tokens, max_length=max_length, num_return_sequences=1, no_repeat_ngram_size=2)

    # 将生成的tokens转换回文本
    generated_text = tokenizer.decode(output[0], skip_special_tokens=True)

    return generated_text

if __name__ == "__main__":
    prompt = "I have a dream "
    generated_text = generate_text(prompt)
    print(generated_text)
    