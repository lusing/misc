import torch
from transformers import OpenAIGPTLMHeadModel, OpenAIGPTTokenizer

def generate_text(prompt, model_name="openai-gpt", max_length=50):
    # 加载预训练模型及对应的分词器
    tokenizer = OpenAIGPTTokenizer.from_pretrained(model_name)
    model = OpenAIGPTLMHeadModel.from_pretrained(model_name)

    # 使用分词器将文本转换为 tokens
    input_tokens = tokenizer.encode(prompt, return_tensors="pt")

    # 使用模型生成文本
    output = model.generate(input_tokens, max_length=max_length, num_return_sequences=1, no_repeat_ngram_size=2)

    # 将生成的 tokens 转换回文本
    generated_text = tokenizer.decode(output[0], skip_special_tokens=True)

    return generated_text

if __name__ == "__main__":
    prompt = "I have a dream "
    generated_text = generate_text(prompt)
    print(generated_text)
