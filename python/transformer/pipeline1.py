from transformers import pipeline

def generate_text(prompt, model_name="gpt2", max_length=50):
    # 创建文本生成 pipeline
    text_generator = pipeline("text-generation", model=model_name)

    # 使用 pipeline 生成文本
    generated_text = text_generator(prompt, max_length=max_length, num_return_sequences=1, no_repeat_ngram_size=2)

    return generated_text[0]["generated_text"]

if __name__ == "__main__":
    prompt = "Once upon a time in a land far, far away"
    generated_text = generate_text(prompt)
    print(generated_text)

