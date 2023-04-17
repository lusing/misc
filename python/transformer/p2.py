from transformers import pipeline
# text_generator = pipeline("text-generation", model="gpt2-large", max_new_tokens=500)

# text_generator = pipeline("text-generation", model="gpt2-xl", max_new_tokens=250)

# bigscience/bloom-560m

# text_generator = pipeline("text-generation", model="bigscience/bloom-560m", max_new_tokens=500)

# bigscience/bloom-1b1

# text_generator = pipeline("text-generation", model="bigscience/bloom-1b1", max_new_tokens=500)

# xlnet-base-cased
# text_generator = pipeline("text-generation", model="xlnet-base-cased", max_new_tokens=500)

# facebook/opt-350m

# text_generator = pipeline("text-generation", model="facebook/opt-350m", max_new_tokens=500)

text_generator = pipeline("text-generation", model="gpt2-large", max_new_tokens=250)

text_generator.model.config.pad_token_id = text_generator.model.config.eos_token_id

text = text_generator("I have a dream ")[0]["generated_text"]

print(text)
