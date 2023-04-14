# Import libraries
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
from transformers import LlamaTokenizer
tokenizer = LlamaTokenizer.from_pretrained("decapoda-research/llama-7b-hf")

# Load model and tokenizer
#tokenizer = AutoTokenizer.from_pretrained("decapoda-research/llama-7b-hf")
model = AutoModelForCausalLM.from_pretrained("decapoda-research/llama-7b-hf")

# Create a text generation pipeline
text_generator = pipeline("text-generation", model=model, tokenizer=tokenizer)

# Generate text from a prompt
prompt = "如何用Python编写一个Hello World程序"
output = text_generator(prompt, max_length=500, temperature=0.9)[0]

# Print the generated text
print(output["generated_text"])
