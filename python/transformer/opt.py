from transformers import AutoTokenizer, OPTForCausalLM

model = OPTForCausalLM.from_pretrained("facebook/opt-350m")
tokenizer = AutoTokenizer.from_pretrained("facebook/opt-350m")

prompt = "What is the meaning of life?"
inputs = tokenizer(prompt, return_tensors="pt")

# Generate
generate_ids = model.generate(inputs.input_ids, max_length=30)
print(tokenizer.batch_decode(generate_ids, skip_special_tokens=True,
      clean_up_tokenization_spaces=False,do_sample=True)[0])
