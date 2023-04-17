from transformers import LlamaTokenizerFast

tokenizer = LlamaTokenizerFast.from_pretrained("hf-internal-testing/llama-tokenizer")
print(tokenizer.encode("Hello this is a test"))

