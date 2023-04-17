from transformers import AutoTokenizer, BartForConditionalGeneration

model = BartForConditionalGeneration.from_pretrained("facebook/bart-large-cnn")
tokenizer = AutoTokenizer.from_pretrained("facebook/bart-large-cnn")

ARTICLE_TO_SUMMARIZE = (
    """
    We have explored chain-of-thought prompting as a simple and broadly applicable method for enhancing
reasoning in language models. Through experiments on arithmetic, symbolic, and commonsense
reasoning, we find that chain-of-thought reasoning is an emergent property of model scale that allows
sufficiently large language models to perform reasoning tasks that otherwise have flat scaling curves.
Broadening the range of reasoning tasks that language models can perform will hopefully inspire
further work on language-based approaches to reasoning.
    """
)
inputs = tokenizer([ARTICLE_TO_SUMMARIZE],
                   max_length=1024, return_tensors="pt")

# Generate Summary
summary_ids = model.generate(
    inputs["input_ids"], num_beams=2, min_length=0, max_length=100)
print(tokenizer.batch_decode(summary_ids, skip_special_tokens=True,
      clean_up_tokenization_spaces=False)[0])
