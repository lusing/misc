from transformers import AutoTokenizer, BartForConditionalGeneration

model = BartForConditionalGeneration.from_pretrained("facebook/bart-large-cnn")
tokenizer = AutoTokenizer.from_pretrained("facebook/bart-large-cnn")

ARTICLE_TO_SUMMARIZE = (
    """
    translate English to Chinese: 
    I have a dream.
    """
)
inputs = tokenizer([ARTICLE_TO_SUMMARIZE],
                   max_length=1024, return_tensors="pt")

# Generate Summary
summary_ids = model.generate(
    inputs["input_ids"], num_beams=2, min_length=0, max_length=100)
print(tokenizer.batch_decode(summary_ids, skip_special_tokens=True,
      clean_up_tokenization_spaces=False)[0])
