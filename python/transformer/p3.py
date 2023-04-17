from transformers import pipeline

pipe = pipeline("text-classification")
result = pipe("This restaurant is awesome")
print(result)

