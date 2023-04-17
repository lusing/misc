from transformers import AutoModelForCausalLM, AutoTokenizer, ConversationalPipeline, Conversation

# 加载预训练的模型和分词器
model_name = "microsoft/DialoGPT-medium"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)

# 创建 ConversationalPipeline
conversation_pipeline = ConversationalPipeline(model=model, tokenizer=tokenizer)

# 创建对话
conversation_1 = Conversation("你今天过得怎么样？")
conversation_2 = Conversation("我想了解一下人工智能。")

# 使用 pipeline 获取回复
responses = conversation_pipeline([conversation_1])

#conversation_1 = Conversation("Going to the movies tonight - any suggestions?") # 创建一个对话对象
#pipe([conversation_1]) # 传入一个对话对象列表，得到模型的回复
print(conversation_1.generated_responses) # 打印模型的回复
