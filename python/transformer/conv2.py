from transformers import pipeline, Conversation

pipe = pipeline('conversational', model='facebook/blenderbot-1B-distill')

conversation_1 = Conversation("What's your favorite moive?") # 创建一个对话对象
pipe([conversation_1]) # 传入一个对话对象列表，得到模型的回复
print(conversation_1.generated_responses) # 打印模型的回复
conversation_1.add_user_input("Avatar") # 添加用户的输入
pipe([conversation_1]) # 再次传入对话对象列表，得到模型的回复
print(conversation_1.generated_responses) # 打印模型的回复
