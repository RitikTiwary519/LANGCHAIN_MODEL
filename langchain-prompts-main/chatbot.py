from langchain_huggingface import HuggingFaceEndpoint
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
from dotenv import load_dotenv

load_dotenv()

model = HuggingFaceEndpoint(
    repo_id="HuggingFaceH4/zephyr-7b-beta",  # ✅ better chat-style model
    temperature=0.5,
    max_new_tokens=500
)


chat_history = [
    SystemMessage(content='You are a helpful AI assistant')
]

while True:
    user_input = input('You: ')
    chat_history.append(HumanMessage(content=user_input))
    if user_input == 'exit':
        break
    result = model.invoke(chat_history)
    chat_history.append(AIMessage(content=result))
    print("AI: ",result)

print(chat_history)