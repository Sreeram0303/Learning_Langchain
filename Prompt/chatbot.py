from langchain_huggingface import HuggingFaceEndpoint, ChatHuggingFace
from dotenv import load_dotenv
load_dotenv()

llm = HuggingFaceEndpoint(
    repo_id = "HuggingFaceH4/zephyr-7b-beta",
    task = "text-generation"
)

model = ChatHuggingFace(llm = llm)
chat_history = []
while True:
    user_input = input("You: ")
    chat_history.append(user_input)
    if user_input == "exit":
        break
    result = model.invoke(chat_history)
    chat_history.append(result.content)
    print(f"Bot: {result.content}")
print(chat_history)