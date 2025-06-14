from langchain_groq import ChatGroq
from dotenv import load_dotenv
import os

load_dotenv()

llm = ChatGroq(
    model="llama3-70b-8192",  # ✅ supported
    api_key=os.getenv("GROQ_API_KEY")
)

result = llm.invoke("how many of your api can i hit in one hour?")
print(result.content)
