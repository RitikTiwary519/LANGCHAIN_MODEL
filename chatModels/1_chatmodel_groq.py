from langchain_groq import ChatGroq
from dotenv import load_dotenv
import os

load_dotenv()

# Initialize Groq chat model
model = ChatGroq(
    model="llama3-70b-8192",  # ✅ replace with supported model
    api_key=os.getenv("GROQ_API_KEY")
)


result = model.invoke("What is the capital of India?")
print(result)
