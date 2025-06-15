from langchain_groq import ChatGroq
from langchain_core.prompts import PromptTemplate   
from langchain_core.output_parsers import JsonOutputParser
from dotenv import load_dotenv
import os

'''
This script demonstrates how to use the LangChain library with Groq's ChatGroq model to generate a detailed report and a summary based on a given topic.

here are my notes:
1)json output parser is just a fancy way to add that give me output in json forrmat 
2) process for it is :
parser = JsonOutputParser() -> 
in the prompt, we add that {format_instruction} -> , partial_variable we pass as dict in that we pass that is equal to parser.get_format_instructions() , which simple means the same
'''

# Load environment variables
load_dotenv()

# Initialize the model
model = ChatGroq(
    model="llama3-70b-8192",
    api_key=os.getenv("GROQ_API_KEY")
)

# JSON parser and format instructions
parser = JsonOutputParser()
format_instructions = parser.get_format_instructions()

print("Format Instructions:", format_instructions)

# First prompt template for detailed report
template1 = PromptTemplate(
    template='Write a detailed report on {topic}. {format_instruction}',
    input_variables=['topic'],
    partial_variables={'format_instruction': format_instructions}
)

# Second prompt template for 5-line summary
template2 = PromptTemplate(
    template='Write a summary in 5 lines of {topic}. {format_instruction}',
    input_variables=['topic'],
    partial_variables={'format_instruction': format_instructions}
)

# Chain the components
chain = template1 | model | parser | template2 | model | parser

# Run the chain
result = chain.invoke({
    'topic': 'black hole'
})

# Print the result
print(result)
