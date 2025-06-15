# from langchain_huggingface import HuggingFaceEndpoint
# import streamlit as st
# import os
# from dotenv import load_dotenv
# from langchain_core.prompts import load_prompt,PromptTemplate
# # from langchain_output_parsers from StrOutputParser
# from langchain_core.output_parsers import StrOutputParser

# load_dotenv()

# model = HuggingFaceEndpoint(
#     repo_id="HuggingFaceH4/zephyr-7b-beta", 
#     temperature=0.5,
#     max_new_tokens=500
# )

# parser = StrOutputParser() #parser is a class that converts the output of the model to a string


# tempLate1 = PromptTemplate(
#     template='Write a detailed report on {topic}',
#     input_variables=['topic']
# )

# tempLate2= PromptTemplate(
#     template = "Write a 5 line summary on the following text. /n {text}",
#     input_variables=['text']
# )

# chain = tempLate1 | model | parser | tempLate2 | model | parser
# chain.invoke({'topic': 'black hole'})
# result = chain.invoke({'topic': 'black hole'})



from langchain_huggingface import HuggingFaceEndpoint
import streamlit as st
import os
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser

# Load environment variables
load_dotenv()

# Hugging Face model endpoint
model = HuggingFaceEndpoint(
    repo_id="HuggingFaceH4/zephyr-7b-beta", 
    temperature=0.5,
    max_new_tokens=500
)

# Parser to convert model output to string
parser = StrOutputParser()

# First prompt: detailed report
report_prompt = PromptTemplate(
    template='Write a detailed report on {topic}.',
    input_variables=['topic']
)

# Second prompt: summarize the output
summary_prompt = PromptTemplate(
    template="Write a 5 line summary on the following text:\n{text}",
    input_variables=['text']
)

# First stage: Generate report
report_chain = report_prompt | model | parser

# Second stage: Summarize report
summary_chain = summary_prompt | model | parser

# Full pipeline
topic = "black hole"
report = report_chain.invoke({'topic': topic})
summary = summary_chain.invoke({'text': report})

# Optional: Streamlit Output
st.title("Report and Summary Generator")
st.subheader("Detailed Report:")
st.write(report)
st.subheader("5-Line Summary:")
st.write(summary)
