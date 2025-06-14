import streamlit as st
from dotenv import load_dotenv
from langchain_core.prompts import load_prompt
from langchain_huggingface import HuggingFaceEndpoint
import os

load_dotenv()

model = HuggingFaceEndpoint(
    repo_id="HuggingFaceH4/zephyr-7b-beta",  # ✅ better chat-style model
    temperature=0.5,
    max_new_tokens=500
)

st.header('Research Tool')
st.text("Hello I am Ritik Tiwary , currently made an LLM agent that summarises your papers")
paper_input = st.selectbox(
    "Select Research Paper Name",
    [
        "Attention Is All You Need",
        "BERT: Pre-training of Deep Bidirectional Transformers",
        "GPT-3: Language Models are Few-Shot Learners",
        "Diffusion Models Beat GANs on Image Synthesis"
    ]
)

style_input = st.selectbox(
    "Select Explanation Style",
    ["Beginner-Friendly", "Technical", "Code-Oriented", "Mathematical"]
)

length_input = st.selectbox(
    "Select Explanation Length",
    ["Short (1-2 paragraphs)", "Medium (3-5 paragraphs)", "Long (detailed explanation)"]
)

template = load_prompt('template.json')

if st.button('Summarize'):
    chain = template | model
    result = chain.invoke({
        'paper_input': paper_input,
        'style_input': style_input,
        'length_input': length_input
    })
    st.write(result)
