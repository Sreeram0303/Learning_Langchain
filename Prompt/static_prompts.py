from langchain_huggingface import ChatHuggingFace,HuggingFaceEndpoint
from dotenv import load_dotenv
import streamlit as st
from langchain_core.prompts import PromptTemplate
load_dotenv()

llm = HuggingFaceEndpoint(
    repo_id="HuggingFaceH4/zephyr-7b-beta",
    task="text-generation"
)

model = ChatHuggingFace(llm=llm)

st.header("Research Tool")

user_input = st.text_input("Enter your question:")

if st.button("Summarize"):
    if user_input:
        result = model.invoke(user_input)
        st.write("Response:", result.content)
    else:
        st.write("Please enter a question to get a response.")