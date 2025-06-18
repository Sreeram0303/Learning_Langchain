# from langchain_huggingface import HuggingFaceEndpoint, ChatHuggingFace
# from dotenv import load_dotenv
# from typing import TypedDict
# import streamlit as st

# load_dotenv()

# class Review(TypedDict):
#     summary: str
#     sentiment: str

# llm = HuggingFaceEndpoint(
#     repo_id="HuggingFaceH4/zephyr-7b-beta",
#     task="text-generation"
# )

# model = ChatHuggingFace(llm=llm)
# structured_model = model.with_structured_output(Review)

# # Streamlit UI
# st.header("📄 Summarizing Tool with Custom Length")

# user_input = st.text_area("Enter the content to summarize:")
# line_count = st.slider("How many lines should the summary be?", min_value=1, max_value=10, value=5)

# if st.button("Summarize"):
#     if user_input:
#         # Add instruction for summary length
#         modified_prompt = f"""
#         Please summarize the following content into a paragraph with around {line_count} lines. 
#         Also provide the sentiment of the text.

#         Content:
#         {user_input}
#         """

#         result = structured_model.invoke(modified_prompt)
#         st.subheader("📌 Summary")
#         st.write(result["summary"])
#         st.subheader("📊 Sentiment")
#         st.write(result["sentiment"])
#     else:
#         st.warning("Please enter some content to summarize.")

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.messages import HumanMessage
from langchain_huggingface import HuggingFaceEndpoint, ChatHuggingFace
from typing import TypedDict, Annotated
import streamlit as st
from dotenv import load_dotenv

load_dotenv()
# Basic
# class Review(TypedDict):
#     summary: str
#     sentiment: str

# Annotated
class Review(TypedDict):
    summary: Annotated[str, "Summary of the content"]
    sentiment: Annotated[str, "Sentiment of the content, e.g., positive, negative, neutral"]
    
llm = HuggingFaceEndpoint(
    repo_id="HuggingFaceH4/zephyr-7b-beta",
    task="text-generation"
)

chat_model = ChatHuggingFace(llm=llm)
structured_model = chat_model.with_structured_output(Review)

# 👇 Create dynamic prompt template
prompt = ChatPromptTemplate(
    [
        ("system","You are a reliable summarizer and sentiment analyzer. And summarize the following content in approximately {line_count} lines and provide the sentiment. "),
        ("human", "{content}")
    ]
    # """Please summarize the following content in approximately {line_count} lines 
    # and provide the sentiment.

    # Content:
    # {content}
    # """
)

# --- Streamlit UI ---
st.header("🧠 Smart Summarizer with LangChain Prompt")

user_input = st.text_area("Paste your content:")
line_count = st.slider("How many lines should the summary be?", 1, 10, 5)

if st.button("Summarize"):
    if user_input:
        # 👇 Format prompt with user input
        messages = prompt.format_messages(line_count=line_count, content=user_input)
        print(messages)
        # 👇 Send formatted messages to the model
        result = structured_model.invoke(messages)
        print(result)

        st.subheader("📌 Summary")
        st.write(result["summary"])
        st.subheader("📊 Sentiment")
        st.write(result["sentiment"])
    else:
        st.warning("Please enter some content.")
