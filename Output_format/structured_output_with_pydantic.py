
# Will work if used with openai but not with HuggingFace


from langchain_core.prompts import ChatPromptTemplate
from langchain_huggingface import HuggingFaceEndpoint, ChatHuggingFace
from typing import TypedDict, Annotated, Optional, Literal
import streamlit as st
from dotenv import load_dotenv
from pydantic import BaseModel, Field
load_dotenv()

# 🔎 Structured Output Schema
class Review(BaseModel):
    key_themes : list[str] = Field(description="Write all the key themes in the content, separated by commas in a list format")
    summary : str = Field(description="Summary of the content")
    sentiment : Literal["positive", "negative", "neutral"] = Field(description="Sentiment of the content, e.g., positive, negative, neutral")
    pros : Optional[list[str]] = Field(default_factory=list, description="List of pros in the content, if any, otherwise empty list")
    cons : Optional[list[str]] = Field(default_factory=list, description="List of cons in the content, if any, otherwise empty list")
    name : Optional[str] = Field(default=None, description="Name of the reviewer")
# 🚀 Hugging Face LLM Setup
llm = HuggingFaceEndpoint(
    repo_id="HuggingFaceH4/zephyr-7b-beta",
    task="text-generation"
)

chat_model = ChatHuggingFace(llm=llm)
structured_model = chat_model.with_structured_output(Review)

# ✍️ Prompt Template with Role Separation
prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a reliable summarizer and sentiment analyzer. Summarize the following content in approximately {line_count} lines. Also provide sentiment, key themes, and list pros and cons if any."),
    ("human", "{content}")
])

# --- Streamlit UI ---
st.header("🧠 Smart Summarizer with LangChain Prompt")

user_input = st.text_area("Paste your content:")
line_count = st.slider("How many lines should the summary be?", 1, 10, 5)

if st.button("Summarize"):
    if user_input:
        # 🎯 Format the prompt
        messages = prompt.format_messages(line_count=line_count, content=user_input)
        print(messages)
        # 🧠 Get structured output
        result = structured_model.invoke(messages)
        print(result)
        # 📤 Display Results
        st.subheader("🧾 Raw Output (Debug)")
        st.write(result)

        
        st.subheader("📌 Summary")
        st.write(result["summary"] if result["summary"] else "No summary available.")

        st.subheader("📊 Sentiment")
        st.write(result["sentiment"])

        st.subheader("🔑 Key Themes")
        st.write(", ".join(result["key_themes"]) if result["key_themes"] else "No key themes detected.")

        st.subheader("✅ Pros")
        st.write(result["pros"] if result["pros"] else "No pros found.")

        st.subheader("❌ Cons")
        st.write(result["cons"] if result["cons"] else "No cons found.")
    else:
        st.warning("Please enter some content.")

