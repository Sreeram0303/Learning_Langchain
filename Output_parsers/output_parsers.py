
""" Output parsers in LangChain are used to convert the raw output from language models into structured 
formats that can be easily consumed by applications. 
They help in extracting specific information, validating outputs, and ensuring that the data adheres to a defined schema.
Examples include Pydantic parsers, JSON parsers, and custom parsers that can handle specific output formats.
"""

from langchain_huggingface import ChatHuggingFace,HuggingFaceEndpoint
from langchain_core.prompts import PromptTemplate
from dotenv import load_dotenv
load_dotenv()
model = ChatHuggingFace(
    llm = HuggingFaceEndpoint(
        repo_id="HuggingFaceH4/zephyr-7b-beta",
        task="text-generation"
    )
)

template1 = PromptTemplate(
    template="Give me a detailed report on {topic}.",
    input_variables=["topic"]
)

template2 = PromptTemplate(
    template="Write a 5 line summary on the following text: {text}.",
    input_variables=["text"]
)

prompt1 = template1.invoke({"topic" : "black hole"})

result1 = model.invoke(prompt1)
print(result1.content)
prompt2 = template2.invoke({"text": result1.content})

result2 = model.invoke(prompt2)

print(f" Summarized:  {result2.content}")