from langchain_community.document_loaders import TextLoader
from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate   
from langchain_core.output_parsers import StrOutputParser

load_dotenv()

loader = TextLoader('cricket.txt', encoding="utf-8")

docs = loader.load()

model = ChatHuggingFace(
    llm=HuggingFaceEndpoint(
        repo_id="HuggingFaceH4/zephyr-7b-beta",
        task="text-generation"
    )
)
prompt = PromptTemplate(
    template="Summarize the following text: {text}",
    input_variables=["text"]
)

parser = StrOutputParser()

chain = prompt | model | parser

result = chain.invoke({"text" : docs[0].page_content})
print(result)
# print(docs)

# print(docs[0].page_content)
# print(docs[0].metadata)