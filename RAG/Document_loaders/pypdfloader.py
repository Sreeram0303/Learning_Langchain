# Only used for simple text only document loading
from langchain_community.document_loaders import PyPDFLoader
from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser

load_dotenv()

loader  = PyPDFLoader('dl-curriculum.pdf')
docs = loader.load()

print(len(docs))

print(docs[0].page_content)
print(docs[0].metadata)