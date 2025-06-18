from langchain_text_splitters import CharacterTextSplitter
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
load_dotenv()

loader = PyPDFLoader('dl-curriculum.pdf')
docs = loader.load()

text_splitter = CharacterTextSplitter(
    chunk_size = 100,
    chunk_overlap = 0,
    separator  = ''
) 
# text = "".join([doc.page_content for doc in docs])
result = text_splitter.split_documents(docs)
print(result[0])