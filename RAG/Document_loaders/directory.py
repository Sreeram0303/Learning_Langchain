from langchain_community.document_loaders import DirectoryLoader,PyPDFLoader

loader = DirectoryLoader(
    path = "books",
    glob = "*.pdf",
    loader_cls = PyPDFLoader
)

docs = loader.load()

print(f"Number of documents loaded: {len(docs)}")
print(f"First document content: {docs[0].page_content[:500]}")  # Print first 500 characters of the first document
print(f"First document metadata: {docs[0].metadata}")  # Print metadata of the first document