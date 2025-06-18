from langchain_huggingface import HuggingFaceEmbeddings

embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2",
)
documents = [
    "This is the first document.",
    "This is the second document.",
    "And this is the third one.",
    "Is this the first document?",
]
# text = "This is a test sentence."

# result = embeddings.embed_query(text)
result = embeddings.embed_documents(documents)
print(result)   