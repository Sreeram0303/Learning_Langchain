from langchain_huggingface import HuggingFaceEmbeddings
from dotenv import load_dotenv
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
load_dotenv()
embeddings = HuggingFaceEmbeddings(
    model_name ="sentence-transformers/all-MiniLM-L6-v2",
)

documents = [
    "Virat Kohli is a famous Indian cricketer known for his aggressive batting and leadership.",
    "Sachin Tendulkar is often regarded as one of the greatest batsmen in cricket history.",
    "Rohit Sharma is the captain of the Indian cricket team and holds the record for the highest individual score in ODIs.",
    "MS Dhoni is a legendary Indian cricketer known for his calm demeanor and finishing skills.",
    "Ravindra Jadeja is an all-rounder known for his exceptional fielding and left-arm spin bowling.",
]

query = "Tell me about Sachin Tendulkar's cricketing career."

doc_embeddings = embeddings.embed_documents(documents)
query_embedding = embeddings.embed_query(query)

scores = cosine_similarity([query_embedding], doc_embeddings)[0]

index,score  = sorted(list(enumerate(scores)),key=lambda x : x[1], reverse=True)[0]

print(f"Query: {query}")
print(f"Most relevant document: {documents[index]}")
print(f"Score: {score:.4f}")