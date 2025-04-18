import numpy as np
import faiss
from langchain_community.embeddings import LlamaCppEmbeddings

# Initialize the embeddings model
embeddings_model = LlamaCppEmbeddings(
    model_path="/boot/home/Downloads/llm/all-MiniLM-L6-v2.F16.gguf",
    n_ctx=2048
)

# Load the documents (in a real application, you might load from a file)
documents = [
    "Haiku is an open-source operating system",
    "Haiku is inspired by BeOS",
    "Haiku has a unique user interface",
    "Python can be used in Haiku for AI applications"
]

# Load the FAISS index
index = faiss.read_index("document_index.faiss")

# User query
query = "Tell me about programming in Haiku"

# Generate embedding for the query
query_embedding = embeddings_model.embed_query(query)
query_embedding = np.array([query_embedding]).astype('float32')

# Search the index
k = 2  # Number of results to retrieve
distances, indices = index.search(query_embedding, k)

# Get the most relevant documents
most_relevant_docs = [documents[i] for i in indices[0]]

print("Query:", query)
print("Most relevant documents:")
for i, doc in enumerate(most_relevant_docs):
    print(f"{i+1}. {doc} (distance: {distances[0][i]})")
