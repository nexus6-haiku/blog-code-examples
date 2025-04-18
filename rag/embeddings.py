import numpy as np
from langchain_community.embeddings import LlamaCppEmbeddings
import faiss

# Initialize the embeddings model
embeddings_model = LlamaCppEmbeddings(
    model_path="/boot/home/Downloads/llm/all-MiniLM-L6-v2.F16.gguf",
    n_ctx=2048
)

# Sample documents
documents = [
    "Haiku is an open-source operating system",
    "Haiku is inspired by BeOS",
    "Haiku has a unique user interface",
    "Python can be used in Haiku for AI applications"
]

# Generate embeddings for the documents
document_embeddings = [embeddings_model.embed_query(doc) for doc in documents]

# Convert to numpy array
document_embeddings = np.array(document_embeddings).astype('float32')

# Create a FAISS index - using L2 distance
dimension = len(document_embeddings[0])
index = faiss.IndexFlatL2(dimension)

# Add the document embeddings to the index
index.add(document_embeddings)

# Save the index for later use
faiss.write_index(index, "document_index.faiss")

print(document_embeddings)



