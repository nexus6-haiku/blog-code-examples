import numpy as np
import faiss
from langchain_community.embeddings import LlamaCppEmbeddings
from langchain_community.llms import LlamaCpp
from langchain_core.prompts import PromptTemplate
from langchain.chains import LLMChain

# Initialize the models
llm = LlamaCpp(
    model_path="/boot/home/Downloads/llm/DeepSeek-R1-Distill-Qwen-1.5B-Q6_K.gguf",
    temperature=0.7,
    max_tokens=512,
    top_p=0.95,
    n_ctx=2048,
    verbose=True
)

embeddings_model = LlamaCppEmbeddings(
    model_path="/boot/home/Downloads/llm/all-MiniLM-L6-v2.F16.gguf",
    n_ctx=2048
)

# Sample knowledge base
documents = [
    "Haiku is an open-source operating system that specifically targets personal computing.",
    "Haiku is inspired by BeOS and maintains binary compatibility with it.",
    "Haiku has a unique user interface with a clean, simple design.",
    "Haiku's package management system makes it easy to install software.",
    "Python can be used in Haiku for AI applications through various libraries.",
    "Despite lacking GPU acceleration, Haiku can run lightweight AI models using CPU.",
    "Virtual environments in Python help isolate project dependencies.",
    "FAISS is a library for efficient similarity search of embeddings."
]

# Generate and store embeddings
document_embeddings = [embeddings_model.embed_query(doc) for doc in documents]
document_embeddings_np = np.array(document_embeddings).astype('float32')

dimension = len(document_embeddings[0])
index = faiss.IndexFlatL2(dimension)
index.add(document_embeddings_np)

# RAG function
def answer_with_rag(query, k=3):
    # Generate embedding for the query
    query_embedding = embeddings_model.embed_query(query)
    query_embedding_np = np.array([query_embedding]).astype('float32')
    
    # Search for relevant documents
    distances, indices = index.search(query_embedding_np, k)
    relevant_docs = [documents[i] for i in indices[0]]
    
    # Create context from retrieved documents
    context = "\n".join(relevant_docs)
    
    # Create a prompt with the context
    prompt_template = PromptTemplate(
        input_variables=["context", "question"],
        template="""
        You are an assistant for Haiku OS. Use the following pieces of information to answer the user's question.
        If you don't know the answer, just say you don't know.
        
        Context:
        {context}
        
        Question: {question}
        Answer:
        """
    )
    
    # Create a chain and generate answer
    chain = LLMChain(llm=llm, prompt=prompt_template)
    response = chain.invoke({"context": context, "question": query})
    
    return response["text"]

# Example usage
query = "Can I use Python for AI on Haiku OS?"
answer = answer_with_rag(query)
print("Question:", query)
print("Answer:", answer)
