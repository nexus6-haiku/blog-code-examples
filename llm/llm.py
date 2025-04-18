from langchain_community.llms import LlamaCpp
from langchain_core.prompts import PromptTemplate
from langchain.chains import LLMChain

# Initialize the local Llama model
llm = LlamaCpp(
    model_path="/boot/home/Downloads/llm/DeepSeek-R1-Distill-Qwen-1.5B-Q6_K.gguf",
    temperature=0.7,
    max_tokens=512,
    top_p=0.95,
    n_ctx=2048,
    verbose=True
)

# Create a prompt template
prompt = PromptTemplate(
    input_variables=["question"],
    template="Question: {question}\nAnswer:"
)

# Create a chain with the model and prompt
chain = LLMChain(llm=llm, prompt=prompt)

# Use the chain to generate a response
response = chain.invoke({"question": "What is Haiku operating system?"})
print(response["text"])
