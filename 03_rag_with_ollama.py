import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)

from ollama import chat
from langchain_community.vectorstores import FAISS
from langchain_ollama import OllamaEmbeddings

embeddings = OllamaEmbeddings(
    model="nomic-embed-text"
)

vectorstore = FAISS.load_local(
    "expense_index",
    embeddings,
    allow_dangerous_deserialization=True
)

question = "What is the limit for a cab ride to see a client?"

docs = vectorstore.similarity_search(question, k=3)
context = "\n\n".join(d.page_content for d in docs)

prompt = f"""
You are an expense policy assistant.

Context:
{context}

Question:
{question}

Answer only using the supplied context.
"""

response = chat(
    model="phi4",
    messages=[{"role": "user", "content": prompt}]
)

print(f"Question: {question}\n")
print(f"Retrieved Context:\n{context}\n")
print(f"Answer:\n{response['message']['content']}")
