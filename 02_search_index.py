import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)

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

queries = [
    "How much can I claim for a taxi to meet a customer?",  # Test 1: Direct keyword match
    "What is the limit for a cab ride to see a client?",   # Test 2: Semantic match (synonyms)
]

for i, query in enumerate(queries, 1):
    print(f"\n--- Test {i} ---")
    print(f"Question: {query}")
    results = vectorstore.similarity_search(query, k=1)
    for doc in results:
        print(f"Retrieved Result: {doc.page_content}")
