import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)

from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_ollama import OllamaEmbeddings

loader = TextLoader("expenses.txt")
documents = loader.load()

splitter = RecursiveCharacterTextSplitter(
    chunk_size=100,
    chunk_overlap=0
)
chunks = splitter.split_documents(documents)

embeddings = OllamaEmbeddings(
    model="nomic-embed-text"
)

vectorstore = FAISS.from_documents(chunks, embeddings)
vectorstore.save_local("expense_index")

print("Index created")
