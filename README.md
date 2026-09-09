# RAG Python Example: Expense Policy Q&A

A minimal, step-by-step Python demonstration of **Retrieval-Augmented Generation (RAG)** using **LangChain**, **FAISS**, and **Ollama** (`nomic-embed-text` embeddings + `phi4` LLM).

This example demonstrates how to build a context-aware Q&A system over local custom documents (an expense policy) using local vector search and local LLM inference.

---

## 📋 Overview

Retrieval-Augmented Generation (RAG) enhances Large Language Model (LLM) responses by retrieving relevant information from a custom knowledge base before generating an answer.

This project breaks the RAG pipeline down into 3 clear, standalone steps:

1. **Indexing (`01_build_index.py`)**: Reads source text (`expenses.txt`), splits it into chunks, generates vector embeddings via Ollama (`nomic-embed-text`), and stores them in a local FAISS index (`expense_index/`).
2. **Retrieval Search (`02_search_index.py`)**: Queries the FAISS index to test semantic similarity search independently.
3. **RAG Pipeline (`03_rag_with_ollama.py`)**: Retrieves matching policy context and passes it along with a user prompt to a local Ollama model (`phi4`) for grounded answer generation.

---

## 📁 Project Structure

```text
rag_python_example/
├── 01_build_index.py     # Ingests expenses.txt and builds the FAISS vector index via Ollama Embeddings
├── 02_search_index.py    # Loads the index and performs semantic similarity search
├── 03_rag_with_ollama.py # Complete RAG chain querying Ollama (phi4) with context
├── expenses.txt          # Source document containing company expense policy rules
├── requirements.txt      # Python dependencies
└── README.md             # Project documentation
```

---

## 🛠️ Prerequisites

1. **Python 3.9+**
2. **Ollama**: Download and install [Ollama](https://ollama.com/).
3. **Required Ollama Models**: Pull the embedding and generation models via the Ollama CLI:
   ```bash
   ollama pull nomic-embed-text
   ollama pull phi4
   ```

---

## ⚡ Installation & Setup

1. **Clone or navigate to the project directory**:
   ```bash
   cd /path/to/rag_python_example
   ```

2. **Create and activate a Python virtual environment** (recommended):
   ```bash
   python3 -m venv venv
   source venv/bin/activate  # On macOS/Linux
   # venv\Scripts\activate   # On Windows
   ```

3. **Install the dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

---

## 🚀 Step-by-Step Usage

### Step 1: Build the Vector Index
Run `01_build_index.py` to process `expenses.txt`, generate vector embeddings using Ollama (`nomic-embed-text`), and save the index locally to `expense_index/`.

```bash
python 01_build_index.py
```
*Output:*
```text
Index created
```

### Step 2: Test Vector Similarity Search
Run `02_search_index.py` to compare direct keyword matching vs. semantic (synonym) matching.

```bash
python 02_search_index.py
```
*Output:*
```text
--- Test 1 ---
Question: How much can I claim for a taxi to meet a customer?
Retrieved Result: Taxi journeys to customer meetings can be reimbursed up to £50.

--- Test 2 ---
Question: What is the limit for a cab ride to see a client?
Retrieved Result: Taxi journeys to customer meetings can be reimbursed up to £50.
```

> **Note**: Test 2 demonstrates **semantic vector search**—it successfully retrieves *"Taxi journeys to customer meetings"* even though the query uses entirely different synonyms ("cab ride", "client", "limit").

### Step 3: Run the Full RAG Chain with Ollama
Make sure your local Ollama application is running, then execute `03_rag_with_ollama.py`. The script will retrieve the relevant policy context and prompt `phi4` to answer the question using *only* the supplied context.

```bash
python 03_rag_with_ollama.py
```
*Output:*
```text
Question: What is the limit for a cab ride to see a client?

Retrieved Context:
Taxi journeys to customer meetings can be reimbursed up to £50.

Answer:
The limit for a cab ride to see a client is up to £50.
```

---

## ⚙️ How It Works Under the Hood

```
[ expenses.txt ] ──> Chunking ──> Ollama (nomic-embed-text) ──> Local FAISS Index
                                                                        │
                                                                        ▼
[ User Question ] ───────────────────────────────────────────> Similarity Search
                                                                        │
                                                                        ▼
[ Prompt + Context ] ──> Ollama (phi4) ──> Final Grounded Answer
```

- **Document Loader**: `TextLoader` reads plain text files (`expenses.txt`).
- **Text Splitter**: `RecursiveCharacterTextSplitter` breaks long documents into chunks (chunk size: 100, overlap: 0).
- **Embedding Model**: `OllamaEmbeddings(model="nomic-embed-text")` generates 768-dimensional dense vector representations via Ollama.
- **Vector Store**: `FAISS` (Facebook AI Similarity Search) stores vectors locally for fast nearest-neighbor retrieval.
- **Generation**: `ollama.chat` sends the prompt and retrieved context to `phi4`, enforcing that answers are strictly based on the provided context.

---

## 💡 Customization

- **Change Source Data**: Edit `expenses.txt` or point `01_build_index.py` to a different document file.
- **Change the LLM**: Update `model="phi4"` in `03_rag_with_ollama.py` to any other Ollama model you have installed (e.g., `llama3`, `mistral`).
- **Adjust Chunking**: Modify `chunk_size` and `chunk_overlap` in `01_build_index.py` to fine-tune retrieval performance for longer documents.

---

## 📚 References & Inspiration

- [Microsoft Learn: Explore RAG Exercise](https://microsoftlearning.github.io/mslearn-ai-concepts/Instructions/exercises/07-explore-rag.html): Hands-on lab exploring Retrieval-Augmented Generation concepts.
- [Microsoft AI Apps Chat Playground](https://microsoftlearning.github.io/ai-apps/chat-playground/): Interactive web playground for AI chat applications.
- [Microsoft Learning AI Apps GitHub Repository](https://github.com/MicrosoftLearning/ai-apps/tree/main/chat-playground): Source code for Microsoft Learning AI chat application exercises.
- [LangChain RAG Tutorial](https://python.langchain.com/docs/tutorials/rag/): Official guide on building Retrieval-Augmented Generation chains.
- [LangChain Ollama Integration](https://python.langchain.com/docs/integrations/text_embedding/ollama/): Documentation for using local Ollama embeddings and LLMs with LangChain.
- [Ollama Library](https://ollama.com/library): Explore open-source local models including [`nomic-embed-text`](https://ollama.com/library/nomic-embed-text) and [`phi4`](https://ollama.com/library/phi4).
- [Meta FAISS Repository](https://github.com/facebookresearch/faiss): Facebook AI Similarity Search library for fast, efficient vector indexing.
- [Nomic AI](https://www.nomic.ai/): Creators of the open-source `nomic-embed-text` embedding model.
