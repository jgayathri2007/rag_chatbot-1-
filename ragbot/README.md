# RAG-Based Document Chatbot

A beginner-friendly Retrieval-Augmented Generation (RAG) chatbot.

## Architecture

PDF
  ↓
Text Extraction (pypdf)
  ↓
Chunking
  ↓
Embeddings (Sentence Transformers)
  ↓
Vector Similarity Search
  ↓
Relevant Chunks
  ↓
Local LLM (Ollama)
  ↓
Chatbot Answer

## 1. Requirements

- Python 3.10 or newer
- Ollama
- An Ollama model such as `llama3.2:3b`

Install Ollama from the official website:
https://ollama.com/

Then run:

```bash
ollama pull llama3.2:3b
```

## 2. Create virtual environment

Windows:

```bash
python -m venv venv
venv\Scripts\activate
```

macOS/Linux:

```bash
python3 -m venv venv
source venv/bin/activate
```

## 3. Install Python packages

```bash
pip install -r requirements.txt
```

## 4. Start Ollama

Make sure Ollama is running.

The application expects:

```text
http://localhost:11434
```

You can change the model with:

Windows PowerShell:
```powershell
$env:OLLAMA_MODEL="llama3.2:3b"
```

macOS/Linux:
```bash
export OLLAMA_MODEL=llama3.2:3b
```

## 5. Run the chatbot

```bash
streamlit run app.py
```

A browser window will open.

## 6. How to use

1. Upload one or more PDFs.
2. Click **Process Documents**.
3. Ask questions in the chat box.
4. Expand **Sources** to see which PDF pages were retrieved.

## Important

This starter uses a JSON file as a simple local vector index. For a larger production project, replace it with a vector database such as FAISS, Chroma, Qdrant, or PostgreSQL/pgvector.

## Project structure

```text
rag_chatbot/
├── app.py
├── rag_engine.py
├── requirements.txt
├── README.md
└── data/
```

## Possible upgrades

- OCR for scanned PDFs
- Login/authentication
- Multiple users
- Chat history database
- Better vector database
- Streaming LLM responses
- Citations with highlighted source text
- DOCX/PPTX/TXT support
- Tamil + English multilingual retrieval
- Admin dashboard
