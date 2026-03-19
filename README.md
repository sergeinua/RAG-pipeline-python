# Legal RAG — Document Q&A with LangGraph Agent

A production-ready RAG (Retrieval-Augmented Generation) SaaS pipeline that lets users upload PDF documents and ask questions about their content. Built with LangChain, LangGraph, Chroma, and OpenRouter.

---

## Architecture

```
PDF Upload
    ↓
[Ingest] PyMuPDF → Chunking → Embeddings → Chroma DB
                                                ↓
User Question → [LangGraph Agent] → search_documents (tool)
                        ↓
                   LLM reasons over retrieved chunks
                        ↓
                   Final Answer
```

The agent uses a reasoning loop — it decides when to search, whether to search again with a different query, and when it has enough context to answer. This is more robust than a simple RAG chain that always retrieves exactly once.

---

## Tech Stack

| Component | Tool | Why |
|---|---|---|
| PDF parsing | PyMuPDF | Fast, handles complex layouts |
| Chunking | LangChain RecursiveCharacterTextSplitter | Respects paragraph/sentence boundaries |
| Embeddings | sentence-transformers (all-MiniLM-L6-v2) | Free, runs locally, good quality |
| Vector store | Chroma | Simple local setup, no cloud required |
| LLM | OpenRouter (llama-3.1-8b-instruct:free) | Free tier, OpenAI-compatible API |
| Agent framework | LangGraph | Stateful reasoning loop with tool use |
| API | FastAPI | Async, auto-generated docs at /docs |

---

## Project Structure

```
legal-rag/
├── .env                  # API keys — never commit this
├── .gitignore
├── requirements.txt
├── ingest.py             # PDF parsing, chunking, and vectorstore ingestion
├── rag.py                # Vectorstore loading, RAG chain, format utilities
├── agent.py              # LangGraph agent with search tools
├── api.py                # FastAPI endpoints — upload PDF, ask questions
└── data/                 # Place your PDF files here (gitignored)
```

---

## Prerequisites

- Python 3.9+
- An OpenRouter API key — get one free at [openrouter.ai](https://openrouter.ai)

---

## Setup

### 1. Clone the repository

```bash
git clone https://github.com/your-username/legal-rag.git
cd legal-rag
```

### 2. Create a virtual environment

```bash
python3.9 -m venv venv
source venv/bin/activate      # macOS / Linux
# venv\Scripts\activate       # Windows
```

> **Important:** Always activate the venv before running any commands. You will see `(venv)` in your terminal prompt when it is active.

### 3. Install dependencies

```bash
pip install --upgrade pip
pip install langchain langchain-community langchain-openai
pip install langchain-huggingface langchain-chroma
pip install chromadb pymupdf sentence-transformers
pip install fastapi uvicorn python-dotenv python-multipart
pip install langgraph
```

### 4. Configure environment variables

Create a `.env` file in the project root:

```env
OPENROUTER_API_KEY=sk-or-xxxxxxxxxxxxxxxx
OPENROUTER_BASE_URL=https://openrouter.ai/api/v1
LLM_MODEL=meta-llama/llama-3.1-8b-instruct:free
```

Get your free API key at [openrouter.ai/keys](https://openrouter.ai/keys). The model above is free — you can swap it for any OpenRouter-supported model.

---

## Usage

### Option A — API (recommended)

Start the server:

```bash
uvicorn api:app --reload
```

The API is now running at `http://localhost:8000`. Open `http://localhost:8000/docs` for the interactive Swagger UI.

**Step 1 — Upload a PDF:**

```bash
curl -X POST http://localhost:8000/upload \
  -F "file=@data/your_document.pdf"
```

Response:
```json
{"status": "ok", "pages": 12, "chunks": 47}
```

**Step 2 — Ask a question:**

```bash
curl -X POST http://localhost:8000/ask \
  -H "Content-Type: application/json" \
  -d '{"question": "What are the main obligations described in the contract?"}'
```

Response:
```json
{"answer": "According to page 3, the main obligations include..."}
```

**Check server health:**

```bash
curl http://localhost:8000/health
```

### Option B — Command line

Ingest a PDF directly:

```bash
python ingest.py data/your_document.pdf
```

Test the RAG chain without the agent:

```bash
python rag.py
```

Test the full agent in the terminal:

```bash
python agent.py
```

---

## API Reference

| Method | Endpoint | Description |
|---|---|---|
| `GET` | `/health` | Check if server is running and documents are loaded |
| `POST` | `/upload` | Upload and index a PDF file |
| `POST` | `/ask` | Ask a question about uploaded documents |

### POST /upload

- **Body:** `multipart/form-data` with a `file` field (PDF only)
- **Response:** `{"status": "ok", "pages": int, "chunks": int}`

### POST /ask

- **Body:** `{"question": "string"}`
- **Response:** `{"answer": "string"}`

---

## How the Agent Works

The LangGraph agent uses a stateful reasoning loop:

```
[llm node] — reads the question, decides whether to search
     ↓
should_continue?
     ├── tool_calls present → [tools node] — executes search_documents
     │                              ↓
     │                        [llm node] — reads search results, reasons
     │                              ↓
     │                        should_continue? (loop again if needed)
     └── no tool_calls → END — returns final answer
```

Available tools:

- `search_documents(query)` — semantic search across all indexed documents, returns top-5 relevant chunks with page references
- `search_documents_targeted(query, page_hint)` — same search with an optional page filter for when the user references a specific section

---

## Security

- User input is always wrapped in `<user_input>` tags in prompts, separated from system instructions
- Document content retrieved from the vectorstore is explicitly labeled as data, not instructions, to prevent prompt injection
- The system prompt instructs the agent to treat all tool results as untrusted document data

---

## Customisation

**Swap the LLM** — change `LLM_MODEL` in `.env` to any model available on OpenRouter:

```env
LLM_MODEL=openai/gpt-4o
LLM_MODEL=anthropic/claude-3-5-sonnet
LLM_MODEL=google/gemini-flash-1.5
```

**Adjust chunk size** — edit `ingest.py`:

```python
splitter = RecursiveCharacterTextSplitter(
    chunk_size=1024,   # increase for more context per chunk
    chunk_overlap=100, # increase to reduce information loss at boundaries
)
```

**Change retrieval depth** — edit `agent.py`:

```python
retriever = vs.as_retriever(search_kwargs={"k": 10})  # retrieve more chunks
```

---

## Troubleshooting

**`ModuleNotFoundError: No module named 'fitz'`**
Make sure you are inside the virtual environment (`source venv/bin/activate`) and run `pip install pymupdf`.

**`No module named 'sentence_transformers'`**
Run `pip install sentence-transformers`. If you see import errors related to `transformers`, install inside a clean venv.

**OpenRouter returns 401**
Check that `OPENROUTER_API_KEY` in `.env` is correct and has no extra spaces.

**Agent gives answers not based on documents**
The free llama model occasionally ignores tool-use instructions. Try a stronger model like `openai/gpt-4o-mini` or add more explicit instructions to `SYSTEM_PROMPT` in `agent.py`.

---

## Next Steps

- [ ] Add hybrid search (BM25 + semantic) for better retrieval on exact terms
- [ ] Add conversation memory so the agent remembers previous questions
- [ ] Integrate LangSmith for agent tracing and observability
- [ ] Add RAGAS evaluation to measure answer quality
- [ ] Deploy to Azure Container Apps with Azure OpenAI Service

---

## License

MIT