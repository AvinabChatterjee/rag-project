# Agentic RAG

3-Agent Agentic RAG system built with LangGraph, FastAPI, and OpenAI `gpt-4o-mini`.  
All files (CSV, Excel, PDF) are stored on the local drive.

See [EXECUTION_PLAN.md](./EXECUTION_PLAN.md) for the full implementation plan.

## Prerequisites

- Python 3.12+
- OpenAI API key

## Setup

1. **Activate virtual environment**

   ```powershell
   cd C:\Users\avina\Desktop\Projects\rag-project
   .venv\Scripts\activate
   ```

2. **Install dependencies**

   ```powershell
   pip install -r requirements.txt
   ```

3. **Configure environment**

   Copy `.env.example` to `.env` and set your OpenAI API key:

   ```powershell
   copy .env.example .env
   ```

   Edit `.env` and replace `your_openai_api_key_here` with your real key.

## Run the API

```powershell
uvicorn app.main:app --reload
```

API docs: http://localhost:8000/docs

## Phase 0 Endpoints

| Method | Endpoint   | Description                          |
|--------|------------|--------------------------------------|
| GET    | `/health`  | Health check                         |
| POST   | `/upload`  | Upload a file to `data/uploads/`     |
| POST   | `/ask`     | Validate question + file (stub)      |

### Example: upload a file

```powershell
curl -X POST "http://localhost:8000/upload" -F "file=@C:\path\to\sales.csv"
```

### Example: ask (validation only for now)

```powershell
curl -X POST "http://localhost:8000/ask" ^
  -H "Content-Type: application/json" ^
  -d "{\"question\": \"What is total revenue?\", \"file_path\": \"C:/Users/avina/Desktop/Projects/rag-project/data/uploads/your_file.csv\"}"
```

## Project Structure

```
rag-project/
├── app/
│   ├── main.py
│   ├── config.py
│   ├── api/routes.py
│   └── utils/file_utils.py
├── data/
│   ├── uploads/      # uploaded files
│   └── vector_db/    # Chroma (Phase 5)
├── .env
├── requirements.txt
└── EXECUTION_PLAN.md
```
