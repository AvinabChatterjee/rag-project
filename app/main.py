from fastapi import FastAPI

from app.api.routes import router
from app.config import get_settings

settings = get_settings()

app = FastAPI(
    title="Agentic RAG",
    description="3-Agent Agentic RAG with LangGraph (local files only)",
    version="0.1.0",
)

app.include_router(router)


@app.on_event("startup")
def on_startup() -> None:
    settings.ensure_directories()
