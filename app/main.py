from contextlib import asynccontextmanager

from fastapi import FastAPI

from app.api.routes import router
from app.config import get_settings
from app.graph.workflow import init_workflow_engine, shutdown_workflow_engine

settings = get_settings()


@asynccontextmanager
async def lifespan(_: FastAPI):
    settings.ensure_directories()
    await init_workflow_engine()
    yield
    await shutdown_workflow_engine()


app = FastAPI(
    title="Agentic RAG",
    description="3-Agent Agentic RAG with LangGraph (local files only)",
    version="0.1.0",
    lifespan=lifespan,
)

app.include_router(router)
