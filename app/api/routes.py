from typing import Literal

from fastapi import APIRouter, File, HTTPException, UploadFile
from pydantic import BaseModel, Field

from app.config import get_settings
from app.graph.workflow import rag_graph
from app.llm.openai_client import verify_openai_connection
from app.rag.ingest import ingest_document
from app.utils.file_utils import detect_file_type, save_upload, validate_local_file

router = APIRouter()


class HealthResponse(BaseModel):
    status: str = "ok"


class LlmHealthResponse(BaseModel):
    ok: bool
    model: str
    response: str | None = None
    error: str | None = None


class UploadResponse(BaseModel):
    file_path: str
    file_type: str
    filename: str


class IngestRequest(BaseModel):
    file_path: str = Field(..., min_length=1)


class IngestResponse(BaseModel):
    file_path: str
    chunks_indexed: int


class AvailableFile(BaseModel):
    file_path: str
    file_name: str
    file_type: str


class AskRequest(BaseModel):
    question: str = Field(..., min_length=1)
    data_folder: str | None = Field(
        default=None,
        description="Optional override. Defaults to DATA_FOLDER in .env.",
    )


class AnalystOutput(BaseModel):
    final_answer: str
    error_message: str | None = None
    confidence: Literal["high", "medium", "low"]


class ExecutionResult(BaseModel):
    success: bool
    attempts: int | None = None
    raw_result: object | None = None
    executed_code: str | None = None
    error: str | None = None


class AskResponse(BaseModel):
    workflow_id: str
    status: str
    question: str
    data_folder: str
    available_files: list[AvailableFile]
    route: str | None = None
    selected_file_path: str | None = None
    answer: str | None = None
    error: str | None = None
    execution_result: ExecutionResult | None = None
    analyst_output: AnalystOutput | None = None
    message: str


def build_ask_input_state(question: str, data_folder: str | None = None) -> dict:
    user_question = question.strip()
    if not user_question:
        raise ValueError("question is required.")

    input_state: dict[str, str] = {"user_question": user_question}
    if data_folder is not None:
        folder = data_folder.strip()
        if not folder:
            raise ValueError("data_folder must not be blank when provided.")
        input_state["data_folder"] = folder
    return input_state


def build_ask_response(final_state: dict) -> AskResponse:
    available_files = [
        AvailableFile(**file_info)
        for file_info in final_state.get("available_files", [])
    ]

    analyst_raw = final_state.get("analyst_output") or {}
    analyst_output = AnalystOutput(**analyst_raw) if analyst_raw else None
    execution_raw = final_state.get("execution_result") or {}
    execution_result = ExecutionResult(**execution_raw) if execution_raw else None
    route = final_state.get("route")

    return AskResponse(
        workflow_id=final_state["workflow_id"],
        status=final_state["status"],
        question=final_state["user_question"],
        data_folder=final_state["data_folder"],
        available_files=available_files,
        route=route,
        selected_file_path=final_state.get("selected_file_path"),
        answer=analyst_raw.get("final_answer"),
        error=analyst_raw.get("error_message"),
        execution_result=execution_result,
        analyst_output=analyst_output,
        message=f"Workflow completed via {route or 'unknown'} route.",
    )


@router.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    return HealthResponse()


@router.get("/health/llm", response_model=LlmHealthResponse)
async def health_llm() -> LlmHealthResponse:
    result = await verify_openai_connection()
    return LlmHealthResponse(**result)


@router.post("/upload", response_model=UploadResponse)
async def upload(file: UploadFile = File(...)) -> UploadResponse:
    filename = (file.filename or "").strip()
    if not filename:
        raise HTTPException(status_code=400, detail="Filename is required.")

    settings = get_settings()
    try:
        content = await file.read()
        saved_path = save_upload(content, filename, settings.upload_dir)
        file_type = detect_file_type(saved_path)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    return UploadResponse(
        file_path=str(saved_path),
        file_type=file_type,
        filename=saved_path.name,
    )


@router.post("/ingest", response_model=IngestResponse)
async def ingest(request: IngestRequest) -> IngestResponse:
    file_path = request.file_path.strip()
    if not file_path:
        raise HTTPException(status_code=400, detail="file_path is required.")

    try:
        resolved_path = validate_local_file(file_path)
        chunks_indexed = await ingest_document(resolved_path)
    except (FileNotFoundError, ValueError) as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    return IngestResponse(
        file_path=str(resolved_path.resolve()),
        chunks_indexed=chunks_indexed,
    )


@router.post("/ask", response_model=AskResponse)
async def ask(request: AskRequest) -> AskResponse:
    try:
        input_state = build_ask_input_state(request.question, request.data_folder)
        final_state = await rag_graph.ainvoke(input_state)
    except (FileNotFoundError, ValueError) as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    return build_ask_response(final_state)
