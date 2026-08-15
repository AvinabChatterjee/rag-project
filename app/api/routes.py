from fastapi import APIRouter, File, HTTPException, UploadFile
from pydantic import BaseModel, Field

from app.config import get_settings
from app.graph.workflow import rag_graph
from app.llm.openai_client import verify_openai_connection
from app.utils.file_utils import detect_file_type, save_upload

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
    final_answer: str | None = None
    error_message: str | None = None
    confidence: str | None = None


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
    analyst_output: AnalystOutput | None = None
    message: str


@router.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    return HealthResponse()


@router.get("/health/llm", response_model=LlmHealthResponse)
async def health_llm() -> LlmHealthResponse:
    result = await verify_openai_connection()
    return LlmHealthResponse(**result)


@router.post("/upload", response_model=UploadResponse)
async def upload(file: UploadFile = File(...)) -> UploadResponse:
    if not file.filename:
        raise HTTPException(status_code=400, detail="Filename is required.")

    settings = get_settings()
    try:
        content = await file.read()
        saved_path = save_upload(content, file.filename, settings.upload_dir)
        file_type = detect_file_type(saved_path)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    return UploadResponse(
        file_path=str(saved_path),
        file_type=file_type,
        filename=saved_path.name,
    )


@router.post("/ask", response_model=AskResponse)
async def ask(request: AskRequest) -> AskResponse:
    try:
        input_state: dict = {"user_question": request.question}
        if request.data_folder is not None:
            input_state["data_folder"] = request.data_folder

        final_state = await rag_graph.ainvoke(input_state)
    except (FileNotFoundError, ValueError) as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    available_files = [
        AvailableFile(**file_info)
        for file_info in final_state.get("available_files", [])
    ]

    analyst_raw = final_state.get("analyst_output") or {}
    analyst_output = AnalystOutput(**analyst_raw) if analyst_raw else None

    return AskResponse(
        workflow_id=final_state["workflow_id"],
        status=final_state["status"],
        question=final_state["user_question"],
        data_folder=final_state["data_folder"],
        available_files=available_files,
        route=final_state.get("route"),
        selected_file_path=final_state.get("selected_file_path"),
        answer=analyst_raw.get("final_answer"),
        error=analyst_raw.get("error_message"),
        analyst_output=analyst_output,
        message=(
            f"Workflow completed via {final_state.get('route', 'unknown')} route "
            "(Phase 2 stubs)."
        ),
    )
