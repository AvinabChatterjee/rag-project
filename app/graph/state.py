from typing import Any, Literal, TypedDict

from app.utils.folder_utils import ScannedFile

WorkflowStatus = Literal[
    "pending",
    "initialized",
    "planning",
    "executing",
    "retrieving",
    "analyzing",
    "completed",
    "failed",
]

Route = Literal["tabular", "document"]
FileType = Literal["csv", "excel", "document"]


class WorkflowState(TypedDict, total=False):
    """
    LangGraph state — mirrors the workflow JSON passed between nodes.

    Nodes return partial updates; they must not replace the full state.

    Option A: user only asks a question. data_folder is resolved from DATA_FOLDER
    in .env (fixed location). Agent 1 picks the right file from available_files.
    """

    # --- Set by FastAPI before graph invoke ---
    user_question: str  # from user request
    data_folder: str  # from .env DATA_FOLDER (not from user each time)

    # --- Set by init_workflow ---
    workflow_id: str
    created_at: str
    status: WorkflowStatus
    available_files: list[ScannedFile]

    # --- Set by Agent 1 (Query Planner) in Phase 3+ ---
    selected_file_path: str | None
    selected_file_type: FileType | None
    route: Route | None
    planner_output: dict[str, Any] | None

    # --- Set by Agent 2 (Code Executor) in Phase 4+ ---
    execution_result: dict[str, Any] | None
    execution_attempts: int
    max_execution_attempts: int

    # --- Set by Document RAG pipeline in Phase 5+ ---
    retrieval_result: dict[str, Any] | None
    cache_hit: bool

    # --- Set by Agent 3 (Data Analyst) in Phase 6+ ---
    analyst_output: dict[str, Any] | None

    # --- Trace, timings, model info ---
    metadata: dict[str, Any]
