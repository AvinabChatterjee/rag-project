from datetime import datetime, timezone
from typing import Any
from uuid import uuid4

from app.config import get_settings
from app.graph.state import WorkflowState
from app.utils.folder_utils import scan_data_folder


def init_workflow_node(state: WorkflowState) -> dict[str, Any]:
    """
    First graph node — creates workflow JSON and scans the fixed data folder.

    Expects user_question in state. data_folder comes from state override or .env.
    """
    settings = get_settings()
    folder_override = state.get("data_folder")
    folder = settings.resolve_data_folder(folder_override)

    available_files = scan_data_folder(folder)
    if not available_files:
        raise ValueError(
            f"No supported files found in data folder: {folder}. "
            f"Add .csv, .xlsx, .xls, .pdf, .txt, or .docx files."
        )

    trace_entry = {
        "node": "init_workflow",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "files_found": len(available_files),
        "data_folder": str(folder),
    }

    return {
        "workflow_id": str(uuid4()),
        "created_at": datetime.now(timezone.utc).isoformat(),
        "status": "initialized",
        "data_folder": str(folder),
        "available_files": available_files,
        "selected_file_path": None,
        "selected_file_type": None,
        "route": None,
        "planner_output": None,
        "execution_result": None,
        "retrieval_result": None,
        "analyst_output": None,
        "cache_hit": False,
        "execution_attempts": 0,
        "max_execution_attempts": 2,
        "metadata": {
            "model": settings.openai_model,
            "agent_trace": [trace_entry],
        },
    }
