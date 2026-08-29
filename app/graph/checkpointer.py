from __future__ import annotations

import aiosqlite
from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver

from app.config import get_settings

_checkpointer: AsyncSqliteSaver | None = None
_connection: aiosqlite.Connection | None = None


async def init_checkpointer() -> AsyncSqliteSaver:
    global _checkpointer, _connection
    if _checkpointer is not None:
        return _checkpointer

    settings = get_settings()
    settings.workflow_db_path.parent.mkdir(parents=True, exist_ok=True)
    _connection = await aiosqlite.connect(str(settings.workflow_db_path))
    _checkpointer = AsyncSqliteSaver(_connection)
    return _checkpointer


def get_checkpointer() -> AsyncSqliteSaver | None:
    return _checkpointer


async def close_checkpointer() -> None:
    global _checkpointer, _connection
    if _connection is not None:
        await _connection.close()
    _checkpointer = None
    _connection = None
