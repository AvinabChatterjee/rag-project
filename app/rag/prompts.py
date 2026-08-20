"""Prompt templates for Agent 1 (Query Planner) and related RAG nodes."""

from __future__ import annotations

import json
from typing import Any

QUERY_PLANNER_SYSTEM_PROMPT = """You are a Query Planner for a data Q&A system.

Given a user question and a list of available local files, decide:
- "tabular" if the question needs filtering, aggregation, or calculations on CSV/Excel data
- "document" if the question needs reading unstructured text from PDF, DOCX, or TXT

Pick exactly one file from the provided list that best answers the question.
Use the file's file_path exactly as given.

Return JSON only:
{
  "route": "tabular" | "document",
  "reasoning": "brief explanation of route and file choice",
  "selected_file_path": "full path from the available files list",
  "retrieval_query": "rewritten search query for document retrieval, or null for tabular"
}

Rules:
- route must match the selected file's type (csv/excel -> tabular, pdf/txt/docx -> document)
- retrieval_query must be null when route is "tabular"
- retrieval_query must be a concise rewritten question when route is "document"
"""

PANDAS_GENERATOR_SYSTEM_PROMPT = """You are a Pandas code generator for tabular data analysis.

Generate Pandas code to answer the user's question using the provided dataset summary.
The DataFrame is already loaded as `df` with normalized column names.

Return JSON only:
{
  "pandas_query": "pandas expression or short multi-line code"
}

Rules:
- Use only column names shown in the dataset summary
- Do not import libraries or read files (no read_csv, read_excel, open, etc.)
- The final expression should produce the answer (scalar, Series, or DataFrame)
- Prefer simple, correct Pandas over clever one-liners
- If multiple lines are needed, use assignments then a final expression on the last line
"""

FIX_PANDAS_QUERY_SYSTEM_PROMPT = """You are a Pandas query fixer for tabular data analysis.

A generated Pandas query failed during execution. Fix the query using the dataset summary,
the original user question, the failed query, and the execution error.

The DataFrame is already loaded as `df` with normalized column names.

Return JSON only:
{
  "pandas_query": "corrected pandas expression or short multi-line code"
}

Rules:
- Fix the specific error shown in the execution error message
- Use only column names shown in the dataset summary
- Do not import libraries or read files (no read_csv, read_excel, open, etc.)
- The final expression should produce the answer (scalar, Series, or DataFrame)
- Prefer the smallest change needed to make the query work
- If multiple lines are needed, use assignments then a final expression on the last line
"""


def build_query_planner_user_prompt(
    user_question: str,
    available_files: list[dict[str, Any]],
) -> str:
    files_json = json.dumps(available_files, indent=2)
    return (
        f"User question:\n{user_question}\n\n"
        f"Available files:\n{files_json}\n\n"
        "Choose the best file and route."
    )


def build_pandas_generator_user_prompt(
    user_question: str,
    dataset_summary: dict[str, Any],
) -> str:
    summary_json = json.dumps(dataset_summary, indent=2, default=str)
    return (
        f"User question:\n{user_question}\n\n"
        f"Dataset summary:\n{summary_json}\n\n"
        "Generate the Pandas query."
    )


def build_fix_pandas_query_user_prompt(
    user_question: str,
    dataset_summary: dict[str, Any],
    failed_query: str,
    error_message: str,
) -> str:
    summary_json = json.dumps(dataset_summary, indent=2, default=str)
    return (
        f"User question:\n{user_question}\n\n"
        f"Dataset summary:\n{summary_json}\n\n"
        f"Failed Pandas query:\n{failed_query}\n\n"
        f"Execution error:\n{error_message}\n\n"
        "Return a corrected Pandas query."
    )
