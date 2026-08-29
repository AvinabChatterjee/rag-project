import asyncio
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from app.api.routes import AskRequest, ask
from app.graph.workflow import rag_graph


def _visited_nodes(final_state: dict) -> list[str]:
    trace = final_state.get("metadata", {}).get("agent_trace", [])
    return [entry["node"] for entry in trace]


class Phase2WorkflowTests(unittest.TestCase):
  def test_tabular_stub_path(self) -> None:
    with tempfile.TemporaryDirectory() as folder:
      Path(folder, "sales.csv").write_text("region,revenue\nNorth,100\n", encoding="utf-8")

      final_state = asyncio.run(
        rag_graph.ainvoke(
          {
            "user_question": "What is total revenue?",
            "data_folder": folder,
          }
        )
      )

    self.assertEqual(final_state["status"], "completed")
    self.assertEqual(final_state["route"], "tabular")
    self.assertIsNotNone(final_state.get("analyst_output", {}).get("final_answer"))

    visited = _visited_nodes(final_state)
    self.assertEqual(
      visited,
      [
        "init_workflow",
        "query_planner",
        "inspect_dataset",
        "generate_pandas",
        "code_executor",
        "data_analyst",
      ],
    )

  def test_document_stub_path(self) -> None:
    with tempfile.TemporaryDirectory() as folder:
      pdf_path = Path(folder, "policy.pdf")
      pdf_path.write_bytes(b"%PDF-1.4\n")
      fake_chunks = [
        {
          "chunk_id": "chunk-1",
          "text": "Plastics must be recyclable by 2026.",
          "score": 0.9,
          "file_path": str(pdf_path.resolve()),
        }
      ]

      with patch(
        "app.graph.nodes.retriever.retrieve_chunks",
        return_value=fake_chunks,
      ):
        final_state = asyncio.run(
          rag_graph.ainvoke(
            {
              "user_question": "What is the plastics policy?",
              "data_folder": folder,
            }
          )
        )

    self.assertEqual(final_state["status"], "completed")
    self.assertEqual(final_state["route"], "document")
    self.assertIsNotNone(final_state.get("analyst_output", {}).get("final_answer"))

    visited = _visited_nodes(final_state)
    self.assertEqual(
      visited,
      [
        "init_workflow",
        "query_planner",
        "cache_lookup",
        "retriever",
        "reranker",
        "llm_answer",
        "cache_store",
        "data_analyst",
      ],
    )

  def test_ask_endpoint_returns_completed_response(self) -> None:
    with tempfile.TemporaryDirectory() as folder:
      Path(folder, "sales.csv").write_text("region,revenue\nNorth,100\n", encoding="utf-8")

      response = asyncio.run(
        ask(
          AskRequest(
            question="What is total revenue?",
            data_folder=folder,
          )
        )
      )

    self.assertEqual(response.status, "completed")
    self.assertEqual(response.route, "tabular")
    self.assertIsNotNone(response.answer)
    self.assertIsNotNone(response.analyst_output)


if __name__ == "__main__":
  unittest.main()
