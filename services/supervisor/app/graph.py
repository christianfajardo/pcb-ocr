"""LangGraph state graph definition for the OCR pipeline."""

from __future__ import annotations

from langgraph.graph import END, StateGraph

from .nodes import (
    glm_ocr_extract_node,
    ingest_pdf_node,
    pymupdf_extract_node,
    qwen_vl_extract_node,
    reconcile_node,
    tesseract_extract_node,
    validate_node,
)
from .state import PipelineState


def build_graph() -> StateGraph:
    """Build the OCR pipeline graph.

    Returns:
        Compiled StateGraph.
    """
    graph = StateGraph(PipelineState)

    # Add nodes
    graph.add_node("ingest_pdf", ingest_pdf_node)
    graph.add_node("tesseract_extract", tesseract_extract_node)
    graph.add_node("glm_ocr_extract", glm_ocr_extract_node)
    graph.add_node("qwen_vl_extract", qwen_vl_extract_node)
    graph.add_node("pymupdf_extract", pymupdf_extract_node)
    graph.add_node("reconcile", reconcile_node)
    graph.add_node("validate", validate_node)

    # Set entry point
    graph.set_entry_point("ingest_pdf")

    # Parallel fan-out: all 4 extraction nodes run after ingestion
    graph.add_edge("ingest_pdf", "tesseract_extract")
    graph.add_edge("ingest_pdf", "glm_ocr_extract")
    graph.add_edge("ingest_pdf", "qwen_vl_extract")
    graph.add_edge("ingest_pdf", "pymupdf_extract")

    # Fan-in: reconcile waits for all 4
    graph.add_edge("tesseract_extract", "reconcile")
    graph.add_edge("glm_ocr_extract", "reconcile")
    graph.add_edge("qwen_vl_extract", "reconcile")
    graph.add_edge("pymupdf_extract", "reconcile")

    # Validate -> END
    graph.add_edge("reconcile", "validate")
    graph.add_edge("validate", END)

    return graph


# Pre-compile the graph
graph = build_graph()
compiled = graph.compile()
