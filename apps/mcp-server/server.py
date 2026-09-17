#!/usr/bin/env python3
"""MCP server exposing rag-api's /query, /ingest/upload and /ingest as tools.

Thin proxy only — no RAG logic lives here, no direct Qdrant/LLM access.
Every call is forwarded to an already-running rag-api instance over HTTP,
so this server has exactly the same behavior (and failure modes) as
curling rag-api directly. Point RAG_API_URL at whichever environment you
want an MCP client (Claude Desktop, Claude Code, or any other MCP host)
to talk to.

Runs over stdio, so it's launched as a local subprocess by the MCP host,
not deployed as a k8s pod — see the root .mcp.json for the Claude Code
wiring. ingest_document takes a filesystem path rather than base64 content:
the path must be readable from wherever this server process itself runs
(today, your machine, since the host launches it locally) — not
necessarily from wherever the MCP client is, if they ever differ.
"""
import os
from pathlib import Path

import httpx
from mcp.server.fastmcp import FastMCP

RAG_API_URL = os.environ.get("RAG_API_URL", "http://rag.local")
TIMEOUT = float(os.environ.get("RAG_API_TIMEOUT", "60"))

mcp = FastMCP("rag-platform")


@mcp.tool()
def query(question: str) -> dict:
    """Ask the RAG platform a question and get back a generated answer
    grounded on whatever's been ingested into its knowledge base."""
    resp = httpx.post(f"{RAG_API_URL}/query", json={"question": question}, timeout=TIMEOUT)
    resp.raise_for_status()
    return resp.json()


@mcp.tool()
def ingest_document(file_path: str) -> dict:
    """Ingest a single document (PDF, DOCX, PNG/JPG, or TXT/MD) into the RAG
    platform's knowledge base. file_path must be readable from wherever
    this MCP server process is running, not from the MCP client's machine
    if they differ."""
    path = Path(file_path).expanduser()
    if not path.is_file():
        raise FileNotFoundError(f"No such file: {path}")

    with path.open("rb") as f:
        resp = httpx.post(
            f"{RAG_API_URL}/ingest/upload",
            files={"file": (path.name, f)},
            timeout=max(TIMEOUT, 120),
        )
    resp.raise_for_status()
    return resp.json()


@mcp.tool()
def ingest_all() -> dict:
    """Batch-ingest every file already sitting in the ingestion-worker's
    mounted documents directory (not files sent through this tool — those
    dropped on disk out-of-band, e.g. via the hostPath volume or docker
    compose bind mount). No arguments: it processes everything it finds."""
    resp = httpx.post(f"{RAG_API_URL}/ingest", timeout=max(TIMEOUT, 600))
    resp.raise_for_status()
    return resp.json()


if __name__ == "__main__":
    mcp.run()
