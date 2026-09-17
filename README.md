# Local RAG Platform

GitLab CI/CD -> GitLab Container Registry -> GitOps repo -> Argo CD -> k3s.
Agentic RAG via LangGraph. LLM via the Gemini API (OpenAI-compatible),
with automatic fallback to a local Ollama model if Gemini errors (rate limit, 5xx,
timeout). Embeddings always via Gemini. LLM calls traced with LangSmith (optional).
5 pods: `rag-api`, `ingestion-worker`, `qdrant`, `ollama`, `chat-ui` (Django,
serving a built Angular chat interface and proxying its API calls to `rag-api`
server-side — the browser never talks to `rag-api` directly).

## Repo layout
- `apps/rag-api` — FastAPI + LangGraph agentic RAG (retrieve -> generate), calls
  Gemini with Ollama fallback for chat, Gemini-only for embeddings.
- `apps/ingestion-worker` — OCR/parsing (pdfplumber/pytesseract/python-docx) + embedding + upsert into Qdrant.
- `apps/chat-ui` — Django app that serves the Angular chat UI (built at Docker
  build time, in `apps/chat-ui/angular-app/`) and proxies `/api/query` and
  `/api/ingest/upload` to `rag-api` server-side.
- `apps/mcp-server` — thin MCP proxy in front of `rag-api`'s `/query` and
  `/ingest/upload`, so MCP clients (Claude Code, Claude Desktop, other agents)
  can use this RAG platform as a tool. Not deployed to k3s — see "MCP server" below.
- `gitops/` — copy this into your **separate** GitOps repo (chart + Argo CD Applications).
- `nginx/`, `docker-compose.yml` — local exposure/testing without k3s.
- `.gitlab-ci.yml` — builds/pushes only the service(s) that changed, then bumps that service's tag in the GitOps repo.

## One-time setup

0. Install k3s (single node is fine for local use). k3s ships with Traefik as its
   ingress controller, which is what this chart is configured for by default —
   nothing extra to install:
   ```
   curl -sfL https://get.k3s.io | sh -
   sudo cat /etc/rancher/k3s/k3s.yaml   # kubeconfig — copy to ~/.kube/config or export KUBECONFIG
   kubectl get nodes                    # sanity check
   ```
   Then install Argo CD into the cluster:
   ```
   kubectl create namespace argocd
   kubectl apply -n argocd -f https://raw.githubusercontent.com/argoproj/argo-cd/stable/manifests/install.yaml
   kubectl -n argocd get pods -w        # wait until all Running
   ```

1. Local dev (no k3s needed for this step): `cp .env.example .env` and fill in
   `GEMINI_API_KEY` (key from https://aistudio.google.com/app/apikey).
   `LANGSMITH_API_KEY` is optional (account at https://smith.langchain.com) —
   leave blank to skip tracing entirely. No key needed for Ollama, it's local.
   `docker compose up --build` then open `http://localhost:4200` for the chat UI,
   or hit the API directly at `http://localhost:8080/query` (via nginx) or
   `http://localhost:8000/query` (direct). First startup pulls the Ollama model
   (~1.7GB, gemma2:2b) — this can take a minute or two.

2. CI/CD variables/secrets on the **app** repo — set these on whichever remote(s)
   you actually push to:

   **GitLab** (Settings > CI/CD > Variables, masked + protected):
   - `GITOPS_TOKEN`, `GITOPS_REPO_HOST`, `GITOPS_REPO_PATH`
   - (Container Registry vars are auto-provided by GitLab)

   **GitHub** (Settings > Secrets and variables > Actions), if mirroring this repo
   to GitHub too:
   - `GITOPS_TOKEN` — same GitLab PAT/Project Access Token as above (`write_repository`)
   - `GITOPS_REPO_HOST`, `GITOPS_REPO_PATH` — same values as above
   - Pushing to `ghcr.io` needs no extra secret — GitHub Actions' built-in
     `GITHUB_TOKEN` covers it automatically.

3. k3s secret (create once per namespace, keys never live in git):
   ```
   kubectl create ns rag-platform
   kubectl create secret generic rag-secrets -n rag-platform \
     --from-literal=GEMINI_API_KEY=xxxx \
     --from-literal=LANGSMITH_API_KEY=xxxx
   kubectl create ns rag-platform-test
   kubectl create secret generic rag-secrets -n rag-platform-test \
     --from-literal=GEMINI_API_KEY=xxxx \
     --from-literal=LANGSMITH_API_KEY=xxxx
   ```
   `LANGSMITH_API_KEY` is optional — omit the flag entirely to run without tracing.
   You can optionally also add `--from-literal=DJANGO_SECRET_KEY=<random-string>`
   for the chat UI's Django backend — it falls back to an insecure dev key if
   omitted, which is fine here since this proxy has no sessions/auth/cookies that
   would depend on it, but a real value is good practice regardless.

4. Push `gitops/charts/rag-platform`, `gitops/argocd`, and `gitops/README.md` to
   your GitOps repo — see `gitops/README.md` for the full first-time setup checklist
   (placeholders to replace, secrets, Argo CD repo registration). Then:
   ```
   kubectl apply -f argocd/application-test.yaml
   kubectl apply -f argocd/application-main.yaml
   ```

## Flow
Push to `test` or `main` branch of the **app** repo -> CI (GitLab CI and/or GitHub
Actions, whichever remote you push to) detects which service(s) actually changed
(`apps/rag-api`, `apps/ingestion-worker`, `apps/chat-ui` are built independently —
a change to one doesn't rebuild the others) -> builds/pushes only those images ->
bumps only their tag(s) in the GitOps repo's `values-<branch>.yaml`, commits ->
Argo CD (auto-sync) rolls out the new image(s) to the matching namespace on k3s.

## Query & ingest documents
Add `rag.local` (or `rag-test.local`) to `/etc/hosts` pointing at your k3s node IP,
or `curl --resolve` it, since Traefik routes by Host header.

Ask a question:
```
curl -X POST http://rag.local/query -d '{"question":"..."}' -H 'Content-Type: application/json'
```

Ingest a document straight from your machine (PDF, DOCX, PNG/JPG, or TXT/MD):
```
curl -X POST http://rag.local/ingest/upload -F 'file=@/path/to/document.pdf'
```

Alternatively, batch-ingest every file in the documents directory on the k3s node
(default `/srv/rag-platform/documents`, configurable via `ingestionWorker.hostPath`
in the Helm chart — see `gitops/README.md`). Drop files in (subdirectories are
walked too), then trigger a batch run — no path argument needed, it processes
everything it finds:
```
curl -X POST http://rag.local/ingest
```
Response includes a per-file chunk count and any files that failed to parse:
```json
{"files_ingested": 3, "chunks_per_file": {"reports/q3.pdf": 12}, "files_failed": {}}
```
For local `docker compose` dev, drop files into `./documents/` in this repo instead
(bind-mounted to the same place).

## Chat UI

`apps/chat-ui` is a Django app that serves a built Angular chat window plus an
"Add document" button, and proxies their API calls to `rag-api` itself:

- Angular calls `/api/query` and `/api/ingest/upload` — same origin, relative paths.
- Django's `ragproxy` app forwards those server-side to `rag-api`'s `/query` and
  `/ingest/upload` (`RAG_API_URL` env var, defaults to `http://rag-api:8000`).
- Django also serves the Angular static files (via WhiteNoise) and falls back to
  `index.html` for any unmatched path, so the Angular app loads correctly however
  the URL is reached.

**Local dev:** `docker compose up --build` and open `http://localhost:4200`.

**k3s:** the UI is served from its own Ingress host —
add it to `/etc/hosts` pointing at your k3s node IP:
```
<node-ip>  rag-chat.local
```
(swap in `rag-chat-test.local` for the test namespace). Open
`http://rag-chat.local` in a browser. `rag.local` (routing straight to `rag-api`)
is still available too, for direct `curl`-based testing as shown above.

**Rebuilding after an Angular change:** the Angular app lives in
`apps/chat-ui/angular-app/` and is built as part of `apps/chat-ui`'s own Docker
build (multi-stage: Node builds Angular, then it's copied into the Django image)
— there's no separate image or deploy step for the frontend.

## MCP server

`apps/mcp-server` exposes three tools that proxy straight to `rag-api` — it
carries no RAG logic of its own, same requests/responses as curling
`rag-api` directly, just reachable from any MCP client instead of only
`curl`/the chat UI:
- `query(question)` -> `rag-api`'s `/query`
- `ingest_document(file_path)` -> `rag-api`'s `/ingest/upload`, for a single
  file already on disk
- `ingest_all()` -> `rag-api`'s `/ingest`, batch-ingesting whatever's already
  sitting in the ingestion-worker's mounted documents directory

It talks stdio today — the MCP host (Claude Code, Claude Desktop, ...) spawns
it as a local subprocess and pipes JSON-RPC over its stdin/stdout, no port or
network involved. It gets its own venv so it doesn't depend on whatever
happens to be on your global `python3`:
```
python3 -m venv apps/mcp-server/.venv
apps/mcp-server/.venv/bin/pip install -r apps/mcp-server/requirements.txt
```
The repo's `.mcp.json` already points Claude Code at that venv's interpreter
(`apps/mcp-server/.venv/bin/python`) — `RAG_API_URL` defaults to
`http://rag.local` (the k3s Ingress host); override it to
`http://localhost:8000` for a `docker compose` setup, or to
`http://rag-test.local` to point at the test namespace instead. For Claude
Desktop, add the same `command`/`args`/`env` under `mcpServers` in its own
config file (adjust the interpreter path if the venv lives somewhere else on
that machine).

`ingest_document` takes a `file_path` — it must be readable from wherever
the MCP server process itself runs (your machine, for a local stdio
server), not necessarily from wherever the MCP client is. Passing raw
content instead (e.g. base64) was tried and reverted: pushing file bytes
through a tool-call argument means an LLM client has to read, chunk, and
re-emit the entire payload as text to make the call, which is slow, easy
to get wrong on anything but small files, and unnecessary as long as
client and server share a filesystem.

## Gemini fallback to local Ollama

`rag-api` calls Gemini for chat/generation first. If that call raises **any**
error — rate limit (429), server error (5xx), timeout, network failure — it
automatically retries the same request against a local Ollama model instead, so
`/query` keeps working even if Gemini's rate limit is throttling you. Embeddings
always go through Gemini (switching embedding models would break similarity search
against existing Qdrant vectors, so there's no embeddings fallback).

Default model is **gemma2:2b** (~1.7GB at Q4 quantization) — the smallest model
that's still genuinely usable for RAG answer generation, chosen to leave real
headroom in an 8GB total RAM budget shared with the other 3 pods. If you have
more headroom and want better fallback quality, swap `ollama.model` in
`values.yaml` (or `OLLAMA_MODEL` in `.env` for local dev) for something larger,
e.g. `phi4-mini` (~3.2GB, stronger reasoning).

GPU passthrough is **not** configured by default (`ollama.gpu: false`) — Ollama
runs on CPU unless enabled.

**k3s**: this chart includes the `RuntimeClass` and NVIDIA device plugin needed —
see `gitops/README.md`'s "Enabling GPU inference" section for the node-level
driver/toolkit setup steps that can't be automated by Helm.

**Docker Compose**: install the [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html)
on the Docker host first, then run with the GPU overlay:
```
docker compose -f docker-compose.yml -f docker-compose.gpu.yml up --build
```
`docker-compose.gpu.yml` is a separate file, not merged into the base
`docker-compose.yml`, because the GPU reservation makes `docker compose up` fail
outright on a machine with no NVIDIA driver registered — it doesn't gracefully
fall back to CPU. Keeping it as an opt-in overlay means the default
`docker compose up --build` still works on any machine.

CPU inference on gemma2:2b is slower but functional either way. First pull of the
model (either environment) takes a minute or two and is cached afterward (Docker
named volume / k3s PVC).

## Monitoring with LangSmith

Every chat/embedding call is traced via LangSmith's `@traceable` decorator (works
independently of LangGraph's own tracing, since the LLM clients use the raw
`openai` SDK). Traces show which provider actually served each request — you'll
see `chat-gemini` spans normally, and `chat-ollama-fallback` spans whenever the
fallback triggers, so you can monitor how often you're hitting Gemini's rate limit.

Set `LANGSMITH_API_KEY` to enable it; leave blank to disable tracing with zero
code changes. Sign up at https://smith.langchain.com
