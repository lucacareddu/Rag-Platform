# Local, Free RAG Platform

GitLab CI/CD -> GitLab Container Registry -> GitOps repo -> Argo CD -> k3s.
Agentic RAG via LangGraph. LLM via the Gemini API (free tier, OpenAI-compatible),
with automatic fallback to a local Ollama model if Gemini errors (rate limit, 5xx,
timeout). Embeddings always via Gemini. LLM calls traced with LangSmith (optional).
4 pods: `rag-api`, `ingestion-worker`, `qdrant`, `ollama`.

## Repo layout
- `apps/rag-api` — FastAPI + LangGraph agentic RAG (retrieve -> generate), calls
  Gemini with Ollama fallback for chat, Gemini-only for embeddings.
- `apps/ingestion-worker` — OCR/parsing (pdfplumber/pytesseract/python-docx) + embedding + upsert into Qdrant.
- `gitops/` — copy this into your **separate** GitOps repo (chart + Argo CD Applications).
- `nginx/`, `docker-compose.yml` — local exposure/testing without k3s.
- `.gitlab-ci.yml` — builds/pushes images, then bumps image tag in the GitOps repo.

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
   `GEMINI_API_KEY` (free key from https://aistudio.google.com/app/apikey).
   `LANGSMITH_API_KEY` is optional (free account at https://smith.langchain.com) —
   leave blank to skip tracing entirely. No key needed for Ollama, it's local.
   `docker compose up --build` then hit `http://localhost:8080/query`. First
   startup pulls the Ollama model (~1.7GB, gemma2:2b) — this can take a minute or two.

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

4. Push `gitops/charts/rag-platform`, `gitops/argocd`, and `gitops/README.md` to
   your GitOps repo — see `gitops/README.md` for the full first-time setup checklist
   (placeholders to replace, secrets, Argo CD repo registration). Then:
   ```
   kubectl apply -f argocd/application-test.yaml
   kubectl apply -f argocd/application-main.yaml
   ```

## Flow
Push to `test` or `main` branch of the **app** repo -> CI (GitLab CI and/or GitHub
Actions, whichever remote you push to) builds/pushes both images -> the pipeline
clones the GitOps repo, bumps `values-<branch>.yaml` image registry/tag, commits
-> Argo CD (auto-sync) rolls out the new image to the matching namespace on k3s.

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

## Gemini fallback to local Ollama

`rag-api` calls Gemini for chat/generation first. If that call raises **any**
error — rate limit (429), server error (5xx), timeout, network failure — it
automatically retries the same request against a local Ollama model instead, so
`/query` keeps working even if the Gemini free tier is throttling you. Embeddings
always go through Gemini (switching embedding models would break similarity search
against existing Qdrant vectors, so there's no embeddings fallback).

Default model is **gemma2:2b** (~1.7GB at Q4 quantization) — the smallest model
that's still genuinely usable for RAG answer generation, chosen to leave real
headroom in an 8GB total RAM budget shared with the other 3 pods. If you have
more headroom and want better fallback quality, swap `ollama.model` in
`values.yaml` (or `OLLAMA_MODEL` in `.env` for local dev) for something larger,
e.g. `phi4-mini` (~3.2GB, stronger reasoning).

GPU passthrough is **not** configured by default (`ollama.gpu: false`) — Ollama
runs on CPU unless you've set up the NVIDIA Container Toolkit (Docker Compose) or
the NVIDIA device plugin (k3s) yourself; CPU inference on gemma2:2b is slower but
functional. First pull of the model (either environment) takes a minute or two and
is cached afterward (Docker named volume / k3s PVC).

## Monitoring with LangSmith

Every chat/embedding call is traced via LangSmith's `@traceable` decorator (works
independently of LangGraph's own tracing, since the LLM clients use the raw
`openai` SDK). Traces show which provider actually served each request — you'll
see `chat-gemini` spans normally, and `chat-ollama-fallback` spans whenever the
fallback triggers, so you can monitor how often you're hitting Gemini's rate limit.

Set `LANGSMITH_API_KEY` to enable it; leave blank to disable tracing with zero
code changes. Free tier account: https://smith.langchain.com
