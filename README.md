# Local RAG Platform

> **Experiment branch — not for production.** The handcrafted Qdrant + Neo4j
> retrieval arm, kept as the baseline the library implementations on
> `feat_neo4j` and `feat_graphrag` are measured against.

GitLab CI/CD -> GitLab Container Registry -> GitOps repo -> Argo CD -> k3s.
Agentic **GraphRAG** via LangGraph: vector retrieval from Qdrant, then expansion
through a Neo4j knowledge graph built from the same documents. LLM via the Gemini
API (OpenAI-compatible), with automatic fallback to a local Ollama model if Gemini
errors (rate limit, 5xx, timeout). Embeddings always via Gemini. LLM calls traced
with LangSmith (optional).
6 pods: `rag-api`, `ingestion-worker`, `qdrant`, `neo4j`, `ollama`, `chat-ui`
(Django, serving a built Angular chat interface and proxying its API calls to
`rag-api` server-side — the browser never talks to `rag-api` directly).

## Repo layout
- `apps/rag-api` — FastAPI + LangGraph agentic GraphRAG (retrieve -> graph -> generate),
  calls Gemini with Ollama fallback for chat, Gemini-only for embeddings.
- `apps/ingestion-worker` — OCR/parsing (pdfplumber/pytesseract/python-docx) + embedding
  + upsert into Qdrant + knowledge-graph extraction into Neo4j.
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
     --from-literal=LANGSMITH_API_KEY=xxxx \
     --from-literal=NEO4J_PASSWORD=xxxxxxxx \
     --from-literal=NEO4J_AUTH=neo4j/xxxxxxxx
   kubectl create ns rag-platform-test
   kubectl create secret generic rag-secrets -n rag-platform-test \
     --from-literal=GEMINI_API_KEY=xxxx \
     --from-literal=LANGSMITH_API_KEY=xxxx \
     --from-literal=NEO4J_PASSWORD=xxxxxxxx \
     --from-literal=NEO4J_AUTH=neo4j/xxxxxxxx
   ```
   `NEO4J_PASSWORD` must be **8+ characters** or the Neo4j container refuses to
   start. It appears twice on purpose: the apps read `NEO4J_PASSWORD`, while the
   Neo4j container itself wants user and password joined into a single
   `NEO4J_AUTH=neo4j/<password>` string, which can't be assembled from separate
   keys inside the chart. Keep the two in sync — same password in both.
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
Response includes per-file chunk and entity counts, plus any files that failed to parse:
```json
{"files_ingested": 3, "chunks_per_file": {"reports/q3.pdf": 12},
 "entities_per_file": {"reports/q3.pdf": 27}, "files_failed": {}}
```
For local `docker compose` dev, drop files into `./documents/` in this repo instead
(bind-mounted to the same place).

## GraphRAG (Neo4j)

Qdrant still does vector search exactly as before. Neo4j holds a knowledge graph
built from the same documents, and retrieval walks from the vector hits into that
graph before generating. The two stores are joined by a **shared chunk id**:
`ingestion-worker` derives each point id deterministically
(`uuid5(doc_id + chunk_index)`) and writes a `(:Chunk)` node under the same id,
so a Qdrant hit maps straight onto a graph node.

**At ingest** (`ingestion-worker`):
1. OCR/parse -> chunk -> embed -> upsert to Qdrant (as before, now with a
   `{doc_id, chunk_index}` payload). Embedding goes out in batches of
   `EMBED_BATCH_SIZE` (default 32): the API caps inputs per request, and a
   250-page manual is ~850 chunks, so one request per document fails outright.
2. Entities and relations are extracted by Gemini over **successive overlapping
   windows covering the entire document** (`EXTRACT_WINDOW_CHARS`, default 60k,
   with a 2k overlap so a relation straddling a boundary is still seen whole),
   then merged and deduplicated. Every page contributes, not just the opening
   ones — a 250-page manual takes 12 windows.
   Windows are a per-request size limit, *not* a return to chunk-level
   extraction: an 800-char chunk pass fragments the graph, since the same entity
   comes back under different surface forms and relations crossing a boundary are
   never seen. A 60k window is large enough that entities and their relations
   almost always sit inside one window. It is also deliberately far below the
   model's context limit — asking for entities over a huge span makes the model
   skim and silently drop them, so the constraint is extraction quality, not
   context size.
3. The result is written as
   `(:Document)-[:HAS_CHUNK]->(:Chunk)-[:MENTIONS]->(:Entity)`, with
   `(:Entity)-[:RELATES {type}]->(:Entity)` for the relations and `[:NEXT]`
   edges between consecutive chunks. Entities extracted at document level are
   linked back down to the chunks that mention them by surface-form match.

**At query time** (`rag-api`'s workflow: `retrieve` -> `expand_graph` -> `fuse` ->
`generate`):
1. `retrieve`: vector top-k from Qdrant, as before.
2. `expand_graph`: anchor entities = entities mentioned by those chunks, plus
   **query-side entity linking** — the question's terms are also matched against a
   Neo4j fulltext index on `Entity.name`, adding entities the vector search
   missed. This is the cheap, graph-native half of query expansion: no extra LLM
   call, and it covers the case where the question names an entity explicitly but
   no chunk phrased it similarly enough to rank. (Generic multi-query rewriting
   isn't done — it costs an LLM call and a multiple of the vector searches per
   query for little gain at this corpus size.) From those entities it also pulls
   `GRAPH_MAX_CHUNKS` (default 6) chunks ranked by how many of them each chunk
   mentions — **not** limited to chunks vector search missed, on purpose (see
   next step).
3. `fuse`: the vector list and the graph list are combined via **Reciprocal Rank
   Fusion** rather than concatenated as two separate blocks — each chunk's score
   is `1/(k+rank)` summed across whichever list(s) it appears in. Cosine
   similarity and entity-overlap counts aren't on a comparable scale, so RRF
   compares rank positions instead of raw scores. Because the graph list isn't
   restricted to new chunks, a chunk *both* retrievers rank highly gets a
   genuinely summed, higher score — real cross-retriever consensus, not just an
   interleave of two disjoint lists.
4. `generate`: the fused chunks plus the entity/relation subgraph (as separate
   narrative text, since those aren't retrieval candidates to rerank) are handed
   to the LLM.

**Degradation is deliberate**: if Neo4j is unreachable, extraction returns junk, or
`GRAPH_ENABLED=false`, ingest still writes vectors and `/query` still answers from
vector context alone — the graph only ever *adds* context, it's never required.
Same idea as the Gemini->Ollama fallback on the generation side.

Check the graph is actually being populated:
```
curl http://rag.local/graph/stats
```
```json
{"enabled": true, "documents": 3, "chunks": 41, "entities": 68, "relations": 52}
```

Re-ingesting the same file is now **idempotent** — `doc_id` is a content hash, point
ids are derived from it, and both stores prune chunks left over from a longer
previous version. Editing a file makes a new `doc_id`, so the old version's chunks
stay until you delete them; there's no cleanup of orphaned entities yet either.

Browse the graph visually with Neo4j Browser at `http://localhost:7474` under
`docker compose`, or via port-forward on k3s:
```
kubectl port-forward -n rag-platform svc/neo4j 7474:7474 7687:7687
```
It's intentionally **not** exposed through the Ingress. Useful starting query:
```cypher
MATCH (e:Entity)-[r:RELATES]->(t:Entity) RETURN e, r, t LIMIT 100
```

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
headroom in an 8GB total RAM budget shared with the other 4 pods. If you have
more headroom and want better fallback quality, swap `ollama.model` in
`values.yaml` (or `OLLAMA_MODEL` in `.env` for local dev) for something larger,
e.g. `phi4-mini` (~3.2GB, stronger reasoning).

GPU passthrough is **on** by default (`ollama.gpu: true` in `values.yaml`).

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
named volume / k3s PVC). After the pull, the model is also loaded into memory
immediately (`OLLAMA_KEEP_ALIVE=-1`, plus a warmup call at startup) so the first
real request isn't the one paying for the load.

## Monitoring with LangSmith

Every chat/embedding call is traced via LangSmith's `@traceable` decorator (works
independently of LangGraph's own tracing, since the LLM clients use the raw
`openai` SDK). Traces show which provider actually served each request — you'll
see `chat-gemini` spans normally, and `chat-ollama-fallback` spans whenever the
fallback triggers, so you can monitor how often you're hitting Gemini's rate limit.

Set `LANGSMITH_API_KEY` to enable it; leave blank to disable tracing with zero
code changes. Sign up at https://smith.langchain.com

## Eval harness

- `eval/build_test_book.py` — builds the test book from the live corpus, storing
  gold contexts as resolvable chunk ids rather than pasted text, which
  re-chunking silently invalidated.
- `eval/compare.py` — vector-only vs vector+graph retrieval on five non-LLM ragas
  metrics (context precision/recall, string similarity, BLEU, ROUGE), so scoring
  costs no LLM quota.
- `eval/compare_deepeval.py` — same two retrievers scored with DeepEval's
  LLM-judged contextual metrics, judged locally by Ollama for the same reason.

Test books and result JSON are gitignored; rerun the scripts to regenerate them.
