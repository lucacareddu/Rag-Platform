# Local, Free RAG Platform

GitLab CI/CD -> GitLab Container Registry -> GitOps repo -> Argo CD -> k3s.
Agentic RAG via LangGraph. LLM + embeddings via the Gemini API (free tier, OpenAI-compatible).
Only 3 pods: `rag-api`, `ingestion-worker`, `qdrant`.

## Repo layout
- `apps/rag-api` — FastAPI + LangGraph agentic RAG (retrieve -> generate), calls Gemini.
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
   `docker compose up --build` then hit `http://localhost:8080/query`.

2. GitLab CI/CD variables (masked/protected), on the **app** repo:
   - `GITOPS_TOKEN`, `GITOPS_REPO_HOST`, `GITOPS_REPO_PATH`
   - (Container Registry vars are auto-provided by GitLab)

3. k3s secret (create once per namespace, keys never live in git):
   ```
   kubectl create ns rag-platform
   kubectl create secret generic rag-secrets -n rag-platform \
     --from-literal=GEMINI_API_KEY=xxxx
   kubectl create ns rag-platform-test
   kubectl create secret generic rag-secrets -n rag-platform-test \
     --from-literal=GEMINI_API_KEY=xxxx
   ```

4. Push `gitops/charts/rag-platform` and `gitops/argocd` to your GitOps repo, then:
   ```
   kubectl apply -f argocd/application-test.yaml
   kubectl apply -f argocd/application-main.yaml
   ```

## Flow
Push to `test` or `main` branch of the **app** repo -> GitLab CI builds/pushes both
images -> CI clones the GitOps repo, bumps `values-<branch>.yaml` image tag, commits
-> Argo CD (auto-sync) rolls out the new tag to the matching namespace on k3s.

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

Alternatively, if the file is already on a volume mounted into the `ingestion-worker`
pod, ingest by path instead:
```
curl -X POST http://rag.local/ingest -d '{"source_path":"/data/file.pdf"}' -H 'Content-Type: application/json'
```
