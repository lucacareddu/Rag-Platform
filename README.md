# Local, Free RAG Platform

GitLab CI/CD -> GitLab Container Registry -> GitOps repo -> Argo CD -> K3s.
Agentic RAG via LangGraph. LLM + embeddings via GitHub Models API (free tier).
Only 3 pods: `rag-api`, `ingestion-worker`, `qdrant`.

## Repo layout
- `apps/rag-api` — FastAPI + LangGraph agentic RAG (retrieve -> generate), calls GitHub Models.
- `apps/ingestion-worker` — OCR/parsing (unstructured) + embedding + upsert into Qdrant.
- `gitops/` — copy this into your **separate** GitOps repo (chart + Argo CD Applications).
- `nginx/`, `docker-compose.yml` — local exposure/testing without k3s.
- `.gitlab-ci.yml` — builds/pushes images, then bumps image tag in the GitOps repo.

## One-time setup
1. Local dev: `cp .env.example .env` and fill in `GITHUB_TOKEN` (GitHub Models PAT).
   `docker compose up --build` then hit `http://localhost:8080/query`.

2. GitLab CI/CD variables (masked/protected), on the **app** repo:
   - `GITOPS_TOKEN`, `GITOPS_REPO_HOST`, `GITOPS_REPO_PATH`
   - (Container Registry vars are auto-provided by GitLab)

3. K3s secret (create once per namespace, keys never live in git):
   ```
   kubectl create ns rag-platform
   kubectl create secret generic rag-secrets -n rag-platform \
     --from-literal=GITHUB_TOKEN=xxxx
   kubectl create ns rag-platform-test
   kubectl create secret generic rag-secrets -n rag-platform-test \
     --from-literal=GITHUB_TOKEN=xxxx
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

## Query
```
curl -X POST http://rag.local/query -d '{"question":"..."}' -H 'Content-Type: application/json'
curl -X POST http://rag.local/ingest -d '{"source_path":"/data/file.pdf"}' -H 'Content-Type: application/json'
```
