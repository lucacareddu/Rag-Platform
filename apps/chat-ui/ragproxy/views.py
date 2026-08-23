import json

import httpx
from django.conf import settings
from django.http import JsonResponse
from django.views.decorators.http import require_POST


@require_POST
def query(request):
    """Proxies to rag-api's /query. angular calls this same-origin path instead
    of rag-api directly."""
    try:
        payload = json.loads(request.body)
    except json.JSONDecodeError:
        return JsonResponse({"error": "invalid JSON body"}, status=400)

    question = payload.get("question", "").strip()
    if not question:
        return JsonResponse({"error": "question is required"}, status=400)

    try:
        resp = httpx.post(
            f"{settings.RAG_API_URL}/query",
            json={"question": question},
            timeout=120,
        )
        resp.raise_for_status()
    except httpx.HTTPError as e:
        return JsonResponse({"error": f"rag-api request failed: {e}"}, status=502)

    return JsonResponse(resp.json())


@require_POST
def ingest_upload(request):
    """Proxies a multipart file upload to rag-api's /ingest/upload."""
    upload = request.FILES.get("file")
    if not upload:
        return JsonResponse({"error": "file is required"}, status=400)

    try:
        resp = httpx.post(
            f"{settings.RAG_API_URL}/ingest/upload",
            files={"file": (upload.name, upload.read(), upload.content_type)},
            timeout=600,
        )
        resp.raise_for_status()
    except httpx.HTTPError as e:
        return JsonResponse({"error": f"rag-api request failed: {e}"}, status=502)

    return JsonResponse(resp.json())
