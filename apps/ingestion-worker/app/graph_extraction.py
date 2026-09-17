import json
import logging

from openai import OpenAI
from .config import settings

logger = logging.getLogger(__name__)

client = OpenAI(base_url=settings.gemini_base_url, api_key=settings.gemini_api_key)

PROMPT = """Extract a knowledge graph from the document below.

Return JSON with exactly this shape:
{
  "entities": [{"name": "...", "type": "PERSON|ORG|PRODUCT|TECHNOLOGY|LOCATION|EVENT|CONCEPT"}],
  "relations": [{"source": "...", "type": "SHORT_UPPER_SNAKE_VERB", "target": "..."}]
}

Rules:
- Use the entity's most complete surface form as "name" (e.g. "Luca Careddu", not "Luca"),
  and reuse that exact same string everywhere it appears in "relations".
- "source" and "target" must both appear in "entities".
- Only relations actually stated in the document. Do not infer or invent.
- Skip generic filler entities ("the company", "the system", "this document").

Document:
"""


def extract_entities_and_relations(text: str) -> tuple[list[dict], list[dict]]:
    """Extracts over the whole document in overlapping windows, then merges —
    a chunk-level pass would fragment entities/relations across chunk boundaries."""
    windows = _windows(text)
    if not windows:
        return [], []

    entities: dict[str, dict] = {}
    relations: dict[tuple, dict] = {}
    for i, window in enumerate(windows):
        try:
            w_entities, w_relations = _extract_window(window)
        except Exception as e:
            # One bad window shouldn't cost the whole document its graph.
            logger.warning("Extraction failed on window %d/%d (%s), continuing", i + 1, len(windows), e)
            continue
        for e in w_entities:
            entities.setdefault(e["name"].lower(), e)
        for r in w_relations:
            relations.setdefault((r["source"].lower(), r["type"], r["target"].lower()), r)

    return list(entities.values()), list(relations.values())


def _windows(text: str) -> list[str]:
    """Overlapping fixed-size windows covering the entire text."""
    if not text.strip():
        return []
    size = settings.extract_window_chars
    step = max(1, size - settings.extract_window_overlap)
    windows = [text[s:s + size] for s in range(0, len(text), step)]
    # range() emits a final start inside the previous window's tail whenever the
    # text doesn't divide evenly; that trailing window is already covered.
    windows = [w for w in windows if w.strip()]
    if len(windows) > settings.extract_max_windows:
        logger.warning(
            "Document needs %d extraction windows, capping at %d — the tail will not "
            "reach the graph. Raise extract_max_windows to cover it.",
            len(windows), settings.extract_max_windows,
        )
        windows = windows[:settings.extract_max_windows]
    return windows


def _extract_window(text: str) -> tuple[list[dict], list[dict]]:
    resp = client.chat.completions.create(
        model=settings.extract_model,
        messages=[{"role": "user", "content": PROMPT + text}],
        response_format={"type": "json_object"},
    )
    raw = resp.choices[0].message.content or ""

    try:
        data = json.loads(_strip_fences(raw))
    except json.JSONDecodeError:
        logger.warning("Graph extraction returned non-JSON for a window, skipping it")
        return [], []

    return _clean_entities(data.get("entities")), _clean_relations(data.get("relations"))


def _strip_fences(raw: str) -> str:
    """Models occasionally wrap JSON in ```json fences despite response_format."""
    s = raw.strip()
    if s.startswith("```"):
        s = s.split("\n", 1)[-1].rsplit("```", 1)[0]
    return s.strip()


def _clean_entities(entities) -> list[dict]:
    if not isinstance(entities, list):
        return []
    seen, out = set(), []
    for e in entities:
        if not isinstance(e, dict):
            continue
        name = str(e.get("name", "")).strip()
        # Single characters and empty names are always extraction noise.
        if len(name) < 2 or name.lower() in seen:
            continue
        seen.add(name.lower())
        out.append({"name": name, "type": str(e.get("type", "")).strip().upper() or "CONCEPT"})
    return out


def _clean_relations(relations) -> list[dict]:
    if not isinstance(relations, list):
        return []
    seen, out = set(), []
    for r in relations:
        if not isinstance(r, dict):
            continue
        source = str(r.get("source", "")).strip()
        target = str(r.get("target", "")).strip()
        rel = str(r.get("type", "")).strip().upper().replace(" ", "_") or "RELATED_TO"
        if len(source) < 2 or len(target) < 2 or source.lower() == target.lower():
            continue
        key = (source.lower(), rel, target.lower())
        if key in seen:
            continue
        seen.add(key)
        out.append({"source": source, "type": rel, "target": target})
    return out
