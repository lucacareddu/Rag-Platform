"""Collapses surface-form variants of one entity onto a canonical key, and drops extraction noise.
Regex-first: mechanical normalization, not judgement calls, so no LLM call is spent on it.
"""
import re

# "Protect (function)" -> "Protect"
_PARENTHETICAL = re.compile(r"\s*\([^)]*\)")
# "Framework ... Version 1.1" / "v1.1" -> "Framework ..."
_VERSION_SUFFIX = re.compile(r"\s+(?:version|v\.?)\s*\d+(?:\.\d+)*\s*$", re.I)
_LEADING_ARTICLE = re.compile(r"^(?:the|a|an)\s+", re.I)
# CSF calls "Protect" a Function; "Protect Function" and "Protect" are the same node.
_ROLE_SUFFIX = re.compile(
    r"\s+(?:functions?|categor(?:y|ies)|subcategor(?:y|ies)|tiers?|profiles?)\s*$", re.I
)
_PUNCT = re.compile(r"[^\w\s/.:-]")
_WS = re.compile(r"\s+")

# Too generic alone; stripping a role suffix must leave something specific behind.
_GENERIC_STEMS = {
    "framework", "cybersecurity", "security", "risk", "information",
    "implementation", "management", "system", "systems", "organization",
}

_MONTHS = (
    "january|february|march|april|may|june|july|august|september|october|november|december"
)
# Noise the extraction prompt asks the model to skip but that still slips through.
_NOISE = [
    re.compile(r"^\d{4}$"),                                   # "2017"
    re.compile(rf"^(?:{_MONTHS})\s+\d{{1,2}},?\s*\d{{4}}$", re.I),  # "December 5, 2017"
    re.compile(rf"^(?:{_MONTHS})\s+\d{{4}}$", re.I),          # "April 2018"
    re.compile(r"^[\d.,\s-]+$"),                              # pure numerics
    # Standards clause ids: "ISO/IEC 27001:2013 A.16.1.6", "PR.AC-1", "ID.AM-1"
    re.compile(r"^(?:[A-Z]{2}\.[A-Z]{2}(?:-\d+)?)$"),
    re.compile(r"\bA\.\d+\.\d+(?:\.\d+)?\s*$"),
    re.compile(r"^(?:ISO|IEC|NIST|COBIT|ISA)[\s/]*[\d\s:.-]+$", re.I),
]


def is_noise(name: str) -> bool:
    n = name.strip()
    return len(n) < 2 or any(p.search(n) for p in _NOISE)


def canonical_key(name: str) -> str:
    """The merge key. Two names sharing a key are treated as the same entity."""
    s = _PARENTHETICAL.sub("", name).strip()
    s = _VERSION_SUFFIX.sub("", s)
    s = _LEADING_ARTICLE.sub("", s)
    stripped = _ROLE_SUFFIX.sub("", s)
    # Only drop the role noun if something identifying is left behind.
    if stripped and not all(w.lower() in _GENERIC_STEMS for w in stripped.split()):
        s = stripped
    s = _PUNCT.sub(" ", s)
    s = _WS.sub(" ", s).strip().lower()
    s = s.replace(" & ", " and ")
    return s or name.strip().lower()


def resolve(entities: list[dict], relations: list[dict]) -> tuple[list[dict], list[dict]]:
    """Merges variant spellings onto one node and rewrites relations to match.
    Returns entities carrying an explicit `key`, so downstream writes no longer
    key on raw lowercased name."""
    by_key: dict[str, dict] = {}
    alias: dict[str, str] = {}

    for e in entities:
        name = str(e.get("name", "")).strip()
        if is_noise(name):
            continue
        key = canonical_key(name)
        if is_noise(key):
            continue
        alias[name.lower()] = key
        prev = by_key.get(key)
        # Keep the most complete surface form as the display name.
        if prev is None or len(name) > len(prev["name"]):
            by_key[key] = {"key": key, "name": name, "type": e.get("type", "CONCEPT")}

    out_rel, seen = [], set()
    for r in relations:
        src = str(r.get("source", "")).strip()
        tgt = str(r.get("target", "")).strip()
        s_key = alias.get(src.lower()) or canonical_key(src)
        t_key = alias.get(tgt.lower()) or canonical_key(tgt)
        # Variant collapse can turn a real relation into a self-loop.
        if s_key == t_key or s_key not in by_key or t_key not in by_key:
            continue
        sig = (s_key, r.get("type"), t_key)
        if sig in seen:
            continue
        seen.add(sig)
        out_rel.append({"source_key": s_key, "target_key": t_key, "type": r.get("type", "RELATED_TO")})

    return list(by_key.values()), out_rel
