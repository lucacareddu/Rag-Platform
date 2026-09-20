"""Cross-document alias resolution entity_norm.py can't do (world knowledge, not string shape).
One LLM call over the whole entity list, not one per pair (avoids O(n^2) quota cost).
"""
import json
import logging
import sys

from openai import OpenAI

from .config import settings
from .neo4j_client import driver

logger = logging.getLogger(__name__)

client = OpenAI(base_url=settings.gemini_base_url, api_key=settings.gemini_api_key)

PROMPT = """Below is a list of entity names extracted from a set of related documents.
Some names refer to the SAME real-world thing using different wording — often because
different documents name it differently (full title vs common name vs abbreviation).

Group ONLY names that denote the identical thing. Return JSON:
{"groups": [{"canonical": "<the clearest full name>", "aliases": ["...", "..."]}]}

Rules:
- A part is NOT its whole: "Framework Core" and "Framework Profile" are COMPONENTS of
  the Cybersecurity Framework, not aliases of it. Do not group them with it.
- A narrower concept is not an alias of a broader one ("Recovery Planning" is not
  "Cybersecurity Event Recovery").
- Only group when you are confident they are interchangeable references.
- Omit any name that has no alias. Most names will not appear in your output.

Names:
"""


def propose(names: list[str]) -> list[dict]:
    resp = client.chat.completions.create(
        model=settings.extract_model,
        messages=[{"role": "user", "content": PROMPT + "\n".join(f"- {n}" for n in names)}],
        response_format={"type": "json_object"},
    )
    raw = (resp.choices[0].message.content or "").strip()
    if raw.startswith("```"):
        raw = raw.split("\n", 1)[-1].rsplit("```", 1)[0]
    try:
        groups = json.loads(raw).get("groups", [])
    except json.JSONDecodeError:
        logger.warning("Alias pass returned non-JSON, no aliases applied")
        return []

    known = {n.lower() for n in names}
    out = []
    for g in groups:
        canonical = str(g.get("canonical", "")).strip()
        # Only trust names the model was actually given — it otherwise invents
        # plausible-looking canonical forms that match no node in the graph.
        aliases = [a for a in {str(x).strip() for x in g.get("aliases", [])}
                   if a.lower() in known and a.lower() != canonical.lower()]
        if canonical.lower() in known and aliases:
            out.append({"canonical": canonical, "aliases": aliases})
    return out


def apply(groups: list[dict]):
    """Re-points MENTIONS and RELATES from each alias onto the canonical node."""
    with driver.session(database=settings.neo4j_database) as s:
        for g in groups:
            ckey = _key_of(s, g["canonical"])
            if not ckey:
                continue
            for alias in g["aliases"]:
                akey = _key_of(s, alias)
                if not akey or akey == ckey:
                    continue
                s.run("MATCH (c:Chunk)-[:MENTIONS]->(:Entity {key:$a}), (k:Entity {key:$k}) "
                      "MERGE (c)-[:MENTIONS]->(k)", a=akey, k=ckey)
                s.run("MATCH (:Entity {key:$a})-[r:RELATES]->(o:Entity), (k:Entity {key:$k}) "
                      "WHERE o.key <> $k MERGE (k)-[n:RELATES {type:r.type}]->(o)", a=akey, k=ckey)
                s.run("MATCH (o:Entity)-[r:RELATES]->(:Entity {key:$a}), (k:Entity {key:$k}) "
                      "WHERE o.key <> $k MERGE (o)-[n:RELATES {type:r.type}]->(k)", a=akey, k=ckey)
                # Keep the alias as a searchable surface form: mention mapping
                # matches chunk text, so dropping "NIST" in favour of the full
                # name would lose every chunk that only ever writes the acronym.
                s.run("MATCH (k:Entity {key:$k}) "
                      "SET k.aliases = [x IN coalesce(k.aliases, []) + [$name] "
                      "                 WHERE x <> k.name]", k=ckey, name=alias)
                s.run("MATCH (e:Entity {key:$a}) DETACH DELETE e", a=akey)


def _key_of(s, name: str) -> str | None:
    r = s.run("MATCH (e:Entity) WHERE toLower(e.name)=toLower($n) RETURN e.key AS k",
              n=name).single()
    return r["k"] if r else None


def main():
    with driver.session(database=settings.neo4j_database) as s:
        names = [r["n"] for r in s.run("MATCH (e:Entity) RETURN e.name AS n ORDER BY n")]
    print(f"entities: {len(names)}")

    groups = propose(names)
    print(f"alias groups proposed: {len(groups)}")
    for g in groups:
        print(f"  {g['canonical']!r} <- {g['aliases']}")

    if "--apply" in sys.argv:
        apply(groups)
        with driver.session(database=settings.neo4j_database) as s:
            n = s.run("MATCH (e:Entity) RETURN count(e) AS n").single()["n"]
        print(f"applied. entities now: {n}")
    else:
        print("\ndry run — pass --apply to write these merges")


if __name__ == "__main__":
    main()
