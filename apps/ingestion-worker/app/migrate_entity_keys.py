"""One-off: re-key existing Entity nodes onto canonical keys and drop noise
nodes, without re-running extraction. Lets documents ingested before
entity_norm existed benefit from resolution at no API cost.

Run: docker exec <ingestion-worker> python3 -m app.migrate_entity_keys
"""
from .entity_norm import canonical_key, is_noise
from .neo4j_client import driver
from .config import settings


def main():
    with driver.session(database=settings.neo4j_database) as s:
        rows = [dict(r) for r in s.run(
            "MATCH (e:Entity) RETURN e.key AS key, e.name AS name, e.type AS type")]
        print(f"entities before: {len(rows)}")

        dropped = [r for r in rows if is_noise(r["name"]) or is_noise(canonical_key(r["name"]))]
        for r in dropped:
            s.run("MATCH (e:Entity {key:$k}) DETACH DELETE e", k=r["key"])
        print(f"dropped as noise: {len(dropped)}")

        keep = [r for r in rows if r not in dropped]
        groups: dict[str, list[dict]] = {}
        for r in keep:
            groups.setdefault(canonical_key(r["name"]), []).append(r)

        merged = 0
        for ckey, members in groups.items():
            # A member already sitting on the canonical key must be the survivor —
            # re-keying a different node onto it would break the uniqueness constraint.
            existing = next((m for m in members if m["key"] == ckey), None)
            display = max(members, key=lambda m: len(m["name"]))["name"]
            primary = existing or max(members, key=lambda m: len(m["name"]))
            if primary["key"] != ckey:
                s.run("MATCH (e:Entity {key:$old}) SET e.key=$new", old=primary["key"], new=ckey)
            s.run("MATCH (e:Entity {key:$k}) SET e.name=$name", k=ckey, name=display)
            primary = {**primary, "key": ckey}

            for m in members:
                if m["key"] == primary["key"]:
                    continue
                # Re-point both edge types at the surviving node, then remove it.
                s.run(
                    "MATCH (dup:Entity {key:$dup}), (keep:Entity {key:$keep}) "
                    "OPTIONAL MATCH (c:Chunk)-[:MENTIONS]->(dup) "
                    "FOREACH (_ IN CASE WHEN c IS NULL THEN [] ELSE [1] END | "
                    "  MERGE (c)-[:MENTIONS]->(keep))",
                    dup=m["key"], keep=ckey,
                )
                s.run(
                    "MATCH (dup:Entity {key:$dup})-[r:RELATES]->(o:Entity), (keep:Entity {key:$keep}) "
                    "WHERE o.key <> $keep MERGE (keep)-[n:RELATES {type:r.type}]->(o) "
                    "SET n.doc_ids = coalesce(n.doc_ids, r.doc_ids)",
                    dup=m["key"], keep=ckey,
                )
                s.run(
                    "MATCH (o:Entity)-[r:RELATES]->(dup:Entity {key:$dup}), (keep:Entity {key:$keep}) "
                    "WHERE o.key <> $keep MERGE (o)-[n:RELATES {type:r.type}]->(keep) "
                    "SET n.doc_ids = coalesce(n.doc_ids, r.doc_ids)",
                    dup=m["key"], keep=ckey,
                )
                s.run("MATCH (dup:Entity {key:$dup}) DETACH DELETE dup", dup=m["key"])
                merged += 1

        after = s.run("MATCH (e:Entity) RETURN count(e) AS n").single()["n"]
        rels = s.run("MATCH ()-[r:RELATES]->() RETURN count(r) AS n").single()["n"]
        print(f"merged away: {merged}")
        print(f"entities after: {after}, relations: {rels}")


if __name__ == "__main__":
    main()
