"""
Graph memory — entity-relationship knowledge graph.

Stores entities and relations (subject, predicate, object) for
structured recall and reasoning over relationships.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from curio_agent_sdk.memory.base import Memory, MemoryEntry

_CHARS_PER_TOKEN = 4


@dataclass
class Triple:
    """A single (subject, relation, object) fact."""
    subject: str
    relation: str
    obj: str
    metadata: dict[str, Any] | None = None

    def __hash__(self):
        return hash((self.subject, self.relation, self.obj))

    def to_dict(self) -> dict[str, Any]:
        return {
            "subject": self.subject,
            "relation": self.relation,
            "object": self.obj,
            "metadata": self.metadata or {},
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Triple:
        return cls(
            subject=data.get("subject", ""),
            relation=data.get("relation", ""),
            obj=data.get("object", ""),
            metadata=data.get("metadata"),
        )


class GraphMemory(Memory):
    """
    Entity-relationship knowledge graph.

    Stores triples (entity1, relation, entity2) and supports
    query by subject, relation, object, or free text.

    Example:
        memory = GraphMemory()
        await memory.add_entity("Alice", {"type": "person"})
        await memory.add_relation("Alice", "works_at", "Acme")
        triples = await memory.query("Alice")
    """

    def __init__(self):
        self._entities: dict[str, dict[str, Any]] = {}  # entity -> attributes
        self._triples: list[Triple] = []
        self._by_subject: dict[str, list[int]] = {}  # subject -> indices
        self._by_relation: dict[str, list[int]] = {}
        self._by_object: dict[str, list[int]] = {}

    def _index_triple(self, t: Triple, idx: int) -> None:
        self._by_subject.setdefault(t.subject, []).append(idx)
        self._by_relation.setdefault(t.relation, []).append(idx)
        self._by_object.setdefault(t.obj, []).append(idx)

    def _unindex_triple(self, t: Triple, idx: int) -> None:
        for d, key in [
            (self._by_subject, t.subject),
            (self._by_relation, t.relation),
            (self._by_object, t.obj),
        ]:
            if key in d:
                try:
                    d[key].remove(idx)
                except ValueError:
                    pass
                if not d[key]:
                    del d[key]

    async def add_entity(self, entity: str, attributes: dict[str, Any] | None = None) -> None:
        """Register an entity with optional attributes."""
        self._entities[entity] = dict(attributes or {})

    async def add_relation(self, entity1: str, relation: str, entity2: str) -> None:
        """Add a triple (entity1, relation, entity2)."""
        t = Triple(subject=entity1, relation=relation, obj=entity2)
        idx = len(self._triples)
        self._triples.append(t)
        self._index_triple(t, idx)

    async def query(self, query: str, limit: int = 20) -> list[Triple]:
        """
        Query triples by matching query against subject, relation, or object.
        Returns matching triples.
        """
        q = query.lower().strip()
        seen: set[int] = set()
        for key in (q,):
            for d in (self._by_subject, self._by_relation, self._by_object):
                for k, indices in d.items():
                    if key in k.lower():
                        seen.update(indices)
        # Also match multi-word: treat query as multiple terms
        terms = q.split()
        for t in self._triples:
            combined = f"{t.subject} {t.relation} {t.obj}".lower()
            if any(term in combined for term in terms):
                idx = self._triples.index(t)
                seen.add(idx)
        result = [self._triples[i] for i in sorted(seen)][:limit]
        return result

    async def add(self, content: str, metadata: dict[str, Any] | None = None) -> str:
        """
        Add a fact as a triple. Expects content or metadata to describe
        subject/relation/object. If metadata has subject, relation, object
        use those; else parse content as "subject relation object" (space-separated).
        """
        meta = metadata or {}
        s = meta.get("subject")
        r = meta.get("relation")
        o = meta.get("object")
        if s is None or r is None or o is None:
            parts = content.strip().split(None, 2)
            if len(parts) >= 3:
                s, r, o = parts[0], parts[1], parts[2]
            elif len(parts) == 2:
                s, r, o = parts[0], "related_to", parts[1]
            else:
                s, r, o = content, "is", "known"
        await self.add_entity(s)
        await self.add_entity(o)
        await self.add_relation(s, r, o)
        # Return a synthetic id for Memory interface
        return f"{s}:{r}:{o}"

    async def search(self, query: str, limit: int = 5) -> list[MemoryEntry]:
        """Search triples by query; return as MemoryEntry for compatibility."""
        triples = await self.query(query, limit=limit)
        return [
            MemoryEntry(
                id=f"{t.subject}:{t.relation}:{t.obj}",
                content=f"{t.subject} {t.relation} {t.obj}",
                metadata=t.metadata or {},
                relevance=1.0,
            )
            for t in triples
        ]

    async def get_context(self, query: str, max_tokens: int = 2000) -> str:
        """Get matching triples formatted for prompt inclusion."""
        max_chars = max_tokens * _CHARS_PER_TOKEN
        triples = await self.query(query, limit=50)
        lines = ["[Knowledge Graph]"]
        total_chars = len(lines[0]) + 2
        for t in triples:
            line = f"- {t.subject} --{t.relation}--> {t.obj}"
            if total_chars + len(line) + 2 > max_chars:
                break
            lines.append(line)
            total_chars += len(line) + 2
        if len(lines) == 1:
            return ""
        return "\n".join(lines)

    async def get(self, entry_id: str) -> MemoryEntry | None:
        # entry_id may be "subject:relation:object"
        if ":" in entry_id:
            parts = entry_id.split(":", 2)
            if len(parts) >= 3:
                s, r, o = parts[0], parts[1], parts[2]
                for t in self._triples:
                    if t.subject == s and t.relation == r and t.obj == o:
                        return MemoryEntry(
                            id=entry_id,
                            content=f"{s} {r} {o}",
                            metadata=t.metadata or {},
                        )
        return None

    async def delete(self, entry_id: str) -> bool:
        if ":" not in entry_id:
            return False
        parts = entry_id.split(":", 2)
        if len(parts) < 3:
            return False
        s, r, o = parts[0], parts[1], parts[2]
        for i, t in enumerate(self._triples):
            if t.subject == s and t.relation == r and t.obj == o:
                self._unindex_triple(t, i)
                self._triples.pop(i)
                # Rebuild indices after removal
                self._by_subject.clear()
                self._by_relation.clear()
                self._by_object.clear()
                for idx, tr in enumerate(self._triples):
                    self._index_triple(tr, idx)
                return True
        return False

    async def clear(self) -> None:
        self._entities.clear()
        self._triples.clear()
        self._by_subject.clear()
        self._by_relation.clear()
        self._by_object.clear()

    async def count(self) -> int:
        return len(self._triples)
