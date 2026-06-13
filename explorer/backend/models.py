from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any
from uuid import uuid4


@dataclass(frozen=True)
class Node:
    element_id: str
    id: str
    name: str
    labels: list[str] = field(default_factory=list)
    properties: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class Edge:
    element_id: str
    type: str
    start_element_id: str
    end_element_id: str
    subject: str = ""
    object: str = ""
    properties: dict[str, Any] = field(default_factory=dict)


@dataclass
class Path:
    nodes: list[Node]
    edges: list[Edge]
    score: float = 0.0
    source: str = "semantic"
    seed_subject: str | None = None
    seed_object: str | None = None
    seed_predicate: str | None = None
    evidence_text: str | None = None
    id: str = field(default_factory=lambda: str(uuid4()))

    @property
    def length(self) -> int:
        return len(self.edges)

    @property
    def node_element_ids(self) -> list[str]:
        return [node.element_id for node in self.nodes]

    @property
    def endpoint_element_ids(self) -> list[str]:
        if not self.nodes:
            return []
        if len(self.nodes) == 1:
            return [self.nodes[0].element_id]
        return [self.nodes[0].element_id, self.nodes[-1].element_id]

    def summary(self) -> str:
        if not self.nodes:
            return "(empty path)"

        by_id = {node.element_id: node.name for node in self.nodes}
        parts = [self.nodes[0].name]
        current = self.nodes[0].element_id
        for edge in self.edges:
            if edge.start_element_id == current:
                next_id = edge.end_element_id
                parts.append(f"-[{edge.type}]->")
            else:
                next_id = edge.start_element_id
                parts.append(f"<-[{edge.type}]-")
            parts.append(by_id.get(next_id, next_id))
            current = next_id
        return " ".join(parts)

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "score": self.score,
            "source": self.source,
            "length": self.length,
            "summary": self.summary(),
            "seed_subject": self.seed_subject,
            "seed_object": self.seed_object,
            "seed_predicate": self.seed_predicate,
            "evidence_text": self.evidence_text,
            "nodes": [
                {
                    "element_id": node.element_id,
                    "id": node.id,
                    "name": node.name,
                    "labels": node.labels,
                    "properties": _jsonable(node.properties),
                }
                for node in self.nodes
            ],
            "edges": [
                {
                    "element_id": edge.element_id,
                    "type": edge.type,
                    "start_element_id": edge.start_element_id,
                    "end_element_id": edge.end_element_id,
                    "subject": edge.subject,
                    "object": edge.object,
                    "properties": _jsonable(edge.properties),
                }
                for edge in self.edges
            ],
        }


@dataclass(frozen=True)
class SemanticSearchResult:
    relationships: list[Any]
    nodes: dict[str, list[Any]]


def _jsonable(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items() if key != "embedding"}
    if isinstance(value, list):
        return [_jsonable(item) for item in value]
    if isinstance(value, tuple):
        return [_jsonable(item) for item in value]
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return str(value)
