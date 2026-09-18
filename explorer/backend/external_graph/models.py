from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any

SOURCE_GRAPH_ROBOKOP = "robokop"


@dataclass(frozen=True)
class ExternalNode:
    curie: str
    name: str = ""
    categories: tuple[str, ...] = ()
    source_graph: str = SOURCE_GRAPH_ROBOKOP
    properties: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["categories"] = list(self.categories)
        payload["properties"] = _jsonable(self.properties)
        return payload


@dataclass(frozen=True)
class ExternalEdge:
    edge_id: str
    subject_curie: str
    object_curie: str
    predicate: str
    direction: str
    adjacent_curie: str
    query_curie: str
    subject_name: str = ""
    object_name: str = ""
    subject_categories: tuple[str, ...] = ()
    object_categories: tuple[str, ...] = ()
    qualifiers: dict[str, Any] = field(default_factory=dict)
    primary_knowledge_source: str | None = None
    publications: tuple[str, ...] = ()
    supporting_sentences: tuple[str, ...] = ()
    original_subject_curie: str | None = None
    original_object_curie: str | None = None
    source_graph: str = SOURCE_GRAPH_ROBOKOP
    properties: dict[str, Any] = field(default_factory=dict)
    rank: int | None = None
    score: float | None = None
    ranking_reasons: tuple[str, ...] = ()
    matched_query_tokens: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        for key in (
            "subject_categories",
            "object_categories",
            "publications",
            "supporting_sentences",
            "ranking_reasons",
            "matched_query_tokens",
        ):
            payload[key] = list(payload[key])
        payload["qualifiers"] = _jsonable(self.qualifiers)
        payload["properties"] = _jsonable(self.properties)
        return payload


@dataclass(frozen=True)
class EdgeSummaryItem:
    predicate: str
    category: str
    count: int

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class EdgeSummary:
    query_curie: str
    items: tuple[EdgeSummaryItem, ...]
    provider_mode: str
    source_graph: str = SOURCE_GRAPH_ROBOKOP

    @property
    def total_edges(self) -> int:
        return sum(item.count for item in self.items)

    def categories(self) -> list[str]:
        return sorted({item.category for item in self.items if item.category})

    def predicates(self) -> list[str]:
        return sorted({item.predicate for item in self.items if item.predicate})

    def to_dict(self) -> dict[str, Any]:
        return {
            "query_curie": self.query_curie,
            "items": [item.to_dict() for item in self.items],
            "total_edges": self.total_edges,
            "categories": self.categories(),
            "predicates": self.predicates(),
            "provider_mode": self.provider_mode,
            "source_graph": self.source_graph,
        }


@dataclass(frozen=True)
class ExternalExpansionConfig:
    category: str | None = None
    predicate: str | None = None
    direction: str = "either"
    limit: int = 10
    offset: int = 0
    query: str = ""
    max_limit: int = 25

    def normalized_direction(self) -> str:
        direction = (self.direction or "either").strip().lower()
        if direction not in {"either", "incoming", "outgoing"}:
            raise ValueError("ROBOKOP direction must be either, incoming, or outgoing.")
        return direction

    def normalized_limit(self) -> int:
        return max(1, min(int(self.limit), int(self.max_limit)))

    def normalized_offset(self) -> int:
        return max(0, int(self.offset))

    def to_dict(self) -> dict[str, Any]:
        return {
            "category": self.category or None,
            "predicate": self.predicate or None,
            "direction": self.normalized_direction(),
            "limit": self.normalized_limit(),
            "offset": self.normalized_offset(),
            "query": self.query,
            "max_limit": self.max_limit,
        }


@dataclass(frozen=True)
class ExternalExpansionResult:
    query_curie: str
    edges: tuple[ExternalEdge, ...]
    config: ExternalExpansionConfig
    provider_mode: str
    total_returned: int
    total_available: int | None = None
    source_graph: str = SOURCE_GRAPH_ROBOKOP
    diagnostics: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "query_curie": self.query_curie,
            "edges": [edge.to_dict() for edge in self.edges],
            "config": self.config.to_dict(),
            "provider_mode": self.provider_mode,
            "total_returned": self.total_returned,
            "total_available": self.total_available,
            "source_graph": self.source_graph,
            "diagnostics": _jsonable(self.diagnostics),
        }


def _jsonable(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items() if "embedding" not in str(key)}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return str(value)
