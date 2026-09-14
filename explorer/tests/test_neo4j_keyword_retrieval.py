from __future__ import annotations

from explorer.backend.graph_adapter.neo4j import adapter


class FakeResult:
    def __init__(self, records: list[dict[str, object]]) -> None:
        self.records = records

    def __iter__(self):
        return iter(self.records)

    def single(self):
        return self.records[0] if self.records else None


class FakeSession:
    def __init__(self, captured: dict[str, object], records: list[dict[str, object]] | None = None) -> None:
        self.captured = captured
        self.records = records or []

    def __enter__(self):
        return self

    def __exit__(self, *_args) -> None:
        return None

    def run(self, query: str, **kwargs):
        self.captured["query"] = query
        self.captured["kwargs"] = kwargs
        return FakeResult(self.records)


class FakeDriver:
    def __init__(self, captured: dict[str, object], records: list[dict[str, object]] | None = None) -> None:
        self.captured = captured
        self.records = records

    def session(self) -> FakeSession:
        return FakeSession(self.captured, self.records)

    def close(self) -> None:
        return None


def test_keyword_relationship_search_does_not_score_semantic_text(monkeypatch) -> None:
    captured: dict[str, object] = {}
    monkeypatch.setattr(adapter.GraphDatabase, "driver", lambda *_args, **_kwargs: FakeDriver(captured))
    graph = adapter.Neo4jGraphAdapter(uri="bolt://test", username="neo4j", password="password")

    assert graph.keyword_relationship_search("genes involved in chemoresistance", k=5) == []

    query = str(captured["query"])
    assert "coalesce(rel_props.semantic_text" not in query
    assert "rel_props.semantic_text AS semantic_text" in query
    assert "rel_props.title_text" not in query
    assert "rel_props.context_text" not in query
    assert "rel_props.abstract_text" not in query
    assert "edge_text" in query


def test_neighborhood_filter_options_returns_sorted_kg_labels_and_predicates(monkeypatch) -> None:
    captured: dict[str, object] = {}
    records = [
        {
            "node_categories": ["biolink:Disease", "biolink:Gene", "biolink:Gene", ""],
            "predicates": ["biolink:treats", "biolink:associated_with", "biolink:treats"],
        }
    ]
    monkeypatch.setattr(adapter.GraphDatabase, "driver", lambda *_args, **_kwargs: FakeDriver(captured, records))
    graph = adapter.Neo4jGraphAdapter(uri="bolt://test", username="neo4j", password="password")

    options = graph.neighborhood_filter_options()

    assert options == {
        "node_categories": ["biolink:Disease", "biolink:Gene"],
        "predicates": ["biolink:associated_with", "biolink:treats"],
    }
    query = str(captured["query"])
    assert "UNWIND labels(n) AS category" in query
    assert "collect(DISTINCT type(r)) AS predicates" in query
