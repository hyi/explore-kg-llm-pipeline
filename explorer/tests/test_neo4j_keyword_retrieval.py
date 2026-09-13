from __future__ import annotations

from explorer.backend.graph_adapter.neo4j import adapter


class FakeSession:
    def __init__(self, captured: dict[str, object]) -> None:
        self.captured = captured

    def __enter__(self):
        return self

    def __exit__(self, *_args) -> None:
        return None

    def run(self, query: str, **kwargs):
        self.captured["query"] = query
        self.captured["kwargs"] = kwargs
        return []


class FakeDriver:
    def __init__(self, captured: dict[str, object]) -> None:
        self.captured = captured

    def session(self) -> FakeSession:
        return FakeSession(self.captured)

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
