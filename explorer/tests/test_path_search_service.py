from __future__ import annotations

from langchain_core.documents import Document

from explorer.backend.models import Edge, Node, Path, SemanticSearchResult
from explorer.backend.path_search import PathSearchService
from explorer.backend.path_search.cache import PathSearchCache


class FakeGraph:
    def __init__(self) -> None:
        self.seen_anchor_ids: list[str] = []

    def enrich_relationship_hits(self, candidates):
        return candidates

    def candidate_paths_for_semantic_hit(self, metadata, score, evidence_text=None, limit=3):
        self.seen_anchor_ids.append(metadata["id"])
        node_a = Node(element_id=f"{metadata['id']}-a", id="a", name="A")
        node_b = Node(element_id=f"{metadata['id']}-b", id="b", name="B")
        edge = Edge(
            element_id=f"{metadata['id']}-e",
            type=metadata["predicate"],
            start_element_id=node_a.element_id,
            end_element_id=node_b.element_id,
        )
        return [
            Path(
                id=f"path-{metadata['id']}",
                nodes=[node_a, node_b],
                edges=[edge],
                score=score,
                seed_subject=metadata["id"],
            )
        ]

    def close(self) -> None:
        return None


def test_path_search_uses_semantic_anchor_order_before_max_path_truncation(
    tmp_path,
    monkeypatch,
) -> None:
    monkeypatch.chdir(tmp_path)
    (tmp_path / ".env").write_text("EMBEDDING_PROVIDER=sapbert\nEMBEDDING_MODEL=local-sapbert\n", encoding="utf-8")
    fake_graph = FakeGraph()
    requested_models = []

    class FakeSemanticSearchService:
        def __init__(self, **kwargs) -> None:
            requested_models.append(kwargs.get("model"))
            return None

        def search(self, query: str, relationship_k: int):
            assert relationship_k == 2
            return SemanticSearchResult(
                relationships=[
                    Document(
                        page_content="first selected anchor",
                        metadata={"id": "first", "score": 0.4, "predicate": "biolink:related_to"},
                    ),
                    Document(
                        page_content="second anchor",
                        metadata={"id": "second", "score": 0.9, "predicate": "biolink:related_to"},
                    ),
                ],
                nodes={},
                diagnostics={"raw_candidates": [{"raw_rank": 0, "relationship_identity": "first"}]},
            )

    monkeypatch.setattr("explorer.backend.path_search.service.Neo4jGraphAdapter", lambda: fake_graph)
    monkeypatch.setattr("explorer.backend.semantic_search.SemanticSearchService", FakeSemanticSearchService)

    result = PathSearchService(cache=PathSearchCache(tmp_path / "cache.json")).search(
        query="genes involved in chemoresistance",
        relationship_k=1,
        semantic_fetch_k=2,
        paths_per_hit=1,
    )

    assert result.paths[0]["seed_subject"] == "first"
    assert fake_graph.seen_anchor_ids == ["first"]
    assert requested_models == ["sapbert"]
    payload = (tmp_path / "cache.json").read_text(encoding="utf-8")
    assert "raw_candidates" in payload
