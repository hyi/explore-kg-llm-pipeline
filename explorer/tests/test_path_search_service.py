from __future__ import annotations

import json

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
                anchor_metadata={
                    "relationship_identity": metadata.get("relationship_identity", metadata["id"]),
                    "anchor_score": score,
                    "semantic_score": metadata.get("semantic_score"),
                    "raw_rank": metadata.get("raw_rank"),
                },
            )
        ]

    def close(self) -> None:
        return None


def test_path_search_discovers_all_selected_anchors_before_final_truncation(
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

        def search(self, query: str, relationship_k: int):
            assert relationship_k == 2
            return SemanticSearchResult(
                relationships=[
                    Document(
                        page_content="first selected anchor",
                        metadata={
                            "id": "first",
                            "relationship_identity": "first",
                            "score": 0.4,
                            "semantic_score": 0.3,
                            "raw_rank": 0,
                            "predicate": "biolink:related_to",
                        },
                    ),
                    Document(
                        page_content="second anchor",
                        metadata={
                            "id": "second",
                            "relationship_identity": "second",
                            "score": 0.9,
                            "semantic_score": 0.9,
                            "raw_rank": 1,
                            "predicate": "biolink:related_to",
                        },
                    ),
                ],
                nodes={},
                diagnostics={
                    "raw_candidates": [
                        {"raw_rank": 0, "relationship_identity": "first"},
                        {"raw_rank": 1, "relationship_identity": "second"},
                    ],
                    "raw_semantic_candidates": [
                        {
                            "raw_rank": 0,
                            "rerank_rank": 0,
                            "relationship_identity": "first",
                            "semantic_score": 0.3,
                            "anchor_score": 0.4,
                            "ranking_components": {"retrieval": 0.3, "positive_compatibility_capped": 0.1},
                            "ranking_reasons": ["gene endpoint category match"],
                            "query_intent": {"has_structural_intent": True},
                            "compatibility_components": {"endpoint_category": 0.1},
                            "publication_id": "pub-1",
                            "anchor_entities": {"subject": "first-s", "object": "first-o"},
                            "entered_selected_anchor_set": True,
                            "selection_pass": "primary_diversity_pass",
                            "exclusion_reason": None,
                        },
                        {
                            "raw_rank": 1,
                            "rerank_rank": 1,
                            "relationship_identity": "second",
                            "semantic_score": 0.9,
                            "anchor_score": 0.9,
                            "ranking_components": {"retrieval": 0.9},
                            "ranking_reasons": ["semantic score 0.9000"],
                            "query_intent": {"has_structural_intent": True},
                            "compatibility_components": {},
                            "publication_id": "pub-2",
                            "anchor_entities": {"subject": "second-s", "object": "second-o"},
                            "entered_selected_anchor_set": True,
                            "selection_pass": "primary_diversity_pass",
                            "exclusion_reason": None,
                        },
                    ],
                    "reranked_candidates": [
                        {
                            "raw_rank": 0,
                            "rerank_rank": 0,
                            "relationship_identity": "first",
                            "semantic_score": 0.3,
                            "anchor_score": 0.4,
                            "ranking_components": {"retrieval": 0.3, "positive_compatibility_capped": 0.1},
                            "ranking_reasons": ["gene endpoint category match"],
                            "query_intent": {"has_structural_intent": True},
                            "compatibility_components": {"endpoint_category": 0.1},
                            "publication_id": "pub-1",
                            "anchor_entities": {"subject": "first-s", "object": "first-o"},
                            "entered_selected_anchor_set": True,
                            "selection_pass": "primary_diversity_pass",
                            "exclusion_reason": None,
                        },
                        {
                            "raw_rank": 1,
                            "rerank_rank": 1,
                            "relationship_identity": "second",
                            "semantic_score": 0.9,
                            "anchor_score": 0.9,
                            "ranking_components": {"retrieval": 0.9},
                            "ranking_reasons": ["semantic score 0.9000"],
                            "query_intent": {"has_structural_intent": True},
                            "compatibility_components": {},
                            "publication_id": "pub-2",
                            "anchor_entities": {"subject": "second-s", "object": "second-o"},
                            "entered_selected_anchor_set": True,
                            "selection_pass": "primary_diversity_pass",
                            "exclusion_reason": None,
                        },
                    ],
                    "selected_anchors": [
                        {
                            "raw_rank": 0,
                            "rerank_rank": 0,
                            "relationship_identity": "first",
                            "semantic_score": 0.3,
                            "anchor_score": 0.4,
                            "ranking_components": {"retrieval": 0.3, "positive_compatibility_capped": 0.1},
                            "ranking_reasons": ["gene endpoint category match"],
                            "query_intent": {"has_structural_intent": True},
                            "compatibility_components": {"endpoint_category": 0.1},
                            "publication_id": "pub-1",
                            "anchor_entities": {"subject": "first-s", "object": "first-o"},
                            "entered_selected_anchor_set": True,
                            "selection_pass": "primary_diversity_pass",
                            "exclusion_reason": None,
                        },
                        {
                            "raw_rank": 1,
                            "rerank_rank": 1,
                            "relationship_identity": "second",
                            "semantic_score": 0.9,
                            "anchor_score": 0.9,
                            "ranking_components": {"retrieval": 0.9},
                            "ranking_reasons": ["semantic score 0.9000"],
                            "query_intent": {"has_structural_intent": True},
                            "compatibility_components": {},
                            "publication_id": "pub-2",
                            "anchor_entities": {"subject": "second-s", "object": "second-o"},
                            "entered_selected_anchor_set": True,
                            "selection_pass": "primary_diversity_pass",
                            "exclusion_reason": None,
                        },
                    ],
                },
            )

    monkeypatch.setattr("explorer.backend.path_search.service.Neo4jGraphAdapter", lambda: fake_graph)
    monkeypatch.setattr("explorer.backend.semantic_search.SemanticSearchService", FakeSemanticSearchService)

    result = PathSearchService(cache=PathSearchCache(tmp_path / "cache.json")).search(
        query="genes involved in chemoresistance",
        relationship_k=1,
        semantic_fetch_k=2,
        paths_per_hit=1,
    )

    assert result.paths[0]["seed_subject"] == "second"
    assert fake_graph.seen_anchor_ids == ["first", "second"]
    assert requested_models == ["sapbert"]
    payload = (tmp_path / "cache.json").read_text(encoding="utf-8")
    assert "raw_candidates" in payload


def test_path_search_cache_records_raw_to_final_diagnostics(tmp_path, monkeypatch) -> None:
    monkeypatch.chdir(tmp_path)
    (tmp_path / ".env").write_text("EMBEDDING_PROVIDER=sapbert\nEMBEDDING_MODEL=local-sapbert\n", encoding="utf-8")
    fake_graph = FakeGraph()

    class FakeSemanticSearchService:
        def __init__(self, **_kwargs) -> None:
            return None

        def search(self, query: str, relationship_k: int):
            return SemanticSearchResult(
                relationships=[
                    Document(
                        page_content="selected anchor",
                        metadata={
                            "id": "first",
                            "relationship_identity": "first",
                            "score": 0.7,
                            "semantic_score": 0.6,
                            "raw_rank": 0,
                            "predicate": "biolink:related_to",
                        },
                    )
                ],
                nodes={},
                diagnostics={
                    "raw_semantic_candidates": [
                        {
                            "raw_rank": 0,
                            "rerank_rank": 0,
                            "relationship_identity": "first",
                            "semantic_score": 0.6,
                            "anchor_score": 0.7,
                            "ranking_components": {"retrieval": 0.6, "positive_compatibility_capped": 0.1},
                            "compatibility_components": {"endpoint_category": 0.1},
                            "publication_id": "pub-1",
                            "anchor_entities": {"subject": "first-s", "object": "first-o"},
                            "entered_selected_anchor_set": True,
                            "selection_pass": "primary_diversity_pass",
                        },
                        {
                            "raw_rank": 1,
                            "rerank_rank": 1,
                            "relationship_identity": "excluded",
                            "semantic_score": 0.5,
                            "anchor_score": 0.5,
                            "ranking_components": {"retrieval": 0.5},
                            "compatibility_components": {},
                            "publication_id": "pub-1",
                            "anchor_entities": {"subject": "excluded-s", "object": "excluded-o"},
                            "entered_selected_anchor_set": False,
                            "exclusion_reason": "selection_limit_reached",
                            "exclusion_pass": "primary_diversity_pass",
                        },
                    ],
                    "reranked_candidates": [
                        {
                            "raw_rank": 0,
                            "rerank_rank": 0,
                            "relationship_identity": "first",
                            "semantic_score": 0.6,
                            "anchor_score": 0.7,
                            "ranking_components": {"retrieval": 0.6, "positive_compatibility_capped": 0.1},
                            "compatibility_components": {"endpoint_category": 0.1},
                            "publication_id": "pub-1",
                            "anchor_entities": {"subject": "first-s", "object": "first-o"},
                            "entered_selected_anchor_set": True,
                            "selection_pass": "primary_diversity_pass",
                        },
                        {
                            "raw_rank": 1,
                            "rerank_rank": 1,
                            "relationship_identity": "excluded",
                            "semantic_score": 0.5,
                            "anchor_score": 0.5,
                            "ranking_components": {"retrieval": 0.5},
                            "compatibility_components": {},
                            "publication_id": "pub-1",
                            "anchor_entities": {"subject": "excluded-s", "object": "excluded-o"},
                            "entered_selected_anchor_set": False,
                            "exclusion_reason": "selection_limit_reached",
                            "exclusion_pass": "primary_diversity_pass",
                        },
                    ],
                    "selected_anchors": [
                        {
                            "raw_rank": 0,
                            "rerank_rank": 0,
                            "relationship_identity": "first",
                            "semantic_score": 0.6,
                            "anchor_score": 0.7,
                            "ranking_components": {"retrieval": 0.6, "positive_compatibility_capped": 0.1},
                            "compatibility_components": {"endpoint_category": 0.1},
                            "publication_id": "pub-1",
                            "anchor_entities": {"subject": "first-s", "object": "first-o"},
                            "entered_selected_anchor_set": True,
                            "selection_pass": "primary_diversity_pass",
                        }
                    ],
                },
            )

    monkeypatch.setattr("explorer.backend.path_search.service.Neo4jGraphAdapter", lambda: fake_graph)
    monkeypatch.setattr("explorer.backend.semantic_search.SemanticSearchService", FakeSemanticSearchService)

    PathSearchService(cache=PathSearchCache(tmp_path / "cache.json")).search(
        query="genes involved in chemoresistance",
        relationship_k=1,
        semantic_fetch_k=2,
        paths_per_hit=1,
    )

    payload = json.loads((tmp_path / "cache.json").read_text(encoding="utf-8"))
    metadata = next(iter(payload["entries"].values()))["metadata"]
    diagnostics = metadata["path_search_diagnostics"]

    assert diagnostics["raw_semantic_candidates"][0]["number_of_paths_discovered"] == 1
    assert diagnostics["raw_semantic_candidates"][0]["path_selection_status"] == "has_displayed_path"
    assert diagnostics["raw_semantic_candidates"][1]["path_selection_status"] == "not_selected_as_anchor"
    assert diagnostics["enriched_reranked_candidates"][0]["ranking_components"] == {
        "retrieval": 0.6,
        "positive_compatibility_capped": 0.1,
    }
    assert diagnostics["diversified_semantic_anchors"][0]["relationship_identity"] == "first"
    assert diagnostics["discovered_paths"][0]["anchor_relationship_identity"] == "first"
    assert diagnostics["final_displayed_paths"][0]["selected_for_display"] is True
