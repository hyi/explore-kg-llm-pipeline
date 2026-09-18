from __future__ import annotations

from langchain_core.documents import Document

from explorer.backend.semantic_search.retrieval import (
    RetrievalConfig,
    reciprocal_rank_fusion,
    retrieve_relationship_candidates,
    tokenize_keyword_query,
)


def doc(rel_id: str, score: float, *, predicate: str = "biolink:related_to") -> Document:
    return Document(
        page_content=f"{rel_id} text",
        metadata={
            "id": rel_id,
            "score": score,
            "predicate": predicate,
            "original_subject": f"{rel_id}-subject",
            "original_object": f"{rel_id}-object",
        },
    )


def test_keyword_tokenization_is_deterministic() -> None:
    assert tokenize_keyword_query("Genes involved in CHEMO-resistance, in cancer!") == [
        "genes",
        "involved",
        "chemo",
        "resistance",
        "cancer",
    ]


def test_reciprocal_rank_fusion_deduplicates_and_preserves_channel_metadata() -> None:
    fused = reciprocal_rank_fusion(
        dense_hits=[
            doc("shared", 0.90),
            doc("dense-only", 0.80),
        ],
        keyword_hits=[
            doc("keyword-only", 7.0),
            doc("shared", 5.0),
        ],
        limit=3,
        config=RetrievalConfig(mode="hybrid", dense_weight=1.0, keyword_weight=1.0, rrf_k=60),
    )

    ids = [hit.metadata["id"] for hit in fused]
    assert ids == ["shared", "keyword-only", "dense-only"]
    assert fused[0].metadata["retrieval_channels"] == ["dense", "keyword"]
    assert fused[0].metadata["dense_rank"] == 0
    assert fused[0].metadata["keyword_rank"] == 1
    assert fused[0].metadata["dense_score"] == 0.90
    assert fused[0].metadata["keyword_score"] == 5.0
    assert fused[0].metadata["retrieval_score"] == 1.0
    assert "multi_channel_bonus" not in fused[0].metadata


def test_hybrid_retrieval_records_degraded_mode_when_keyword_channel_unavailable() -> None:
    def dense_retriever(_query: str, k: int, model: str | None = None):
        return [doc("dense", 0.9)]

    result = retrieve_relationship_candidates(
        "genes in cancer",
        k=5,
        dense_retriever=dense_retriever,
        keyword_retriever=None,
        model="sapbert",
        config=RetrievalConfig(mode="hybrid"),
    )

    assert [hit.metadata["id"] for hit in result.candidates] == ["dense"]
    assert result.diagnostics["degraded"] is True
    assert result.diagnostics["degraded_reasons"] == ["keyword_retriever_unavailable"]
    assert "dense" in result.diagnostics["channels"]


def test_keyword_only_retrieval_uses_keyword_channel_and_fusion_score() -> None:
    def dense_retriever(_query: str, k: int, model: str | None = None):
        raise AssertionError("dense retriever should not run")

    def keyword_retriever(_query: str, k: int):
        return [doc("keyword", 4.0)]

    result = retrieve_relationship_candidates(
        "genes in cancer",
        k=5,
        dense_retriever=dense_retriever,
        keyword_retriever=keyword_retriever,
        config=RetrievalConfig(mode="keyword"),
    )

    assert [hit.metadata["id"] for hit in result.candidates] == ["keyword"]
    assert result.candidates[0].metadata["retrieval_channels"] == ["keyword"]
    assert result.candidates[0].metadata["retrieval_score"] == 1.0
    assert result.candidates[0].metadata["semantic_score"] == 0.0
    assert result.diagnostics["original_query"] == "genes in cancer"
    assert result.diagnostics["keyword_query"]["final_query"] == "cancer"


def test_retrieval_passes_original_query_to_dense_but_reports_bm25_content_query() -> None:
    seen_dense_queries = []

    def dense_retriever(query: str, k: int, model: str | None = None):
        seen_dense_queries.append(query)
        return [doc("dense", 0.9)]

    result = retrieve_relationship_candidates(
        "genes involved in chemoresistance in cancer",
        k=5,
        dense_retriever=dense_retriever,
        keyword_retriever=None,
        config=RetrievalConfig(mode="dense"),
    )

    assert seen_dense_queries == ["genes involved in chemoresistance in cancer"]
    assert result.diagnostics["keyword_query"]["final_query"] == "chemoresistance cancer"
    assert result.diagnostics["keyword_tokens"] == ["chemoresistance", "cancer"]
