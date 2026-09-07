from __future__ import annotations

from langchain_core.documents import Document

from explorer.backend.semantic_search import AnchorRankingConfig, SemanticSearchService


def test_semantic_search_overfetches_and_returns_requested_count() -> None:
    requested_ks = []
    requested_models = []

    def retriever(_query: str, k: int, model: str | None = None) -> list[Document]:
        requested_ks.append(k)
        requested_models.append(model)
        return [
            Document(
                page_content=f"edge {index}",
                metadata={
                    "id": f"rel-{index}",
                    "score": 1.0 - index * 0.01,
                    "predicate": "biolink:related_to",
                    "original_subject": f"s{index}",
                    "original_object": f"o{index}",
                },
            )
            for index in range(10)
        ]

    service = SemanticSearchService(
        config=AnchorRankingConfig(
            candidate_multiplier=4,
            minimum_candidate_pool=5,
            maximum_candidate_pool=50,
            max_per_publication=10,
        ),
        relationship_retriever=retriever,
        model="sapbert",
    )

    result = service.search("broad cancer mechanisms", relationship_k=3)

    assert requested_ks == [12]
    assert requested_models == ["sapbert"]
    assert len(result.relationships) == 3
    assert [hit.metadata["id"] for hit in result.relationships] == ["rel-0", "rel-1", "rel-2"]


def test_semantic_search_applies_batch_metadata_enrichment_before_ranking() -> None:
    def retriever(_query: str, k: int, model: str | None = None) -> list[Document]:
        return [
            Document(
                page_content="PTEN cancer",
                metadata={
                    "id": "pten",
                    "score": 0.84,
                    "predicate": "biolink:associated_with",
                    "original_subject": "NCBIGene:5728",
                    "original_object": "MONDO:0004992",
                },
            ),
            Document(
                page_content="drug neuropathy",
                metadata={
                    "id": "drug",
                    "score": 0.95,
                    "predicate": "biolink:related_to",
                    "original_subject": "CHEBI:1",
                    "original_object": "MONDO:1",
                },
            ),
        ]

    def enricher(candidates: list[Document]) -> list[Document]:
        enriched = []
        for candidate in candidates:
            metadata = dict(candidate.metadata)
            if metadata["id"] == "pten":
                metadata["subject_labels"] = ["biolink:Gene"]
            else:
                metadata["subject_labels"] = ["biolink:Drug"]
            enriched.append(candidate.model_copy(update={"metadata": metadata}))
        return enriched

    service = SemanticSearchService(
        relationship_retriever=retriever,
        metadata_enricher=enricher,
        config=AnchorRankingConfig(max_per_publication=10),
    )

    result = service.search("genes involved in chemoresistance in cancer", relationship_k=2)

    assert [hit.metadata["id"] for hit in result.relationships] == ["pten", "drug"]
    assert result.relationships[0].metadata["semantic_score"] == 0.84
    assert result.relationships[0].metadata["anchor_score"] == result.relationships[0].metadata["score"]


def test_semantic_search_records_raw_candidate_diagnostics() -> None:
    def retriever(_query: str, k: int, model: str | None = None) -> list[Document]:
        return [
            Document(
                page_content="PTEN cancer",
                metadata={
                    "id": "pten",
                    "score": 0.84,
                    "predicate": "biolink:associated_with",
                    "retrieval_embedding_property": "sapbert_embedding",
                    "retrieval_expected_dimensions": 768,
                    "retrieval_method": "neo4j_vector_index",
                    "retrieval_index_name": "biolink:associated_with_sapbert_vector_idx",
                },
            )
        ]

    result = SemanticSearchService(relationship_retriever=retriever, model="sapbert").search(
        "genes involved in chemoresistance in cancer",
        relationship_k=1,
    )

    assert result.diagnostics["retrieval_model"] == "sapbert"
    assert result.diagnostics["raw_candidate_k"] == 25
    assert result.diagnostics["raw_candidates"][0]["raw_rank"] == 0
    assert result.diagnostics["raw_candidates"][0]["semantic_score"] == 0.84
    assert result.diagnostics["raw_candidates"][0]["retrieval_embedding_property"] == "sapbert_embedding"
