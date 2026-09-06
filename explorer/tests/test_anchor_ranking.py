from __future__ import annotations

from langchain_core.documents import Document

from explorer.backend.semantic_search.ranking import (
    AnchorRankingConfig,
    candidate_pool_size,
    parse_query_hints,
    rerank_and_diversify_relationships,
)


def doc(
    rel_id: str,
    score: float,
    *,
    subject: str = "subject",
    obj: str = "object",
    predicate: str = "biolink:related_to",
    subject_labels: list[str] | None = None,
    object_labels: list[str] | None = None,
    publication: str | None = None,
    text: str | None = None,
) -> Document:
    metadata = {
        "id": rel_id,
        "score": score,
        "predicate": predicate,
        "original_subject": subject,
        "original_object": obj,
        "llm_subject": subject,
        "llm_object": obj,
    }
    if subject_labels is not None:
        metadata["subject_labels"] = subject_labels
    if object_labels is not None:
        metadata["object_labels"] = object_labels
    if publication:
        metadata["publications"] = [publication]
        metadata["abstract_title"] = f"title {publication}"
    return Document(page_content=text or f"{subject} {predicate} {obj}", metadata=metadata)


def test_candidate_pool_size_is_bounded() -> None:
    config = AnchorRankingConfig(candidate_multiplier=5, minimum_candidate_pool=25, maximum_candidate_pool=100)

    assert candidate_pool_size(0, config) == 0
    assert candidate_pool_size(3, config) == 25
    assert candidate_pool_size(30, config) == 100


def test_parse_query_hints_detects_explicit_terms_only() -> None:
    hints = parse_query_hints("Genes involved in cancer chemoresistance mutations")

    assert hints.wants_gene is True
    assert hints.wants_resistance_or_response is True
    assert parse_query_hints("broad cancer mechanisms").has_active_hint is False


def test_gene_labeled_edge_outranks_higher_raw_similarity_non_gene_edge() -> None:
    ranked = rerank_and_diversify_relationships(
        query="genes involved in chemoresistance in cancer",
        candidates=[
            doc("non-gene", 0.95, subject="paclitaxel", obj="neuropathy", object_labels=["biolink:Disease"]),
            doc("gene", 0.84, subject="PTEN", obj="cancer", subject_labels=["biolink:Gene"]),
        ],
        requested_k=2,
        config=AnchorRankingConfig(max_per_publication=10),
    )

    assert [hit.metadata["id"] for hit in ranked] == ["gene", "non-gene"]
    assert "gene endpoint label match" in ranked[0].metadata["ranking_reasons"]


def test_explicit_response_predicate_outranks_comparable_generic_predicate() -> None:
    ranked = rerank_and_diversify_relationships(
        query="drug resistance response in cancer",
        candidates=[
            doc("generic", 0.85, predicate="biolink:affects"),
            doc("explicit", 0.80, predicate="biolink:associated_with_resistance_to"),
        ],
        requested_k=2,
        config=AnchorRankingConfig(max_per_publication=10),
    )

    assert [hit.metadata["id"] for hit in ranked] == ["explicit", "generic"]
    assert "resistance/response predicate match" in ranked[0].metadata["ranking_reasons"]


def test_results_fall_back_to_semantic_similarity_when_no_hint_matches() -> None:
    ranked = rerank_and_diversify_relationships(
        query="genes involved in chemoresistance in cancer",
        candidates=[
            doc("high", 0.90, subject="drug-a", obj="disease-a", subject_labels=["biolink:Drug"]),
            doc("low", 0.80, subject="drug-b", obj="disease-b", subject_labels=["biolink:Drug"]),
        ],
        requested_k=2,
        config=AnchorRankingConfig(max_per_publication=10),
    )

    assert [hit.metadata["id"] for hit in ranked] == ["high", "low"]
    assert all(hit.metadata["is_semantic_fallback"] for hit in ranked)


def test_diversification_limits_publication_when_alternatives_exist() -> None:
    candidates = [
        doc("same-1", 0.99, subject="s1", obj="o1", publication="PMID:1"),
        doc("same-2", 0.98, subject="s2", obj="o2", publication="PMID:1"),
        doc("same-3", 0.97, subject="s3", obj="o3", publication="PMID:1"),
        doc("alt-1", 0.60, subject="s4", obj="o4", publication="PMID:2"),
        doc("alt-2", 0.59, subject="s5", obj="o5", publication="PMID:3"),
    ]

    ranked = rerank_and_diversify_relationships(
        query="broad cancer mechanisms",
        candidates=candidates,
        requested_k=4,
        config=AnchorRankingConfig(max_per_publication=2),
    )

    selected_pubs = [hit.metadata["publication_id"] for hit in ranked]
    assert selected_pubs.count("PMID:1") == 2
    assert len(ranked) == 4


def test_diversity_selection_is_deterministic_with_score_ties() -> None:
    candidates = [
        doc("b", 0.8, subject="s2", obj="o2", publication="PMID:2"),
        doc("a", 0.8, subject="s1", obj="o1", publication="PMID:1"),
        doc("c", 0.8, subject="s3", obj="o3", publication="PMID:3"),
    ]

    first = rerank_and_diversify_relationships(
        query="broad cancer mechanisms",
        candidates=candidates,
        requested_k=3,
    )
    second = rerank_and_diversify_relationships(
        query="broad cancer mechanisms",
        candidates=candidates,
        requested_k=3,
    )

    assert [hit.metadata["id"] for hit in first] == [hit.metadata["id"] for hit in second]
    assert [hit.metadata["id"] for hit in first] == ["b", "a", "c"]


def test_scores_and_ranking_reasons_are_preserved() -> None:
    ranked = rerank_and_diversify_relationships(
        query="genes involved in chemoresistance in cancer",
        candidates=[doc("gene", 0.84, subject="PTEN", obj="cancer", subject_labels=["biolink:Gene"])],
        requested_k=1,
    )
    metadata = ranked[0].metadata

    assert metadata["semantic_score"] == 0.84
    assert metadata["anchor_score"] == metadata["score"]
    assert metadata["anchor_score"] > metadata["semantic_score"]
    assert metadata["ranking_reasons"]
    assert metadata["ranking_components"]["semantic"] == 0.84


def test_empty_candidates_and_non_positive_requested_k_are_explicit() -> None:
    assert rerank_and_diversify_relationships("genes", [], requested_k=5) == []
    assert rerank_and_diversify_relationships("genes", [doc("a", 0.1)], requested_k=0) == []
    assert rerank_and_diversify_relationships("genes", [doc("a", 0.1)], requested_k=-1) == []


def test_litcoin_chemotherapy_neuropathy_regression_fixture() -> None:
    candidates = [
        doc(
            f"cipn-{index}",
            0.95 - index * 0.01,
            subject=f"chemotherapy-{index}",
            obj="peripheral neuropathy",
            predicate="biolink:related_to",
            object_labels=["biolink:Disease"],
            publication="PMID:CIPN",
            text="chemotherapy cancer treatment response peripheral neuropathy",
        )
        for index in range(4)
    ]
    candidates.extend(
        [
            doc(
                "pten-cancer",
                0.86,
                subject="PTEN",
                obj="cancer",
                predicate="biolink:associated_with",
                subject_labels=["biolink:Gene"],
                object_labels=["biolink:Disease"],
                publication="PMID:PTEN",
                text="PTEN gene cancer chemoresistance",
            ),
            doc(
                "opioid-genetics",
                0.78,
                subject="OPRM1",
                obj="opioid use disorder",
                predicate="biolink:genetically_associated_with",
                subject_labels=["biolink:Gene"],
                object_labels=["biolink:Disease"],
                publication="PMID:OPIOID",
                text="genetics opioid use",
            ),
        ]
    )

    ranked = rerank_and_diversify_relationships(
        query="genes involved in chemoresistance in cancer",
        candidates=candidates,
        requested_k=4,
        config=AnchorRankingConfig(max_per_publication=2),
    )

    selected_ids = [hit.metadata["id"] for hit in ranked]
    assert selected_ids[0] == "pten-cancer"
    assert selected_ids.index("pten-cancer") < selected_ids.index("cipn-0")
    assert selected_ids.index("pten-cancer") < selected_ids.index("opioid-genetics")
    assert sum(hit.metadata["publication_id"] == "PMID:CIPN" for hit in ranked) <= 2
