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
    assert hints.wants_genetic_variant is True
    assert hints.wants_resistance_or_response is True
    assert hints.wants_cancer is True
    assert parse_query_hints("broad mechanisms").has_active_hint is False


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
    assert "gene endpoint category match" in ranked[0].metadata["ranking_reasons"]


def test_explicit_response_predicate_outranks_comparable_generic_predicate() -> None:
    ranked = rerank_and_diversify_relationships(
        query="drug resistance response in cancer",
        candidates=[
            doc("generic", 0.85, predicate="biolink:affects", subject_labels=["biolink:ChemicalEntity"], text="drug affects tumor sensitivity"),
            doc(
                "explicit",
                0.80,
                predicate="biolink:associated_with_resistance_to",
                subject="drug resistance",
                obj="cancer treatment",
                text="drug resistance cancer treatment",
            ),
        ],
        requested_k=2,
        config=AnchorRankingConfig(max_per_publication=10),
    )

    assert [hit.metadata["id"] for hit in ranked] == ["explicit", "generic"]
    assert "drug response predicate context match" in ranked[0].metadata["ranking_reasons"]


def test_response_predicate_needs_drug_or_resistance_context_for_strong_boost() -> None:
    ranked = rerank_and_diversify_relationships(
        query="drug resistance response in cancer",
        candidates=[
            doc(
                "cell-proliferation-response",
                0.90,
                predicate="biolink:decreases_response_to",
                subject="DACH1",
                obj="cell proliferation",
                subject_labels=["biolink:Gene"],
                object_labels=["biolink:BiologicalProcess"],
                text="DACH1 decreases glioma cell proliferation",
            ),
            doc(
                "drug-sensitivity",
                0.80,
                predicate="biolink:decreases_response_to",
                subject="CHEBI:1",
                obj="tumor sensitivity",
                subject_labels=["biolink:ChemicalEntity"],
                object_labels=["biolink:Disease"],
                text="drug decreases tumor sensitivity",
            ),
        ],
        requested_k=2,
        config=AnchorRankingConfig(max_per_publication=10),
    )

    assert [hit.metadata["id"] for hit in ranked] == ["drug-sensitivity", "cell-proliferation-response"]
    assert ranked[0].metadata["ranking_components"]["predicate"] == 0.30
    assert ranked[1].metadata["ranking_components"]["predicate"] == 0.0


def test_gene_query_prefers_gene_endpoint_over_variant_without_variant_hint() -> None:
    ranked = rerank_and_diversify_relationships(
        query="genes involved in chemoresistance in cancer",
        candidates=[
            doc(
                "variant",
                0.90,
                subject="variant allele",
                obj="phenotype",
                subject_labels=["biolink:GenomicEntity"],
                text="genes involved in development",
            ),
            doc(
                "gene",
                0.80,
                subject="PTEN",
                obj="cancer",
                subject_labels=["biolink:Gene"],
                object_labels=["biolink:Disease"],
                text="PTEN cancer",
            ),
        ],
        requested_k=2,
        config=AnchorRankingConfig(max_per_publication=10),
    )

    assert [hit.metadata["id"] for hit in ranked] == ["gene", "variant"]
    assert ranked[0].metadata["ranking_components"]["gene_endpoint"] == 0.20
    assert ranked[1].metadata["ranking_components"]["gene_endpoint"] == 0.06


def test_cancer_hint_boosts_cancer_context() -> None:
    ranked = rerank_and_diversify_relationships(
        query="genes involved in chemoresistance in cancer",
        candidates=[
            doc("skin", 0.82, subject="GENE1", obj="skin phenotype", subject_labels=["biolink:Gene"]),
            doc("cancer", 0.80, subject="GENE2", obj="lung cancer", subject_labels=["biolink:Gene"], text="GENE2 lung cancer"),
        ],
        requested_k=2,
        config=AnchorRankingConfig(max_per_publication=10),
    )

    assert [hit.metadata["id"] for hit in ranked] == ["cancer", "skin"]
    assert ranked[0].metadata["ranking_components"]["cancer_context"] == 0.10


def test_selected_anchors_cover_available_response_facet() -> None:
    ranked = rerank_and_diversify_relationships(
        query="genes involved in chemoresistance in cancer",
        candidates=[
            doc(f"gene-{index}", 0.90 - index * 0.01, subject=f"GENE{index}", obj="cancer", subject_labels=["biolink:Gene"], text="gene cancer")
            for index in range(3)
        ]
        + [
            doc(
                "resistance",
                0.70,
                predicate="biolink:associated_with_resistance_to",
                subject="resistance phenotype",
                obj="treatment procedure",
                text="chemoresistance resistant treatment response",
            )
        ],
        requested_k=3,
        config=AnchorRankingConfig(max_per_publication=10),
    )

    assert "resistance" in [hit.metadata["id"] for hit in ranked]


def test_hypoxia_responsive_text_does_not_match_chemoresistance_context() -> None:
    ranked = rerank_and_diversify_relationships(
        query="genes involved in chemoresistance in cancer",
        candidates=[
            doc(
                "hypoxia",
                0.90,
                predicate="biolink:increases_response_to",
                subject="puromycin aminonucleoside model",
                obj="hypoxia-responsive transgene expression",
                text="hypoxia-responsive reporter vector in renal disease",
            ),
            doc(
                "treatment-response",
                0.80,
                predicate="biolink:increases_response_to",
                subject="drug",
                obj="tumor response",
                subject_labels=["biolink:ChemicalEntity"],
                text="drug treatment response in tumor cells",
            ),
        ],
        requested_k=2,
        config=AnchorRankingConfig(max_per_publication=10),
    )

    hypoxia = next(hit for hit in ranked if hit.metadata["id"] == "hypoxia")
    treatment_response = next(hit for hit in ranked if hit.metadata["id"] == "treatment-response")

    assert "resistance_response_context" not in hypoxia.metadata["ranking_components"]
    assert hypoxia.metadata["ranking_components"]["predicate"] == 0.0
    assert treatment_response.metadata["ranking_components"]["resistance_response_context"] == 0.10
    assert treatment_response.metadata["ranking_components"]["predicate"] == 0.30


def test_resistance_predicate_without_biomedical_context_gets_no_predicate_or_context_boost() -> None:
    ranked = rerank_and_diversify_relationships(
        query="genes involved in chemoresistance in cancer",
        candidates=[
            doc(
                "opioid-policy",
                0.90,
                predicate="biolink:associated_with_resistance_to",
                subject="High-quality opioid policy data",
                obj="Opioid policy researchers",
                text=(
                    "Methodological challenges facing opioid policy researchers "
                    "when evaluating policy effectiveness using observational data"
                ),
            ),
            doc(
                "adam12",
                0.70,
                predicate="biolink:affects",
                subject="ADAM12",
                obj="breast cancer chemoresistance",
                subject_labels=["biolink:Gene"],
                text="ADAM12 is upregulated in breast cancers and is a predictor of chemoresistance",
            ),
        ],
        requested_k=2,
        config=AnchorRankingConfig(max_per_publication=10),
    )

    opioid_policy = next(hit for hit in ranked if hit.metadata["id"] == "opioid-policy")
    assert opioid_policy.metadata["ranking_components"]["predicate"] == 0.0
    assert "resistance_response_context" not in opioid_policy.metadata["ranking_components"]
    assert ranked[0].metadata["id"] == "adam12"


def test_mixed_chemical_categories_do_not_receive_gene_boost() -> None:
    ranked = rerank_and_diversify_relationships(
        query="genes involved in chemoresistance in cancer",
        candidates=[
            doc(
                "puromycin",
                0.80,
                subject="puromycin aminonucleoside",
                obj="kidney disease",
                subject_labels=[
                    "biolink:ChemicalEntity",
                    "biolink:ChemicalEntityOrGeneOrGeneProduct",
                ],
            ),
            doc(
                "conditioned-media",
                0.79,
                subject="conditioned culture media",
                obj="bone mineralization",
                subject_labels=[
                    "biolink:ChemicalOrDrugOrTreatment",
                    "biolink:ChemicalEntityOrGeneOrGeneProduct",
                ],
                object_labels=["biolink:BiologicalProcess"],
            ),
        ],
        requested_k=2,
        config=AnchorRankingConfig(max_per_publication=10),
    )

    assert [hit.metadata["ranking_components"]["gene_endpoint"] for hit in ranked] == [0.0, 0.0]
    assert all("gene endpoint category match" not in hit.metadata["ranking_reasons"] for hit in ranked)


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
