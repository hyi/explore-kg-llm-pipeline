from __future__ import annotations

from langchain_core.documents import Document

from explorer.backend.semantic_search.query_intent import (
    CATEGORY_BIOLOGICAL_PROCESS,
    CATEGORY_DRUG_OR_CHEMICAL,
    CATEGORY_GENE,
    CATEGORY_PATHWAY,
    CATEGORY_SEQUENCE_VARIANT,
    PREDICATE_ASSOCIATION,
    PREDICATE_CAUSAL,
    PREDICATE_DRUG_RESPONSE,
    PREDICATE_REGULATION,
    PREDICATE_TREATMENT,
    parse_query_intent,
)
from explorer.backend.semantic_search.ranking import (
    ANCHOR_RANKING_STRATEGY,
    AnchorRankingConfig,
    candidate_pool_size,
    rerank_and_diversify_relationships,
    rerank_relationships_with_diagnostics,
)


def doc(
    rel_id: str,
    score: float,
    *,
    subject: str = "subject",
    obj: str = "object",
    predicate: str = "biolink:related_to",
    llm_relationship: str | None = None,
    subject_qualifier: str | None = None,
    object_qualifier: str | None = None,
    statement_qualifier: str | None = None,
    subject_labels: list[str] | None = None,
    object_labels: list[str] | None = None,
    publication: str | None = None,
    text: str | None = None,
    semantic_text: str | None = None,
) -> Document:
    metadata = {
        "id": rel_id,
        "score": score,
        "predicate": predicate,
        "original_subject": subject,
        "original_object": obj,
        "llm_subject": subject,
        "llm_object": obj,
        "llm_relationship": llm_relationship,
    }
    if subject_qualifier is not None:
        metadata["llm_subject_qualifier"] = subject_qualifier
    if object_qualifier is not None:
        metadata["llm_object_qualifier"] = object_qualifier
    if statement_qualifier is not None:
        metadata["llm_statement_qualifier"] = statement_qualifier
    if subject_labels is not None:
        metadata["subject_labels"] = subject_labels
    if object_labels is not None:
        metadata["object_labels"] = object_labels
    if publication:
        metadata["publications"] = [publication]
        metadata["abstract_title"] = f"title {publication}"
    if semantic_text is not None:
        metadata["semantic_text"] = semantic_text
    return Document(page_content=text or f"{subject} {predicate} {obj}", metadata=metadata)


def test_candidate_pool_size_is_bounded() -> None:
    config = AnchorRankingConfig(candidate_multiplier=5, minimum_candidate_pool=25, maximum_candidate_pool=100)

    assert candidate_pool_size(0, config) == 0
    assert candidate_pool_size(3, config) == 25
    assert candidate_pool_size(30, config) == 100


def test_parse_query_intent_supports_generic_category_and_predicate_classes() -> None:
    examples = {
        "genes associated with lung cancer": (CATEGORY_GENE, PREDICATE_ASSOCIATION),
        "drugs that treat glioblastoma": (CATEGORY_DRUG_OR_CHEMICAL, PREDICATE_TREATMENT),
        "variants affecting response to paclitaxel": (CATEGORY_SEQUENCE_VARIANT, PREDICATE_DRUG_RESPONSE),
        "biological processes regulated by PTEN": (CATEGORY_BIOLOGICAL_PROCESS, PREDICATE_REGULATION),
        "chemicals causing neuropathy": (CATEGORY_DRUG_OR_CHEMICAL, PREDICATE_CAUSAL),
        "pathways associated with breast cancer": (CATEGORY_PATHWAY, PREDICATE_ASSOCIATION),
    }

    for query, (category, predicate_family) in examples.items():
        intent = parse_query_intent(query)
        requested_categories = {
            request.family
            for request in (
                *intent.requested_subject_categories,
                *intent.requested_object_categories,
                *intent.requested_endpoint_categories,
            )
        }
        requested_predicates = {request.family for request in intent.requested_predicate_families}
        assert category in requested_categories
        assert predicate_family in requested_predicates
        assert intent.has_structural_intent is True


def test_involved_in_is_relational_scaffolding_not_predicate_family() -> None:
    intent = parse_query_intent("genes involved in chemoresistance in cancer")

    assert {request.family for request in intent.requested_endpoint_categories} == {CATEGORY_GENE}
    assert {request.family for request in intent.requested_predicate_families} == {PREDICATE_DRUG_RESPONSE}
    assert "chemoresistance" in intent.content_terms
    assert any(expression.kind == "relational_scaffold" for expression in intent.recognized_expressions)


def test_broad_topic_query_has_no_structural_intent() -> None:
    intent = parse_query_intent("broad oxidative stress literature")

    assert intent.has_structural_intent is False
    assert intent.requested_endpoint_categories == ()
    assert intent.requested_predicate_families == ()
    assert "oxidative" in intent.content_terms


def test_gene_association_candidate_beats_unrelated_disease_candidate() -> None:
    ranked = rerank_and_diversify_relationships(
        query="genes associated with lung cancer",
        candidates=[
            doc("disease-only", 0.99, subject="smoking", obj="lung cancer", object_labels=["biolink:Disease"]),
            doc(
                "gene-association",
                0.90,
                subject="TP53",
                obj="lung cancer",
                predicate="biolink:associated_with",
                subject_labels=["biolink:Gene"],
                object_labels=["biolink:Disease"],
            ),
        ],
        requested_k=2,
        config=AnchorRankingConfig(max_per_publication=10),
    )

    assert [hit.metadata["id"] for hit in ranked] == ["gene-association", "disease-only"]
    assert ranked[0].metadata["structural_compatibility_status"] == "complete_match"


def test_drug_treatment_role_compatibility_is_conservative() -> None:
    ranked = rerank_and_diversify_relationships(
        query="drugs that treat glioblastoma",
        candidates=[
            doc(
                "reversed",
                0.98,
                subject="glioblastoma",
                obj="temozolomide",
                predicate="biolink:treats",
                subject_labels=["biolink:Disease"],
                object_labels=["biolink:Drug"],
            ),
            doc(
                "drug-treats",
                0.90,
                subject="temozolomide",
                obj="glioblastoma",
                predicate="biolink:treats",
                subject_labels=["biolink:Drug"],
                object_labels=["biolink:Disease"],
            ),
        ],
        requested_k=2,
        config=AnchorRankingConfig(max_per_publication=10),
    )

    assert [hit.metadata["id"] for hit in ranked] == ["drug-treats", "reversed"]
    assert ranked[1].metadata["role_compatibility"]["status"] == "contradiction"
    assert ranked[1].metadata["incompatibility_penalties"]


def test_variant_drug_response_uses_schema_intent_without_special_query_branch() -> None:
    ranked = rerank_and_diversify_relationships(
        query="variants affecting response to paclitaxel",
        candidates=[
            doc("gene-response", 0.96, predicate="biolink:affects_response_to", subject_labels=["biolink:Gene"]),
            doc(
                "variant-response",
                0.88,
                subject="variant allele",
                obj="paclitaxel",
                predicate="biolink:affects_response_to",
                subject_labels=["biolink:SequenceVariant"],
                object_labels=["biolink:Drug"],
            ),
        ],
        requested_k=2,
        config=AnchorRankingConfig(max_per_publication=10),
    )

    assert ranked[0].metadata["id"] == "variant-response"
    assert ranked[0].metadata["predicate_family_compatibility"]["status"] == "match"
    assert ranked[0].metadata["endpoint_category_compatibility"]["matched"] == [
        CATEGORY_DRUG_OR_CHEMICAL,
        CATEGORY_SEQUENCE_VARIANT,
    ]


def test_drug_response_predicate_requires_argument_support_for_complete_match() -> None:
    ranked = rerank_and_diversify_relationships(
        query="genes involved in chemoresistance in cancer",
        candidates=[
            doc(
                "gene-disease-predicate-only",
                0.98,
                subject="SLC34A3",
                obj="Unequal Limb Length",
                predicate="biolink:associated_with_resistance_to",
                llm_relationship="susceptible to",
                statement_qualifier="sequence misalignment during meiosis",
                subject_labels=["biolink:Gene"],
                object_labels=["biolink:Disease"],
            ),
            doc(
                "gene-drug-response",
                0.70,
                subject="TP53",
                obj="paclitaxel",
                predicate="biolink:affects_response_to",
                llm_relationship="affects response to",
                subject_labels=["biolink:Gene"],
                object_labels=["biolink:Drug"],
            ),
        ],
        requested_k=2,
        config=AnchorRankingConfig(max_per_publication=10),
    )

    assert [hit.metadata["id"] for hit in ranked] == ["gene-drug-response", "gene-disease-predicate-only"]
    weak = ranked[1].metadata
    assert weak["structural_compatibility_status"] == "partial_match"
    assert weak["predicate_family_compatibility"]["status"] == "partial"
    support = weak["predicate_family_compatibility"]["drug_response_argument_compatibility"]
    assert support["endpoint_role_support"]["status"] == "partial"
    assert support["qualifier_support"]["matched_terms"] == []


def test_drug_response_qualifier_can_validate_without_drug_endpoint() -> None:
    ranked = rerank_and_diversify_relationships(
        query="genes involved in chemoresistance in cancer",
        candidates=[
            doc(
                "gene-disease-treatment-response",
                0.70,
                subject="EGFR",
                obj="glioblastoma",
                predicate="biolink:associated_with_resistance_to",
                llm_relationship="associated with",
                statement_qualifier="temozolomide treatment resistance",
                subject_labels=["biolink:Gene"],
                object_labels=["biolink:Disease"],
            ),
        ],
        requested_k=1,
    )

    metadata = ranked[0].metadata
    assert metadata["structural_compatibility_status"] == "complete_match"
    assert metadata["predicate_family_compatibility"]["status"] == "match"
    support = metadata["predicate_family_compatibility"]["drug_response_argument_compatibility"]
    assert support["qualifier_support"]["matched_terms"]


def test_biological_process_regulated_by_gene_uses_passive_role_expectation() -> None:
    ranked = rerank_and_diversify_relationships(
        query="biological processes regulated by PTEN",
        candidates=[
            doc("gene-only", 0.94, subject="PTEN", obj="protein", subject_labels=["biolink:Gene"]),
            doc(
                "regulated-process",
                0.88,
                subject="PTEN",
                obj="apoptosis",
                predicate="biolink:regulates",
                subject_labels=["biolink:Gene"],
                object_labels=["biolink:BiologicalProcess"],
            ),
        ],
        requested_k=2,
        config=AnchorRankingConfig(max_per_publication=10),
    )

    assert ranked[0].metadata["id"] == "regulated-process"
    assert ranked[0].metadata["role_compatibility"]["status"] == "match"


def test_chemicals_causing_phenotype_prefers_causal_claim() -> None:
    ranked = rerank_and_diversify_relationships(
        query="chemicals causing neuropathy",
        candidates=[
            doc("chemical-mentioned", 0.96, predicate="biolink:mentions", object_labels=["biolink:ChemicalEntity"]),
            doc(
                "chemical-causes",
                0.86,
                subject="paclitaxel",
                obj="neuropathy",
                predicate="biolink:causes",
                subject_labels=["biolink:ChemicalEntity"],
                object_labels=["biolink:PhenotypicFeature"],
            ),
        ],
        requested_k=2,
        config=AnchorRankingConfig(max_per_publication=10),
    )

    assert ranked[0].metadata["id"] == "chemical-causes"
    assert ranked[0].metadata["relationship_quality_tier"] == "direct_assertion"


def test_query_text_overlap_is_diagnostic_only_not_scored() -> None:
    ranked = rerank_and_diversify_relationships(
        query="broad oxidative stress topic",
        candidates=[
            doc("overlap", 0.80, text="oxidative stress topic", semantic_text="oxidative stress topic"),
            doc("no-overlap", 0.79, text="unrelated claim", semantic_text="unrelated claim"),
        ],
        requested_k=2,
        config=AnchorRankingConfig(max_per_publication=10),
    )

    assert [hit.metadata["id"] for hit in ranked] == ["overlap", "no-overlap"]
    assert "query_text_overlap" not in ranked[0].metadata["ranking_components"]
    assert ranked[0].metadata["claim_term_matches"]


def test_no_broad_abstract_derived_compatibility_boost() -> None:
    ranked = rerank_and_diversify_relationships(
        query="genes associated with lung cancer",
        candidates=[
            doc(
                "abstract-only",
                0.94,
                subject="chemical",
                obj="phenotype",
                subject_labels=["biolink:ChemicalEntity"],
                object_labels=["biolink:PhenotypicFeature"],
                semantic_text="lung cancer gene association abstract context",
            ),
            doc(
                "gene-claim",
                0.86,
                subject="EGFR",
                obj="lung cancer",
                predicate="biolink:associated_with",
                subject_labels=["biolink:Gene"],
                object_labels=["biolink:Disease"],
            ),
        ],
        requested_k=2,
        config=AnchorRankingConfig(max_per_publication=10),
    )

    assert ranked[0].metadata["id"] == "gene-claim"
    assert ranked[1].metadata["endpoint_category_compatibility"]["status"] == "missing"


def test_no_forced_unrelated_facet_coverage_insertion() -> None:
    ranked = rerank_and_diversify_relationships(
        query="genes affecting response to paclitaxel",
        candidates=[
            doc(
                f"gene-association-{index}",
                0.95 - index * 0.01,
                predicate="biolink:associated_with",
                subject=f"GENE{index}",
                obj="disease",
                subject_labels=["biolink:Gene"],
            )
            for index in range(3)
        ]
        + [
            doc(
                "low-response-only",
                0.10,
                predicate="biolink:affects_response_to",
                subject="chemical",
                obj="drug",
                subject_labels=["biolink:ChemicalEntity"],
                object_labels=["biolink:Drug"],
            )
        ],
        requested_k=2,
        config=AnchorRankingConfig(max_per_publication=10),
    )

    assert "low-response-only" not in [hit.metadata["id"] for hit in ranked]


def test_mentions_suppressed_in_primary_pass_and_available_as_labeled_fallback() -> None:
    result = rerank_relationships_with_diagnostics(
        query="genes associated with disease",
        candidates=[
            doc("mention-1", 0.99, predicate="biolink:mentions", subject="GENE1", obj="disease", subject_labels=["biolink:Gene"]),
            doc("mention-2", 0.98, predicate="biolink:mentions", subject="GENE2", obj="disease", subject_labels=["biolink:Gene"]),
            doc("claim", 0.50, predicate="biolink:associated_with", subject="GENE3", obj="disease", subject_labels=["biolink:Gene"]),
        ],
        requested_k=3,
        config=AnchorRankingConfig(max_per_publication=10),
    )

    selected = result.diagnostics["selected_anchors"]
    assert selected[0]["relationship_identity"] == "claim"
    assert selected[0]["selection_pass"] == "primary_diversity_pass"
    assert selected[1]["selection_pass"] == "contextual_mention_fallback_pass"


def test_no_structural_intent_preserves_retrieval_order_except_quality() -> None:
    ranked = rerank_and_diversify_relationships(
        query="broad oxidative stress literature",
        candidates=[
            doc("first", 0.80, predicate="biolink:associated_with"),
            doc("second", 0.79, predicate="biolink:correlated_with"),
            doc("mention", 0.99, predicate="biolink:mentions"),
        ],
        requested_k=3,
        config=AnchorRankingConfig(max_per_publication=10),
    )

    assert [hit.metadata["id"] for hit in ranked] == ["first", "second", "mention"]
    assert all(hit.metadata["structural_compatibility_status"] == "no_structural_intent" for hit in ranked)


def test_diversification_limits_publication_when_alternatives_exist() -> None:
    candidates = [
        doc("same-1", 0.99, subject="s1", obj="o1", publication="PMID:1"),
        doc("same-2", 0.98, subject="s2", obj="o2", publication="PMID:1"),
        doc("same-3", 0.97, subject="s3", obj="o3", publication="PMID:1"),
        doc("alt-1", 0.60, subject="s4", obj="o4", publication="PMID:2"),
        doc("alt-2", 0.59, subject="s5", obj="o5", publication="PMID:3"),
    ]

    ranked = rerank_and_diversify_relationships(
        query="broad mechanisms",
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
        query="broad mechanisms",
        candidates=candidates,
        requested_k=3,
    )
    second = rerank_and_diversify_relationships(
        query="broad mechanisms",
        candidates=candidates,
        requested_k=3,
    )

    assert [hit.metadata["id"] for hit in first] == [hit.metadata["id"] for hit in second]
    assert [hit.metadata["id"] for hit in first] == ["b", "a", "c"]


def test_scores_and_generic_diagnostics_are_preserved() -> None:
    result = rerank_relationships_with_diagnostics(
        query="genes associated with lung cancer",
        candidates=[doc("gene", 0.84, subject="PTEN", obj="cancer", subject_labels=["biolink:Gene"])],
        requested_k=1,
    )
    metadata = result.relationships[0].metadata
    diagnostic = result.diagnostics["selected_anchors"][0]

    assert result.diagnostics["ranking_strategy"] == ANCHOR_RANKING_STRATEGY
    assert result.diagnostics["query_intent"]["has_structural_intent"] is True
    assert metadata["semantic_score"] == 0.84
    assert metadata["anchor_score"] == metadata["score"]
    assert metadata["ranking_reasons"]
    assert "compatibility_components" in diagnostic
    assert "query_intent" in diagnostic


def test_empty_candidates_and_non_positive_requested_k_are_explicit() -> None:
    assert rerank_and_diversify_relationships("genes", [], requested_k=5) == []
    assert rerank_and_diversify_relationships("genes", [doc("a", 0.1)], requested_k=0) == []
    assert rerank_and_diversify_relationships("genes", [doc("a", 0.1)], requested_k=-1) == []


def test_litcoin_chemoresistance_regression_not_tuned_to_cancer_flag() -> None:
    candidates = [
        doc(
            f"cipn-{index}",
            0.95 - index * 0.01,
            subject=f"chemotherapy-{index}",
            obj="peripheral neuropathy",
            predicate="biolink:related_to",
            object_labels=["biolink:Disease"],
            publication="PMID:CIPN",
            text="chemotherapy treatment response peripheral neuropathy",
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
    assert sum(hit.metadata["publication_id"] == "PMID:CIPN" for hit in ranked) <= 2
