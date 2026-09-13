from __future__ import annotations

from explorer.backend.semantic_search.keyword import rank_keyword_records
from explorer.backend.semantic_search.query_intent import build_bm25_content_query
from explorer.backend.semantic_search.retrieval import tokenize_keyword_query


def test_bm25_content_query_removes_structural_terms_not_biomedical_content() -> None:
    query = build_bm25_content_query("genes involved in chemoresistance in cancer")

    assert query.final_query == "chemoresistance cancer"
    assert query.tokens == ("chemoresistance", "cancer")
    assert set(query.removed_or_deemphasized_terms) == {"genes", "in", "involved"}
    assert query.fallback_used is False
    assert {expression.text for expression in query.recognized_structural_expressions} == {
        "genes",
        "involved in",
    }


def test_bm25_content_query_keeps_relation_specific_terms_and_named_entities() -> None:
    query = build_bm25_content_query("variants affecting response to paclitaxel")

    assert query.final_query == "affecting response paclitaxel"
    assert query.tokens == ("affecting", "response", "paclitaxel")
    assert "variants" in query.removed_or_deemphasized_terms
    assert "response" not in query.removed_or_deemphasized_terms


def test_bm25_content_query_falls_back_when_only_structural_terms_remain() -> None:
    query = build_bm25_content_query("genes related to")

    assert query.fallback_used is True
    assert query.tokens == ("genes", "related")
    assert query.final_query == "genes related"


def test_edge_claim_bm25_ignores_context_only_match() -> None:
    records = [
        {
            "relationship_id": "context-only",
            "predicate": "biolink:mentions",
            "edge_text": "unrelated edge claim",
            "title_text": "",
            "context_text": "genes involved in chemoresistance in cancer",
        },
        {
            "relationship_id": "edge-match",
            "predicate": "biolink:affects",
            "edge_text": "gene affects chemoresistance in cancer",
            "title_text": "",
            "context_text": "",
        },
    ]

    ranked = rank_keyword_records(
        records,
        list(build_bm25_content_query("genes involved in chemoresistance in cancer").tokens),
        limit=2,
    )

    assert [row["relationship_id"] for row in ranked] == ["edge-match"]
    assert ranked[0]["keyword_scoring_method"] == "edge_claim_bm25_v1"
    assert ranked[0]["keyword_components"]["weighted_edge"] > 0
    assert "weighted_context" not in ranked[0]["keyword_components"]
    assert ranked[0]["keyword_context_matches"] == []


def test_edge_claim_bm25_uses_idf_to_demote_common_terms() -> None:
    records = [
        {
            "relationship_id": "rare",
            "predicate": "biolink:affects",
            "edge_text": "chemoresistance gene cancer",
            "title_text": "",
            "context_text": "",
        },
        {
            "relationship_id": "common-1",
            "predicate": "biolink:mentions",
            "edge_text": "gene cancer",
            "title_text": "",
            "context_text": "",
        },
        {
            "relationship_id": "common-2",
            "predicate": "biolink:mentions",
            "edge_text": "gene cancer",
            "title_text": "",
            "context_text": "",
        },
    ]

    ranked = rank_keyword_records(
        records,
        tokenize_keyword_query("gene chemoresistance cancer"),
        limit=3,
    )

    assert ranked[0]["relationship_id"] == "rare"
    assert "chemoresistance" in ranked[0]["keyword_edge_matches"]
