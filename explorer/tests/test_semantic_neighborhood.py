from __future__ import annotations

from explorer.backend.semantic_search.neighborhood import (
    NeighborhoodExpansionConfig,
    rank_neighborhood_candidates,
)


def candidate(
    edge_id: str,
    *,
    predicate: str = "biolink:associated_with",
    direction: str = "outgoing",
    focus_labels: list[str] | None = None,
    neighbor_labels: list[str] | None = None,
    neighbor_label: str = "neighbor",
    edge_text: str = "",
) -> dict:
    return {
        "predicate": predicate,
        "direction": direction,
        "focus": {
            "id": "focus",
            "label": "focus",
            "labels": focus_labels or [],
        },
        "neighbor": {
            "id": f"neighbor-{edge_id}",
            "label": neighbor_label,
            "labels": neighbor_labels or [],
            "properties": {},
        },
        "edge": {
            "id": edge_id,
            "source": "focus",
            "target": f"neighbor-{edge_id}",
            "label": predicate,
            "type": predicate,
            "text": edge_text,
            "properties": {},
        },
    }


def test_query_conditioned_neighbor_ranking_prefers_matching_gene_response_edge() -> None:
    result = rank_neighborhood_candidates(
        [
            candidate(
                "generic",
                predicate="biolink:associated_with",
                neighbor_labels=["biolink:Disease"],
                neighbor_label="peripheral neuropathy",
                edge_text="cancer chemotherapy toxicity",
            ),
            candidate(
                "gene-response",
                predicate="biolink:affects_response_to",
                neighbor_labels=["biolink:Gene"],
                neighbor_label="PTEN",
                edge_text="PTEN affects response to drug treatment in cancer chemoresistance",
            ),
        ],
        query="genes involved in chemoresistance in cancer",
        config=NeighborhoodExpansionConfig(limit=2),
    )

    assert [item["edge"]["id"] for item in result.candidates] == ["gene-response", "generic"]
    assert "gene_endpoint" in result.candidates[0]["matched_query_facets"]
    assert "response_predicate" in result.candidates[0]["matched_query_facets"]
    assert "cancer_context" in result.candidates[0]["matched_query_facets"]


def test_neighborhood_filters_by_category_predicate_direction_and_limit() -> None:
    result = rank_neighborhood_candidates(
        [
            candidate(
                "keep-1",
                predicate="biolink:affects_response_to",
                direction="outgoing",
                neighbor_labels=["biolink:Gene"],
                edge_text="gene response cancer",
            ),
            candidate(
                "drop-direction",
                predicate="biolink:affects_response_to",
                direction="incoming",
                neighbor_labels=["biolink:Gene"],
                edge_text="gene response cancer",
            ),
            candidate(
                "drop-predicate",
                predicate="biolink:associated_with",
                direction="outgoing",
                neighbor_labels=["biolink:Gene"],
                edge_text="gene response cancer",
            ),
            candidate(
                "drop-category",
                predicate="biolink:affects_response_to",
                direction="outgoing",
                neighbor_labels=["biolink:ChemicalEntity"],
                edge_text="gene response cancer",
            ),
            candidate(
                "keep-2",
                predicate="biolink:affects_response_to",
                direction="outgoing",
                neighbor_labels=["biolink:Gene"],
                edge_text="gene cancer",
            ),
        ],
        query="gene response cancer",
        config=NeighborhoodExpansionConfig(
            limit=1,
            node_categories=("biolink:Gene",),
            predicates=("biolink:affects_response_to",),
            direction="outgoing",
        ),
    )

    assert [item["edge"]["id"] for item in result.candidates] == ["keep-1"]
    assert result.diagnostics["eligible_count"] == 2
    assert [item["exclusion_reason"] for item in result.diagnostics["excluded_candidates"]] == [
        "direction_filter",
        "predicate_filter",
        "node_category_filter",
    ]
    assert result.diagnostics["ranked_candidates"][1]["exclusion_reason"] == "limit"


def test_neighborhood_ranking_is_deterministic_for_score_ties() -> None:
    result = rank_neighborhood_candidates(
        [
            candidate("b", edge_text="cancer"),
            candidate("a", edge_text="cancer"),
        ],
        query="cancer",
        config=NeighborhoodExpansionConfig(limit=2),
    )

    assert [item["edge"]["id"] for item in result.candidates] == ["b", "a"]
    assert result.candidates[0]["ranking_components"]["query_text_overlap"] == 1.0
