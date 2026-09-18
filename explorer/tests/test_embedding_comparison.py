from __future__ import annotations

import json
import math
import sys
from types import SimpleNamespace

import pytest

from analysis.embedding_comparison import (
    DuplicateRelationshipIdError,
    InconsistentEmbeddingDimensionError,
    MissingRelationshipIdError,
    RetrievalResultSet,
    add_query_projection_marker,
    apply_query_highlights,
    build_embedding_collection,
    build_retrieval_comparison_table,
    compare_query_neighborhoods,
    create_neighbor_agreement_figure,
    create_projection_figure,
    deduplicate_exact_duplicate_rows,
    load_retrieval_result_sets_from_path_search_cache,
    match_embedding_collections,
    nearest_neighbor_agreement_rows,
    nearest_neighbor_jaccard,
    nearest_neighbors,
    project_embeddings,
    projection_figure_html,
    query_nearest_relationships,
    same_metadata_fraction,
    sanitize_payload,
    top_k_overlap,
)


def row(
    rel_id: str,
    vector: list[float],
    *,
    publication_id: str = "pmid:1",
    predicate: str = "biolink:related_to",
) -> dict:
    return {
        "rel_id": rel_id,
        "embedding": vector,
        "subject": f"{rel_id}-subject",
        "object": f"{rel_id}-object",
        "predicate": predicate,
        "publication_id": publication_id,
        "abstract_title": f"{publication_id} title",
        "semantic_text": f"{rel_id} semantic text",
    }


def test_edge_matching_is_stable_regardless_of_input_order() -> None:
    openai = build_embedding_collection(
        [row("r2", [0.0, 1.0]), row("r1", [1.0, 0.0])],
        model_name="openai",
    )
    sapbert = build_embedding_collection(
        [row("r1", [1.0, 1.0]), row("r2", [1.0, -1.0])],
        model_name="sapbert",
    )

    matched = match_embedding_collections(openai, sapbert)

    assert matched.relationship_ids == ("r1", "r2")
    assert [record.relationship_id for record in matched.left_records] == ["r1", "r2"]
    assert [record.relationship_id for record in matched.right_records] == ["r1", "r2"]


def test_missing_and_duplicate_relationship_ids_are_reported_clearly() -> None:
    with pytest.raises(MissingRelationshipIdError, match="missing a relationship ID"):
        build_embedding_collection([{"embedding": [1.0, 0.0]}], model_name="openai")

    with pytest.raises(DuplicateRelationshipIdError, match="duplicate relationship ID 'r1'"):
        build_embedding_collection(
            [row("r1", [1.0, 0.0]), row("r1", [0.0, 1.0])],
            model_name="openai",
        )


def test_exact_duplicate_rows_can_be_normalized_explicitly() -> None:
    duplicate = row("r1", [1.0, 0.0])
    rows, report = deduplicate_exact_duplicate_rows([duplicate, dict(duplicate)], model_name="openai")

    assert rows == [duplicate]
    assert report["input_row_count"] == 2
    assert report["output_row_count"] == 1
    assert report["exact_duplicate_row_count"] == 1

    with pytest.raises(DuplicateRelationshipIdError, match="conflicting duplicate relationship ID 'r1'"):
        deduplicate_exact_duplicate_rows(
            [row("r1", [1.0, 0.0]), row("r1", [0.0, 1.0])],
            model_name="openai",
        )


def test_embedding_dimension_inconsistencies_are_reported() -> None:
    with pytest.raises(InconsistentEmbeddingDimensionError, match="inconsistent embedding dimensions"):
        build_embedding_collection(
            [row("r1", [1.0, 0.0]), row("r2", [1.0, 0.0, 0.0])],
            model_name="openai",
        )


def test_mismatched_model_coverage_is_reported_not_silently_discarded() -> None:
    openai = build_embedding_collection(
        [row("r1", [1.0, 0.0]), row("r2", [0.0, 1.0])],
        model_name="openai",
    )
    sapbert = build_embedding_collection(
        [row("r2", [0.0, 1.0]), row("r3", [1.0, 1.0])],
        model_name="sapbert",
    )

    matched = match_embedding_collections(openai, sapbert)

    assert matched.relationship_ids == ("r2",)
    assert matched.coverage_report["same_population"] is False
    assert matched.coverage_report["left_only_ids"] == ["r1"]
    assert matched.coverage_report["right_only_ids"] == ["r3"]


def test_nearest_neighbor_and_top_k_overlap_metrics_return_known_values() -> None:
    openai = build_embedding_collection(
        [
            row("r1", [1.0, 0.0]),
            row("r2", [0.9, 0.1]),
            row("r3", [0.0, 1.0]),
        ],
        model_name="openai",
    )
    sapbert = build_embedding_collection(
        [
            row("r1", [1.0, 0.0]),
            row("r2", [0.0, 1.0]),
            row("r3", [0.9, 0.1]),
        ],
        model_name="sapbert",
    )

    openai_neighbors = nearest_neighbors(openai, k=1)
    sapbert_neighbors = nearest_neighbors(sapbert, k=1)
    agreement = nearest_neighbor_jaccard(openai_neighbors, sapbert_neighbors, k=1)
    overlap = top_k_overlap(["r1", "r2", "r3"], ["r2", "r4", "r1"], k=2)

    assert openai_neighbors["r1"][0].relationship_id == "r2"
    assert sapbert_neighbors["r1"][0].relationship_id == "r3"
    assert agreement["per_edge"]["r1"] == 0.0
    assert overlap["shared_ids"] == ["r2"]
    assert overlap["jaccard"] == pytest.approx(1 / 3)


def test_query_nearest_relationships_returns_known_cosine_ranking() -> None:
    collection = build_embedding_collection(
        [
            row("r1", [1.0, 0.0], predicate="biolink:mentions"),
            row("r2", [0.8, 0.2], predicate="biolink:affects"),
            row("r3", [0.0, 1.0], predicate="biolink:treats"),
        ],
        model_name="openai",
    )

    results = query_nearest_relationships(collection, [1.0, 0.0], k=2)

    assert [item["relationship_id"] for item in results] == ["r1", "r2"]
    assert [item["rank"] for item in results] == [1, 2]
    assert results[0]["similarity"] == pytest.approx(1.0)
    assert results[0]["predicate_family"] == "unknown"


def test_query_neighborhood_comparison_reports_overlap() -> None:
    left = build_embedding_collection(
        [
            row("r1", [1.0, 0.0]),
            row("r2", [0.8, 0.2]),
            row("r3", [0.0, 1.0]),
        ],
        model_name="openai",
    )
    right = build_embedding_collection(
        [
            row("r1", [0.0, 1.0]),
            row("r2", [1.0, 0.0]),
            row("r3", [0.8, 0.2]),
        ],
        model_name="sapbert",
    )

    comparison = compare_query_neighborhoods(
        query="test query",
        left=left,
        right=right,
        left_query_embedding=[1.0, 0.0],
        right_query_embedding=[1.0, 0.0],
        k=2,
    )

    assert [item["relationship_id"] for item in comparison.left_results] == ["r1", "r2"]
    assert [item["relationship_id"] for item in comparison.right_results] == ["r2", "r3"]
    assert comparison.overlap["shared_ids"] == ["r2"]
    assert comparison.overlap["jaccard"] == pytest.approx(1 / 3)


def test_query_embedding_dimension_mismatch_is_reported() -> None:
    collection = build_embedding_collection([row("r1", [1.0, 0.0])], model_name="openai")

    with pytest.raises(InconsistentEmbeddingDimensionError, match="Query embedding"):
        query_nearest_relationships(collection, [1.0], k=1)


def test_projection_figure_uses_same_marker_shape_and_disables_hover() -> None:
    collection = build_embedding_collection(
        [
            row("r1", [1.0, 0.0]),
            row("r2", [0.0, 1.0]),
        ],
        model_name="openai",
    )
    projection = project_embeddings(collection)
    projection.rows[0]["is_highlighted"] = True
    projection.rows[0]["highlight_rank"] = 1

    figure = create_projection_figure([projection])

    assert {trace.marker.symbol for trace in figure.data} <= {"circle", "circle-open"}
    assert all(trace.hoverinfo == "none" for trace in figure.data)
    assert figure.layout.legend.title.text == "predicate_family"


def test_query_projection_marker_is_distinct_and_projected_with_edges() -> None:
    collection = build_embedding_collection(
        [
            row("r1", [1.0, 0.0]),
            row("r2", [0.0, 1.0]),
        ],
        model_name="openai",
    )
    projection = project_embeddings(collection, method="pca")

    rows = add_query_projection_marker(
        projection.rows,
        collection,
        [1.0, 0.0],
        query="gene query",
    )
    figure = create_projection_figure([{"rows": rows}])
    query_row = rows[-1]

    assert query_row["relationship_id"] == "query"
    assert query_row["is_query_marker"] is True
    assert query_row["query_text"] == "gene query"
    assert figure.data[-1].name == "input query"
    assert figure.data[-1].marker.symbol == "star-diamond"


def test_umap_projection_uses_cosine_metric(monkeypatch) -> None:
    captured = {}

    class FakeUMAP:
        def __init__(self, *, n_components, random_state, metric):
            captured["n_components"] = n_components
            captured["random_state"] = random_state
            captured["metric"] = metric

        def fit_transform(self, matrix):
            return [[float(index), 0.0] for index, _row in enumerate(matrix)]

    monkeypatch.setitem(sys.modules, "umap", SimpleNamespace(UMAP=FakeUMAP))
    collection = build_embedding_collection(
        [
            row("r1", [1.0, 0.0]),
            row("r2", [0.0, 1.0]),
        ],
        model_name="openai",
    )

    projection = project_embeddings(collection, method="umap", seed=42)

    assert captured == {"n_components": 2, "random_state": 42, "metric": "cosine"}
    assert projection.parameters["metric"] == "cosine"


def test_projection_html_includes_click_detail_handler() -> None:
    collection = build_embedding_collection([row("r1", [1.0, 0.0])], model_name="openai")
    projection = project_embeddings(collection)

    html = projection_figure_html([projection])

    assert "plotly_click" in html
    assert "selectedpoints" in html
    assert "Clear selection" in html
    assert "selectedRelationshipId === relationshipId" in html
    assert "Click a point to inspect relationship details." in html
    assert "relationship_id: r1" in html
    assert '"r1","relationship_id: r1' in html


def test_neighbor_agreement_rows_support_visual_summary() -> None:
    collection = build_embedding_collection(
        [
            row("r1", [1.0, 0.0], predicate="biolink:affects"),
            row("r2", [0.0, 1.0], predicate="biolink:treats"),
        ],
        model_name="openai",
    )

    rows = nearest_neighbor_agreement_rows(collection, {"k": 2, "per_edge": {"r2": 1.0, "r1": 0.0}})
    figure = create_neighbor_agreement_figure(rows)

    assert [item["relationship_id"] for item in rows] == ["r1", "r2"]
    assert rows[0]["neighbor_jaccard"] == 0.0
    assert rows[0]["predicate_family"] == "regulation"
    assert len(figure.data) == 2


def test_retrieval_comparison_table_retains_mode_and_result_type_labels() -> None:
    table = build_retrieval_comparison_table(
        [
            RetrievalResultSet(
                query="genes in cancer",
                retrieval_mode="hybrid",
                model_name="openai",
                result_set_type="final_anchor",
                results=[
                    {
                        "relationship_id": "r1",
                        "score": 0.7,
                        "predicate": "biolink:affects_response_to",
                        "embedding": [0.1] * 12,
                    }
                ],
            )
        ]
    )

    assert table[0]["retrieval_mode"] == "hybrid"
    assert table[0]["retrieval_model"] == "openai"
    assert table[0]["result_set_type"] == "final_anchor"
    assert "embedding" not in table[0]["metadata"]


def test_path_search_cache_diagnostics_load_as_retrieval_result_sets(tmp_path) -> None:
    cache_path = tmp_path / "path_search_cache.json"
    cache_path.write_text(
        json.dumps(
            {
                "entries": {
                    "cache-key": {
                        "key_payload": {
                            "query": "genes in cancer",
                            "relationship_k": 5,
                            "options": {
                                "embedding_provider": "openai",
                                "retrieval": {"mode": "hybrid"},
                            },
                        },
                        "metadata": {
                            "retrieval_diagnostics": {
                                "retrieval_model": "openai",
                                "retrieval": {"retrieval_mode": "hybrid"},
                                "raw_semantic_candidates": [{"relationship_identity": "r1", "semantic_score": 0.9}],
                                "reranked_candidates": [{"relationship_identity": "r1", "anchor_score": 1.0}],
                                "selected_anchors": [{"relationship_identity": "r1", "anchor_score": 1.0}],
                            }
                        },
                    }
                }
            }
        ),
        encoding="utf-8",
    )

    result_sets = load_retrieval_result_sets_from_path_search_cache(cache_path)
    table = build_retrieval_comparison_table(result_sets)

    assert [result.result_set_type for result in result_sets] == ["raw", "reranked", "final_anchor"]
    assert {row["retrieval_mode"] for row in table} == {"hybrid"}
    assert {row["retrieval_model"] for row in table} == {"openai"}


def test_same_publication_fraction_is_calculated_correctly() -> None:
    collection = build_embedding_collection(
        [
            row("r1", [1.0, 0.0], publication_id="p1"),
            row("r2", [0.9, 0.1], publication_id="p1"),
            row("r3", [0.0, 1.0], publication_id="p2"),
        ],
        model_name="openai",
    )
    neighbors = {
        "r1": ["r2", "r3"],
        "r2": ["r1", "r3"],
        "r3": ["r1", "r2"],
    }

    fraction = same_metadata_fraction(collection, neighbors, metadata_field="publication_id", k=2)

    assert fraction["per_edge"]["r1"] == 0.5
    assert fraction["per_edge"]["r2"] == 0.5
    assert fraction["per_edge"]["r3"] == 0.0
    assert fraction["mean_fraction"] == pytest.approx(1 / 3)


def test_projection_and_sampling_are_reproducible_with_fixed_seed() -> None:
    collection = build_embedding_collection(
        [
            row("r1", [1.0, 0.0, 0.0]),
            row("r2", [0.0, 1.0, 0.0]),
            row("r3", [0.0, 0.0, 1.0]),
            row("r4", [1.0, 1.0, 0.0]),
        ],
        model_name="openai",
    )

    first = project_embeddings(collection, seed=42, max_points=3)
    second = project_embeddings(collection, seed=42, max_points=3)

    assert first.parameters == second.parameters
    assert first.rows == second.rows
    assert len(first.rows) == 3


def test_projection_rows_include_derived_color_fields() -> None:
    collection = build_embedding_collection(
        [
            {
                **row("r1", [1.0, 0.0], predicate="biolink:mentions"),
                "subject": "PMID:1",
                "object": "NCBIGene:1",
                "subject_labels": ["biolink:Publication"],
                "object_labels": ["biolink:Gene"],
            }
        ],
        model_name="openai",
    )

    projection = project_embeddings(collection)

    assert projection.rows[0]["subject_prefix"] == "PMID"
    assert projection.rows[0]["object_prefix"] == "NCBIGene"
    assert projection.rows[0]["endpoint_prefix_pair"] == "PMID-NCBIGene"
    assert projection.rows[0]["endpoint_label_pair"] == "biolink:Publication-biolink:Gene"
    assert projection.rows[0]["is_mentions_edge"] == "True"
    assert projection.rows[0]["relationship_kind"] == "publication_mention"


def test_query_result_highlighting_uses_stable_relationship_ids() -> None:
    projection_rows = [
        {"relationship_id": "r1", "x": 0.0, "y": 0.0},
        {"relationship_id": "r2", "x": 1.0, "y": 1.0},
    ]
    result_rows = [
        {"relationship_id": "r2", "rank": 1},
        {"relationship_id": "r3", "rank": 2},
    ]

    highlighted = apply_query_highlights(projection_rows, result_rows, top_k=1, query="drug resistance")

    assert highlighted[0]["is_highlighted"] is False
    assert highlighted[1]["is_highlighted"] is True
    assert highlighted[1]["highlight_rank"] == 1


def test_raw_and_reranked_result_sets_remain_distinguishable() -> None:
    raw = RetrievalResultSet(
        query="drug resistance",
        retrieval_mode="dense",
        model_name="sapbert",
        result_set_type="raw",
        results=[{"relationship_id": "r1", "semantic_score": 0.9}],
    )
    reranked = RetrievalResultSet(
        query="drug resistance",
        retrieval_mode="dense",
        model_name="sapbert",
        result_set_type="final_anchor",
        results=[{"relationship_id": "r1", "anchor_score": 1.0}],
    )

    table = build_retrieval_comparison_table([raw, reranked])

    assert [row["result_set_type"] for row in table] == ["raw", "final_anchor"]
    assert table[0]["semantic_score"] == 0.9
    assert table[1]["anchor_score"] == 1.0


def test_embedding_arrays_are_excluded_from_exported_payloads() -> None:
    payload = sanitize_payload(
        {
            "relationship_id": "r1",
            "embedding": [0.1] * 12,
            "nested": {"sapbert_embedding": [0.2] * 12},
            "short_numeric_list": [1, 2],
        }
    )

    assert "embedding" not in payload
    assert "sapbert_embedding" not in payload["nested"]
    assert payload["short_numeric_list"] == [1, 2]


def test_empty_projection_and_overlap_are_explicit() -> None:
    collection = build_embedding_collection([], model_name="openai")

    projection = project_embeddings(collection)
    overlap = top_k_overlap([], [], k=10)

    assert projection.rows == []
    assert projection.parameters["embedding_dimension"] == 0
    assert overlap["jaccard"] == 1.0
    assert math.isnan(nearest_neighbor_jaccard({}, {}, k=5)["mean_jaccard"])
