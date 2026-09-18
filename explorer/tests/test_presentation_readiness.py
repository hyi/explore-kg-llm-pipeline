from __future__ import annotations

import csv
import json
from pathlib import Path

from analysis.presentation_readiness import (
    build_presentation_summary,
    demo_readiness_rows,
    summarize_retrieval_rows,
    write_presentation_package,
)


def test_retrieval_summary_groups_modes_and_preserves_result_type() -> None:
    rows = [
        retrieval_row(
            query="genes in cancer",
            mode="dense",
            model="openai",
            result_type="raw",
            rank=2,
            relationship_id="r2",
            publication_id="PMID:2",
            predicate="biolink:related_to",
            metadata={"is_semantic_fallback": True},
        ),
        retrieval_row(
            query="genes in cancer",
            mode="dense",
            model="openai",
            result_type="raw",
            rank=1,
            relationship_id="r1",
            publication_id="PMID:1",
            predicate="biolink:affects_response_to",
            metadata={"structural_compatibility_status": "complete_match"},
        ),
        retrieval_row(
            query="genes in cancer",
            mode="dense",
            model="openai",
            result_type="final_anchor",
            rank=1,
            relationship_id="r3",
            publication_id="PMID:1",
            predicate="biolink:affects_response_to",
            metadata={"structural_compatibility_status": "partial_match"},
        ),
    ]

    summary = summarize_retrieval_rows(rows)

    raw = next(row for row in summary if row["result_set_type"] == "raw")
    final_anchor = next(row for row in summary if row["result_set_type"] == "final_anchor")
    assert raw["result_count"] == 2
    assert raw["unique_publications"] == 2
    assert raw["complete_structural_matches"] == 1
    assert raw["semantic_fallbacks"] == 1
    assert raw["top_relationship_id"] == "r1"
    assert final_anchor["partial_structural_matches"] == 1


def test_write_presentation_package_copies_artifacts_and_writes_handoff(tmp_path: Path) -> None:
    source = tmp_path / "analysis_outputs"
    source.mkdir()
    write_minimal_artifacts(source)
    output = tmp_path / "presentation"

    package = write_presentation_package(source, output)

    assert not package.missing_source_artifacts
    assert (output / "README.md").exists()
    assert (output / "demo_runbook.md").exists()
    assert (output / "conclusions.md").exists()
    assert (output / "artifacts" / "manifest.json").exists()
    assert (output / "tables" / "retrieval_summary.csv").exists()
    summary_json = (output / "summary.json").read_text(encoding="utf-8")
    assert "sapbert_embedding" not in summary_json
    assert "openai_embedding" not in summary_json
    assert "genes involved in chemoresistance in cancer" in (output / "demo_runbook.md").read_text(encoding="utf-8")


def test_build_presentation_summary_reports_coverage_and_overlap(tmp_path: Path) -> None:
    source = tmp_path / "analysis_outputs"
    source.mkdir()
    write_minimal_artifacts(source)

    summary = build_presentation_summary(source)

    assert summary["embedding_coverage"][0]["matched_count"] == 2
    assert summary["embedding_coverage"][0]["same_population"] is True
    assert summary["nearest_neighbor_summary"]["mean_jaccard"] == 0.5
    assert summary["query_overlap_summary"]["intersection_count"] == 1
    assert summary["demo_queries"]["primary"] == "genes involved in chemoresistance in cancer"


def test_demo_readiness_marks_missing_artifacts() -> None:
    rows = demo_readiness_rows({"embedding_coverage": [{"same_population": True}]}, ["pca_projection.html"])

    assert rows[0]["status"] == "needs attention"
    assert "pca_projection.html" in rows[0]["notes"]
    assert rows[1]["status"] == "ready"


def write_minimal_artifacts(source: Path) -> None:
    (source / "manifest.json").write_text(
        json.dumps(
            {
                "coverage": {
                    "matched_count": 2,
                    "same_population": True,
                    "left_only_count": 0,
                    "right_only_count": 0,
                },
                "models": [
                    {"model_name": "openai", "relationship_count": 2, "embedding_dimension": 1536},
                    {"model_name": "sapbert", "relationship_count": 2, "embedding_dimension": 768},
                ],
                "queries": ["drug resistance in cancer", "genes involved in chemoresistance in cancer"],
                "random_seed": 13,
                "projection_parameters": {"method": "pca", "random_seed": 13},
                "embedding_arrays_included": False,
            }
        ),
        encoding="utf-8",
    )
    (source / "nearest_neighbor_metrics.json").write_text(
        json.dumps({"nearest_neighbor_jaccard": {"k": 10, "mean_jaccard": 0.5}}),
        encoding="utf-8",
    )
    (source / "query_neighbor_comparison.json").write_text(
        json.dumps(
            {
                "query": "genes involved in chemoresistance in cancer",
                "top_k": 15,
                "overlap": {"intersection_count": 1, "union_count": 3, "jaccard": 1 / 3},
                "rank_correlation": {"shared_count": 1},
            }
        ),
        encoding="utf-8",
    )
    write_csv(
        source / "anchor_diversity_summary.csv",
        [
            {
                "query": "genes involved in chemoresistance in cancer",
                "unique_publications": 2,
                "unique_predicates": 2,
                "query_compatible_predicate_matches": 1,
            }
        ],
    )
    write_csv(
        source / "retrieval_comparison.csv",
        [
            retrieval_row(
                query="genes involved in chemoresistance in cancer",
                mode="dense",
                model="openai",
                result_type="final_anchor",
                rank=1,
                relationship_id="r1",
                publication_id="PMID:1",
                predicate="biolink:affects_response_to",
                metadata={"structural_compatibility_status": "complete_match"},
            )
        ],
    )
    for artifact in (
        "pca_projection.html",
        "umap_projection.html",
        "query_projection.html",
        "nearest_neighbor_agreement.html",
    ):
        (source / artifact).write_text("<html></html>", encoding="utf-8")


def write_csv(path: Path, rows: list[dict]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def retrieval_row(
    *,
    query: str,
    mode: str,
    model: str,
    result_type: str,
    rank: int,
    relationship_id: str,
    publication_id: str,
    predicate: str,
    metadata: dict,
) -> dict:
    return {
        "query": query,
        "retrieval_mode": mode,
        "retrieval_model": model,
        "result_set_type": result_type,
        "rank": rank,
        "relationship_id": relationship_id,
        "publication_id": publication_id,
        "predicate": predicate,
        "predicate_family": "drug_response",
        "score": 0.9,
        "anchor_score": 0.9,
        "semantic_score": 0.8,
        "metadata": json.dumps(metadata),
    }
