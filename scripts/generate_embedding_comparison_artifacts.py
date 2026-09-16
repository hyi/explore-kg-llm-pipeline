from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from analysis.embedding_comparison import (
    build_embedding_collection,
    comparison_manifest,
    deduplicate_exact_duplicate_rows,
    match_embedding_collections,
    nearest_neighbor_agreement_rows,
    nearest_neighbor_jaccard,
    nearest_neighbors,
    project_embeddings,
    read_jsonl_rows,
    same_metadata_fraction,
    write_json,
    write_neighbor_agreement_html,
    write_projection_html,
    write_rows_csv,
)

DEFAULT_QUERIES = (
    "drug resistance in cancer",
    "genes involved in chemoresistance in cancer",
    "PTEN cancer chemoresistance",
    "therapeutic response relationships",
)


def generate_artifacts(args: argparse.Namespace) -> dict:
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    openai_rows, openai_normalization = deduplicate_exact_duplicate_rows(
        read_jsonl_rows(args.openai_relationships),
        model_name="openai",
    )
    sapbert_rows, sapbert_normalization = deduplicate_exact_duplicate_rows(
        read_jsonl_rows(args.sapbert_relationships),
        model_name="sapbert",
    )
    openai_edges = build_embedding_collection(
        openai_rows,
        model_name="openai",
        embedding_key="embedding",
    )
    sapbert_edges = build_embedding_collection(
        sapbert_rows,
        model_name="sapbert",
        embedding_key="sapbert_embedding",
    )
    matched = match_embedding_collections(openai_edges, sapbert_edges)

    openai_projection = project_embeddings(
        openai_edges,
        relationship_ids=matched.relationship_ids,
        method=args.projection_method,
        seed=args.seed,
        max_points=args.max_projection_points,
    )
    sapbert_projection = project_embeddings(
        sapbert_edges,
        relationship_ids=matched.relationship_ids,
        method=args.projection_method,
        seed=args.seed,
        max_points=args.max_projection_points,
    )
    projection_rows = openai_projection.rows + sapbert_projection.rows
    write_rows_csv(projection_rows, output_dir / f"{args.projection_method}_projection_rows.csv")

    html_path = None
    try:
        html_path = write_projection_html(
            [openai_projection, sapbert_projection],
            output_dir / f"{args.projection_method}_projection.html",
            color_field=args.color_field,
            title="LitCoin relationship embedding comparison",
        )
    except Exception as exc:  # noqa: BLE001 - HTML output is optional; CSV/JSON artifacts still matter.
        html_path = f"not generated: {type(exc).__name__}: {exc}"

    openai_neighbors = nearest_neighbors(openai_edges, k=args.top_k, relationship_ids=matched.relationship_ids)
    sapbert_neighbors = nearest_neighbors(sapbert_edges, k=args.top_k, relationship_ids=matched.relationship_ids)
    nearest_neighbor_agreement = nearest_neighbor_jaccard(openai_neighbors, sapbert_neighbors, k=args.top_k)
    neighbor_metrics = {
        "top_k": args.top_k,
        "nearest_neighbor_jaccard": _json_safe(nearest_neighbor_agreement),
        "openai_same_publication_fraction": _json_safe(
            same_metadata_fraction(openai_edges, openai_neighbors, metadata_field="publication_id", k=args.top_k)
        ),
        "sapbert_same_publication_fraction": _json_safe(
            same_metadata_fraction(sapbert_edges, sapbert_neighbors, metadata_field="publication_id", k=args.top_k)
        ),
        "openai_same_predicate_family_fraction": _json_safe(
            same_metadata_fraction(openai_edges, openai_neighbors, metadata_field="predicate_family", k=args.top_k)
        ),
        "sapbert_same_predicate_family_fraction": _json_safe(
            same_metadata_fraction(sapbert_edges, sapbert_neighbors, metadata_field="predicate_family", k=args.top_k)
        ),
    }
    write_json(neighbor_metrics, output_dir / "nearest_neighbor_metrics.json")
    agreement_rows = nearest_neighbor_agreement_rows(openai_edges, nearest_neighbor_agreement)
    write_rows_csv(agreement_rows, output_dir / "nearest_neighbor_agreement_rows.csv")
    agreement_html_path = None
    try:
        agreement_html_path = write_neighbor_agreement_html(
            agreement_rows,
            output_dir / "nearest_neighbor_agreement.html",
            title=f"OpenAI/SapBERT top-{args.top_k} nearest-neighbor agreement",
        )
    except Exception as exc:  # noqa: BLE001 - HTML output is optional; CSV/JSON artifacts still matter.
        agreement_html_path = f"not generated: {type(exc).__name__}: {exc}"

    manifest = comparison_manifest(
        collections=[openai_edges, sapbert_edges],
        projection_parameters={
            "method": args.projection_method,
            "random_seed": args.seed,
            "top_k": args.top_k,
            "max_projection_points": args.max_projection_points,
        },
        queries=args.query,
        seed=args.seed,
    )
    manifest["coverage"] = matched.coverage_report
    manifest["input_normalization"] = {
        "openai": openai_normalization,
        "sapbert": sapbert_normalization,
    }
    manifest["artifacts"] = {
        "projection_rows": str(output_dir / f"{args.projection_method}_projection_rows.csv"),
        "projection_html": str(html_path),
        "nearest_neighbor_metrics": str(output_dir / "nearest_neighbor_metrics.json"),
        "nearest_neighbor_agreement_rows": str(output_dir / "nearest_neighbor_agreement_rows.csv"),
        "nearest_neighbor_agreement_html": str(agreement_html_path),
    }
    manifest["metadata_limitations"] = metadata_limitations(openai_edges.records)
    write_json(manifest, output_dir / "manifest.json")
    return manifest


def metadata_limitations(records) -> list[str]:
    if not records:
        return ["No relationship records were loaded."]
    available = set().union(*(record.metadata.keys() for record in records))
    limitations = []
    required_fields = {
        "subject_labels": "endpoint labels are absent",
        "object_labels": "endpoint labels are absent",
        "publication_id": "stable publication IDs are absent",
        "abstract_title": "abstract titles are absent",
    }
    for field, message in required_fields.items():
        if field not in available and message not in limitations:
            limitations.append(message)
    if "semantic_text" not in available:
        limitations.append("semantic text is absent")
    return limitations


def _json_safe(value):
    if isinstance(value, dict):
        return {key: _json_safe(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_json_safe(item) for item in value]
    if isinstance(value, float) and math.isnan(value):
        return None
    return value


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate Workstream 2 embedding comparison artifacts.")
    parser.add_argument(
        "--openai-relationships",
        type=Path,
        default=Path("scripts/data/relationship_embeddings.jsonl"),
        help="OpenAI relationship embedding JSONL export.",
    )
    parser.add_argument(
        "--sapbert-relationships",
        type=Path,
        default=Path("scripts/data/sapbert_relationship_embeddings.jsonl"),
        help="SapBERT relationship embedding JSONL export.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("analysis/outputs/embedding_comparison"),
        help="Directory for generated CSV, JSON, and HTML artifacts.",
    )
    parser.add_argument("--projection-method", choices=("pca", "umap"), default="pca")
    parser.add_argument(
        "--color-field",
        default="predicate_family",
        help=(
            "Projection row field used for color. Useful values include predicate, "
            "predicate_family, relationship_kind, is_mentions_edge, endpoint_prefix_pair, "
            "endpoint_label_pair, and publication_id."
        ),
    )
    parser.add_argument("--seed", type=int, default=13)
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument(
        "--max-projection-points",
        type=int,
        default=None,
        help="Optional deterministic sample size for large projection figures.",
    )
    parser.add_argument("--query", action="append", default=list(DEFAULT_QUERIES))
    return parser.parse_args()


if __name__ == "__main__":
    generated_manifest = generate_artifacts(parse_args())
    print(
        "Generated embedding comparison artifacts: "
        f"{generated_manifest['artifacts']} with coverage {generated_manifest['coverage']}"
    )
