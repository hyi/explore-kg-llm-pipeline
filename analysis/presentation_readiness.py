from __future__ import annotations

import csv
import json
import shutil
from collections import defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

DEFAULT_ANALYSIS_OUTPUT_DIR = Path("analysis/outputs/embedding_comparison")
DEFAULT_PRESENTATION_DIR = Path("presentation")
DEFAULT_PRIMARY_QUERY = "genes involved in chemoresistance in cancer"
DEFAULT_BACKUP_QUERY = "drug resistance in cancer"

PRESENTATION_ARTIFACTS = (
    "pca_projection.html",
    "umap_projection.html",
    "query_projection.html",
    "nearest_neighbor_agreement.html",
    "retrieval_comparison.csv",
    "anchor_diversity_summary.csv",
    "query_neighbor_comparison.json",
    "nearest_neighbor_metrics.json",
    "manifest.json",
)


@dataclass(frozen=True)
class PresentationPackage:
    output_dir: Path
    generated_files: tuple[Path, ...]
    missing_source_artifacts: tuple[str, ...]
    summary: dict[str, Any]


def build_presentation_summary(source_dir: str | Path = DEFAULT_ANALYSIS_OUTPUT_DIR) -> dict[str, Any]:
    source = Path(source_dir)
    manifest = _read_json_if_exists(source / "manifest.json")
    neighbor_metrics = _read_json_if_exists(source / "nearest_neighbor_metrics.json")
    query_comparison = _read_json_if_exists(source / "query_neighbor_comparison.json")
    anchor_diversity = _read_csv_if_exists(source / "anchor_diversity_summary.csv")
    retrieval_rows = _read_csv_if_exists(source / "retrieval_comparison.csv")

    return {
        "created_at_utc": datetime.now(UTC).isoformat(timespec="seconds"),
        "analysis_output_dir": str(source),
        "embedding_coverage": _embedding_coverage_rows(manifest),
        "nearest_neighbor_summary": _nearest_neighbor_summary(neighbor_metrics),
        "query_overlap_summary": _query_overlap_summary(query_comparison),
        "anchor_diversity": anchor_diversity,
        "retrieval_summary": summarize_retrieval_rows(retrieval_rows),
        "demo_queries": {
            "primary": _pick_query(retrieval_rows, preferred=DEFAULT_PRIMARY_QUERY),
            "backup": _backup_query(manifest, retrieval_rows),
        },
        "metadata_limitations": manifest.get("metadata_limitations", []),
        "reproducibility": {
            "random_seed": manifest.get("random_seed"),
            "projection_parameters": manifest.get("projection_parameters", {}),
            "embedding_arrays_included": manifest.get("embedding_arrays_included", False),
        },
    }


def summarize_retrieval_rows(rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    groups: dict[tuple[str, str, str, str], list[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        key = (
            str(row.get("query") or ""),
            str(row.get("retrieval_mode") or ""),
            str(row.get("retrieval_model") or ""),
            str(row.get("result_set_type") or ""),
        )
        groups[key].append(row)

    summary_rows = []
    for (query, mode, model, result_type), group in sorted(groups.items()):
        ranked = sorted(group, key=lambda item: (_safe_int(item.get("rank"), default=10**9), str(item.get("relationship_id") or "")))
        top = ranked[0] if ranked else {}
        metadata_rows = [_metadata_from_row(item) for item in ranked]
        complete_matches = sum(
            1 for metadata in metadata_rows
            if metadata.get("structural_compatibility_status") == "complete_match"
        )
        partial_matches = sum(
            1 for metadata in metadata_rows
            if metadata.get("structural_compatibility_status") == "partial_match"
        )
        fallback_count = sum(1 for metadata in metadata_rows if metadata.get("is_semantic_fallback") is True)
        summary_rows.append(
            {
                "query": query,
                "retrieval_mode": mode,
                "retrieval_model": model,
                "result_set_type": result_type,
                "result_count": len(ranked),
                "unique_publications": _unique_count(ranked, "publication_id"),
                "unique_predicates": _unique_count(ranked, "predicate"),
                "unique_predicate_families": _unique_count(ranked, "predicate_family"),
                "complete_structural_matches": complete_matches,
                "partial_structural_matches": partial_matches,
                "semantic_fallbacks": fallback_count,
                "top_relationship_id": top.get("relationship_id", ""),
                "top_predicate": top.get("predicate", ""),
                "top_publication_id": top.get("publication_id", ""),
                "top_score": _safe_float(top.get("score")),
                "top_anchor_score": _safe_float(top.get("anchor_score")),
                "top_semantic_score": _safe_float(top.get("semantic_score")),
            }
        )
    return summary_rows


def write_presentation_package(
    source_dir: str | Path = DEFAULT_ANALYSIS_OUTPUT_DIR,
    output_dir: str | Path = DEFAULT_PRESENTATION_DIR,
) -> PresentationPackage:
    source = Path(source_dir)
    destination = Path(output_dir)
    artifacts_dir = destination / "artifacts"
    tables_dir = destination / "tables"
    artifacts_dir.mkdir(parents=True, exist_ok=True)
    tables_dir.mkdir(parents=True, exist_ok=True)

    generated: list[Path] = []
    missing: list[str] = []
    for artifact in PRESENTATION_ARTIFACTS:
        source_path = source / artifact
        if not source_path.exists():
            missing.append(artifact)
            continue
        target_path = artifacts_dir / artifact
        shutil.copy2(source_path, target_path)
        generated.append(target_path)

    summary = build_presentation_summary(source)
    summary_path = destination / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    generated.append(summary_path)

    retrieval_summary_path = tables_dir / "retrieval_summary.csv"
    _write_csv(retrieval_summary_path, summary["retrieval_summary"])
    generated.append(retrieval_summary_path)

    coverage_path = tables_dir / "embedding_coverage.csv"
    _write_csv(coverage_path, summary["embedding_coverage"])
    generated.append(coverage_path)

    readiness_path = tables_dir / "demo_readiness_matrix.csv"
    _write_csv(readiness_path, demo_readiness_rows(summary, missing))
    generated.append(readiness_path)

    readme_path = destination / "README.md"
    readme_path.write_text(render_presentation_readme(summary, missing), encoding="utf-8")
    generated.append(readme_path)

    runbook_path = destination / "demo_runbook.md"
    runbook_path.write_text(render_demo_runbook(summary), encoding="utf-8")
    generated.append(runbook_path)

    conclusions_path = destination / "conclusions.md"
    conclusions_path.write_text(render_conclusions(summary), encoding="utf-8")
    generated.append(conclusions_path)

    return PresentationPackage(
        output_dir=destination,
        generated_files=tuple(generated),
        missing_source_artifacts=tuple(missing),
        summary=summary,
    )


def demo_readiness_rows(summary: Mapping[str, Any], missing_artifacts: Sequence[str] = ()) -> list[dict[str, str]]:
    coverage = summary.get("embedding_coverage") or []
    same_population = "unknown"
    if coverage:
        same_population = str(coverage[0].get("same_population", "unknown"))
    return [
        {
            "area": "Evidence notebook",
            "status": "ready" if not missing_artifacts else "needs attention",
            "check": "Analysis artifacts copied into presentation/artifacts.",
            "notes": "Missing: " + ", ".join(missing_artifacts) if missing_artifacts else "Core HTML, CSV, and JSON artifacts are present.",
        },
        {
            "area": "Embedding coverage",
            "status": "ready" if same_population == "True" else "needs review",
            "check": "OpenAI and SapBERT rows cover the same relationship IDs.",
            "notes": f"same_population={same_population}",
        },
        {
            "area": "Primary query",
            "status": "ready",
            "check": "Use the configured primary query for the live explorer demo.",
            "notes": str((summary.get("demo_queries") or {}).get("primary") or DEFAULT_PRIMARY_QUERY),
        },
        {
            "area": "ROBOKOP fixture",
            "status": "ready",
            "check": "Run the app with ROBOKOP_MODE=fixture for deterministic development fallback.",
            "notes": "Fixture mode is explicit in provider metadata and UI mode badges.",
        },
        {
            "area": "ROBOKOP live",
            "status": "opt-in",
            "check": "Run explorer/tests/test_robokop_live_integration.py with RUN_LIVE_ROBOKOP=1 before the presentation.",
            "notes": "Live availability is not part of the default pytest suite.",
        },
    ]


def render_presentation_readme(summary: Mapping[str, Any], missing_artifacts: Sequence[str] = ()) -> str:
    coverage_rows = summary.get("embedding_coverage") or []
    coverage_text = "\n".join(f"- {row['model_name']}: {row['relationship_count']} edges, {row['embedding_dimension']} dimensions" for row in coverage_rows)
    nn_summary = summary.get("nearest_neighbor_summary") or {}
    overlap = summary.get("query_overlap_summary") or {}
    missing_text = ", ".join(missing_artifacts) if missing_artifacts else "none"

    return f"""# Group Presentation Package

Generated from `{summary.get('analysis_output_dir')}` at `{summary.get('created_at_utc')}`.

## What Is Included

- `artifacts/`: copied interactive HTML figures and source CSV/JSON evidence from the embedding comparison run.
- `tables/retrieval_summary.csv`: compact retrieval-mode summary for slide tables.
- `tables/embedding_coverage.csv`: model coverage and embedding dimensions.
- `tables/demo_readiness_matrix.csv`: preflight status for the notebook and live demo.
- `demo_runbook.md`: commands, environment variables, demo path, and recovery steps.
- `conclusions.md`: slide-ready observations, interpretations, caveats, and future directions.

## Evidence Snapshot

{coverage_text or "- Embedding coverage metadata was not available."}

- Same edge population: `{_same_population(summary)}`.
- Mean top-{nn_summary.get('top_k', 'k')} nearest-neighbor Jaccard: `{_format_number(nn_summary.get('mean_jaccard'))}`.
- Query overlap for `{overlap.get('query', 'unavailable')}`: `{overlap.get('intersection_count', 'n/a')}` shared top-{overlap.get('top_k', 'k')} edges, Jaccard `{_format_number(overlap.get('jaccard'))}`.
- Missing source artifacts: `{missing_text}`.

## Primary Demo Query

`{(summary.get('demo_queries') or {}).get('primary') or DEFAULT_PRIMARY_QUERY}`

## Backup Demo Query

`{(summary.get('demo_queries') or {}).get('backup') or DEFAULT_BACKUP_QUERY}`

## Interpretation Guardrails

- Compare OpenAI and SapBERT by matched relationship IDs, retrieved neighborhoods, and rankings, not by absolute 2D projection coordinates.
- Treat projection figures as exploratory diagnostics, not proof that one embedding model is globally better.
- Treat bounded ROBOKOP pages as partial neighborhoods; pagination, filters, and mode badges are part of the demo story.
"""


def render_demo_runbook(summary: Mapping[str, Any]) -> str:
    primary_query = (summary.get("demo_queries") or {}).get("primary") or DEFAULT_PRIMARY_QUERY
    backup_query = (summary.get("demo_queries") or {}).get("backup") or DEFAULT_BACKUP_QUERY
    return f"""# Demo Runbook

## Preflight

1. Regenerate analysis artifacts if source data changed:

```bash
uv run python scripts/generate_embedding_comparison_artifacts.py --projection-method pca
uv run python scripts/build_presentation_package.py
```

2. Run default validation:

```bash
env PYTHONDONTWRITEBYTECODE=1 uv run pytest
env PYTHONDONTWRITEBYTECODE=1 uv run ruff check
```

3. Optional live ROBOKOP check:

```bash
env PYTHONDONTWRITEBYTECODE=1 RUN_LIVE_ROBOKOP=1 uv run pytest explorer/tests/test_robokop_live_integration.py
```

## Environment

- Required for LitCoin search: `NEO4J_URI`, `NEO4J_USERNAME`, `NEO4J_PASSWORD`, and the embedding provider settings already used by the app.
- Deterministic ROBOKOP development route: `ROBOKOP_MODE=fixture`.
- Live ROBOKOP route: `ROBOKOP_MODE=live`, optional `ROBOKOP_URL`, optional `ROBOKOP_TIMEOUT_SECONDS`.
- The app defaults to `HOST=0.0.0.0` and `PORT=8050`.

## Start The App

Fixture-backed development/demo fallback:

```bash
env ROBOKOP_MODE=fixture PORT=8050 uv run python -m explorer.frontend.dash_app
```

Live demo route:

```bash
env ROBOKOP_MODE=live PORT=8050 uv run python -m explorer.frontend.dash_app
```

## Primary Story

1. Search: `{primary_query}`.
2. Select a candidate path whose anchor reasons include gene endpoint or drug-response compatibility.
3. Accept or bookmark the path to show that session state is preserved.
4. Select a normalized LitCoin entity with a visible CURIE, preferably `NCBIGene:5728` / PTEN if present.
5. Use `Fetch ROBOKOP summary`; confirm the UI shows mode `fixture` or `live` explicitly.
6. Choose a category/predicate filter, fetch a bounded ROBOKOP page, and point out limit/offset pagination.
7. Add one remote edge to the graph and identify its ROBOKOP source badge and primary knowledge source.

## Backup Story

Use `{backup_query}` if the primary query does not surface a clean bridge quickly. The fallback discussion is still valid: LitCoin retrieves an interpretable local anchor, then ROBOKOP expansion starts only after a normalized CURIE is confirmed.

## Expected Latency

- Cached LitCoin path searches should usually respond faster than first-run searches.
- ROBOKOP fixture mode should be effectively local.
- ROBOKOP live mode depends on the remote service; keep the requested page size at 10 or lower for the presentation.

## Recovery Steps

- If LitCoin search is slow or unavailable, use the copied HTML figures and CSV summaries in `presentation/artifacts`.
- If live ROBOKOP fails, restart with `ROBOKOP_MODE=fixture`; do not present fixture data as live data.
- If a selected node has no bridge CURIE, choose another path node with an explicit normalized identifier. Display-name equality is intentionally rejected.
"""


def render_conclusions(summary: Mapping[str, Any]) -> str:
    nn_summary = summary.get("nearest_neighbor_summary") or {}
    overlap = summary.get("query_overlap_summary") or {}
    diversity_rows = summary.get("anchor_diversity") or []
    diversity = diversity_rows[0] if diversity_rows else {}
    retrieval_summaries = summary.get("retrieval_summary") or []
    final_rows = [row for row in retrieval_summaries if row.get("result_set_type") == "final_anchor"]
    best_diverse = max(final_rows, key=lambda row: int(row.get("unique_publications") or 0), default={})

    return f"""# Presentation Conclusions

## Direct Observations

- The compared exports contain `{_matched_count(summary)}` matched LitCoin relationship embeddings across OpenAI and SapBERT with same-population coverage `{_same_population(summary)}`.
- OpenAI and SapBERT nearest-neighbor sets are related but not identical: mean top-{nn_summary.get('top_k', 'k')} Jaccard is `{_format_number(nn_summary.get('mean_jaccard'))}`.
- For `{overlap.get('query', 'the highlighted query')}`, top-{overlap.get('top_k', 'k')} query results share `{overlap.get('intersection_count', 'n/a')}` relationships with Jaccard `{_format_number(overlap.get('jaccard'))}`.
- The current anchor-diversity run reports `{diversity.get('unique_publications', 'n/a')}` unique publications, `{diversity.get('unique_predicates', 'n/a')}` unique predicates, and `{diversity.get('query_compatible_predicate_matches', 'n/a')}` query-compatible predicate matches.
- Among final-anchor result sets, `{best_diverse.get('retrieval_mode', 'n/a')}` / `{best_diverse.get('retrieval_model', 'n/a')}` has the highest publication diversity in this artifact set with `{best_diverse.get('unique_publications', 'n/a')}` unique publications.

## Interpretations

- The models organize the same LitCoin edges differently enough that retrieval diagnostics should compare ranked neighborhoods rather than relying on one projection view.
- Query-aware reranking is useful for separating structural compatibility from raw text similarity, especially for gene and treatment-response queries.
- Hybrid/keyword retrieval gives a practical recovery path when dense retrieval over-concentrates on one publication context.
- Bounded ROBOKOP expansion is best presented as human-guided follow-up from a confirmed bridge CURIE, not as global ROBOKOP search.

## Limitations And Future Directions

- The evidence is based on the exported relationship population and configured queries, so claims should stay within that scope.
- 2D projections can show local patterns and outliers, but cannot establish global model superiority.
- Live ROBOKOP availability and latency remain external risks; fixture mode is deterministic but must be clearly labeled.
- Grant-worthy next steps: evaluate retrieval quality with human labels, add explicit bridge-selection UX for ambiguous IDs, and compare whether external expansion improves time-to-useful-path.
"""


def _embedding_coverage_rows(manifest: Mapping[str, Any]) -> list[dict[str, Any]]:
    coverage = manifest.get("coverage") or {}
    models = manifest.get("models") or []
    rows = []
    for model in models:
        rows.append(
            {
                "model_name": model.get("model_name", ""),
                "relationship_count": model.get("relationship_count", ""),
                "embedding_dimension": model.get("embedding_dimension", ""),
                "matched_count": coverage.get("matched_count", ""),
                "same_population": coverage.get("same_population", ""),
                "left_only_count": coverage.get("left_only_count", ""),
                "right_only_count": coverage.get("right_only_count", ""),
            }
        )
    return rows


def _nearest_neighbor_summary(metrics: Mapping[str, Any]) -> dict[str, Any]:
    agreement = metrics.get("nearest_neighbor_jaccard") or {}
    return {
        "top_k": agreement.get("k"),
        "mean_jaccard": agreement.get("mean_jaccard"),
        "openai_same_publication_fraction": (metrics.get("openai_same_publication_fraction") or {}).get("mean_fraction"),
        "sapbert_same_publication_fraction": (metrics.get("sapbert_same_publication_fraction") or {}).get("mean_fraction"),
        "openai_same_predicate_family_fraction": (metrics.get("openai_same_predicate_family_fraction") or {}).get("mean_fraction"),
        "sapbert_same_predicate_family_fraction": (metrics.get("sapbert_same_predicate_family_fraction") or {}).get("mean_fraction"),
    }


def _query_overlap_summary(comparison: Mapping[str, Any]) -> dict[str, Any]:
    overlap = comparison.get("overlap") or {}
    rank_correlation = comparison.get("rank_correlation") or {}
    return {
        "query": comparison.get("query"),
        "top_k": comparison.get("top_k") or overlap.get("k"),
        "intersection_count": overlap.get("intersection_count"),
        "union_count": overlap.get("union_count"),
        "jaccard": overlap.get("jaccard"),
        "spearman": rank_correlation.get("spearman"),
        "shared_count": rank_correlation.get("shared_count"),
    }


def _pick_query(rows: Sequence[Mapping[str, Any]], *, preferred: str) -> str:
    queries = {str(row.get("query") or "") for row in rows if row.get("query")}
    if preferred in queries:
        return preferred
    return min(queries) if queries else preferred


def _backup_query(manifest: Mapping[str, Any], rows: Sequence[Mapping[str, Any]]) -> str:
    primary = _pick_query(rows, preferred=DEFAULT_PRIMARY_QUERY)
    queries = [str(query) for query in manifest.get("queries", []) if str(query)]
    for query in queries:
        if query != primary:
            return query
    return DEFAULT_BACKUP_QUERY if primary != DEFAULT_BACKUP_QUERY else "PTEN cancer chemoresistance"


def _metadata_from_row(row: Mapping[str, Any]) -> dict[str, Any]:
    raw = row.get("metadata")
    if not isinstance(raw, str) or not raw.strip():
        return {}
    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError:
        return {}
    return parsed if isinstance(parsed, dict) else {}


def _read_json_if_exists(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    value = json.loads(path.read_text(encoding="utf-8"))
    return value if isinstance(value, dict) else {}


def _read_csv_if_exists(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    with path.open(newline="", encoding="utf-8") as handle:
        return [dict(row) for row in csv.DictReader(handle)]


def _write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    fieldnames = sorted({key for row in rows for key in row})
    with path.open("w", newline="", encoding="utf-8") as handle:
        if not fieldnames:
            handle.write("")
            return
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _unique_count(rows: Sequence[Mapping[str, Any]], key: str) -> int:
    return len({str(row.get(key) or "") for row in rows if row.get(key)})


def _safe_int(value: Any, *, default: int) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _safe_float(value: Any) -> float | str:
    try:
        return float(value)
    except (TypeError, ValueError):
        return ""


def _format_number(value: Any) -> str:
    try:
        return f"{float(value):.3f}"
    except (TypeError, ValueError):
        return "n/a"


def _matched_count(summary: Mapping[str, Any]) -> Any:
    coverage = summary.get("embedding_coverage") or []
    return coverage[0].get("matched_count", "n/a") if coverage else "n/a"


def _same_population(summary: Mapping[str, Any]) -> Any:
    coverage = summary.get("embedding_coverage") or []
    return coverage[0].get("same_population", "n/a") if coverage else "n/a"
