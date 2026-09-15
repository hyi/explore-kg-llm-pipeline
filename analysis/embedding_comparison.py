from __future__ import annotations

import csv
import json
import math
import random
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np

from explorer.backend.semantic_search.query_intent import (
    parse_query_intent,
    predicate_family,
)
from explorer.backend.semantic_search.ranking import (
    publication_identity,
    relationship_identity,
)

DEFAULT_RANDOM_SEED = 13
DEFAULT_PROJECTION_METHOD = "pca"
DEFAULT_HOVER_FIELDS = (
    "relationship_id",
    "subject",
    "object",
    "subject_name",
    "object_name",
    "predicate",
    "predicate_family",
    "publication_id",
    "abstract_title",
    "semantic_text",
)
EMBEDDING_KEYS = frozenset(
    {
        "embedding",
        "sapbert_embedding",
        "openai_embedding",
        "relationship_embedding",
        "vector",
    }
)
RELATIONSHIP_ID_KEYS = ("relationship_id", "rel_id", "id", "element_id")


class EmbeddingComparisonError(ValueError):
    """Base class for recoverable embedding-comparison data errors."""


class MissingRelationshipIdError(EmbeddingComparisonError):
    """Raised when an embedding row has no stable relationship identifier."""


class DuplicateRelationshipIdError(EmbeddingComparisonError):
    """Raised when one model export contains duplicate relationship IDs."""


class InconsistentEmbeddingDimensionError(EmbeddingComparisonError):
    """Raised when one model export contains inconsistent vector dimensions."""


class CoverageMismatchError(EmbeddingComparisonError):
    """Raised when full coverage is required but two model exports differ."""


class ProjectionUnavailableError(EmbeddingComparisonError):
    """Raised when a requested optional projection backend is unavailable."""


@dataclass(frozen=True)
class EmbeddingRecord:
    relationship_id: str
    model_name: str
    embedding: tuple[float, ...]
    metadata: dict[str, Any]


@dataclass(frozen=True)
class EmbeddingCollection:
    model_name: str
    embedding_dimension: int
    records: tuple[EmbeddingRecord, ...]

    @property
    def ids(self) -> tuple[str, ...]:
        return tuple(record.relationship_id for record in self.records)

    def by_id(self) -> dict[str, EmbeddingRecord]:
        return {record.relationship_id: record for record in self.records}

    def matrix(self, relationship_ids: Sequence[str] | None = None) -> np.ndarray:
        records_by_id = self.by_id()
        records = (
            [records_by_id[relationship_id] for relationship_id in relationship_ids]
            if relationship_ids is not None
            else list(self.records)
        )
        if not records:
            return np.empty((0, self.embedding_dimension), dtype=float)
        return np.asarray([record.embedding for record in records], dtype=float)


@dataclass(frozen=True)
class MatchedEmbeddingSet:
    left_model: str
    right_model: str
    relationship_ids: tuple[str, ...]
    left_records: tuple[EmbeddingRecord, ...]
    right_records: tuple[EmbeddingRecord, ...]
    coverage_report: dict[str, Any]


@dataclass(frozen=True)
class Neighbor:
    relationship_id: str
    similarity: float


@dataclass(frozen=True)
class ProjectionResult:
    model_name: str
    method: str
    seed: int
    rows: list[dict[str, Any]]
    parameters: dict[str, Any]


@dataclass(frozen=True)
class RetrievalResultSet:
    query: str
    retrieval_mode: str
    results: Sequence[Any]
    model_name: str | None = None
    result_set_type: str = "raw"


def load_embedding_jsonl(
    path: str | Path,
    *,
    model_name: str,
    embedding_key: str | None = None,
) -> EmbeddingCollection:
    return build_embedding_collection(
        read_jsonl_rows(path),
        model_name=model_name,
        embedding_key=embedding_key,
    )


def read_jsonl_rows(path: str | Path) -> list[dict[str, Any]]:
    rows = []
    with Path(path).open() as handle:
        for line_number, line in enumerate(handle, start=1):
            stripped = line.strip()
            if not stripped:
                continue
            try:
                rows.append(json.loads(stripped))
            except json.JSONDecodeError as exc:
                raise EmbeddingComparisonError(f"{path}:{line_number} is not valid JSONL.") from exc
    return rows


def deduplicate_exact_duplicate_rows(
    rows: Iterable[Mapping[str, Any]],
    *,
    model_name: str,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    first_by_id: dict[str, dict[str, Any]] = {}
    fingerprint_by_id: dict[str, str] = {}
    duplicate_count = 0
    duplicate_ids: set[str] = set()
    total_count = 0

    for index, row in enumerate(rows):
        total_count += 1
        relationship_id = _relationship_id_from_row(row)
        if not relationship_id:
            raise MissingRelationshipIdError(f"Row {index} for model '{model_name}' is missing a relationship ID.")
        fingerprint = json.dumps(row, sort_keys=True)
        if relationship_id not in first_by_id:
            first_by_id[relationship_id] = dict(row)
            fingerprint_by_id[relationship_id] = fingerprint
            continue
        duplicate_count += 1
        duplicate_ids.add(relationship_id)
        if fingerprint_by_id[relationship_id] != fingerprint:
            raise DuplicateRelationshipIdError(
                f"Model '{model_name}' contains conflicting duplicate relationship ID '{relationship_id}'."
            )

    report = {
        "model_name": model_name,
        "input_row_count": total_count,
        "output_row_count": len(first_by_id),
        "exact_duplicate_row_count": duplicate_count,
        "duplicate_id_count": len(duplicate_ids),
        "duplicate_ids_sample": sorted(duplicate_ids)[:20],
    }
    return list(first_by_id.values()), report


def build_embedding_collection(
    rows: Iterable[Mapping[str, Any]],
    *,
    model_name: str,
    embedding_key: str | None = None,
) -> EmbeddingCollection:
    records: list[EmbeddingRecord] = []
    seen_ids: set[str] = set()
    expected_dimension: int | None = None

    for index, row in enumerate(rows):
        relationship_id = _relationship_id_from_row(row)
        if not relationship_id:
            raise MissingRelationshipIdError(f"Row {index} for model '{model_name}' is missing a relationship ID.")
        if relationship_id in seen_ids:
            raise DuplicateRelationshipIdError(
                f"Model '{model_name}' contains duplicate relationship ID '{relationship_id}'."
            )
        seen_ids.add(relationship_id)

        key = embedding_key or _detect_embedding_key(row)
        embedding = _coerce_embedding(row.get(key), model_name=model_name, relationship_id=relationship_id)
        if expected_dimension is None:
            expected_dimension = len(embedding)
        elif len(embedding) != expected_dimension:
            raise InconsistentEmbeddingDimensionError(
                f"Model '{model_name}' has inconsistent embedding dimensions: "
                f"expected {expected_dimension}, found {len(embedding)} for relationship '{relationship_id}'."
            )

        metadata = sanitize_payload(dict(row))
        metadata["relationship_id"] = relationship_id
        metadata.setdefault("predicate_family", predicate_family(metadata.get("predicate")))
        records.append(
            EmbeddingRecord(
                relationship_id=relationship_id,
                model_name=model_name,
                embedding=embedding,
                metadata=metadata,
            )
        )

    return EmbeddingCollection(
        model_name=model_name,
        embedding_dimension=expected_dimension or 0,
        records=tuple(records),
    )


def match_embedding_collections(
    left: EmbeddingCollection,
    right: EmbeddingCollection,
    *,
    require_full_coverage: bool = False,
) -> MatchedEmbeddingSet:
    left_by_id = left.by_id()
    right_by_id = right.by_id()
    left_ids = set(left_by_id)
    right_ids = set(right_by_id)
    common_ids = tuple(sorted(left_ids & right_ids))
    only_left = tuple(sorted(left_ids - right_ids))
    only_right = tuple(sorted(right_ids - left_ids))

    coverage_report = {
        "left_model": left.model_name,
        "right_model": right.model_name,
        "left_count": len(left.records),
        "right_count": len(right.records),
        "matched_count": len(common_ids),
        "left_only_count": len(only_left),
        "right_only_count": len(only_right),
        "left_only_ids": list(only_left),
        "right_only_ids": list(only_right),
        "same_population": not only_left and not only_right,
    }
    if require_full_coverage and not coverage_report["same_population"]:
        raise CoverageMismatchError(
            f"Model coverage differs: {left.model_name} has {len(only_left)} unmatched IDs; "
            f"{right.model_name} has {len(only_right)} unmatched IDs."
        )

    return MatchedEmbeddingSet(
        left_model=left.model_name,
        right_model=right.model_name,
        relationship_ids=common_ids,
        left_records=tuple(left_by_id[relationship_id] for relationship_id in common_ids),
        right_records=tuple(right_by_id[relationship_id] for relationship_id in common_ids),
        coverage_report=coverage_report,
    )


def nearest_neighbors(
    collection: EmbeddingCollection,
    *,
    k: int,
    relationship_ids: Sequence[str] | None = None,
) -> dict[str, list[Neighbor]]:
    ids = tuple(relationship_ids) if relationship_ids is not None else collection.ids
    if k <= 0 or not ids:
        return {relationship_id: [] for relationship_id in ids}

    matrix = collection.matrix(ids)
    similarities = _cosine_similarity_matrix(matrix)
    results: dict[str, list[Neighbor]] = {}
    for row_index, relationship_id in enumerate(ids):
        candidates = []
        for col_index, neighbor_id in enumerate(ids):
            if row_index == col_index:
                continue
            candidates.append(Neighbor(relationship_id=neighbor_id, similarity=float(similarities[row_index, col_index])))
        candidates.sort(key=lambda item: (-item.similarity, item.relationship_id))
        results[relationship_id] = candidates[:k]
    return results


def top_k_overlap(left_results: Sequence[Any], right_results: Sequence[Any], *, k: int) -> dict[str, Any]:
    left_ids = _ordered_result_ids(left_results)[: max(0, k)]
    right_ids = _ordered_result_ids(right_results)[: max(0, k)]
    left_set = set(left_ids)
    right_set = set(right_ids)
    shared = sorted(left_set & right_set)
    union = left_set | right_set
    return {
        "k": k,
        "left_ids": left_ids,
        "right_ids": right_ids,
        "shared_ids": shared,
        "intersection_count": len(shared),
        "union_count": len(union),
        "jaccard": len(shared) / len(union) if union else 1.0,
    }


def nearest_neighbor_jaccard(
    left_neighbors: Mapping[str, Sequence[Neighbor | str]],
    right_neighbors: Mapping[str, Sequence[Neighbor | str]],
    *,
    k: int,
) -> dict[str, Any]:
    per_edge: dict[str, float] = {}
    for relationship_id in sorted(set(left_neighbors) & set(right_neighbors)):
        left = set(_neighbor_ids(left_neighbors[relationship_id])[: max(0, k)])
        right = set(_neighbor_ids(right_neighbors[relationship_id])[: max(0, k)])
        union = left | right
        per_edge[relationship_id] = len(left & right) / len(union) if union else 1.0
    values = list(per_edge.values())
    return {
        "k": k,
        "mean_jaccard": sum(values) / len(values) if values else math.nan,
        "per_edge": per_edge,
    }


def rank_correlation_shared_candidates(
    left_results: Sequence[Any],
    right_results: Sequence[Any],
    *,
    k: int | None = None,
) -> dict[str, Any]:
    left_ids = _ordered_result_ids(left_results)
    right_ids = _ordered_result_ids(right_results)
    if k is not None:
        left_ids = left_ids[: max(0, k)]
        right_ids = right_ids[: max(0, k)]
    left_ranks = {relationship_id: rank for rank, relationship_id in enumerate(left_ids, start=1)}
    right_ranks = {relationship_id: rank for rank, relationship_id in enumerate(right_ids, start=1)}
    shared_ids = sorted(set(left_ranks) & set(right_ranks))
    if len(shared_ids) < 2:
        return {"shared_count": len(shared_ids), "spearman": math.nan, "shared_ids": shared_ids}

    left_values = np.asarray([left_ranks[relationship_id] for relationship_id in shared_ids], dtype=float)
    right_values = np.asarray([right_ranks[relationship_id] for relationship_id in shared_ids], dtype=float)
    coefficient = _pearson(left_values, right_values)
    return {
        "shared_count": len(shared_ids),
        "spearman": coefficient,
        "shared_ids": shared_ids,
    }


def same_metadata_fraction(
    collection: EmbeddingCollection,
    neighbors: Mapping[str, Sequence[Neighbor | str]],
    *,
    metadata_field: str,
    k: int,
) -> dict[str, Any]:
    records_by_id = collection.by_id()
    per_edge: dict[str, float] = {}
    for relationship_id in sorted(set(neighbors) & set(records_by_id)):
        base_value = _metadata_value(records_by_id[relationship_id].metadata, metadata_field)
        neighbor_ids = _neighbor_ids(neighbors[relationship_id])[: max(0, k)]
        comparable = [
            neighbor_id
            for neighbor_id in neighbor_ids
            if neighbor_id in records_by_id and _metadata_value(records_by_id[neighbor_id].metadata, metadata_field) is not None
        ]
        if base_value is None or not comparable:
            continue
        same_count = sum(
            1
            for neighbor_id in comparable
            if _metadata_value(records_by_id[neighbor_id].metadata, metadata_field) == base_value
        )
        per_edge[relationship_id] = same_count / len(comparable)
    values = list(per_edge.values())
    return {
        "metadata_field": metadata_field,
        "k": k,
        "mean_fraction": sum(values) / len(values) if values else math.nan,
        "per_edge": per_edge,
    }


def project_embeddings(
    collection: EmbeddingCollection,
    *,
    method: str = DEFAULT_PROJECTION_METHOD,
    seed: int = DEFAULT_RANDOM_SEED,
    relationship_ids: Sequence[str] | None = None,
    max_points: int | None = None,
    hover_fields: Sequence[str] = DEFAULT_HOVER_FIELDS,
) -> ProjectionResult:
    selected_ids = tuple(relationship_ids) if relationship_ids is not None else collection.ids
    selected_ids = _sample_ids(selected_ids, max_points=max_points, seed=seed)
    matrix = collection.matrix(selected_ids)
    method = method.lower()
    if method == "pca":
        coordinates = _pca_2d(matrix)
        parameters = {"method": "pca", "n_components": 2, "random_seed": seed}
    elif method == "umap":
        coordinates = _umap_2d(matrix, seed=seed)
        parameters = {"method": "umap", "n_components": 2, "random_seed": seed}
    else:
        raise ValueError("Projection method must be 'pca' or 'umap'.")

    records_by_id = collection.by_id()
    rows = []
    for index, relationship_id in enumerate(selected_ids):
        metadata = records_by_id[relationship_id].metadata
        hover_metadata = _select_hover_metadata(metadata, hover_fields)
        rows.append(
            {
                "relationship_id": relationship_id,
                "model": collection.model_name,
                "projection_method": method,
                "x": float(coordinates[index, 0]) if len(coordinates) else 0.0,
                "y": float(coordinates[index, 1]) if len(coordinates) else 0.0,
                "hover": hover_metadata,
                "publication_id": publication_identity(metadata),
                "predicate": metadata.get("predicate"),
                "predicate_family": metadata.get("predicate_family") or predicate_family(metadata.get("predicate")),
                "subject": _first_present(metadata, "subject", "subject_name", "original_subject", "llm_subject"),
                "object": _first_present(metadata, "object", "object_name", "original_object", "llm_object"),
            }
        )
    return ProjectionResult(
        model_name=collection.model_name,
        method=method,
        seed=seed,
        rows=rows,
        parameters={
            **parameters,
            "model": collection.model_name,
            "embedding_dimension": collection.embedding_dimension,
            "point_count": len(rows),
            "max_points": max_points,
        },
    )


def side_by_side_projection(
    left: EmbeddingCollection,
    right: EmbeddingCollection,
    *,
    method: str = DEFAULT_PROJECTION_METHOD,
    seed: int = DEFAULT_RANDOM_SEED,
    max_points: int | None = None,
    highlight_ids: Iterable[str] = (),
) -> dict[str, Any]:
    matched = match_embedding_collections(left, right)
    sampled_ids = _sample_ids(matched.relationship_ids, max_points=max_points, seed=seed)
    highlight_set = set(highlight_ids)
    left_projection = project_embeddings(left, method=method, seed=seed, relationship_ids=sampled_ids)
    right_projection = project_embeddings(right, method=method, seed=seed, relationship_ids=sampled_ids)
    return {
        "coverage": matched.coverage_report,
        "projection_warning": (
            "Panels are fitted separately. Compare membership, neighborhoods, and highlighted IDs; "
            "do not compare absolute axis direction, rotation, or inter-panel coordinates."
        ),
        "left": _with_projection_highlight(left_projection, highlight_set),
        "right": _with_projection_highlight(right_projection, highlight_set),
    }


def create_projection_figure(
    projections: Sequence[ProjectionResult | Mapping[str, Any]],
    *,
    color_field: str = "predicate_family",
    title: str = "Embedding comparison",
) -> Any:
    try:
        import plotly.express as px
    except ImportError as exc:  # pragma: no cover - optional presentation dependency.
        raise ProjectionUnavailableError("Plotly is required to create interactive HTML figures.") from exc

    rows: list[dict[str, Any]] = []
    for projection in projections:
        projection_rows = projection.rows if isinstance(projection, ProjectionResult) else projection["rows"]
        rows.extend(projection_rows)
    if not rows:
        rows = [{"x": 0.0, "y": 0.0, "model": "empty", color_field: "empty", "hover_text": "No rows"}]
    for row in rows:
        row.setdefault("hover_text", _hover_text(row.get("hover", {})))
        row.setdefault(color_field, row.get("hover", {}).get(color_field))

    figure = px.scatter(
        rows,
        x="x",
        y="y",
        color=color_field,
        symbol="model",
        facet_col="model",
        hover_name="relationship_id",
        hover_data={"hover_text": True, "x": ":.3f", "y": ":.3f"},
        title=title,
    )
    figure.update_layout(
        annotations=[
            {
                "text": (
                    "Each panel is projected and axis-scaled separately; compare highlighted membership and "
                    "neighborhood patterns, not absolute coordinates or apparent spread."
                ),
                "xref": "paper",
                "yref": "paper",
                "x": 0,
                "y": -0.18,
                "showarrow": False,
                "align": "left",
            }
        ]
    )
    figure.update_xaxes(matches=None)
    figure.update_yaxes(matches=None)
    return figure


def write_projection_html(
    projections: Sequence[ProjectionResult | Mapping[str, Any]],
    path: str | Path,
    *,
    color_field: str = "predicate_family",
    title: str = "Embedding comparison",
) -> Path:
    figure = create_projection_figure(projections, color_field=color_field, title=title)
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.write_html(output_path, include_plotlyjs="cdn")
    return output_path


def normalize_result_rows(
    *,
    query: str,
    retrieval_mode: str,
    results: Sequence[Any],
    model_name: str | None = None,
    result_set_type: str = "raw",
) -> list[dict[str, Any]]:
    rows = []
    for rank, result in enumerate(results, start=1):
        metadata = _result_metadata(result)
        relationship_id = relationship_identity(metadata, fallback=f"{retrieval_mode}:{result_set_type}:{rank}")
        score = _first_present(metadata, "anchor_score", "retrieval_score", "semantic_score", "score", default=math.nan)
        rows.append(
            {
                "query": query,
                "retrieval_mode": retrieval_mode,
                "retrieval_model": model_name or metadata.get("retrieval_model"),
                "result_set_type": result_set_type,
                "rank": rank,
                "relationship_id": relationship_id,
                "score": score,
                "semantic_score": metadata.get("semantic_score"),
                "anchor_score": metadata.get("anchor_score"),
                "publication_id": publication_identity(metadata),
                "predicate": metadata.get("predicate"),
                "predicate_family": metadata.get("predicate_family") or predicate_family(metadata.get("predicate")),
                "subject": _first_present(metadata, "subject", "subject_name", "original_subject", "llm_subject"),
                "object": _first_present(metadata, "object", "object_name", "original_object", "llm_object"),
                "retrieval_channels": metadata.get("retrieval_channels"),
                "ranking_reasons": list(metadata.get("ranking_reasons", []) or []),
                "metadata": sanitize_payload(metadata),
            }
        )
    return rows


def build_retrieval_comparison_table(result_sets: Sequence[RetrievalResultSet]) -> list[dict[str, Any]]:
    table: list[dict[str, Any]] = []
    for result_set in result_sets:
        table.extend(
            normalize_result_rows(
                query=result_set.query,
                retrieval_mode=result_set.retrieval_mode,
                results=result_set.results,
                model_name=result_set.model_name,
                result_set_type=result_set.result_set_type,
            )
        )
    return table


def apply_query_highlights(
    projection_rows: Sequence[Mapping[str, Any]],
    result_rows: Sequence[Mapping[str, Any]],
    *,
    top_k: int,
    query: str,
    label: str | None = None,
) -> list[dict[str, Any]]:
    ranked_ids = {
        str(row["relationship_id"]): int(row.get("rank", index + 1))
        for index, row in enumerate(result_rows[: max(0, top_k)])
        if row.get("relationship_id")
    }
    highlighted = []
    for row in projection_rows:
        relationship_id = str(row.get("relationship_id", ""))
        copy = dict(row)
        copy["highlight_query"] = query
        copy["highlight_label"] = label or query
        copy["highlight_rank"] = ranked_ids.get(relationship_id)
        copy["is_highlighted"] = relationship_id in ranked_ids
        highlighted.append(copy)
    return highlighted


def summarize_anchor_diversity(
    result_rows: Sequence[Mapping[str, Any]],
    *,
    query: str | None = None,
) -> dict[str, Any]:
    rows = list(result_rows)
    endpoint_neighborhoods = {
        (
            _normalize_key(row.get("subject")),
            _normalize_key(row.get("object")),
        )
        for row in rows
    }
    endpoint_neighborhoods.discard(("", ""))
    entity_match_count = 0
    predicate_match_count = 0
    if query:
        intent = parse_query_intent(query)
        requested_predicates = {request.family for request in intent.requested_predicate_families}
        for row in rows:
            metadata = dict(row.get("metadata", {}) or {})
            endpoint_status = (
                metadata.get("endpoint_category_compatibility", {}).get("status")
                if isinstance(metadata.get("endpoint_category_compatibility"), dict)
                else None
            )
            predicate_status = (
                metadata.get("predicate_family_compatibility", {}).get("status")
                if isinstance(metadata.get("predicate_family_compatibility"), dict)
                else None
            )
            if endpoint_status in {"match", "partial"}:
                entity_match_count += 1
            elif not endpoint_status and intent.requested_endpoint_categories:
                labels = set(metadata.get("subject_labels", []) or []) | set(metadata.get("object_labels", []) or [])
                entity_match_count += int(any(request.family in labels for request in intent.requested_endpoint_categories))
            if predicate_status in {"match", "partial"} or (
                requested_predicates and row.get("predicate_family") in requested_predicates
            ):
                predicate_match_count += 1

    return {
        "query": query,
        "result_count": len(rows),
        "unique_publications": len({row.get("publication_id") for row in rows if row.get("publication_id")}),
        "unique_predicates": len({row.get("predicate") for row in rows if row.get("predicate")}),
        "unique_predicate_families": len({row.get("predicate_family") for row in rows if row.get("predicate_family")}),
        "unique_endpoint_neighborhoods": len(endpoint_neighborhoods),
        "requested_entity_type_matches": entity_match_count,
        "query_compatible_predicate_matches": predicate_match_count,
        "raw_result_count": sum(1 for row in rows if row.get("result_set_type") == "raw"),
        "reranked_result_count": sum(1 for row in rows if row.get("result_set_type") in {"reranked", "final_anchor"}),
    }


def comparison_manifest(
    *,
    collections: Sequence[EmbeddingCollection],
    projection_parameters: Mapping[str, Any] | None = None,
    queries: Sequence[str] = (),
    seed: int = DEFAULT_RANDOM_SEED,
) -> dict[str, Any]:
    return {
        "created_at_utc": datetime.now(UTC).replace(microsecond=0).isoformat(),
        "random_seed": seed,
        "models": [
            {
                "model_name": collection.model_name,
                "relationship_count": len(collection.records),
                "embedding_dimension": collection.embedding_dimension,
            }
            for collection in collections
        ],
        "projection_parameters": dict(projection_parameters or {}),
        "queries": list(queries),
        "embedding_arrays_included": False,
    }


def write_rows_csv(rows: Sequence[Mapping[str, Any]], path: str | Path) -> Path:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = sorted({key for row in rows for key in row})
    with output_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: _csv_value(row.get(key)) for key in fieldnames})
    return output_path


def write_json(data: Any, path: str | Path) -> Path:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w") as handle:
        json.dump(sanitize_payload(data), handle, indent=2, sort_keys=True)
        handle.write("\n")
    return output_path


def sanitize_payload(value: Any) -> Any:
    if isinstance(value, Mapping):
        cleaned = {}
        for key, item in value.items():
            key_string = str(key)
            if key_string in EMBEDDING_KEYS or (key_string.endswith("_embedding") and _looks_like_vector(item)):
                continue
            cleaned[key_string] = sanitize_payload(item)
        return cleaned
    if isinstance(value, list):
        if _looks_like_vector(value):
            return "[numeric_vector_omitted]"
        return [sanitize_payload(item) for item in value]
    if isinstance(value, tuple):
        if _looks_like_vector(value):
            return "[numeric_vector_omitted]"
        return [sanitize_payload(item) for item in value]
    return value


def _detect_embedding_key(row: Mapping[str, Any]) -> str:
    candidates = [key for key in EMBEDDING_KEYS if key in row and row[key] is not None]
    if not candidates:
        raise InconsistentEmbeddingDimensionError("Embedding row does not contain a recognized embedding vector field.")
    return min(candidates)


def _relationship_id_from_row(row: Mapping[str, Any]) -> str | None:
    for key in RELATIONSHIP_ID_KEYS:
        value = row.get(key)
        if value is not None and str(value).strip():
            return str(value)
    metadata = row.get("metadata")
    if isinstance(metadata, Mapping):
        for key in RELATIONSHIP_ID_KEYS:
            value = metadata.get(key)
            if value is not None and str(value).strip():
                return str(value)
    return None


def _coerce_embedding(value: Any, *, model_name: str, relationship_id: str) -> tuple[float, ...]:
    if not isinstance(value, Sequence) or isinstance(value, str) or not value:
        raise InconsistentEmbeddingDimensionError(
            f"Model '{model_name}' relationship '{relationship_id}' has no usable embedding vector."
        )
    try:
        return tuple(float(item) for item in value)
    except (TypeError, ValueError) as exc:
        raise InconsistentEmbeddingDimensionError(
            f"Model '{model_name}' relationship '{relationship_id}' has a non-numeric embedding vector."
        ) from exc


def _cosine_similarity_matrix(matrix: np.ndarray) -> np.ndarray:
    if matrix.size == 0:
        return np.empty((0, 0), dtype=float)
    norms = np.linalg.norm(matrix, axis=1)
    safe_norms = np.where(norms == 0, 1.0, norms)
    normalized = matrix / safe_norms[:, None]
    similarities = normalized @ normalized.T
    zero_rows = norms == 0
    if np.any(zero_rows):
        similarities[zero_rows, :] = 0.0
        similarities[:, zero_rows] = 0.0
    return similarities


def _pca_2d(matrix: np.ndarray) -> np.ndarray:
    if matrix.size == 0:
        return np.empty((0, 2), dtype=float)
    if matrix.shape[0] == 1:
        return np.zeros((1, 2), dtype=float)
    centered = matrix - matrix.mean(axis=0, keepdims=True)
    _u, _s, vt = np.linalg.svd(centered, full_matrices=False)
    components = vt[:2].T
    if components.shape[1] < 2:
        components = np.pad(components, ((0, 0), (0, 2 - components.shape[1])))
    components = _stabilize_component_signs(centered, components)
    coordinates = centered @ components
    if coordinates.shape[1] < 2:
        coordinates = np.pad(coordinates, ((0, 0), (0, 2 - coordinates.shape[1])))
    return coordinates[:, :2]


def _umap_2d(matrix: np.ndarray, *, seed: int) -> np.ndarray:
    try:
        import umap
    except ImportError as exc:  # pragma: no cover - optional exploratory dependency.
        raise ProjectionUnavailableError("Install umap-learn to use UMAP projections.") from exc
    if matrix.size == 0:
        return np.empty((0, 2), dtype=float)
    reducer = umap.UMAP(n_components=2, random_state=seed)
    return np.asarray(reducer.fit_transform(matrix), dtype=float)


def _stabilize_component_signs(matrix: np.ndarray, components: np.ndarray) -> np.ndarray:
    stabilized = components.copy()
    for column in range(stabilized.shape[1]):
        component = stabilized[:, column]
        if not len(component):
            continue
        max_index = int(np.argmax(np.abs(component)))
        if component[max_index] < 0:
            stabilized[:, column] *= -1
    return stabilized


def _sample_ids(ids: Sequence[str], *, max_points: int | None, seed: int) -> tuple[str, ...]:
    normalized = tuple(str(relationship_id) for relationship_id in ids)
    if max_points is None or max_points <= 0 or len(normalized) <= max_points:
        return normalized
    sampled = random.Random(seed).sample(sorted(normalized), max_points)
    return tuple(sorted(sampled))


def _ordered_result_ids(results: Sequence[Any]) -> list[str]:
    ids = []
    for index, result in enumerate(results):
        if isinstance(result, str):
            ids.append(result)
            continue
        metadata = _result_metadata(result)
        ids.append(relationship_identity(metadata, fallback=f"result:{index}"))
    return ids


def _neighbor_ids(neighbors: Sequence[Neighbor | str]) -> list[str]:
    return [neighbor.relationship_id if isinstance(neighbor, Neighbor) else str(neighbor) for neighbor in neighbors]


def _result_metadata(result: Any) -> dict[str, Any]:
    if isinstance(result, Mapping):
        if isinstance(result.get("metadata"), Mapping):
            metadata = dict(result["metadata"])
            for key, value in result.items():
                if key != "metadata":
                    metadata.setdefault(str(key), value)
            return metadata
        return dict(result)
    return dict(getattr(result, "metadata", {}) or {})


def _metadata_value(metadata: Mapping[str, Any], field: str) -> Any:
    if field == "publication_id":
        return publication_identity(dict(metadata))
    if field == "predicate_family":
        return metadata.get("predicate_family") or predicate_family(metadata.get("predicate"))
    value = metadata.get(field)
    if isinstance(value, list):
        return tuple(value)
    return value


def _pearson(left: np.ndarray, right: np.ndarray) -> float:
    left_centered = left - left.mean()
    right_centered = right - right.mean()
    denominator = float(np.linalg.norm(left_centered) * np.linalg.norm(right_centered))
    if denominator == 0.0:
        return math.nan
    return float(np.dot(left_centered, right_centered) / denominator)


def _select_hover_metadata(metadata: Mapping[str, Any], fields: Sequence[str]) -> dict[str, Any]:
    selected = {}
    for field in fields:
        value = _metadata_value(metadata, field)
        if value is None:
            continue
        selected[field] = _truncate_text(value)
    return sanitize_payload(selected)


def _hover_text(metadata: Mapping[str, Any]) -> str:
    return "<br>".join(f"{key}: {value}" for key, value in metadata.items())


def _with_projection_highlight(projection: ProjectionResult, highlight_ids: set[str]) -> dict[str, Any]:
    rows = []
    for row in projection.rows:
        copy = dict(row)
        copy["is_highlighted"] = str(row["relationship_id"]) in highlight_ids
        rows.append(copy)
    return {
        "model_name": projection.model_name,
        "method": projection.method,
        "seed": projection.seed,
        "rows": rows,
        "parameters": dict(projection.parameters),
    }


def _first_present(mapping: Mapping[str, Any], *keys: str, default: Any = None) -> Any:
    for key in keys:
        value = mapping.get(key)
        if value is not None and value != "":
            return value
    return default


def _normalize_key(value: Any) -> str:
    return str(value or "").casefold().strip()


def _looks_like_vector(value: Any) -> bool:
    if not isinstance(value, Sequence) or isinstance(value, str):
        return False
    if len(value) < 8:
        return False
    return all(isinstance(item, int | float) for item in value)


def _truncate_text(value: Any, *, max_length: int = 240) -> Any:
    if not isinstance(value, str) or len(value) <= max_length:
        return value
    return f"{value[: max_length - 3]}..."


def _csv_value(value: Any) -> Any:
    if isinstance(value, dict | list | tuple):
        return json.dumps(sanitize_payload(value), sort_keys=True)
    return value


def manifest_from_projection_results(
    projections: Sequence[ProjectionResult],
    *,
    queries: Sequence[str] = (),
    seed: int = DEFAULT_RANDOM_SEED,
) -> dict[str, Any]:
    return comparison_manifest(
        collections=[],
        projection_parameters={
            projection.model_name: projection.parameters
            for projection in projections
        },
        queries=queries,
        seed=seed,
    )


def result_set_from_rows(
    *,
    query: str,
    retrieval_mode: str,
    rows: Sequence[Mapping[str, Any]],
    model_name: str | None = None,
    result_set_type: str = "raw",
) -> RetrievalResultSet:
    return RetrievalResultSet(
        query=query,
        retrieval_mode=retrieval_mode,
        model_name=model_name,
        result_set_type=result_set_type,
        results=tuple(rows),
    )


def dataclass_to_dict(value: Any) -> dict[str, Any]:
    return sanitize_payload(asdict(value))
