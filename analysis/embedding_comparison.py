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


@dataclass(frozen=True)
class QueryNeighborhoodComparison:
    query: str
    top_k: int
    left_model: str
    right_model: str
    left_results: list[dict[str, Any]]
    right_results: list[dict[str, Any]]
    overlap: dict[str, Any]
    rank_correlation: dict[str, Any]


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


def nearest_neighbor_agreement_rows(
    collection: EmbeddingCollection,
    neighbor_agreement: Mapping[str, Any],
) -> list[dict[str, Any]]:
    per_edge = neighbor_agreement.get("per_edge", neighbor_agreement)
    if not isinstance(per_edge, Mapping):
        raise EmbeddingComparisonError("Neighbor agreement must be a mapping or contain a 'per_edge' mapping.")

    records_by_id = collection.by_id()
    rows = []
    for relationship_id, value in sorted(per_edge.items(), key=lambda item: (float(item[1]), str(item[0]))):
        metadata = records_by_id.get(str(relationship_id), EmbeddingRecord(str(relationship_id), collection.model_name, (), {})).metadata
        rows.append(
            {
                "relationship_id": str(relationship_id),
                "neighbor_jaccard": float(value),
                "model_reference": collection.model_name,
                "publication_id": publication_identity(metadata),
                "predicate": metadata.get("predicate"),
                "predicate_family": metadata.get("predicate_family") or predicate_family(metadata.get("predicate")),
                "subject": _first_present(metadata, "subject", "subject_name", "original_subject", "llm_subject"),
                "object": _first_present(metadata, "object", "object_name", "original_object", "llm_object"),
                "semantic_text": _truncate_text(metadata.get("semantic_text"), max_length=320),
            }
        )
    return rows


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


def query_nearest_relationships(
    collection: EmbeddingCollection,
    query_embedding: Sequence[float],
    *,
    k: int,
    relationship_ids: Sequence[str] | None = None,
) -> list[dict[str, Any]]:
    ids = tuple(relationship_ids) if relationship_ids is not None else collection.ids
    if k <= 0 or not ids:
        return []
    query_vector = np.asarray([float(value) for value in query_embedding], dtype=float)
    if len(query_vector) != collection.embedding_dimension:
        raise InconsistentEmbeddingDimensionError(
            f"Query embedding for model '{collection.model_name}' has dimension {len(query_vector)}; "
            f"expected {collection.embedding_dimension}."
        )

    matrix = collection.matrix(ids)
    query_norm = float(np.linalg.norm(query_vector))
    matrix_norms = np.linalg.norm(matrix, axis=1)
    safe_denominator = np.where(matrix_norms == 0, 1.0, matrix_norms) * (query_norm or 1.0)
    similarities = (matrix @ query_vector) / safe_denominator
    if query_norm == 0.0:
        similarities = np.zeros_like(similarities)
    similarities = np.where(matrix_norms == 0, 0.0, similarities)

    records_by_id = collection.by_id()
    ranked = []
    for relationship_id, similarity in zip(ids, similarities, strict=True):
        metadata = records_by_id[relationship_id].metadata
        ranked.append(
            {
                "relationship_id": relationship_id,
                "model": collection.model_name,
                "similarity": float(similarity),
                "publication_id": publication_identity(metadata),
                "predicate": metadata.get("predicate"),
                "predicate_family": metadata.get("predicate_family") or predicate_family(metadata.get("predicate")),
                "subject": _first_present(metadata, "subject", "subject_name", "original_subject", "llm_subject"),
                "object": _first_present(metadata, "object", "object_name", "original_object", "llm_object"),
                "semantic_text": metadata.get("semantic_text"),
                "metadata": sanitize_payload(metadata),
            }
        )
    ranked.sort(key=lambda item: (-item["similarity"], item["relationship_id"]))
    for rank, item in enumerate(ranked[:k], start=1):
        item["rank"] = rank
    return ranked[:k]


def compare_query_neighborhoods(
    *,
    query: str,
    left: EmbeddingCollection,
    right: EmbeddingCollection,
    left_query_embedding: Sequence[float],
    right_query_embedding: Sequence[float],
    k: int,
    relationship_ids: Sequence[str] | None = None,
) -> QueryNeighborhoodComparison:
    ids = tuple(relationship_ids) if relationship_ids is not None else match_embedding_collections(left, right).relationship_ids
    left_results = query_nearest_relationships(left, left_query_embedding, k=k, relationship_ids=ids)
    right_results = query_nearest_relationships(right, right_query_embedding, k=k, relationship_ids=ids)
    return QueryNeighborhoodComparison(
        query=query,
        top_k=k,
        left_model=left.model_name,
        right_model=right.model_name,
        left_results=left_results,
        right_results=right_results,
        overlap=top_k_overlap(left_results, right_results, k=k),
        rank_correlation=rank_correlation_shared_candidates(left_results, right_results, k=k),
    )


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
        parameters = {"method": "umap", "n_components": 2, "random_seed": seed, "metric": "cosine"}
    else:
        raise ValueError("Projection method must be 'pca' or 'umap'.")

    records_by_id = collection.by_id()
    rows = []
    for index, relationship_id in enumerate(selected_ids):
        metadata = records_by_id[relationship_id].metadata
        hover_metadata = _select_hover_metadata(metadata, hover_fields)
        subject = _first_present(metadata, "subject", "subject_name", "original_subject", "llm_subject")
        obj = _first_present(metadata, "object", "object_name", "original_object", "llm_object")
        subject_prefix = _curie_prefix(subject)
        object_prefix = _curie_prefix(obj)
        candidate_predicate = metadata.get("predicate")
        is_mentions_edge = candidate_predicate == "biolink:mentions"
        rows.append(
            {
                "relationship_id": relationship_id,
                "model": collection.model_name,
                "projection_method": method,
                "x": float(coordinates[index, 0]) if len(coordinates) else 0.0,
                "y": float(coordinates[index, 1]) if len(coordinates) else 0.0,
                "hover": hover_metadata,
                "publication_id": publication_identity(metadata),
                "predicate": candidate_predicate,
                "predicate_family": metadata.get("predicate_family") or predicate_family(candidate_predicate),
                "subject": subject,
                "object": obj,
                "subject_prefix": subject_prefix,
                "object_prefix": object_prefix,
                "endpoint_prefix_pair": f"{subject_prefix}-{object_prefix}",
                "endpoint_label_pair": _endpoint_label_pair(metadata),
                "is_mentions_edge": str(is_mentions_edge),
                "relationship_kind": "publication_mention" if is_mentions_edge or subject_prefix == "PMID" else "claim_edge",
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


def add_query_projection_marker(
    projection_rows: Sequence[Mapping[str, Any]],
    collection: EmbeddingCollection,
    query_embedding: Sequence[float],
    *,
    query: str,
    seed: int = DEFAULT_RANDOM_SEED,
) -> list[dict[str, Any]]:
    rows = [dict(row) for row in projection_rows if not row.get("is_query_marker")]
    relationship_ids = tuple(
        str(row["relationship_id"])
        for row in rows
        if row.get("relationship_id") and not row.get("is_query_marker")
    )
    if not relationship_ids:
        return rows

    method = str(rows[0].get("projection_method") or DEFAULT_PROJECTION_METHOD).lower()
    query_vector = np.asarray([float(value) for value in query_embedding], dtype=float)
    if len(query_vector) != collection.embedding_dimension:
        raise InconsistentEmbeddingDimensionError(
            f"Query embedding for model '{collection.model_name}' has dimension {len(query_vector)}; "
            f"expected {collection.embedding_dimension}."
        )
    query_coordinate = _project_extra_vector_2d(
        collection.matrix(relationship_ids),
        query_vector,
        method=method,
        seed=seed,
    )
    rows.append(
        {
            "relationship_id": "query",
            "model": rows[0].get("model") or collection.model_name,
            "projection_method": method,
            "x": float(query_coordinate[0]),
            "y": float(query_coordinate[1]),
            "hover": {
                "point_type": "input query",
                "query_text": query,
                "model": collection.model_name,
            },
            "publication_id": None,
            "predicate": "input query",
            "predicate_family": "input query",
            "subject": "input query",
            "object": query,
            "subject_prefix": "query",
            "object_prefix": "query",
            "endpoint_prefix_pair": "query-query",
            "endpoint_label_pair": "query-query",
            "is_mentions_edge": "False",
            "relationship_kind": "input_query",
            "is_query_marker": True,
            "query_text": query,
        }
    )
    return rows


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
        import plotly.graph_objects as go
        from plotly.colors import qualitative
        from plotly.subplots import make_subplots
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

    models = _ordered_unique(row.get("model", "unknown") for row in rows)
    categories = _ordered_unique(
        row.get(color_field) or "missing"
        for row in rows
        if not row.get("is_query_marker")
    )
    color_map = {
        category: qualitative.Plotly[index % len(qualitative.Plotly)]
        for index, category in enumerate(categories)
    }
    figure = make_subplots(
        rows=1,
        cols=len(models),
        subplot_titles=[str(model) for model in models],
        horizontal_spacing=0.08,
    )
    for col_index, model in enumerate(models, start=1):
        model_rows = [row for row in rows if row.get("model", "unknown") == model]
        edge_rows = [row for row in model_rows if not row.get("is_query_marker")]
        for category in categories:
            category_rows = [row for row in edge_rows if (row.get(color_field) or "missing") == category]
            if not category_rows:
                continue
            figure.add_trace(
                go.Scatter(
                    x=[row["x"] for row in category_rows],
                    y=[row["y"] for row in category_rows],
                    mode="markers",
                    name=str(category),
                    legendgroup=str(category),
                    showlegend=col_index == 1,
                    customdata=[_point_customdata(row, color_field=color_field) for row in category_rows],
                    hoverinfo="none",
                    marker={
                        "symbol": "circle",
                        "size": 6,
                        "opacity": 0.72,
                        "color": color_map[category],
                        "line": {"width": 0},
                    },
                ),
                row=1,
                col=col_index,
            )

        highlighted_rows = [row for row in edge_rows if row.get("is_highlighted")]
        if highlighted_rows:
            figure.add_trace(
                go.Scatter(
                    x=[row["x"] for row in highlighted_rows],
                    y=[row["y"] for row in highlighted_rows],
                    mode="markers+text",
                    name="query top-k",
                    legendgroup="query top-k",
                    showlegend=col_index == 1,
                    text=[
                        str(row.get("highlight_rank") or "")
                        for row in highlighted_rows
                    ],
                    textposition="top center",
                    customdata=[_point_customdata(row, color_field=color_field) for row in highlighted_rows],
                    hoverinfo="none",
                    marker={
                        "symbol": "circle-open",
                        "size": 13,
                        "color": "black",
                        "line": {"width": 2},
                    },
                ),
                row=1,
                col=col_index,
            )

        query_rows = [row for row in model_rows if row.get("is_query_marker")]
        if query_rows:
            figure.add_trace(
                go.Scatter(
                    x=[row["x"] for row in query_rows],
                    y=[row["y"] for row in query_rows],
                    mode="markers+text",
                    name="input query",
                    legendgroup="input query",
                    showlegend=col_index == 1,
                    text=["Q" for _row in query_rows],
                    textposition="middle center",
                    customdata=[_point_customdata(row, color_field=color_field) for row in query_rows],
                    hoverinfo="none",
                    marker={
                        "symbol": "star-diamond",
                        "size": 22,
                        "color": "#d62728",
                        "opacity": 1.0,
                        "line": {"width": 2, "color": "#111111"},
                    },
                ),
                row=1,
                col=col_index,
            )

    figure.update_layout(
        title=title,
        clickmode="event+select",
        legend_title_text=color_field,
        annotations=[
            {
                "text": (
                    "Each panel is projected and axis-scaled separately; compare highlighted membership and "
                    "neighborhood patterns, not absolute coordinates or apparent spread. Click a point for details."
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
    figure.update_xaxes(matches=None, showticklabels=False, title_text="")
    figure.update_yaxes(matches=None, showticklabels=False, title_text="")
    return figure


def create_neighbor_agreement_figure(
    rows: Sequence[Mapping[str, Any]],
    *,
    title: str = "Nearest-neighbor agreement by relationship",
) -> Any:
    try:
        import plotly.graph_objects as go
        from plotly.subplots import make_subplots
    except ImportError as exc:  # pragma: no cover - optional presentation dependency.
        raise ProjectionUnavailableError("Plotly is required to create interactive HTML figures.") from exc

    values = [float(row["neighbor_jaccard"]) for row in rows if row.get("neighbor_jaccard") is not None]
    figure = make_subplots(
        rows=2,
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.09,
        row_heights=[0.35, 0.65],
        subplot_titles=["Distribution", "Per-edge values by predicate family"],
    )
    figure.add_trace(
        go.Histogram(
            x=values,
            nbinsx=20,
            name="edge count",
            marker={"color": "#4c78a8"},
        ),
        row=1,
        col=1,
    )
    figure.add_trace(
        go.Scatter(
            x=values,
            y=[str(row.get("predicate_family") or "missing") for row in rows],
            mode="markers",
            name="relationship",
            customdata=[[_click_detail_text(row, color_field="neighbor_jaccard")] for row in rows],
            hoverinfo="none",
            marker={
                "symbol": "circle",
                "size": 7,
                "opacity": 0.62,
                "color": values,
                "colorscale": "Viridis",
                "cmin": 0,
                "cmax": 1,
                "colorbar": {"title": "Jaccard"},
            },
        ),
        row=2,
        col=1,
    )
    mean_value = sum(values) / len(values) if values else math.nan
    if not math.isnan(mean_value):
        figure.add_vline(
            x=mean_value,
            line_dash="dash",
            line_color="#222222",
            annotation_text=f"mean={mean_value:.3f}",
            annotation_position="top right",
        )
    figure.update_layout(
        title=title,
        clickmode="event+select",
        showlegend=False,
        annotations=[
            {
                "text": (
                    "Jaccard compares each edge's top-k nearest-neighbor set across models. "
                    "0 means no shared neighbors; 1 means identical top-k neighbor membership."
                ),
                "xref": "paper",
                "yref": "paper",
                "x": 0,
                "y": -0.2,
                "showarrow": False,
                "align": "left",
            }
        ],
    )
    figure.update_xaxes(range=[-0.02, 1.02], title_text="OpenAI/SapBERT nearest-neighbor Jaccard")
    figure.update_yaxes(title_text="", row=2, col=1)
    return figure


def write_neighbor_agreement_html(
    rows: Sequence[Mapping[str, Any]],
    path: str | Path,
    *,
    title: str = "Nearest-neighbor agreement by relationship",
) -> Path:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure = create_neighbor_agreement_figure(rows, title=title)
    output_path.write_text(
        figure.to_html(
            include_plotlyjs="cdn",
            full_html=True,
            post_script=_click_detail_post_script(),
        )
    )
    return output_path


def write_projection_html(
    projections: Sequence[ProjectionResult | Mapping[str, Any]],
    path: str | Path,
    *,
    color_field: str = "predicate_family",
    title: str = "Embedding comparison",
) -> Path:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        projection_figure_html(projections, color_field=color_field, title=title, full_html=True)
    )
    return output_path


def projection_figure_html(
    projections: Sequence[ProjectionResult | Mapping[str, Any]],
    *,
    color_field: str = "predicate_family",
    title: str = "Embedding comparison",
    full_html: bool = False,
) -> str:
    figure = create_projection_figure(projections, color_field=color_field, title=title)
    return figure.to_html(
        include_plotlyjs="cdn",
        full_html=full_html,
        post_script=_click_detail_post_script(link_by_relationship_id=True),
    )


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


def load_retrieval_result_sets_from_path_search_cache(
    path: str | Path = "/tmp/kg_explorer/path_search_cache.json",
) -> list[RetrievalResultSet]:
    cache_path = Path(path)
    if not cache_path.exists():
        return []

    with cache_path.open(encoding="utf-8") as handle:
        payload = json.load(handle)

    entries = payload.get("entries", {})
    if not isinstance(entries, Mapping):
        return []

    result_sets: list[RetrievalResultSet] = []
    for entry in entries.values():
        if not isinstance(entry, Mapping):
            continue
        key_payload = entry.get("key_payload", {})
        metadata = entry.get("metadata", {})
        if not isinstance(key_payload, Mapping) or not isinstance(metadata, Mapping):
            continue

        query = str(key_payload.get("query") or "")
        options = key_payload.get("options", {})
        retrieval_diagnostics = metadata.get("retrieval_diagnostics", {})
        if not isinstance(options, Mapping) or not isinstance(retrieval_diagnostics, Mapping):
            continue

        retrieval_options = options.get("retrieval", {})
        retrieval_metadata = retrieval_diagnostics.get("retrieval", {})
        retrieval_mode = _first_present(
            retrieval_metadata if isinstance(retrieval_metadata, Mapping) else {},
            "retrieval_mode",
            default=None,
        ) or _first_present(
            retrieval_options if isinstance(retrieval_options, Mapping) else {},
            "mode",
            default="unknown",
        )
        model_name = str(retrieval_diagnostics.get("retrieval_model") or options.get("embedding_provider") or "")
        model_name = model_name or None

        for result_set_type, keys in (
            ("raw", ("raw_semantic_candidates", "raw_candidates")),
            ("reranked", ("reranked_candidates",)),
            ("final_anchor", ("selected_anchors",)),
        ):
            rows = _first_sequence(retrieval_diagnostics, keys)
            if not rows:
                continue
            result_sets.append(
                result_set_from_rows(
                    query=query,
                    retrieval_mode=str(retrieval_mode),
                    rows=rows,
                    model_name=model_name,
                    result_set_type=result_set_type,
                )
            )
    return result_sets


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
    mean, components = _pca_fit_2d(matrix)
    if matrix.size == 0:
        return np.empty((0, 2), dtype=float)
    return (matrix - mean) @ components


def _pca_fit_2d(matrix: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    if matrix.size == 0:
        return np.empty((0,), dtype=float), np.empty((0, 2), dtype=float)
    mean = matrix.mean(axis=0, keepdims=True)
    if matrix.shape[0] == 1:
        return mean, np.zeros((matrix.shape[1], 2), dtype=float)
    centered = matrix - mean
    _u, _s, vt = np.linalg.svd(centered, full_matrices=False)
    components = vt[:2].T
    if components.shape[1] < 2:
        components = np.pad(components, ((0, 0), (0, 2 - components.shape[1])))
    return mean, _stabilize_component_signs(centered, components)


def _umap_2d(matrix: np.ndarray, *, seed: int) -> np.ndarray:
    try:
        import umap
    except ImportError as exc:  # pragma: no cover - optional exploratory dependency.
        raise ProjectionUnavailableError("Install umap-learn to use UMAP projections.") from exc
    if matrix.size == 0:
        return np.empty((0, 2), dtype=float)
    reducer = umap.UMAP(n_components=2, random_state=seed, metric="cosine")
    return np.asarray(reducer.fit_transform(matrix), dtype=float)


def _project_extra_vector_2d(
    matrix: np.ndarray,
    vector: np.ndarray,
    *,
    method: str,
    seed: int,
) -> np.ndarray:
    if matrix.size == 0:
        return np.zeros(2, dtype=float)
    if method == "pca":
        mean, components = _pca_fit_2d(matrix)
        return ((vector.reshape(1, -1) - mean) @ components)[0]
    if method == "umap":
        try:
            import umap
        except ImportError as exc:  # pragma: no cover - optional exploratory dependency.
            raise ProjectionUnavailableError("Install umap-learn to use UMAP projections.") from exc
        reducer = umap.UMAP(n_components=2, random_state=seed, metric="cosine")
        reducer.fit(matrix)
        return np.asarray(reducer.transform(vector.reshape(1, -1)), dtype=float)[0]
    raise ValueError("Projection method must be 'pca' or 'umap'.")


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


def _curie_prefix(value: Any) -> str:
    text = str(value or "")
    if ":" not in text:
        return "missing"
    return text.split(":", 1)[0] or "missing"


def _endpoint_label_pair(metadata: Mapping[str, Any]) -> str:
    subject = _primary_label(metadata.get("subject_labels"))
    obj = _primary_label(metadata.get("object_labels"))
    return f"{subject}-{obj}"


def _primary_label(value: Any) -> str:
    if isinstance(value, list) and value:
        return str(value[0])
    if isinstance(value, str) and value:
        return value
    return "unlabeled"


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


def _click_detail_text(row: Mapping[str, Any], *, color_field: str) -> str:
    detail = {
        "relationship_id": row.get("relationship_id"),
        "model": row.get("model"),
        "point_type": row.get("relationship_kind"),
        "query_text": row.get("query_text"),
        color_field: row.get(color_field),
        "neighbor_jaccard": row.get("neighbor_jaccard"),
        "query_rank": row.get("highlight_rank"),
        "subject": row.get("subject") or row.get("hover", {}).get("subject"),
        "object": row.get("object") or row.get("hover", {}).get("object"),
        "predicate": row.get("predicate") or row.get("hover", {}).get("predicate"),
        "predicate_family": row.get("predicate_family") or row.get("hover", {}).get("predicate_family"),
        "publication_id": row.get("publication_id") or row.get("hover", {}).get("publication_id"),
        "semantic_text": row.get("semantic_text") or row.get("hover", {}).get("semantic_text"),
    }
    return "\n".join(f"{key}: {value}" for key, value in detail.items() if value not in {None, ""})


def _point_customdata(row: Mapping[str, Any], *, color_field: str) -> list[str]:
    return [
        str(row.get("relationship_id") or ""),
        _click_detail_text(row, color_field=color_field),
    ]


def _click_detail_post_script(*, link_by_relationship_id: bool = False) -> str:
    linked_selection = ""
    if link_by_relationship_id:
        linked_selection = """
        const relationshipId = Array.isArray(point.customdata) ? point.customdata[0] : '';
        if (relationshipId) {
          if (selectedRelationshipId === relationshipId) {
            clearSelection();
            return;
          }
          selectedRelationshipId = relationshipId;
          const selectedPoints = plot.data.map(function(trace) {
            const matches = [];
            const customdata = trace.customdata || [];
            for (let i = 0; i < customdata.length; i += 1) {
              const row = customdata[i];
              const rowId = Array.isArray(row) ? row[0] : '';
              if (rowId === relationshipId) matches.push(i);
            }
            return matches;
          });
          Plotly.restyle(plot, {
            selectedpoints: selectedPoints,
            selected: {marker: {opacity: 1, size: 13, line: {width: 2, color: '#111111'}}},
            unselected: {marker: {opacity: 0.18}}
          });
        }
        """
    script = """
    (function() {
      const plot = document.getElementById('{plot_id}');
      if (!plot) return;
      let selectedRelationshipId = null;
      const defaultDetailText = 'Click a point to inspect relationship details. Click the selected point again or use Clear selection to reset.';
      const clearButton = document.createElement('button');
      clearButton.textContent = 'Clear selection';
      clearButton.type = 'button';
      clearButton.style.margin = '14px 0 0 0';
      clearButton.style.padding = '6px 10px';
      clearButton.style.border = '1px solid #d0d7de';
      clearButton.style.borderRadius = '6px';
      clearButton.style.background = '#ffffff';
      clearButton.style.cursor = 'pointer';
      const detail = document.createElement('pre');
      detail.textContent = defaultDetailText;
      detail.style.whiteSpace = 'pre-wrap';
      detail.style.border = '1px solid #d0d7de';
      detail.style.borderRadius = '8px';
      detail.style.padding = '12px';
      detail.style.margin = '8px 0 0 0';
      detail.style.maxHeight = '220px';
      detail.style.overflow = 'auto';
      detail.style.fontSize = '12px';
      detail.style.background = '#f6f8fa';
      function clearSelection() {
        selectedRelationshipId = null;
        detail.textContent = defaultDetailText;
        if (plot.data && plot.data.length) {
          Plotly.restyle(plot, 'selectedpoints', plot.data.map(function() { return null; }));
        }
      }
      clearButton.addEventListener('click', clearSelection);
      plot.parentNode.insertBefore(detail, plot.nextSibling);
      plot.parentNode.insertBefore(clearButton, detail);
      plot.on('plotly_click', function(eventData) {
        if (!eventData || !eventData.points || !eventData.points.length) return;
        const point = eventData.points[0];
        const text = Array.isArray(point.customdata)
          ? (point.customdata.length > 1 ? point.customdata[1] : point.customdata[0])
          : '';
        detail.textContent = text || 'No detail payload available for this point.';
        __LINKED_SELECTION__
      });
    })();
    """
    return script.replace("__LINKED_SELECTION__", linked_selection)


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


def _ordered_unique(values: Iterable[Any]) -> list[Any]:
    seen = set()
    ordered = []
    for value in values:
        if value in seen:
            continue
        seen.add(value)
        ordered.append(value)
    return ordered


def _first_sequence(mapping: Mapping[str, Any], keys: Sequence[str]) -> Sequence[Mapping[str, Any]]:
    for key in keys:
        value = mapping.get(key)
        if isinstance(value, Sequence) and not isinstance(value, str):
            return value
    return ()


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
