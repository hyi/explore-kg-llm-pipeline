from __future__ import annotations

import copy
from dataclasses import asdict, dataclass, field
from typing import Any

from explorer.backend.semantic_search.ranking import (
    BROAD_RESPONSE_PREDICATES,
    CANCER_CONTEXT_TERMS,
    EXPLICIT_RESPONSE_PREDICATES,
    GENE_CATEGORY_LABELS,
    GENE_PRODUCT_CATEGORY_LABELS,
    GENOMIC_VARIANT_CATEGORY_LABELS,
    QueryHints,
    parse_query_hints,
)
from explorer.backend.semantic_search.retrieval import tokenize_keyword_query

NEIGHBORHOOD_RANKING_STRATEGY = "query_ranked_one_hop_v1"
NEIGHBOR_QUERY_TEXT_WEIGHT = 1.0
NEIGHBOR_EXPLICIT_RESPONSE_PREDICATE_BOOST = 0.45
NEIGHBOR_BROAD_RESPONSE_PREDICATE_BOOST = 0.18
NEIGHBOR_GENE_ENDPOINT_BOOST = 0.35
NEIGHBOR_GENE_PRODUCT_ENDPOINT_BOOST = 0.30
NEIGHBOR_GENOMIC_VARIANT_ENDPOINT_BOOST = 0.24
NEIGHBOR_CANCER_CONTEXT_BOOST = 0.18
DEFAULT_NEIGHBOR_EXPANSION_LIMIT = 12
MAX_NEIGHBOR_EXPANSION_LIMIT = 50
SUPPORTED_DIRECTIONS = frozenset({"either", "incoming", "outgoing"})


@dataclass(frozen=True)
class NeighborhoodExpansionConfig:
    limit: int = DEFAULT_NEIGHBOR_EXPANSION_LIMIT
    node_categories: tuple[str, ...] = field(default_factory=tuple)
    predicates: tuple[str, ...] = field(default_factory=tuple)
    direction: str = "either"

    def normalized_direction(self) -> str:
        direction = (self.direction or "either").strip().lower()
        if direction not in SUPPORTED_DIRECTIONS:
            raise ValueError(f"Unsupported neighborhood direction '{self.direction}'. Use either, incoming, or outgoing.")
        return direction

    def normalized_limit(self) -> int:
        return max(0, min(int(self.limit), MAX_NEIGHBOR_EXPANSION_LIMIT))

    def normalized_node_categories(self) -> tuple[str, ...]:
        return tuple(sorted({category for category in self.node_categories if category}))

    def normalized_predicates(self) -> tuple[str, ...]:
        return tuple(sorted({predicate for predicate in self.predicates if predicate}))

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["limit"] = self.normalized_limit()
        payload["direction"] = self.normalized_direction()
        payload["node_categories"] = list(self.normalized_node_categories())
        payload["predicates"] = list(self.normalized_predicates())
        return payload


@dataclass(frozen=True)
class NeighborhoodRankingResult:
    candidates: list[dict[str, Any]]
    diagnostics: dict[str, Any]


def rank_neighborhood_candidates(
    candidates: list[dict[str, Any]],
    query: str,
    config: NeighborhoodExpansionConfig | None = None,
) -> NeighborhoodRankingResult:
    config = config or NeighborhoodExpansionConfig()
    hints = parse_query_hints(query)
    query_tokens = tokenize_keyword_query(query)
    scored: list[dict[str, Any]] = []
    excluded: list[dict[str, Any]] = []

    for original_rank, candidate in enumerate(candidates):
        normalized = _copy_candidate(candidate)
        normalized["original_rank"] = original_rank
        exclusion_reason = _filter_exclusion_reason(normalized, config)
        if exclusion_reason:
            excluded.append(_diagnostic_row(normalized, excluded=True, exclusion_reason=exclusion_reason))
            continue
        scored.append(_score_candidate(normalized, query_tokens=query_tokens, hints=hints))

    scored.sort(
        key=lambda candidate: (
            -float(candidate["neighborhood_score"]),
            int(candidate["original_rank"]),
            str(candidate.get("edge", {}).get("id") or ""),
            str(candidate.get("neighbor", {}).get("id") or ""),
        )
    )
    selected = scored[: config.normalized_limit()]
    selected_ids = {_candidate_identity(candidate) for candidate in selected}
    diagnostics = {
        "ranking_strategy": NEIGHBORHOOD_RANKING_STRATEGY,
        "query_hints": asdict(hints),
        "query_tokens": query_tokens,
        "config": config.to_dict(),
        "candidate_count": len(candidates),
        "eligible_count": len(scored),
        "selected_count": len(selected),
        "ranked_candidates": [
            _diagnostic_row(
                candidate,
                excluded=False,
                exclusion_reason=None if _candidate_identity(candidate) in selected_ids else "limit",
            )
            for candidate in scored
        ],
        "excluded_candidates": excluded,
    }
    return NeighborhoodRankingResult(candidates=selected, diagnostics=diagnostics)


def _score_candidate(
    candidate: dict[str, Any],
    *,
    query_tokens: list[str],
    hints: QueryHints,
) -> dict[str, Any]:
    text = _candidate_text(candidate)
    text_tokens = set(tokenize_keyword_query(text))
    matched_tokens = sorted(set(query_tokens) & text_tokens)
    components: dict[str, float] = {}
    reasons: list[str] = []

    if query_tokens:
        text_component = NEIGHBOR_QUERY_TEXT_WEIGHT * len(matched_tokens) / len(set(query_tokens))
        components["query_text_overlap"] = text_component
        if matched_tokens:
            reasons.append("query text overlap")

    if hints.wants_gene:
        gene_component, gene_reason = _gene_endpoint_component(candidate)
        components["gene_endpoint"] = gene_component
        if gene_component:
            reasons.append(gene_reason)

    if hints.wants_resistance_or_response:
        predicate_component, predicate_reason = _response_predicate_component(candidate)
        components["response_predicate"] = predicate_component
        if predicate_component:
            reasons.append(predicate_reason)

    if hints.wants_cancer:
        cancer_component = NEIGHBOR_CANCER_CONTEXT_BOOST if text_tokens & CANCER_CONTEXT_TERMS else 0.0
        components["cancer_context"] = cancer_component
        if cancer_component:
            reasons.append("cancer context match")

    score = sum(components.values())
    candidate["neighborhood_score"] = score
    candidate["ranking_components"] = components
    candidate["ranking_reasons"] = reasons or ["query fallback"]
    candidate["matched_query_tokens"] = matched_tokens
    candidate["matched_query_facets"] = _matched_query_facets(candidate, hints)
    return candidate


def _filter_exclusion_reason(candidate: dict[str, Any], config: NeighborhoodExpansionConfig) -> str | None:
    direction = config.normalized_direction()
    if direction != "either" and candidate.get("direction") != direction:
        return "direction_filter"

    predicates = set(config.normalized_predicates())
    if predicates and str(candidate.get("predicate") or "") not in predicates:
        return "predicate_filter"

    categories = set(config.normalized_node_categories())
    if categories and not (categories & set(_neighbor_labels(candidate))):
        return "node_category_filter"

    return None


def _gene_endpoint_component(candidate: dict[str, Any]) -> tuple[float, str]:
    labels = set(_focus_labels(candidate)) | set(_neighbor_labels(candidate))
    if labels & GENE_CATEGORY_LABELS:
        return NEIGHBOR_GENE_ENDPOINT_BOOST, "gene endpoint category match"
    if labels & GENE_PRODUCT_CATEGORY_LABELS:
        return NEIGHBOR_GENE_PRODUCT_ENDPOINT_BOOST, "gene product endpoint category match"
    if labels & GENOMIC_VARIANT_CATEGORY_LABELS:
        return NEIGHBOR_GENOMIC_VARIANT_ENDPOINT_BOOST, "genomic variant endpoint category match"
    return 0.0, "no gene endpoint match"


def _response_predicate_component(candidate: dict[str, Any]) -> tuple[float, str]:
    predicate = str(candidate.get("predicate") or "")
    if predicate in EXPLICIT_RESPONSE_PREDICATES:
        return NEIGHBOR_EXPLICIT_RESPONSE_PREDICATE_BOOST, "explicit response predicate match"
    if predicate in BROAD_RESPONSE_PREDICATES:
        return NEIGHBOR_BROAD_RESPONSE_PREDICATE_BOOST, "broad response predicate match"
    return 0.0, "no response predicate match"


def _matched_query_facets(candidate: dict[str, Any], hints: QueryHints) -> list[str]:
    components = candidate.get("ranking_components", {})
    facets: list[str] = []
    if hints.wants_gene and components.get("gene_endpoint", 0.0) > 0:
        facets.append("gene_endpoint")
    if hints.wants_resistance_or_response and components.get("response_predicate", 0.0) > 0:
        facets.append("response_predicate")
    if hints.wants_cancer and components.get("cancer_context", 0.0) > 0:
        facets.append("cancer_context")
    if components.get("query_text_overlap", 0.0) > 0:
        facets.append("query_text_overlap")
    return facets


def _candidate_text(candidate: dict[str, Any]) -> str:
    edge = candidate.get("edge", {})
    focus = candidate.get("focus", {})
    neighbor = candidate.get("neighbor", {})
    parts = [
        candidate.get("predicate"),
        edge.get("label"),
        edge.get("type"),
        edge.get("text"),
        edge.get("semantic_text"),
        edge.get("abstract_title"),
        edge.get("supporting_sentence"),
        focus.get("label"),
        focus.get("id"),
        neighbor.get("label"),
        neighbor.get("id"),
    ]
    return " ".join(str(part) for part in parts if part)


def _diagnostic_row(
    candidate: dict[str, Any],
    *,
    excluded: bool,
    exclusion_reason: str | None,
) -> dict[str, Any]:
    edge = candidate.get("edge", {})
    neighbor = candidate.get("neighbor", {})
    return {
        "edge_id": edge.get("id"),
        "neighbor_id": neighbor.get("id"),
        "predicate": candidate.get("predicate"),
        "direction": candidate.get("direction"),
        "original_rank": candidate.get("original_rank"),
        "neighborhood_score": candidate.get("neighborhood_score"),
        "ranking_components": candidate.get("ranking_components", {}),
        "ranking_reasons": candidate.get("ranking_reasons", []),
        "matched_query_tokens": candidate.get("matched_query_tokens", []),
        "matched_query_facets": candidate.get("matched_query_facets", []),
        "excluded": excluded,
        "exclusion_reason": exclusion_reason,
    }


def _candidate_identity(candidate: dict[str, Any]) -> str:
    edge = candidate.get("edge", {})
    neighbor = candidate.get("neighbor", {})
    return f"{edge.get('id') or ''}|{neighbor.get('id') or ''}"


def _focus_labels(candidate: dict[str, Any]) -> list[str]:
    return list((candidate.get("focus") or {}).get("labels") or [])


def _neighbor_labels(candidate: dict[str, Any]) -> list[str]:
    return list((candidate.get("neighbor") or {}).get("labels") or [])


def _copy_candidate(candidate: dict[str, Any]) -> dict[str, Any]:
    return copy.deepcopy(candidate)
