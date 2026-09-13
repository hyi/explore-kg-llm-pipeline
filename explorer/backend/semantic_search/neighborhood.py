from __future__ import annotations

import copy
from dataclasses import asdict, dataclass, field
from typing import Any

from explorer.backend.semantic_search.query_intent import (
    QUALITY_CONTEXTUAL_MENTION,
    QueryIntent,
    category_families_from_labels,
    parse_query_intent,
    predicate_family,
    relationship_quality_tier,
)
from explorer.backend.semantic_search.retrieval import tokenize_keyword_query

NEIGHBORHOOD_RANKING_STRATEGY = "query_intent_ranked_one_hop_v1"
NEIGHBOR_ENDPOINT_CATEGORY_MATCH = 0.35
NEIGHBOR_PREDICATE_FAMILY_MATCH = 0.30
NEIGHBOR_CONTEXTUAL_MENTION_PENALTY = -0.20
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
    intent = parse_query_intent(query)
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
        scored.append(_score_candidate(normalized, query_tokens=query_tokens, intent=intent))

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
        "query_intent": intent.to_dict(),
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
    intent: QueryIntent,
) -> dict[str, Any]:
    text = _candidate_text(candidate)
    text_tokens = set(tokenize_keyword_query(text))
    matched_tokens = sorted(set(query_tokens) & text_tokens)
    components: dict[str, float] = {}
    reasons: list[str] = []

    endpoint_component = _endpoint_category_component(candidate, intent)
    if endpoint_component:
        components["endpoint_category_compatibility"] = endpoint_component
        reasons.append("endpoint category compatibility")

    predicate_component = _predicate_family_component(candidate, intent)
    if predicate_component:
        components["predicate_family_compatibility"] = predicate_component
        reasons.append("predicate family compatibility")

    quality_tier = relationship_quality_tier(candidate.get("predicate"))
    if quality_tier == QUALITY_CONTEXTUAL_MENTION:
        components["relationship_quality"] = NEIGHBOR_CONTEXTUAL_MENTION_PENALTY
        reasons.append("contextual mention relationship")

    score = sum(components.values())
    candidate["neighborhood_score"] = score
    candidate["ranking_components"] = components
    candidate["ranking_reasons"] = reasons or ["query fallback"]
    candidate["matched_query_tokens"] = matched_tokens
    candidate["query_intent"] = intent.to_dict()
    candidate["relationship_quality_tier"] = quality_tier
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


def _endpoint_category_component(candidate: dict[str, Any], intent: QueryIntent) -> float:
    requested = {
        request.family
        for request in (
            *intent.requested_subject_categories,
            *intent.requested_object_categories,
            *intent.requested_endpoint_categories,
        )
    }
    if not requested:
        return 0.0
    labels = _focus_labels(candidate) + _neighbor_labels(candidate)
    observed = category_families_from_labels(labels)
    return NEIGHBOR_ENDPOINT_CATEGORY_MATCH if requested & observed else 0.0


def _predicate_family_component(candidate: dict[str, Any], intent: QueryIntent) -> float:
    requested = {request.family for request in intent.requested_predicate_families}
    if not requested:
        return 0.0
    return NEIGHBOR_PREDICATE_FAMILY_MATCH if predicate_family(candidate.get("predicate")) in requested else 0.0


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
        "query_intent": candidate.get("query_intent", {}),
        "relationship_quality_tier": candidate.get("relationship_quality_tier"),
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
