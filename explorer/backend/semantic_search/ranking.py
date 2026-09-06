from __future__ import annotations

import copy
import re
from collections.abc import Iterable
from dataclasses import asdict, dataclass
from typing import Any

ANCHOR_RANKING_STRATEGY = "dense_query_aware_v1"

# Initial transparent heuristics. These weights are intentionally simple and
# should be evaluated against LitCoin retrieval examples before being treated
# as empirically validated.
SEMANTIC_SCORE_WEIGHT = 1.0
GENE_ENDPOINT_LABEL_BOOST = 0.20
GENE_ENDPOINT_LEXICAL_FALLBACK_BOOST = 0.08
EXPLICIT_RESPONSE_PREDICATE_BOOST = 0.18
BROAD_RESPONSE_PREDICATE_BOOST = 0.06
QUERY_TEXT_OVERLAP_MAX_BOOST = 0.04
MISSING_ACTIVE_HINT_PENALTY = -0.02

GENE_HINT_TERMS = frozenset({"gene", "genes", "genetic", "mutation", "mutations"})
RESISTANCE_RESPONSE_HINT_TERMS = frozenset(
    {"resistance", "resistant", "chemoresistance", "sensitivity", "response"}
)

EXPLICIT_RESPONSE_PREDICATES = frozenset(
    {
        "biolink:affects_response_to",
        "biolink:increases_response_to",
        "biolink:decreases_response_to",
        "biolink:associated_with_resistance_to",
    }
)
BROAD_RESPONSE_PREDICATES = frozenset(
    {
        "biolink:affects",
        "biolink:regulates",
        "biolink:contributes_to",
        "biolink:causes",
        "biolink:associated_with",
        "biolink:genetically_associated_with",
        "biolink:correlated_with",
        "biolink:positively_correlated_with",
    }
)

GENE_LIKE_LABEL_TERMS = (
    "gene",
    "geneorgeneproduct",
    "genomicentity",
    "protein",
)
GENE_LIKE_ID_PREFIXES = (
    "ncbigene:",
    "hgnc:",
    "ensembl:",
    "uniprotkb:",
)
STOPWORDS = frozenset(
    {
        "a",
        "an",
        "and",
        "are",
        "for",
        "in",
        "into",
        "is",
        "of",
        "or",
        "the",
        "to",
        "with",
    }
)


@dataclass(frozen=True)
class QueryHints:
    wants_gene: bool = False
    wants_resistance_or_response: bool = False

    @property
    def has_active_hint(self) -> bool:
        return self.wants_gene or self.wants_resistance_or_response


@dataclass(frozen=True)
class AnchorRankingConfig:
    candidate_multiplier: int = 5
    minimum_candidate_pool: int = 25
    maximum_candidate_pool: int = 100
    max_per_publication: int = 2

    def to_cache_dict(self) -> dict[str, int]:
        return asdict(self)


def parse_query_hints(query: str) -> QueryHints:
    tokens = set(_tokens(query))
    return QueryHints(
        wants_gene=bool(tokens & GENE_HINT_TERMS),
        wants_resistance_or_response=bool(tokens & RESISTANCE_RESPONSE_HINT_TERMS),
    )


def candidate_pool_size(requested_k: int, config: AnchorRankingConfig) -> int:
    if requested_k <= 0:
        return 0
    return min(
        max(int(requested_k) * int(config.candidate_multiplier), int(config.minimum_candidate_pool)),
        int(config.maximum_candidate_pool),
    )


def rerank_and_diversify_relationships(
    query: str,
    candidates: list[Any],
    requested_k: int,
    config: AnchorRankingConfig | None = None,
) -> list[Any]:
    config = config or AnchorRankingConfig()
    if requested_k <= 0 or not candidates:
        return []

    hints = parse_query_hints(query)
    scored = [
        _score_candidate(query=query, candidate=candidate, raw_rank=rank, hints=hints)
        for rank, candidate in enumerate(candidates)
    ]
    scored.sort(key=_rank_sort_key)

    selected = _diversified_selection(scored, requested_k=requested_k, config=config)
    return [item["candidate"] for item in selected]


def relationship_identity(metadata: dict[str, Any], fallback: str = "") -> str:
    for key in ("relationship_identity", "id", "rel_id", "relationship_id", "element_id"):
        value = metadata.get(key)
        if value:
            return str(value)

    predicate = str(metadata.get("predicate") or "")
    subject = _endpoint_value(metadata, "subject")
    obj = _endpoint_value(metadata, "object")
    publication = publication_identity(metadata) or ""
    identity = "|".join(part for part in (predicate, subject, obj, publication) if part)
    return identity or fallback


def publication_identity(metadata: dict[str, Any]) -> str | None:
    value = metadata.get("publication_id")
    if value:
        return str(value)

    publications = metadata.get("publications")
    if isinstance(publications, list) and publications:
        return str(publications[0])
    if isinstance(publications, str) and publications:
        return publications

    value = metadata.get("llm_abstract_id")
    if value:
        return str(value)

    value = metadata.get("abstract_title")
    if value:
        return str(value)

    return None


def endpoint_neighborhood(metadata: dict[str, Any]) -> tuple[str, str]:
    return (_normalize_key(_endpoint_value(metadata, "subject")), _normalize_key(_endpoint_value(metadata, "object")))


def compact_anchor_metadata(metadata: dict[str, Any]) -> dict[str, Any]:
    keys = (
        "anchor_score",
        "semantic_score",
        "ranking_reasons",
        "ranking_components",
        "is_semantic_fallback",
        "publication_id",
        "relationship_identity",
        "raw_rank",
        "predicate",
        "subject_labels",
        "object_labels",
    )
    return {key: metadata[key] for key in keys if key in metadata}


def _score_candidate(query: str, candidate: Any, raw_rank: int, hints: QueryHints) -> dict[str, Any]:
    raw_metadata = dict(getattr(candidate, "metadata", {}) or {})
    semantic_score = float(raw_metadata.get("semantic_score", raw_metadata.get("score", 0.0)) or 0.0)
    components: dict[str, float] = {"semantic": semantic_score * SEMANTIC_SCORE_WEIGHT}
    reasons = [f"semantic score {semantic_score:.4f}"]
    positive_hint_match = False

    if hints.wants_gene:
        gene_component, gene_reason = _gene_endpoint_component(raw_metadata)
        components["gene_endpoint"] = gene_component
        if gene_component > 0:
            positive_hint_match = True
            reasons.append(gene_reason)
        else:
            components["missing_gene_hint"] = MISSING_ACTIVE_HINT_PENALTY
            reasons.append("no gene endpoint match")

    if hints.wants_resistance_or_response:
        predicate_component, predicate_reason = _predicate_component(raw_metadata.get("predicate"))
        components["predicate"] = predicate_component
        if predicate_component > 0:
            positive_hint_match = True
            reasons.append(predicate_reason)
        else:
            components["missing_response_hint"] = MISSING_ACTIVE_HINT_PENALTY
            reasons.append("no resistance/response predicate match")

    overlap = _query_text_overlap_component(query, raw_metadata, getattr(candidate, "page_content", ""))
    if overlap:
        components["query_text_overlap"] = overlap
        reasons.append("query/context term overlap")

    anchor_score = sum(components.values())
    metadata = dict(raw_metadata)
    identity = relationship_identity(metadata, fallback=f"raw-rank:{raw_rank}")
    publication_id = publication_identity(metadata)
    metadata.update(
        {
            "semantic_score": semantic_score,
            "anchor_score": anchor_score,
            "score": anchor_score,
            "ranking_reasons": reasons,
            "ranking_components": components,
            "is_semantic_fallback": hints.has_active_hint and not positive_hint_match,
            "publication_id": publication_id,
            "relationship_identity": identity,
            "raw_rank": raw_rank,
            "anchor_ranking_strategy": ANCHOR_RANKING_STRATEGY,
        }
    )
    return {
        "candidate": _with_metadata(candidate, metadata),
        "metadata": metadata,
        "raw_rank": raw_rank,
        "identity": identity,
        "publication_id": publication_id,
        "neighborhood": endpoint_neighborhood(metadata),
    }


def _gene_endpoint_component(metadata: dict[str, Any]) -> tuple[float, str]:
    subject_labels = _labels(metadata, "subject")
    object_labels = _labels(metadata, "object")
    labels_available = bool(subject_labels or object_labels)

    if _has_gene_like_label(subject_labels) or _has_gene_like_label(object_labels):
        return GENE_ENDPOINT_LABEL_BOOST, "gene endpoint label match"

    if labels_available:
        return 0.0, "no gene endpoint match"

    if _has_gene_lexical_fallback(metadata):
        return GENE_ENDPOINT_LEXICAL_FALLBACK_BOOST, "gene endpoint lexical fallback"

    return 0.0, "no gene endpoint match"


def _predicate_component(predicate: Any) -> tuple[float, str]:
    predicate_text = str(predicate or "")
    if predicate_text in EXPLICIT_RESPONSE_PREDICATES:
        return EXPLICIT_RESPONSE_PREDICATE_BOOST, "resistance/response predicate match"
    if predicate_text in BROAD_RESPONSE_PREDICATES:
        return BROAD_RESPONSE_PREDICATE_BOOST, "broad response-related predicate match"
    return 0.0, "no resistance/response predicate match"


def _query_text_overlap_component(query: str, metadata: dict[str, Any], page_content: Any) -> float:
    query_tokens = [token for token in _tokens(query) if token not in STOPWORDS]
    if not query_tokens:
        return 0.0

    text_parts = [
        page_content,
        metadata.get("semantic_text"),
        metadata.get("abstract_title"),
        metadata.get("llm_subject"),
        metadata.get("llm_object"),
        metadata.get("llm_relationship"),
    ]
    text_tokens = set(_tokens(" ".join(str(part) for part in text_parts if part)))
    if not text_tokens:
        return 0.0

    matches = len(set(query_tokens) & text_tokens)
    if not matches:
        return 0.0
    return min(QUERY_TEXT_OVERLAP_MAX_BOOST, QUERY_TEXT_OVERLAP_MAX_BOOST * matches / len(set(query_tokens)))


def _diversified_selection(
    ranked_items: list[dict[str, Any]],
    requested_k: int,
    config: AnchorRankingConfig,
) -> list[dict[str, Any]]:
    selected: list[dict[str, Any]] = []
    selected_identities: set[str] = set()
    publication_counts: dict[str, int] = {}
    used_neighborhoods: set[tuple[str, str]] = set()

    def try_add(item: dict[str, Any], enforce_publication: bool, enforce_neighborhood: bool) -> None:
        if len(selected) >= requested_k:
            return
        identity = item["identity"]
        if identity in selected_identities:
            return
        publication_id = item["publication_id"]
        if (
            enforce_publication
            and publication_id
            and publication_counts.get(publication_id, 0) >= config.max_per_publication
        ):
            return
        neighborhood = item["neighborhood"]
        if enforce_neighborhood and all(neighborhood) and neighborhood in used_neighborhoods:
            return

        selected.append(item)
        selected_identities.add(identity)
        if publication_id:
            publication_counts[publication_id] = publication_counts.get(publication_id, 0) + 1
        if all(neighborhood):
            used_neighborhoods.add(neighborhood)

    for item in ranked_items:
        try_add(item, enforce_publication=True, enforce_neighborhood=True)
    for item in ranked_items:
        try_add(item, enforce_publication=True, enforce_neighborhood=False)
    for item in ranked_items:
        try_add(item, enforce_publication=False, enforce_neighborhood=False)

    return selected


def _rank_sort_key(item: dict[str, Any]) -> tuple[float, int, str, str, str, str]:
    metadata = item["metadata"]
    subject, obj = item["neighborhood"]
    return (
        -float(metadata["anchor_score"]),
        int(item["raw_rank"]),
        str(item["identity"]),
        str(metadata.get("predicate") or ""),
        subject,
        obj,
    )


def _labels(metadata: dict[str, Any], endpoint: str) -> list[str]:
    for key in (f"{endpoint}_labels", f"{endpoint}_node_labels"):
        value = metadata.get(key)
        if isinstance(value, list):
            return [str(item) for item in value]
    return []


def _has_gene_like_label(labels: Iterable[str]) -> bool:
    for label in labels:
        compact_label = re.sub(r"[^a-z0-9]", "", label.casefold())
        if any(term in compact_label for term in GENE_LIKE_LABEL_TERMS):
            return True
    return False


def _has_gene_lexical_fallback(metadata: dict[str, Any]) -> bool:
    for endpoint in ("subject", "object"):
        endpoint_id = _endpoint_value(metadata, endpoint).casefold()
        if endpoint_id.startswith(GENE_LIKE_ID_PREFIXES):
            return True
        endpoint_type = str(metadata.get(f"llm_{endpoint}_type") or "").casefold()
        if endpoint_type in {"gene", "protein"}:
            return True
    return False


def _endpoint_value(metadata: dict[str, Any], endpoint: str) -> str:
    keys = (
        f"original_{endpoint}",
        endpoint,
        f"llm_{endpoint}",
        f"{endpoint}_id",
        f"{endpoint}_name",
    )
    for key in keys:
        value = metadata.get(key)
        if value:
            return str(value)
    return ""


def _normalize_key(value: str) -> str:
    return " ".join(str(value or "").casefold().split())


def _tokens(text: str) -> list[str]:
    return re.findall(r"[a-z0-9]+", (text or "").casefold())


def _with_metadata(candidate: Any, metadata: dict[str, Any]) -> Any:
    if hasattr(candidate, "model_copy"):
        return candidate.model_copy(update={"metadata": metadata})

    cloned = copy.copy(candidate)
    cloned.metadata = metadata
    return cloned
