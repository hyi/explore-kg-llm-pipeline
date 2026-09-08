from __future__ import annotations

import copy
import re
from collections.abc import Iterable
from dataclasses import asdict, dataclass
from typing import Any

ANCHOR_RANKING_STRATEGY = "dense_query_aware_facets_v2"

# Initial transparent heuristics. These weights are intentionally simple and
# should be evaluated against LitCoin retrieval examples before being treated
# as empirically validated.
SEMANTIC_SCORE_WEIGHT = 1.0
GENE_ENDPOINT_LABEL_BOOST = 0.20
GENE_PRODUCT_ENDPOINT_LABEL_BOOST = 0.18
GENOMIC_VARIANT_ENDPOINT_LABEL_BOOST = 0.16
GENOMIC_VARIANT_PARTIAL_GENE_BOOST = 0.06
GENE_ENDPOINT_LEXICAL_FALLBACK_BOOST = 0.08
EXPLICIT_RESPONSE_PREDICATE_BOOST = 0.30
BROAD_RESPONSE_PREDICATE_BOOST = 0.10
RESISTANCE_RESPONSE_TEXT_BOOST = 0.10
CANCER_CONTEXT_BOOST = 0.10
QUERY_TEXT_OVERLAP_MAX_BOOST = 0.04
MISSING_GENE_HINT_PENALTY = -0.05
MISSING_RESPONSE_HINT_PENALTY = -0.08
MISSING_CANCER_HINT_PENALTY = -0.04

GENE_HINT_TERMS = frozenset({"gene", "genes", "genetic"})
GENETIC_VARIANT_HINT_TERMS = frozenset(
    {"mutation", "mutations", "variant", "variants", "allele", "alleles", "polymorphism", "polymorphisms"}
)
RESISTANCE_RESPONSE_HINT_TERMS = frozenset(
    {"resistance", "resistant", "chemoresistance", "chemoresistant", "sensitivity", "response"}
)
CANCER_HINT_TERMS = frozenset(
    {
        "cancer",
        "cancers",
        "tumor",
        "tumors",
        "tumour",
        "tumours",
        "neoplasm",
        "neoplasms",
        "carcinoma",
        "carcinomas",
        "glioma",
        "glioblastoma",
        "melanoma",
        "leukemia",
        "lymphoma",
    }
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

GENE_CATEGORY_LABELS = frozenset(
    {
        "biolink:Gene",
    }
)
GENE_PRODUCT_CATEGORY_LABELS = frozenset(
    {
        "biolink:GeneOrGeneProduct",
        "biolink:Protein",
        "biolink:Polypeptide",
        "biolink:GeneProductMixin",
    }
)
GENOMIC_VARIANT_CATEGORY_LABELS = frozenset(
    {
        "biolink:GenomicEntity",
        "biolink:SequenceVariant",
        "biolink:Allele",
        "biolink:Haplotype",
    }
)
CHEMICAL_CATEGORY_LABELS = frozenset(
    {
        "biolink:ChemicalEntity",
        "biolink:Drug",
        "biolink:ChemicalOrDrugOrTreatment",
        "biolink:SmallMolecule",
    }
)
BIOLOGICAL_PROCESS_CATEGORY_LABELS = frozenset(
    {
        "biolink:BiologicalProcess",
        "biolink:BiologicalProcessOrActivity",
        "biolink:MolecularActivity",
    }
)
DRUG_CONTEXT_TERMS = frozenset(
    "drug treatment therapy therapeutic chemotherapy chemotherapeutic chemoresistance "
    "resistance resistant sensitivity sensitive response".split()
)
RESISTANCE_RESPONSE_CONTEXT_TERMS = frozenset(
    {"chemoresistance", "chemoresistant", "resistance", "resistant", "sensitivity", "sensitive"}
)
GENERIC_RESPONSE_TERMS = frozenset({"response", "responsive"})
THERAPEUTIC_CONTEXT_TERMS = frozenset(
    {"drug", "drugs", "treatment", "treatments", "therapy", "therapies", "therapeutic", "chemotherapy", "chemotherapeutic"}
)
CANCER_CONTEXT_TERMS = CANCER_HINT_TERMS | frozenset(
    {
        "oncology",
        "oncogenic",
        "malignancy",
        "malignant",
        "metastasis",
        "metastatic",
        "carcinogenesis",
        "tumorigenesis",
        "tumorigenic",
    }
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
    wants_genetic_variant: bool = False
    wants_resistance_or_response: bool = False
    wants_cancer: bool = False

    @property
    def has_active_hint(self) -> bool:
        return self.wants_gene or self.wants_genetic_variant or self.wants_resistance_or_response or self.wants_cancer


@dataclass(frozen=True)
class AnchorRankingConfig:
    candidate_multiplier: int = 5
    minimum_candidate_pool: int = 25
    maximum_candidate_pool: int = 100
    max_per_publication: int = 2

    def to_cache_dict(self) -> dict[str, int]:
        return asdict(self)


@dataclass(frozen=True)
class AnchorRankingResult:
    relationships: list[Any]
    diagnostics: dict[str, Any]


def parse_query_hints(query: str) -> QueryHints:
    tokens = set(_tokens(query))
    wants_genetic_variant = bool(tokens & GENETIC_VARIANT_HINT_TERMS)
    return QueryHints(
        wants_gene=bool(tokens & GENE_HINT_TERMS) or wants_genetic_variant,
        wants_genetic_variant=wants_genetic_variant,
        wants_resistance_or_response=bool(tokens & RESISTANCE_RESPONSE_HINT_TERMS),
        wants_cancer=bool(tokens & CANCER_HINT_TERMS),
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
    return rerank_relationships_with_diagnostics(
        query=query,
        candidates=candidates,
        requested_k=requested_k,
        config=config,
    ).relationships


def rerank_relationships_with_diagnostics(
    query: str,
    candidates: list[Any],
    requested_k: int,
    config: AnchorRankingConfig | None = None,
) -> AnchorRankingResult:
    config = config or AnchorRankingConfig()
    hints = parse_query_hints(query)
    if requested_k <= 0 or not candidates:
        return AnchorRankingResult(
            relationships=[],
            diagnostics={
                "ranking_strategy": ANCHOR_RANKING_STRATEGY,
                "query_hints": asdict(hints),
                "requested_k": requested_k,
                "candidate_count": len(candidates),
                "ranked_candidates": [],
                "selected_anchors": [],
            },
        )

    scored = [
        _score_candidate(query=query, candidate=candidate, raw_rank=rank, hints=hints)
        for rank, candidate in enumerate(candidates)
    ]
    scored.sort(key=_rank_sort_key)
    for rerank_rank, item in enumerate(scored):
        item["rerank_rank"] = rerank_rank

    selected, selection_status = _diversified_selection_with_diagnostics(
        scored,
        requested_k=requested_k,
        config=config,
        hints=hints,
    )
    ranked_candidates = [
        _ranking_candidate_diagnostic(
            item,
            hints=hints,
            selection_status=selection_status.get(item["raw_rank"], {}),
        )
        for item in scored
    ]
    selected_anchors = [
        _ranking_candidate_diagnostic(
            item,
            hints=hints,
            selection_status=selection_status.get(item["raw_rank"], {}),
        )
        for item in selected
    ]
    return AnchorRankingResult(
        relationships=[item["candidate"] for item in selected],
        diagnostics={
            "ranking_strategy": ANCHOR_RANKING_STRATEGY,
            "query_hints": asdict(hints),
            "requested_k": requested_k,
            "candidate_count": len(candidates),
            "ranked_candidates": ranked_candidates,
            "selected_anchors": selected_anchors,
        },
    )


def _ranking_candidate_diagnostic(
    item: dict[str, Any],
    hints: QueryHints,
    selection_status: dict[str, Any],
) -> dict[str, Any]:
    metadata = item["metadata"]
    subject_labels = _labels(metadata, "subject")
    object_labels = _labels(metadata, "object")
    selected = bool(selection_status.get("selected"))
    return {
        "raw_rank": item["raw_rank"],
        "rerank_rank": int(item.get("rerank_rank", 0)),
        "relationship_identity": item["identity"],
        "semantic_score": float(metadata.get("semantic_score", 0.0) or 0.0),
        "anchor_score": float(metadata.get("anchor_score", 0.0) or 0.0),
        "ranking_components": dict(metadata.get("ranking_components", {}) or {}),
        "ranking_reasons": list(metadata.get("ranking_reasons", []) or []),
        "query_hints": asdict(hints),
        "matched_query_facets": _matched_query_facets(metadata, hints),
        "is_semantic_fallback": bool(metadata.get("is_semantic_fallback", False)),
        "predicate": metadata.get("predicate"),
        "endpoint_categories": {
            "subject": {
                "labels": subject_labels,
                "tier": _endpoint_category_tier(subject_labels),
            },
            "object": {
                "labels": object_labels,
                "tier": _endpoint_category_tier(object_labels),
            },
        },
        "publication_id": item["publication_id"],
        "retrieval_model": metadata.get("retrieval_model"),
        "retrieval_method": metadata.get("retrieval_method"),
        "retrieval_index_name": metadata.get("retrieval_index_name"),
        "retrieval_embedding_property": metadata.get("retrieval_embedding_property"),
        "retrieval_expected_dimensions": metadata.get("retrieval_expected_dimensions"),
        "retrieval_query_embedding_dimensions": metadata.get("retrieval_query_embedding_dimensions"),
        "anchor_entities": {
            "subject": _endpoint_value(metadata, "subject"),
            "object": _endpoint_value(metadata, "object"),
            "neighborhood": list(item["neighborhood"]),
        },
        "entered_selected_anchor_set": selected,
        "selection_pass": selection_status.get("selection_pass"),
        "exclusion_reason": None if selected else selection_status.get("exclusion_reason"),
        "exclusion_pass": None if selected else selection_status.get("exclusion_pass"),
        "diversity_pass": selection_status.get("selection_pass") == "diversity_pass",
        "fallback_pass": selection_status.get("selection_pass") in {
            "publication_fallback_pass",
            "semantic_fallback_pass",
        },
    }


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
        gene_component, gene_reason = _gene_endpoint_component(raw_metadata, hints)
        components["gene_endpoint"] = gene_component
        if gene_component > 0:
            positive_hint_match = True
            reasons.append(gene_reason)
        else:
            components["missing_gene_hint"] = MISSING_GENE_HINT_PENALTY
            reasons.append("no gene endpoint match")

    if hints.wants_resistance_or_response:
        predicate_component, predicate_reason = _predicate_component(raw_metadata)
        components["predicate"] = predicate_component
        if predicate_component > 0:
            positive_hint_match = True
            reasons.append(predicate_reason)
        else:
            components["missing_response_hint"] = MISSING_RESPONSE_HINT_PENALTY
            reasons.append("no resistance/response predicate match")
        response_text_component = _resistance_response_text_component(raw_metadata, getattr(candidate, "page_content", ""))
        if response_text_component:
            components["resistance_response_context"] = response_text_component
            positive_hint_match = True
            reasons.append("resistance/response context text match")

    if hints.wants_cancer:
        cancer_component, cancer_reason = _cancer_context_component(raw_metadata, getattr(candidate, "page_content", ""))
        components["cancer_context"] = cancer_component
        if cancer_component > 0:
            positive_hint_match = True
            reasons.append(cancer_reason)
        else:
            components["missing_cancer_hint"] = MISSING_CANCER_HINT_PENALTY
            reasons.append("no cancer context match")

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


def _gene_endpoint_component(metadata: dict[str, Any], hints: QueryHints) -> tuple[float, str]:
    subject_labels = _labels(metadata, "subject")
    object_labels = _labels(metadata, "object")
    labels_available = bool(subject_labels or object_labels)

    subject_tier = _endpoint_category_tier(subject_labels)
    object_tier = _endpoint_category_tier(object_labels)
    matched_tier = _best_gene_compatible_tier(subject_tier, object_tier)
    if matched_tier == "gene":
        return GENE_ENDPOINT_LABEL_BOOST, "gene endpoint category match"
    if matched_tier == "gene product":
        return GENE_PRODUCT_ENDPOINT_LABEL_BOOST, "gene product endpoint category match"
    if matched_tier == "genomic variant":
        if hints.wants_genetic_variant:
            return GENOMIC_VARIANT_ENDPOINT_LABEL_BOOST, "genomic variant endpoint category match"
        return GENOMIC_VARIANT_PARTIAL_GENE_BOOST, "genomic variant endpoint partial gene match"

    if labels_available:
        return 0.0, "no gene endpoint match"

    if _has_gene_lexical_fallback(metadata):
        return GENE_ENDPOINT_LEXICAL_FALLBACK_BOOST, "gene endpoint lexical fallback"

    return 0.0, "no gene endpoint match"


def _predicate_component(metadata: dict[str, Any]) -> tuple[float, str]:
    predicate_text = str(metadata.get("predicate") or "")
    has_drug_response_context = _has_drug_response_context(metadata)
    if predicate_text in EXPLICIT_RESPONSE_PREDICATES:
        if has_drug_response_context:
            return EXPLICIT_RESPONSE_PREDICATE_BOOST, "drug response predicate context match"
        return 0.0, "response predicate lacks drug/resistance context"
    if predicate_text in BROAD_RESPONSE_PREDICATES and has_drug_response_context:
        return BROAD_RESPONSE_PREDICATE_BOOST, "broad drug response context match"
    return 0.0, "no resistance/response predicate match"


def _resistance_response_text_component(metadata: dict[str, Any], page_content: Any) -> float:
    edge_tokens = set(_tokens(_edge_specific_text(metadata)))
    if _has_resistance_response_context_tokens(edge_tokens):
        return RESISTANCE_RESPONSE_TEXT_BOOST

    context_tokens = set(_tokens(_compact_context_text(metadata, page_content)))
    if _has_resistance_response_context_tokens(context_tokens):
        return RESISTANCE_RESPONSE_TEXT_BOOST
    return 0.0


def _has_resistance_response_context_tokens(tokens: set[str]) -> bool:
    if tokens & RESISTANCE_RESPONSE_CONTEXT_TERMS:
        return True
    return bool(tokens & GENERIC_RESPONSE_TERMS) and bool(tokens & THERAPEUTIC_CONTEXT_TERMS)


def _cancer_context_component(metadata: dict[str, Any], page_content: Any) -> tuple[float, str]:
    endpoint_text = " ".join(
        str(part)
        for part in (
            _endpoint_value(metadata, "subject"),
            _endpoint_value(metadata, "object"),
            metadata.get("llm_subject_type"),
            metadata.get("llm_object_type"),
        )
        if part
    )
    endpoint_tokens = set(_tokens(endpoint_text))
    if endpoint_tokens & CANCER_CONTEXT_TERMS:
        return CANCER_CONTEXT_BOOST, "cancer endpoint/context match"

    context_tokens = set(_tokens(_compact_context_text(metadata, page_content)))
    if context_tokens & CANCER_CONTEXT_TERMS:
        return CANCER_CONTEXT_BOOST, "cancer context text match"
    return 0.0, "no cancer context match"


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


def _matched_query_facets(metadata: dict[str, Any], hints: QueryHints) -> list[str]:
    components = dict(metadata.get("ranking_components", {}) or {})
    facets: list[str] = []
    if hints.wants_gene and components.get("gene_endpoint", 0.0) > 0:
        facets.append("gene_endpoint")
    if hints.wants_resistance_or_response and components.get("predicate", 0.0) > 0:
        facets.append("resistance_or_response_predicate")
    if hints.wants_resistance_or_response and components.get("resistance_response_context", 0.0) > 0:
        facets.append("resistance_or_response_context")
    if hints.wants_cancer and components.get("cancer_context", 0.0) > 0:
        facets.append("cancer_context")
    if components.get("query_text_overlap", 0.0) > 0:
        facets.append("query_text_overlap")
    return facets


def _diversified_selection(
    ranked_items: list[dict[str, Any]],
    requested_k: int,
    config: AnchorRankingConfig,
) -> list[dict[str, Any]]:
    selected, _selection_status = _diversified_selection_with_diagnostics(
        ranked_items,
        requested_k=requested_k,
        config=config,
        hints=None,
    )
    return selected


def _diversified_selection_with_diagnostics(
    ranked_items: list[dict[str, Any]],
    requested_k: int,
    config: AnchorRankingConfig,
    hints: QueryHints | None = None,
) -> tuple[list[dict[str, Any]], dict[int, dict[str, Any]]]:
    selected: list[dict[str, Any]] = []
    selected_identities: set[str] = set()
    publication_counts: dict[str, int] = {}
    used_neighborhoods: set[tuple[str, str]] = set()
    selection_status: dict[int, dict[str, Any]] = {}

    def record_exclusion(item: dict[str, Any], reason: str, pass_name: str) -> None:
        status = selection_status.setdefault(item["raw_rank"], {"selected": False})
        if not status.get("selected"):
            status["exclusion_reason"] = reason
            status["exclusion_pass"] = pass_name

    def try_add(
        item: dict[str, Any],
        enforce_publication: bool,
        enforce_neighborhood: bool,
        pass_name: str,
    ) -> None:
        if len(selected) >= requested_k:
            record_exclusion(item, "selection_limit_reached", pass_name)
            return
        identity = item["identity"]
        if identity in selected_identities:
            record_exclusion(item, "duplicate_relationship_identity", pass_name)
            return
        publication_id = item["publication_id"]
        if (
            enforce_publication
            and publication_id
            and publication_counts.get(publication_id, 0) >= config.max_per_publication
        ):
            record_exclusion(item, "publication_limit", pass_name)
            return
        neighborhood = item["neighborhood"]
        if enforce_neighborhood and all(neighborhood) and neighborhood in used_neighborhoods:
            record_exclusion(item, "duplicate_neighborhood", pass_name)
            return

        selected.append(item)
        selected_identities.add(identity)
        selection_status[item["raw_rank"]] = {
            "selected": True,
            "selection_pass": pass_name,
        }
        if publication_id:
            publication_counts[publication_id] = publication_counts.get(publication_id, 0) + 1
        if all(neighborhood):
            used_neighborhoods.add(neighborhood)

    for item in ranked_items:
        try_add(
            item,
            enforce_publication=True,
            enforce_neighborhood=True,
            pass_name="diversity_pass",
        )
    for item in ranked_items:
        try_add(
            item,
            enforce_publication=True,
            enforce_neighborhood=False,
            pass_name="publication_fallback_pass",
        )
    for item in ranked_items:
        try_add(
            item,
            enforce_publication=False,
            enforce_neighborhood=False,
            pass_name="semantic_fallback_pass",
        )

    if hints is not None:
        _ensure_required_facet_coverage(
            selected=selected,
            ranked_items=ranked_items,
            selection_status=selection_status,
            hints=hints,
        )

    return selected, selection_status


def _ensure_required_facet_coverage(
    selected: list[dict[str, Any]],
    ranked_items: list[dict[str, Any]],
    selection_status: dict[int, dict[str, Any]],
    hints: QueryHints,
) -> None:
    required_facets = _required_coverage_facets(hints)
    if not required_facets or not selected:
        return

    selected_identities = {item["identity"] for item in selected}
    for required_facet in required_facets:
        selected_coverage = _coverage_counts(selected, hints)
        if selected_coverage.get(required_facet, 0) > 0:
            continue

        replacement = next(
            (
                item
                for item in ranked_items
                if item["identity"] not in selected_identities
                and required_facet in _coverage_facets(item["metadata"], hints)
            ),
            None,
        )
        if replacement is None:
            continue

        replace_index = _replacement_index_for_facet(selected, hints)
        removed = selected[replace_index]
        selected_identities.discard(removed["identity"])
        selection_status[removed["raw_rank"]] = {
            "selected": False,
            "exclusion_reason": "facet_coverage_replacement",
            "exclusion_pass": "facet_coverage_pass",
        }

        selected[replace_index] = replacement
        selected_identities.add(replacement["identity"])
        selection_status[replacement["raw_rank"]] = {
            "selected": True,
            "selection_pass": "facet_coverage_pass",
        }


def _required_coverage_facets(hints: QueryHints) -> list[str]:
    facets: list[str] = []
    if hints.wants_gene:
        facets.append("gene")
    if hints.wants_resistance_or_response:
        facets.append("resistance_or_response")
    if hints.wants_cancer:
        facets.append("cancer")
    return facets


def _coverage_facets(metadata: dict[str, Any], hints: QueryHints) -> set[str]:
    components = dict(metadata.get("ranking_components", {}) or {})
    facets: set[str] = set()
    if hints.wants_gene and components.get("gene_endpoint", 0.0) > 0:
        facets.add("gene")
    if hints.wants_resistance_or_response and (
        components.get("predicate", 0.0) > 0
        or components.get("resistance_response_context", 0.0) > 0
    ):
        facets.add("resistance_or_response")
    if hints.wants_cancer and components.get("cancer_context", 0.0) > 0:
        facets.add("cancer")
    return facets


def _coverage_counts(items: list[dict[str, Any]], hints: QueryHints) -> dict[str, int]:
    counts: dict[str, int] = {}
    for item in items:
        for facet in _coverage_facets(item["metadata"], hints):
            counts[facet] = counts.get(facet, 0) + 1
    return counts


def _replacement_index_for_facet(
    selected: list[dict[str, Any]],
    hints: QueryHints,
) -> int:
    coverage_counts = _coverage_counts(selected, hints)
    for index in range(len(selected) - 1, -1, -1):
        item_facets = _coverage_facets(selected[index]["metadata"], hints)
        if all(coverage_counts.get(facet, 0) > 1 for facet in item_facets):
            return index
    return len(selected) - 1


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
    return bool(_gene_compatible_tier(_endpoint_category_tier(labels)))


def _endpoint_category_tier(labels: Iterable[str]) -> str | None:
    label_set = {str(label) for label in labels}
    if label_set & GENE_CATEGORY_LABELS:
        return "gene"
    if label_set & GENE_PRODUCT_CATEGORY_LABELS:
        return "gene product"
    if label_set & GENOMIC_VARIANT_CATEGORY_LABELS:
        return "genomic variant"
    if label_set & CHEMICAL_CATEGORY_LABELS:
        return "chemical"
    if label_set & BIOLOGICAL_PROCESS_CATEGORY_LABELS:
        return "biological process"
    return None


def _gene_compatible_tier(tier: str | None) -> str | None:
    if tier in {"gene", "gene product", "genomic variant"}:
        return tier
    return None


def _best_gene_compatible_tier(*tiers: str | None) -> str | None:
    priority = {"gene": 0, "gene product": 1, "genomic variant": 2}
    compatible = [tier for tier in tiers if tier in priority]
    if not compatible:
        return None
    return min(compatible, key=lambda tier: priority[tier])


def _has_gene_lexical_fallback(metadata: dict[str, Any]) -> bool:
    for endpoint in ("subject", "object"):
        endpoint_id = _endpoint_value(metadata, endpoint).casefold()
        if endpoint_id.startswith(GENE_LIKE_ID_PREFIXES):
            return True
        endpoint_type = str(metadata.get(f"llm_{endpoint}_type") or "").casefold()
        if endpoint_type in {"gene", "protein"}:
            return True
    return False


def _has_drug_response_context(metadata: dict[str, Any]) -> bool:
    subject_tier = _endpoint_category_tier(_labels(metadata, "subject"))
    object_tier = _endpoint_category_tier(_labels(metadata, "object"))
    has_drug_like_endpoint = "chemical" in {subject_tier, object_tier}
    text = _edge_specific_text(metadata)
    tokens = set(_tokens(text))
    has_response_text = bool(tokens & DRUG_CONTEXT_TERMS)
    has_therapeutic_response_text = (
        bool(tokens & THERAPEUTIC_CONTEXT_TERMS)
        and _has_resistance_response_context_tokens(tokens)
    )
    return (has_drug_like_endpoint and has_response_text) or has_therapeutic_response_text


def _edge_specific_text(metadata: dict[str, Any]) -> str:
    keys = (
        "llm_subject",
        "llm_subject_qualifier",
        "llm_relationship",
        "llm_object",
        "llm_object_qualifier",
        "llm_statement_qualifier",
        "llm_subject_type",
        "llm_object_type",
        "original_subject",
        "original_object",
    )
    return " ".join(str(metadata.get(key) or "") for key in keys)


def _compact_context_text(metadata: dict[str, Any], page_content: Any) -> str:
    keys = (
        "abstract_title",
        "llm_statement",
        "llm_relationship",
        "llm_subject_qualifier",
        "llm_object_qualifier",
        "llm_statement_qualifier",
        "semantic_text",
    )
    return " ".join(
        str(part)
        for part in (
            page_content,
            *(metadata.get(key) for key in keys),
        )
        if part
    )


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
