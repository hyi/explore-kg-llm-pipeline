from __future__ import annotations

import copy
import re
from dataclasses import asdict, dataclass
from typing import Any

from explorer.backend.semantic_search.query_intent import (
    CATEGORY_BIOLOGICAL_PROCESS,
    CATEGORY_DISEASE,
    CATEGORY_DRUG_OR_CHEMICAL,
    CATEGORY_GENE,
    CATEGORY_GENE_PRODUCT,
    CATEGORY_PHENOTYPE,
    CATEGORY_SEQUENCE_VARIANT,
    PREDICATE_DRUG_RESPONSE,
    PREDICATE_UNKNOWN,
    QUALITY_BROAD_ASSOCIATION,
    QUALITY_CONTEXTUAL_MENTION,
    QUALITY_DIRECT_ASSERTION,
    QueryIntent,
    category_families_from_identifier,
    category_families_from_labels,
    is_directional_predicate_family,
    is_symmetric_predicate_family,
    parse_query_intent,
    predicate_family,
    relationship_quality_tier,
)

ANCHOR_RANKING_STRATEGY = "graph_intent_compatibility_v2"

# Initial transparent heuristics. These weights are intentionally simple and
# should be evaluated against LitCoin retrieval examples before being treated
# as empirically validated.
SEMANTIC_SCORE_WEIGHT = 1.0
MAX_COMPATIBILITY_ADJUSTMENT = 0.25
ENDPOINT_CATEGORY_MATCH = 0.08
ENDPOINT_CATEGORY_PARTIAL_MATCH = 0.04
PREDICATE_FAMILY_MATCH = 0.08
PREDICATE_FAMILY_PARTIAL_MATCH = 0.04
ROLE_MATCH = 0.04
ROLE_REVERSED_COMPATIBLE = 0.02
CONJUNCTIVE_MATCH_BONUS = 0.05
INCOMPATIBILITY_PENALTY = -0.20
DIRECT_ASSERTION_QUALITY_ADJUSTMENT = 0.03
BROAD_ASSOCIATION_QUALITY_ADJUSTMENT = 0.0
CONTEXTUAL_MENTION_QUALITY_ADJUSTMENT = -0.08
UNKNOWN_QUALITY_ADJUSTMENT = 0.0
STRUCTURAL_COMPATIBILITY_TIER_ORDER = {
    "complete_match": 0,
    "partial_match": 1,
    "retrieval_only": 2,
    "contextual_mention": 3,
    "no_structural_intent": 0,
}

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
    [
        "drug",
        "treatment",
        "therapy",
        "therapeutic",
        "chemotherapy",
        "chemotherapeutic",
        "chemoresistance",
        "resistance",
        "resistant",
        "sensitivity",
        "sensitive",
        "response",
    ]
)
RESISTANCE_RESPONSE_CONTEXT_TERMS = frozenset(
    {"chemoresistance", "chemoresistant", "resistance", "resistant", "sensitivity", "sensitive"}
)
GENERIC_RESPONSE_TERMS = frozenset({"response", "responsive"})
THERAPEUTIC_CONTEXT_TERMS = frozenset(
    {"drug", "drugs", "treatment", "treatments", "therapy", "therapies", "therapeutic", "chemotherapy", "chemotherapeutic"}
)
DRUG_RESPONSE_EVIDENCE_TERMS = frozenset(
    {
        "chemoresistance",
        "chemoresistant",
        "efficacy",
        "resistance",
        "resistant",
        "response",
        "responsive",
        "sensitivity",
        "sensitive",
    }
) | THERAPEUTIC_CONTEXT_TERMS
DRUG_RESPONSE_GENETIC_CATEGORIES = frozenset({CATEGORY_GENE, CATEGORY_GENE_PRODUCT, CATEGORY_SEQUENCE_VARIANT})
DRUG_RESPONSE_RESPONSE_BEARING_CATEGORIES = frozenset({CATEGORY_PHENOTYPE, CATEGORY_BIOLOGICAL_PROCESS})
DRUG_RESPONSE_DISEASE_CONTEXT_CATEGORIES = frozenset({CATEGORY_DISEASE})
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
class AnchorRankingConfig:
    candidate_multiplier: int = 5
    minimum_candidate_pool: int = 25
    maximum_candidate_pool: int = 100
    max_per_publication: int = 2
    compatibility_profile: str = "generic_graph_intent_v1"

    def to_cache_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class AnchorRankingResult:
    relationships: list[Any]
    diagnostics: dict[str, Any]


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
    intent = parse_query_intent(query)
    if requested_k <= 0 or not candidates:
        return AnchorRankingResult(
            relationships=[],
            diagnostics={
                "ranking_strategy": ANCHOR_RANKING_STRATEGY,
                "query_intent": intent.to_dict(),
                "requested_k": requested_k,
                "candidate_count": len(candidates),
                "ranked_candidates": [],
                "selected_anchors": [],
            },
        )

    scored = [
        _score_candidate(
            query=query,
            candidate=candidate,
            raw_rank=rank,
            candidate_count=len(candidates),
            intent=intent,
        )
        for rank, candidate in enumerate(candidates)
    ]
    scored.sort(key=_rank_sort_key)
    for rerank_rank, item in enumerate(scored):
        item["rerank_rank"] = rerank_rank

    selected, selection_status = _diversified_selection_with_diagnostics(
        scored,
        requested_k=requested_k,
        config=config,
        intent=intent,
    )
    ranked_candidates = [
        _ranking_candidate_diagnostic(
            item,
            intent=intent,
            selection_status=selection_status.get(item["raw_rank"], {}),
        )
        for item in scored
    ]
    selected_anchors = [
        _ranking_candidate_diagnostic(
            item,
            intent=intent,
            selection_status=selection_status.get(item["raw_rank"], {}),
        )
        for item in selected
    ]
    return AnchorRankingResult(
        relationships=[item["candidate"] for item in selected],
        diagnostics={
            "ranking_strategy": ANCHOR_RANKING_STRATEGY,
            "query_intent": intent.to_dict(),
            "requested_k": requested_k,
            "candidate_count": len(candidates),
            "ranked_candidates": ranked_candidates,
            "selected_anchors": selected_anchors,
        },
    )


def _ranking_candidate_diagnostic(
    item: dict[str, Any],
    intent: QueryIntent,
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
        "query_intent": intent.to_dict(),
        "recognized_query_expressions": [item.to_dict() for item in intent.recognized_expressions],
        "unrecognized_query_terms": list(intent.unrecognized_terms),
        "content_terms": list(intent.content_terms),
        "compatibility_components": dict(metadata.get("compatibility_components", {}) or {}),
        "endpoint_category_compatibility": dict(metadata.get("endpoint_category_compatibility", {}) or {}),
        "predicate_family_compatibility": dict(metadata.get("predicate_family_compatibility", {}) or {}),
        "role_compatibility": dict(metadata.get("role_compatibility", {}) or {}),
        "relationship_quality_tier": metadata.get("relationship_quality_tier"),
        "conjunctive_compatibility": metadata.get("conjunctive_compatibility"),
        "incompatibility_penalties": list(metadata.get("incompatibility_penalties", []) or []),
        "claim_term_matches": list(metadata.get("claim_term_matches", []) or []),
        "structural_compatibility_status": metadata.get("structural_compatibility_status"),
        "fallback_status": metadata.get("fallback_status"),
        "compatibility_tier_rank": metadata.get("compatibility_tier_rank"),
        "is_semantic_fallback": bool(metadata.get("is_semantic_fallback", False)),
        "predicate": metadata.get("predicate"),
        "predicate_family": metadata.get("predicate_family"),
        "endpoint_categories": {
            "subject": {
                "labels": subject_labels,
                "families": sorted(_endpoint_categories(metadata, "subject")),
            },
            "object": {
                "labels": object_labels,
                "families": sorted(_endpoint_categories(metadata, "object")),
            },
        },
        "publication_id": item["publication_id"],
        "retrieval_score": metadata.get("retrieval_score"),
        "fusion_score": metadata.get("fusion_score"),
        "retrieval_channels": metadata.get("retrieval_channels"),
        "dense_rank": metadata.get("dense_rank"),
        "dense_score": metadata.get("dense_score"),
        "keyword_rank": metadata.get("keyword_rank"),
        "keyword_score": metadata.get("keyword_score"),
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
        "diversity_pass": selection_status.get("selection_pass") == "primary_diversity_pass",
        "fallback_pass": selection_status.get("selection_pass") in {
            "primary_publication_fallback_pass",
            "primary_relationship_fallback_pass",
            "contextual_mention_fallback_pass",
            "contextual_mention_relaxed_fallback_pass",
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
        "retrieval_score",
        "fusion_score",
        "retrieval_channels",
        "dense_rank",
        "dense_score",
        "keyword_rank",
        "keyword_score",
        "ranking_reasons",
        "ranking_components",
        "compatibility_components",
        "endpoint_category_compatibility",
        "predicate_family_compatibility",
        "role_compatibility",
        "relationship_quality_tier",
        "predicate_family",
        "conjunctive_compatibility",
        "incompatibility_penalties",
        "claim_term_matches",
        "structural_compatibility_status",
        "fallback_status",
        "compatibility_tier_rank",
        "is_semantic_fallback",
        "publication_id",
        "relationship_identity",
        "raw_rank",
        "predicate",
        "subject_labels",
        "object_labels",
    )
    return {key: metadata[key] for key in keys if key in metadata}


def _score_candidate(
    query: str,
    candidate: Any,
    raw_rank: int,
    candidate_count: int,
    intent: QueryIntent,
) -> dict[str, Any]:
    raw_metadata = dict(getattr(candidate, "metadata", {}) or {})
    semantic_score = float(raw_metadata.get("semantic_score", raw_metadata.get("dense_score", raw_metadata.get("score", 0.0))) or 0.0)
    retrieval_score = _normalized_retrieval_score(raw_metadata, raw_rank=raw_rank, candidate_count=candidate_count)
    compatibility = _candidate_compatibility(
        metadata=raw_metadata,
        intent=intent,
        page_content=getattr(candidate, "page_content", ""),
    )
    quality_tier = relationship_quality_tier(raw_metadata.get("predicate"))
    quality_adjustment = _relationship_quality_adjustment(quality_tier)
    positive_compatibility = min(
        MAX_COMPATIBILITY_ADJUSTMENT,
        compatibility["positive_compatibility"],
    )
    incompatibility_penalty = sum(compatibility["incompatibility_penalties"])
    graph_adjustment = positive_compatibility + incompatibility_penalty + quality_adjustment
    anchor_score = retrieval_score + graph_adjustment

    components: dict[str, float] = {
        "retrieval": retrieval_score,
        "positive_compatibility_capped": positive_compatibility,
        "incompatibility_penalty": incompatibility_penalty,
        "relationship_quality": quality_adjustment,
        "graph_compatibility_adjustment": graph_adjustment,
    }
    reasons = [f"retrieval score {retrieval_score:.4f}"]
    reasons.extend(compatibility["reasons"])
    reasons.append(f"relationship quality: {quality_tier}")

    metadata = dict(raw_metadata)
    identity = relationship_identity(metadata, fallback=f"raw-rank:{raw_rank}")
    publication_id = publication_identity(metadata)
    fallback_status = _fallback_status(intent, compatibility, quality_tier)
    compatibility_tier_rank = _compatibility_tier_rank(fallback_status, structural_intent=intent.has_structural_intent)
    metadata.update(
        {
            "semantic_score": semantic_score,
            "retrieval_score": retrieval_score,
            "anchor_score": anchor_score,
            "score": anchor_score,
            "ranking_reasons": reasons,
            "ranking_components": components,
            "compatibility_components": compatibility["components"],
            "endpoint_category_compatibility": compatibility["endpoint_category_compatibility"],
            "predicate_family_compatibility": compatibility["predicate_family_compatibility"],
            "role_compatibility": compatibility["role_compatibility"],
            "relationship_quality_tier": quality_tier,
            "predicate_family": predicate_family(metadata.get("predicate")),
            "conjunctive_compatibility": compatibility["conjunctive_compatibility"],
            "incompatibility_penalties": compatibility["incompatibility_penalty_reasons"],
            "claim_term_matches": _claim_term_matches(intent, metadata, getattr(candidate, "page_content", "")),
            "structural_compatibility_status": compatibility["status"],
            "fallback_status": fallback_status,
            "compatibility_tier_rank": compatibility_tier_rank,
            "is_semantic_fallback": fallback_status in {"partial_match", "retrieval_only", "contextual_mention"},
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


def _normalized_retrieval_score(metadata: dict[str, Any], *, raw_rank: int, candidate_count: int) -> float:
    if "retrieval_score" in metadata:
        return _clamp01(float(metadata.get("retrieval_score", 0.0) or 0.0))
    return _clamp01(float(metadata.get("semantic_score", metadata.get("score", 0.0)) or 0.0))


def _candidate_compatibility(metadata: dict[str, Any], intent: QueryIntent, page_content: Any) -> dict[str, Any]:
    subject_categories = _endpoint_categories(metadata, "subject")
    object_categories = _endpoint_categories(metadata, "object")
    candidate_predicate_family = predicate_family(metadata.get("predicate"))
    endpoint_result = _endpoint_category_compatibility(intent, subject_categories, object_categories)
    predicate_result = _predicate_family_compatibility(
        intent,
        candidate_predicate_family,
        metadata=metadata,
        subject_categories=subject_categories,
        object_categories=object_categories,
    )
    role_result = _role_compatibility(intent, subject_categories, object_categories, candidate_predicate_family)
    conjunctive_score = _conjunctive_compatibility(endpoint_result, predicate_result, role_result, intent)
    penalties = _incompatibility_penalties(endpoint_result, role_result)
    positive = endpoint_result["score"] + predicate_result["score"] + role_result["score"] + conjunctive_score
    status = _structural_compatibility_status(intent, endpoint_result, predicate_result, role_result)
    reasons = []
    for result in (endpoint_result, predicate_result, role_result):
        if result["reason"]:
            reasons.append(result["reason"])
    if conjunctive_score:
        reasons.append("candidate-level conjunctive compatibility")
    reasons.extend(reason for _value, reason in penalties)
    return {
        "positive_compatibility": positive,
        "components": {
            "endpoint_category": endpoint_result["score"],
            "predicate_family": predicate_result["score"],
            "role": role_result["score"],
            "conjunctive": conjunctive_score,
        },
        "endpoint_category_compatibility": endpoint_result,
        "predicate_family_compatibility": predicate_result,
        "role_compatibility": role_result,
        "conjunctive_compatibility": conjunctive_score,
        "incompatibility_penalties": [value for value, _reason in penalties],
        "incompatibility_penalty_reasons": [reason for _value, reason in penalties],
        "status": status,
        "claim_term_matches": _claim_term_matches(intent, metadata, page_content),
        "reasons": reasons,
    }


def _endpoint_category_compatibility(
    intent: QueryIntent,
    subject_categories: set[str],
    object_categories: set[str],
) -> dict[str, Any]:
    requested = _all_requested_category_families(intent)
    if not requested:
        return {"status": "not_requested", "score": 0.0, "matched": [], "missing": [], "reason": None}
    endpoint_categories = subject_categories | object_categories
    matched = sorted(request for request in requested if _category_matches(request, endpoint_categories))
    partial = sorted(request for request in requested if request not in matched and _category_partially_matches(request, endpoint_categories))
    missing = sorted(request for request in requested if request not in matched and request not in partial)
    if matched:
        return {
            "status": "match",
            "score": ENDPOINT_CATEGORY_MATCH,
            "matched": matched,
            "partial": partial,
            "missing": missing,
            "reason": f"endpoint category match: {', '.join(matched)}",
        }
    if partial:
        return {
            "status": "partial",
            "score": ENDPOINT_CATEGORY_PARTIAL_MATCH,
            "matched": [],
            "partial": partial,
            "missing": missing,
            "reason": f"partial endpoint category match: {', '.join(partial)}",
        }
    return {
        "status": "missing",
        "score": 0.0,
        "matched": [],
        "partial": [],
        "missing": missing,
        "reason": "no requested endpoint category match",
    }


def _predicate_family_compatibility(
    intent: QueryIntent,
    candidate_family: str,
    *,
    metadata: dict[str, Any],
    subject_categories: set[str],
    object_categories: set[str],
) -> dict[str, Any]:
    requested = sorted({request.family for request in intent.requested_predicate_families})
    if not requested:
        return {
            "status": "not_requested",
            "score": 0.0,
            "requested": [],
            "candidate": candidate_family,
            "reason": None,
        }
    if candidate_family in requested:
        if candidate_family == PREDICATE_DRUG_RESPONSE:
            return _drug_response_predicate_compatibility(
                requested=requested,
                metadata=metadata,
                subject_categories=subject_categories,
                object_categories=object_categories,
            )
        return {
            "status": "match",
            "score": PREDICATE_FAMILY_MATCH,
            "requested": requested,
            "candidate": candidate_family,
            "reason": f"predicate family match: {candidate_family}",
        }
    if candidate_family == PREDICATE_UNKNOWN:
        status = "unknown"
        reason = "predicate family unknown"
    else:
        status = "mismatch"
        reason = f"predicate family mismatch: {candidate_family}"
    return {
        "status": status,
        "score": 0.0,
        "requested": requested,
        "candidate": candidate_family,
        "reason": reason,
    }


def _drug_response_predicate_compatibility(
    *,
    requested: list[str],
    metadata: dict[str, Any],
    subject_categories: set[str],
    object_categories: set[str],
) -> dict[str, Any]:
    support = _drug_response_argument_support(metadata, subject_categories, object_categories)
    classification = support["classification"]
    if classification == "validated":
        status = "match"
        score = PREDICATE_FAMILY_MATCH
        reason = "predicate family match: drug_response with argument support"
    elif classification == "partial":
        status = "partial"
        score = PREDICATE_FAMILY_PARTIAL_MATCH
        reason = "partial predicate family match: drug_response lacks complete argument support"
    elif classification == "incompatible":
        status = "mismatch"
        score = 0.0
        reason = "predicate family mismatch: drug_response argument context incompatible"
    else:
        status = "unknown"
        score = 0.0
        reason = "predicate family unknown: drug_response argument context unavailable"
    return {
        "status": status,
        "score": score,
        "requested": requested,
        "candidate": PREDICATE_DRUG_RESPONSE,
        "reason": reason,
        "drug_response_argument_compatibility": support,
    }


def _drug_response_argument_support(
    metadata: dict[str, Any],
    subject_categories: set[str],
    object_categories: set[str],
) -> dict[str, Any]:
    endpoint_support = _drug_response_endpoint_role_support(subject_categories, object_categories)
    original_relationship_support = _term_support(
        _metadata_text(metadata, ("llm_relationship", "original_relationship", "relationship")),
        DRUG_RESPONSE_EVIDENCE_TERMS,
    )
    qualifier_support = _term_support(
        _metadata_text(
            metadata,
            (
                "llm_subject_qualifier",
                "llm_object_qualifier",
                "llm_statement_qualifier",
                "subject_qualifier",
                "object_qualifier",
                "statement_qualifier",
                "qualifiers",
            ),
        ),
        DRUG_RESPONSE_EVIDENCE_TERMS,
    )
    has_claim_evidence = bool(original_relationship_support["matched_terms"] or qualifier_support["matched_terms"])

    if (
        endpoint_support["status"] == "match"
        or qualifier_support["matched_terms"]
        or (original_relationship_support["matched_terms"] and endpoint_support["status"] == "partial")
    ):
        classification = "validated"
    elif endpoint_support["status"] == "incompatible" and not has_claim_evidence:
        classification = "incompatible"
    elif endpoint_support["status"] == "unknown" and not has_claim_evidence:
        classification = "unknown"
    else:
        classification = "partial"

    return {
        "classification": classification,
        "predicate_family_support": "normalized_biolink_drug_response_predicate",
        "original_relationship_support": original_relationship_support,
        "endpoint_role_support": endpoint_support,
        "qualifier_support": qualifier_support,
        "final_reason": _drug_response_final_reason(classification, endpoint_support, original_relationship_support, qualifier_support),
    }


def _drug_response_endpoint_role_support(
    subject_categories: set[str],
    object_categories: set[str],
) -> dict[str, Any]:
    if not subject_categories and not object_categories:
        return {
            "status": "unknown",
            "reason": "endpoint categories unavailable",
            "subject_categories": [],
            "object_categories": [],
        }

    subject_genetic = bool(subject_categories & DRUG_RESPONSE_GENETIC_CATEGORIES)
    object_genetic = bool(object_categories & DRUG_RESPONSE_GENETIC_CATEGORIES)
    subject_drug = CATEGORY_DRUG_OR_CHEMICAL in subject_categories
    object_drug = CATEGORY_DRUG_OR_CHEMICAL in object_categories
    subject_response_bearing = bool(subject_categories & DRUG_RESPONSE_RESPONSE_BEARING_CATEGORIES)
    object_response_bearing = bool(object_categories & DRUG_RESPONSE_RESPONSE_BEARING_CATEGORIES)
    subject_disease = bool(subject_categories & DRUG_RESPONSE_DISEASE_CONTEXT_CATEGORIES)
    object_disease = bool(object_categories & DRUG_RESPONSE_DISEASE_CONTEXT_CATEGORIES)

    if (subject_genetic and object_drug) or (subject_drug and object_genetic):
        status = "match"
        reason = "gene/variant endpoint paired with drug/chemical endpoint"
    elif (subject_drug and object_response_bearing) or (object_drug and subject_response_bearing):
        status = "match"
        reason = "drug/chemical endpoint paired with response-bearing phenotype or process"
    elif (subject_genetic and (object_response_bearing or object_disease)) or (
        object_genetic and (subject_response_bearing or subject_disease)
    ):
        status = "partial"
        reason = "gene/variant endpoint paired with disease or phenotype without drug/treatment endpoint"
    elif (subject_drug and object_disease) or (object_drug and subject_disease):
        status = "partial"
        reason = "drug/chemical endpoint paired with disease without explicit response-bearing context"
    elif subject_categories or object_categories:
        status = "incompatible"
        reason = "endpoints do not match documented drug-response argument patterns"
    else:
        status = "unknown"
        reason = "endpoint categories unavailable"

    return {
        "status": status,
        "reason": reason,
        "subject_categories": sorted(subject_categories),
        "object_categories": sorted(object_categories),
    }


def _drug_response_final_reason(
    classification: str,
    endpoint_support: dict[str, Any],
    original_relationship_support: dict[str, Any],
    qualifier_support: dict[str, Any],
) -> str:
    if classification == "validated":
        if endpoint_support["status"] == "match":
            return endpoint_support["reason"]
        if qualifier_support["matched_terms"]:
            return "claim qualifier supplies explicit treatment/response evidence"
        return "original relationship supplies explicit treatment/response evidence"
    if classification == "partial":
        return endpoint_support["reason"]
    if classification == "incompatible":
        return endpoint_support["reason"]
    if original_relationship_support["matched_terms"]:
        return "original relationship has response terms but endpoint context is unavailable"
    return "drug-response predicate label has insufficient argument evidence"


def _role_compatibility(
    intent: QueryIntent,
    subject_categories: set[str],
    object_categories: set[str],
    candidate_predicate_family: str,
) -> dict[str, Any]:
    subject_requests = {request.family for request in intent.requested_subject_categories}
    object_requests = {request.family for request in intent.requested_object_categories}
    if not subject_requests and not object_requests:
        return {"status": "not_requested", "score": 0.0, "reason": None, "matches": []}

    matches = []
    reversed_matches = []
    contradictions = []
    for family in sorted(subject_requests):
        if _category_matches(family, subject_categories):
            matches.append(f"subject:{family}")
        elif _category_matches(family, object_categories):
            reversed_matches.append(f"subject:{family}")
    for family in sorted(object_requests):
        if _category_matches(family, object_categories):
            matches.append(f"object:{family}")
        elif _category_matches(family, subject_categories):
            reversed_matches.append(f"object:{family}")

    if matches:
        return {
            "status": "match",
            "score": ROLE_MATCH,
            "reason": f"subject/object role match: {', '.join(matches)}",
            "matches": matches,
            "reversed_matches": reversed_matches,
            "contradictions": [],
        }
    if reversed_matches and is_symmetric_predicate_family(candidate_predicate_family):
        return {
            "status": "symmetric_compatible",
            "score": ROLE_REVERSED_COMPATIBLE,
            "reason": f"symmetric role-compatible match: {', '.join(reversed_matches)}",
            "matches": [],
            "reversed_matches": reversed_matches,
            "contradictions": [],
        }
    if reversed_matches and is_directional_predicate_family(candidate_predicate_family):
        contradictions = reversed_matches
        return {
            "status": "contradiction",
            "score": 0.0,
            "reason": f"clear subject/object role contradiction: {', '.join(contradictions)}",
            "matches": [],
            "reversed_matches": reversed_matches,
            "contradictions": contradictions,
        }
    if reversed_matches:
        return {
            "status": "unknown_direction",
            "score": 0.0,
            "reason": f"role direction unknown for reversed category: {', '.join(reversed_matches)}",
            "matches": [],
            "reversed_matches": reversed_matches,
            "contradictions": [],
        }
    return {
        "status": "unknown",
        "score": 0.0,
        "reason": "requested subject/object role not observed",
        "matches": [],
        "reversed_matches": [],
        "contradictions": [],
    }


def _conjunctive_compatibility(
    endpoint_result: dict[str, Any],
    predicate_result: dict[str, Any],
    role_result: dict[str, Any],
    intent: QueryIntent,
) -> float:
    if not intent.has_structural_intent:
        return 0.0
    matched_dimensions = 0
    if endpoint_result["status"] in {"match", "partial"}:
        matched_dimensions += 1
    if predicate_result["status"] == "match":
        matched_dimensions += 1
    if role_result["status"] in {"match", "symmetric_compatible"}:
        matched_dimensions += 1
    return CONJUNCTIVE_MATCH_BONUS if matched_dimensions >= 2 else 0.0


def _incompatibility_penalties(
    endpoint_result: dict[str, Any],
    role_result: dict[str, Any],
) -> list[tuple[float, str]]:
    penalties = []
    if endpoint_result["status"] == "missing":
        penalties.append((INCOMPATIBILITY_PENALTY / 2, "requested endpoint category missing"))
    if role_result["status"] == "contradiction":
        penalties.append((INCOMPATIBILITY_PENALTY, role_result["reason"]))
    return penalties


def _structural_compatibility_status(
    intent: QueryIntent,
    endpoint_result: dict[str, Any],
    predicate_result: dict[str, Any],
    role_result: dict[str, Any],
) -> str:
    if not intent.has_structural_intent:
        return "no_structural_intent"
    requested_dimensions: list[tuple[bool, bool]] = []
    if _all_requested_category_families(intent):
        requested_dimensions.append(
            (
                endpoint_result["status"] == "match",
                endpoint_result["status"] in {"match", "partial"},
            )
        )
    if intent.requested_predicate_families:
        requested_dimensions.append(
            (
                predicate_result["status"] == "match",
                predicate_result["status"] in {"match", "partial"},
            )
        )
    if intent.requested_subject_categories or intent.requested_object_categories:
        requested_dimensions.append(
            (
                role_result["status"] in {"match", "symmetric_compatible"},
                role_result["status"] in {"match", "symmetric_compatible"},
            )
        )
    if requested_dimensions and all(complete for complete, _partial in requested_dimensions):
        return "complete_match"
    if any(partial for _complete, partial in requested_dimensions):
        return "partial_match"
    return "retrieval_only"


def _fallback_status(intent: QueryIntent, compatibility: dict[str, Any], quality_tier: str) -> str:
    if quality_tier == QUALITY_CONTEXTUAL_MENTION:
        return "contextual_mention"
    if not intent.has_structural_intent:
        return "no_structural_intent"
    return compatibility["status"]


def _compatibility_tier_rank(status: str, *, structural_intent: bool) -> int:
    if not structural_intent:
        return 0
    return STRUCTURAL_COMPATIBILITY_TIER_ORDER.get(status, STRUCTURAL_COMPATIBILITY_TIER_ORDER["retrieval_only"])


def _relationship_quality_adjustment(quality_tier: str) -> float:
    if quality_tier == QUALITY_DIRECT_ASSERTION:
        return DIRECT_ASSERTION_QUALITY_ADJUSTMENT
    if quality_tier == QUALITY_BROAD_ASSOCIATION:
        return BROAD_ASSOCIATION_QUALITY_ADJUSTMENT
    if quality_tier == QUALITY_CONTEXTUAL_MENTION:
        return CONTEXTUAL_MENTION_QUALITY_ADJUSTMENT
    return UNKNOWN_QUALITY_ADJUSTMENT


def _claim_term_matches(intent: QueryIntent, metadata: dict[str, Any], page_content: Any) -> list[str]:
    claim_tokens = set(_tokens(_edge_specific_text(metadata) + " " + str(page_content or "")))
    return sorted(set(intent.content_terms) & claim_tokens)


def _all_requested_category_families(intent: QueryIntent) -> set[str]:
    return {
        request.family
        for request in (
            *intent.requested_subject_categories,
            *intent.requested_object_categories,
            *intent.requested_endpoint_categories,
        )
    }


def _category_matches(requested_family: str, observed_families: set[str]) -> bool:
    return requested_family in observed_families


def _category_partially_matches(requested_family: str, observed_families: set[str]) -> bool:
    if requested_family == CATEGORY_GENE:
        return bool(observed_families & {CATEGORY_GENE_PRODUCT})
    if requested_family == CATEGORY_GENE_PRODUCT:
        return bool(observed_families & {CATEGORY_GENE})
    if requested_family == CATEGORY_DRUG_OR_CHEMICAL:
        return CATEGORY_DRUG_OR_CHEMICAL in observed_families
    if requested_family == CATEGORY_SEQUENCE_VARIANT:
        return False
    return False


def _endpoint_categories(metadata: dict[str, Any], endpoint: str) -> set[str]:
    labels = _labels(metadata, endpoint)
    families = category_families_from_labels(labels)
    if families:
        return families
    return category_families_from_identifier(_endpoint_value(metadata, endpoint))


def _clamp01(value: float) -> float:
    return max(0.0, min(1.0, value))


def _diversified_selection(
    ranked_items: list[dict[str, Any]],
    requested_k: int,
    config: AnchorRankingConfig,
) -> list[dict[str, Any]]:
    selected, _selection_status = _diversified_selection_with_diagnostics(
        ranked_items,
        requested_k=requested_k,
        config=config,
        intent=None,
    )
    return selected


def _diversified_selection_with_diagnostics(
    ranked_items: list[dict[str, Any]],
    requested_k: int,
    config: AnchorRankingConfig,
    intent: QueryIntent | None = None,
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

    primary_items = [
        item
        for item in ranked_items
        if item["metadata"].get("relationship_quality_tier") != QUALITY_CONTEXTUAL_MENTION
    ]
    contextual_items = [
        item
        for item in ranked_items
        if item["metadata"].get("relationship_quality_tier") == QUALITY_CONTEXTUAL_MENTION
    ]

    for item in contextual_items:
        record_exclusion(item, "contextual_mention_suppressed_in_primary_pass", "primary_quality_pass")

    for item in primary_items:
        try_add(
            item,
            enforce_publication=True,
            enforce_neighborhood=True,
            pass_name="primary_diversity_pass",
        )
    for item in primary_items:
        try_add(
            item,
            enforce_publication=True,
            enforce_neighborhood=False,
            pass_name="primary_publication_fallback_pass",
        )
    for item in primary_items:
        try_add(
            item,
            enforce_publication=False,
            enforce_neighborhood=False,
            pass_name="primary_relationship_fallback_pass",
        )
    for item in contextual_items:
        try_add(
            item,
            enforce_publication=True,
            enforce_neighborhood=True,
            pass_name="contextual_mention_fallback_pass",
        )
    for item in contextual_items:
        try_add(
            item,
            enforce_publication=False,
            enforce_neighborhood=False,
            pass_name="contextual_mention_relaxed_fallback_pass",
        )

    return selected, selection_status


def _rank_sort_key(item: dict[str, Any]) -> tuple[int, float, int, str, str, str, str]:
    metadata = item["metadata"]
    subject, obj = item["neighborhood"]
    return (
        int(metadata.get("compatibility_tier_rank", 0)),
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


def _metadata_text(metadata: dict[str, Any], keys: tuple[str, ...]) -> str:
    return " ".join(_stringify_metadata_value(metadata.get(key)) for key in keys)


def _stringify_metadata_value(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, dict):
        return " ".join(
            f"{_stringify_metadata_value(key)} {_stringify_metadata_value(item)}"
            for key, item in value.items()
        )
    if isinstance(value, (list, tuple, set)):
        return " ".join(_stringify_metadata_value(item) for item in value)
    return str(value)


def _term_support(text: str, terms: frozenset[str]) -> dict[str, Any]:
    tokens = set(_tokens(text))
    matched = sorted(term for term in terms if set(_tokens(term)).issubset(tokens))
    return {
        "matched_terms": matched,
        "text_present": bool(str(text or "").strip()),
    }


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
