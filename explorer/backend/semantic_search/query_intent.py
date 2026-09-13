from __future__ import annotations

import re
from dataclasses import asdict, dataclass
from typing import Any

CATEGORY_GENE = "gene"
CATEGORY_GENE_PRODUCT = "gene_product"
CATEGORY_SEQUENCE_VARIANT = "sequence_variant"
CATEGORY_DRUG_OR_CHEMICAL = "drug_or_chemical"
CATEGORY_DISEASE = "disease"
CATEGORY_PHENOTYPE = "phenotype"
CATEGORY_BIOLOGICAL_PROCESS = "biological_process"
CATEGORY_PATHWAY = "pathway"
CATEGORY_UNKNOWN = "unknown"

PREDICATE_ASSOCIATION = "association"
PREDICATE_TREATMENT = "treatment"
PREDICATE_CAUSAL = "causal"
PREDICATE_REGULATION = "regulation"
PREDICATE_PARTICIPATION = "participation"
PREDICATE_DRUG_RESPONSE = "drug_response"
PREDICATE_EXPRESSION = "expression"
PREDICATE_PHYSICAL_INTERACTION = "physical_interaction"
PREDICATE_RELATED = "related"
PREDICATE_UNKNOWN = "unknown"

QUALITY_DIRECT_ASSERTION = "direct_assertion"
QUALITY_BROAD_ASSOCIATION = "broad_association"
QUALITY_CONTEXTUAL_MENTION = "contextual_mention"
QUALITY_UNKNOWN = "unknown"

ROLE_SUBJECT = "subject"
ROLE_OBJECT = "object"
ROLE_ENDPOINT = "endpoint"

QUERY_INTENT_STRATEGY = "schema_query_intent_v1"
BM25_QUERY_STRATEGY = "structural_intent_aware_bm25_query_v1"

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
        "that",
        "which",
        "what",
        "with",
    }
)


@dataclass(frozen=True)
class CategoryRequest:
    family: str
    role: str = ROLE_ENDPOINT
    expressions: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class PredicateFamilyRequest:
    family: str
    expressions: tuple[str, ...] = ()
    direction: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class RecognizedExpression:
    text: str
    kind: str
    normalized: str
    role: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class QueryIntent:
    requested_subject_categories: tuple[CategoryRequest, ...] = ()
    requested_object_categories: tuple[CategoryRequest, ...] = ()
    requested_endpoint_categories: tuple[CategoryRequest, ...] = ()
    requested_predicate_families: tuple[PredicateFamilyRequest, ...] = ()
    requested_direction: str | None = None
    content_terms: tuple[str, ...] = ()
    recognized_expressions: tuple[RecognizedExpression, ...] = ()
    unrecognized_terms: tuple[str, ...] = ()
    strategy: str = QUERY_INTENT_STRATEGY

    @property
    def has_structural_intent(self) -> bool:
        return bool(
            self.requested_subject_categories
            or self.requested_object_categories
            or self.requested_endpoint_categories
            or self.requested_predicate_families
            or self.requested_direction
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "strategy": self.strategy,
            "requested_subject_categories": [request.to_dict() for request in self.requested_subject_categories],
            "requested_object_categories": [request.to_dict() for request in self.requested_object_categories],
            "requested_endpoint_categories": [request.to_dict() for request in self.requested_endpoint_categories],
            "requested_predicate_families": [request.to_dict() for request in self.requested_predicate_families],
            "requested_direction": self.requested_direction,
            "content_terms": list(self.content_terms),
            "recognized_expressions": [expression.to_dict() for expression in self.recognized_expressions],
            "unrecognized_terms": list(self.unrecognized_terms),
            "has_structural_intent": self.has_structural_intent,
        }


@dataclass(frozen=True)
class BM25ContentQuery:
    original_query: str
    final_query: str
    tokens: tuple[str, ...]
    removed_or_deemphasized_terms: tuple[str, ...]
    recognized_structural_expressions: tuple[RecognizedExpression, ...]
    fallback_used: bool = False
    strategy: str = BM25_QUERY_STRATEGY

    def to_dict(self) -> dict[str, Any]:
        return {
            "strategy": self.strategy,
            "original_query": self.original_query,
            "recognized_structural_expressions": [
                expression.to_dict() for expression in self.recognized_structural_expressions
            ],
            "removed_or_deemphasized_terms": list(self.removed_or_deemphasized_terms),
            "final_query": self.final_query,
            "tokens": list(self.tokens),
            "fallback_used": self.fallback_used,
        }


CATEGORY_LABELS: dict[str, frozenset[str]] = {
    CATEGORY_GENE: frozenset({"biolink:Gene"}),
    CATEGORY_GENE_PRODUCT: frozenset(
        {
            "biolink:GeneOrGeneProduct",
            "biolink:Protein",
            "biolink:Polypeptide",
            "biolink:GeneProductMixin",
        }
    ),
    CATEGORY_SEQUENCE_VARIANT: frozenset(
        {
            "biolink:SequenceVariant",
            "biolink:Allele",
            "biolink:Haplotype",
            "biolink:GenomicEntity",
        }
    ),
    CATEGORY_DRUG_OR_CHEMICAL: frozenset(
        {
            "biolink:ChemicalEntity",
            "biolink:Drug",
            "biolink:ChemicalOrDrugOrTreatment",
            "biolink:SmallMolecule",
        }
    ),
    CATEGORY_DISEASE: frozenset({"biolink:Disease"}),
    CATEGORY_PHENOTYPE: frozenset({"biolink:PhenotypicFeature"}),
    CATEGORY_BIOLOGICAL_PROCESS: frozenset(
        {
            "biolink:BiologicalProcess",
            "biolink:BiologicalProcessOrActivity",
            "biolink:MolecularActivity",
        }
    ),
    CATEGORY_PATHWAY: frozenset({"biolink:Pathway"}),
}

CATEGORY_EXPRESSIONS: tuple[tuple[str, str], ...] = (
    ("biological processes", CATEGORY_BIOLOGICAL_PROCESS),
    ("biological process", CATEGORY_BIOLOGICAL_PROCESS),
    ("gene products", CATEGORY_GENE_PRODUCT),
    ("gene product", CATEGORY_GENE_PRODUCT),
    ("sequence variants", CATEGORY_SEQUENCE_VARIANT),
    ("sequence variant", CATEGORY_SEQUENCE_VARIANT),
    ("proteins", CATEGORY_GENE_PRODUCT),
    ("protein", CATEGORY_GENE_PRODUCT),
    ("variants", CATEGORY_SEQUENCE_VARIANT),
    ("variant", CATEGORY_SEQUENCE_VARIANT),
    ("mutations", CATEGORY_SEQUENCE_VARIANT),
    ("mutation", CATEGORY_SEQUENCE_VARIANT),
    ("alleles", CATEGORY_SEQUENCE_VARIANT),
    ("allele", CATEGORY_SEQUENCE_VARIANT),
    ("chemicals", CATEGORY_DRUG_OR_CHEMICAL),
    ("chemical", CATEGORY_DRUG_OR_CHEMICAL),
    ("drugs", CATEGORY_DRUG_OR_CHEMICAL),
    ("drug", CATEGORY_DRUG_OR_CHEMICAL),
    ("diseases", CATEGORY_DISEASE),
    ("disease", CATEGORY_DISEASE),
    ("phenotypes", CATEGORY_PHENOTYPE),
    ("phenotype", CATEGORY_PHENOTYPE),
    ("processes", CATEGORY_BIOLOGICAL_PROCESS),
    ("process", CATEGORY_BIOLOGICAL_PROCESS),
    ("pathways", CATEGORY_PATHWAY),
    ("pathway", CATEGORY_PATHWAY),
    ("genes", CATEGORY_GENE),
    ("gene", CATEGORY_GENE),
)

PREDICATE_EXPRESSIONS: tuple[tuple[str, str, str | None], ...] = (
    ("affects response to", PREDICATE_DRUG_RESPONSE, "active"),
    ("affecting response to", PREDICATE_DRUG_RESPONSE, "active"),
    ("associated with resistance to", PREDICATE_DRUG_RESPONSE, None),
    ("resistance to", PREDICATE_DRUG_RESPONSE, None),
    ("sensitivity to", PREDICATE_DRUG_RESPONSE, None),
    ("chemoresistance", PREDICATE_DRUG_RESPONSE, None),
    ("resistance", PREDICATE_DRUG_RESPONSE, None),
    ("sensitivity", PREDICATE_DRUG_RESPONSE, None),
    ("associated with", PREDICATE_ASSOCIATION, None),
    ("genetically associated with", PREDICATE_ASSOCIATION, None),
    ("treated by", PREDICATE_TREATMENT, "passive"),
    ("treating", PREDICATE_TREATMENT, "active"),
    ("treats", PREDICATE_TREATMENT, "active"),
    ("treat", PREDICATE_TREATMENT, "active"),
    ("caused by", PREDICATE_CAUSAL, "passive"),
    ("causing", PREDICATE_CAUSAL, "active"),
    ("causes", PREDICATE_CAUSAL, "active"),
    ("regulated by", PREDICATE_REGULATION, "passive"),
    ("regulating", PREDICATE_REGULATION, "active"),
    ("regulates", PREDICATE_REGULATION, "active"),
    ("inhibits", PREDICATE_REGULATION, "active"),
    ("activates", PREDICATE_REGULATION, "active"),
    ("interacts with", PREDICATE_PHYSICAL_INTERACTION, None),
    ("interaction with", PREDICATE_PHYSICAL_INTERACTION, None),
    ("expressed in", PREDICATE_EXPRESSION, None),
    ("related to", PREDICATE_RELATED, None),
)

RELATIONAL_SCAFFOLD_EXPRESSIONS = ("involved in", "involving")
BROAD_BM25_SCAFFOLD_EXPRESSIONS = frozenset({"involved in", "involving", "related to"})

PREDICATE_FAMILIES: dict[str, frozenset[str]] = {
    PREDICATE_ASSOCIATION: frozenset(
        {
            "biolink:associated_with",
            "biolink:genetically_associated_with",
            "biolink:correlated_with",
            "biolink:positively_correlated_with",
        }
    ),
    PREDICATE_TREATMENT: frozenset(
        {
            "biolink:treats",
            "biolink:studied_to_treat",
            "biolink:treats_or_applied_or_studied_to_treat",
            "biolink:ameliorates_condition",
            "biolink:preventative_for_condition",
        }
    ),
    PREDICATE_CAUSAL: frozenset({"biolink:causes", "biolink:contributes_to", "biolink:predisposes"}),
    PREDICATE_REGULATION: frozenset(
        {
            "biolink:regulates",
            "biolink:affects",
            "biolink:increases_activity_or_abundance",
            "biolink:decreases_activity_or_abundance",
        }
    ),
    PREDICATE_PARTICIPATION: frozenset(
        {
            "biolink:actively_involved_in",
            "biolink:has_participant",
            "biolink:participates_in",
            "biolink:occurs_in",
        }
    ),
    PREDICATE_DRUG_RESPONSE: frozenset(
        {
            "biolink:affects_response_to",
            "biolink:increases_response_to",
            "biolink:decreases_response_to",
            "biolink:associated_with_resistance_to",
        }
    ),
    PREDICATE_EXPRESSION: frozenset({"biolink:expressed_in", "biolink:expresses"}),
    PREDICATE_PHYSICAL_INTERACTION: frozenset({"biolink:interacts_with", "biolink:physically_interacts_with"}),
    PREDICATE_RELATED: frozenset({"biolink:related_to", "biolink:coexists_with"}),
}

RELATIONSHIP_QUALITY_BY_PREDICATE: dict[str, str] = {
    "biolink:mentions": QUALITY_CONTEXTUAL_MENTION,
    "biolink:related_to": QUALITY_BROAD_ASSOCIATION,
    "biolink:coexists_with": QUALITY_BROAD_ASSOCIATION,
    "biolink:associated_with": QUALITY_BROAD_ASSOCIATION,
    "biolink:correlated_with": QUALITY_BROAD_ASSOCIATION,
    "biolink:positively_correlated_with": QUALITY_BROAD_ASSOCIATION,
}

SYMMETRIC_PREDICATE_FAMILIES = frozenset({PREDICATE_ASSOCIATION, PREDICATE_PHYSICAL_INTERACTION, PREDICATE_RELATED})
DIRECTIONAL_PREDICATE_FAMILIES = frozenset(
    {PREDICATE_TREATMENT, PREDICATE_CAUSAL, PREDICATE_REGULATION, PREDICATE_DRUG_RESPONSE}
)

GENE_LIKE_ID_PREFIXES = ("ncbigene:", "hgnc:", "ensembl:", "uniprotkb:")
CHEMICAL_LIKE_ID_PREFIXES = ("chebi:", "drugbank:", "pubchem.compound:")


def parse_query_intent(query: str) -> QueryIntent:
    text = _normalize_text(query)
    content_terms = tuple(token for token in _tokens(query) if token not in STOPWORDS)
    recognized: list[RecognizedExpression] = []
    unrecognized = set(content_terms)

    subject_requests: list[CategoryRequest] = []
    object_requests: list[CategoryRequest] = []
    endpoint_requests: list[CategoryRequest] = []
    predicate_requests: list[PredicateFamilyRequest] = []

    for expression, family in CATEGORY_EXPRESSIONS:
        if _contains_phrase(text, expression):
            role = _category_role(text, expression)
            request = CategoryRequest(family=family, role=role, expressions=(expression,))
            if role == ROLE_SUBJECT:
                subject_requests.append(request)
            elif role == ROLE_OBJECT:
                object_requests.append(request)
            else:
                endpoint_requests.append(request)
            recognized.append(
                RecognizedExpression(text=expression, kind="category", normalized=family, role=role)
            )
            unrecognized.difference_update(_tokens(expression))

    for expression, family, direction in PREDICATE_EXPRESSIONS:
        if _contains_phrase(text, expression):
            predicate_requests.append(
                PredicateFamilyRequest(family=family, expressions=(expression,), direction=direction)
            )
            recognized.append(
                RecognizedExpression(text=expression, kind="predicate_family", normalized=family, role=direction)
            )
            unrecognized.difference_update(_tokens(expression))
            if family == PREDICATE_DRUG_RESPONSE and "response to" in expression:
                object_requests.append(
                    CategoryRequest(
                        family=CATEGORY_DRUG_OR_CHEMICAL,
                        role=ROLE_OBJECT,
                        expressions=("response to",),
                    )
                )

    for expression in RELATIONAL_SCAFFOLD_EXPRESSIONS:
        if _contains_phrase(text, expression):
            recognized.append(
                RecognizedExpression(text=expression, kind="relational_scaffold", normalized=expression)
            )
            unrecognized.difference_update(_tokens(expression))

    return QueryIntent(
        requested_subject_categories=_dedupe_category_requests(subject_requests),
        requested_object_categories=_dedupe_category_requests(object_requests),
        requested_endpoint_categories=_dedupe_category_requests(endpoint_requests),
        requested_predicate_families=_dedupe_predicate_requests(predicate_requests),
        requested_direction=_requested_direction(predicate_requests),
        content_terms=content_terms,
        recognized_expressions=tuple(recognized),
        unrecognized_terms=tuple(sorted(unrecognized)),
    )


def build_bm25_content_query(query: str) -> BM25ContentQuery:
    """Build the keyword query from content terms, not structural scaffolding."""

    intent = parse_query_intent(query)
    original_tokens = tuple(token for token in _tokens(query) if token not in STOPWORDS)
    structural_expressions = tuple(
        expression
        for expression in intent.recognized_expressions
        if _is_bm25_deemphasized_expression(expression)
    )
    removed_tokens = _tokens_for_expressions(structural_expressions)
    content_tokens = tuple(token for token in original_tokens if token not in removed_tokens)
    fallback_used = False
    if not content_tokens:
        content_tokens = original_tokens
        fallback_used = True

    return BM25ContentQuery(
        original_query=query,
        final_query=" ".join(content_tokens),
        tokens=content_tokens,
        removed_or_deemphasized_terms=tuple(sorted(removed_tokens)),
        recognized_structural_expressions=structural_expressions,
        fallback_used=fallback_used,
    )


def category_families_from_labels(labels: list[str] | tuple[str, ...]) -> set[str]:
    label_set = {str(label) for label in labels}
    families = {
        family
        for family, family_labels in CATEGORY_LABELS.items()
        if label_set & family_labels
    }
    if CATEGORY_GENE in families:
        families.discard(CATEGORY_GENE_PRODUCT)
    if CATEGORY_DISEASE in families:
        families.discard(CATEGORY_PHENOTYPE)
    return families


def category_families_from_identifier(identifier: str) -> set[str]:
    value = str(identifier or "").casefold()
    if value.startswith(GENE_LIKE_ID_PREFIXES):
        return {CATEGORY_GENE}
    if value.startswith(CHEMICAL_LIKE_ID_PREFIXES):
        return {CATEGORY_DRUG_OR_CHEMICAL}
    return set()


def predicate_family(predicate: str | None) -> str:
    predicate_text = str(predicate or "")
    for family, predicates in PREDICATE_FAMILIES.items():
        if predicate_text in predicates:
            return family
    return PREDICATE_UNKNOWN


def relationship_quality_tier(predicate: str | None) -> str:
    predicate_text = str(predicate or "")
    if predicate_text in RELATIONSHIP_QUALITY_BY_PREDICATE:
        return RELATIONSHIP_QUALITY_BY_PREDICATE[predicate_text]
    if predicate_family(predicate_text) != PREDICATE_UNKNOWN:
        return QUALITY_DIRECT_ASSERTION
    return QUALITY_UNKNOWN


def is_symmetric_predicate_family(family: str | None) -> bool:
    return str(family or "") in SYMMETRIC_PREDICATE_FAMILIES


def is_directional_predicate_family(family: str | None) -> bool:
    return str(family or "") in DIRECTIONAL_PREDICATE_FAMILIES


def _category_role(text: str, expression: str) -> str:
    match = re.search(rf"\b{re.escape(expression)}\b", text)
    if not match:
        return ROLE_ENDPOINT

    after = text[match.end() :].strip()
    if after.startswith(("regulated by", "caused by", "treated by")):
        return ROLE_OBJECT
    if re.match(
        r"^(that\s+|which\s+)?(treat|treats|treating|cause|causes|causing|regulate|regulates|regulating|"
        r"affect|affects|affecting|interact|interacts|interacting|associated\s+with)",
        after,
    ):
        return ROLE_SUBJECT
    return ROLE_ENDPOINT


def _requested_direction(predicate_requests: list[PredicateFamilyRequest]) -> str | None:
    directions = {request.direction for request in predicate_requests if request.direction}
    if len(directions) == 1:
        return next(iter(directions))
    return None


def _dedupe_category_requests(requests: list[CategoryRequest]) -> tuple[CategoryRequest, ...]:
    grouped: dict[tuple[str, str], list[str]] = {}
    for request in requests:
        grouped.setdefault((request.family, request.role), []).extend(request.expressions)
    return tuple(
        CategoryRequest(family=family, role=role, expressions=tuple(sorted(set(expressions))))
        for (family, role), expressions in sorted(grouped.items())
    )


def _dedupe_predicate_requests(requests: list[PredicateFamilyRequest]) -> tuple[PredicateFamilyRequest, ...]:
    grouped: dict[tuple[str, str | None], list[str]] = {}
    for request in requests:
        grouped.setdefault((request.family, request.direction), []).extend(request.expressions)
    return tuple(
        PredicateFamilyRequest(family=family, direction=direction, expressions=tuple(sorted(set(expressions))))
        for (family, direction), expressions in sorted(grouped.items())
    )


def _normalize_text(text: str) -> str:
    return " ".join(_tokens(text))


def _contains_phrase(text: str, phrase: str) -> bool:
    return bool(re.search(rf"\b{re.escape(phrase)}\b", text))


def _tokens(text: str) -> list[str]:
    return re.findall(r"[a-z0-9]+", (text or "").casefold())


def _is_bm25_deemphasized_expression(expression: RecognizedExpression) -> bool:
    if expression.kind == "category":
        return True
    if expression.kind == "relational_scaffold":
        return True
    return expression.text in BROAD_BM25_SCAFFOLD_EXPRESSIONS


def _tokens_for_expressions(expressions: tuple[RecognizedExpression, ...]) -> set[str]:
    tokens: set[str] = set()
    for expression in expressions:
        tokens.update(_tokens(expression.text))
    return tokens
