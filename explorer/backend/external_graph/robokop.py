from __future__ import annotations

import hashlib
import json
import os
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Protocol

import httpx

from explorer.backend.external_graph.models import (
    EdgeSummary,
    EdgeSummaryItem,
    ExternalEdge,
    ExternalExpansionConfig,
    ExternalExpansionResult,
    ExternalNode,
)
from explorer.backend.semantic_search.neighborhood import (
    NEIGHBOR_ENDPOINT_CATEGORY_MATCH,
    NEIGHBOR_PREDICATE_FAMILY_MATCH,
)
from explorer.backend.semantic_search.query_intent import (
    category_families_from_labels,
    parse_query_intent,
    predicate_family,
)
from explorer.backend.semantic_search.retrieval import tokenize_keyword_query

DEFAULT_ROBOKOP_URL = "https://automat.renci.org/robokopkg"
DEFAULT_FIXTURE_PATH = Path(__file__).parent / "fixtures" / "robokop_pten.json"
ROBOKOP_RANKING_STRATEGY = "bounded_robokop_keyword_intent_v1"
REMOTE_TOKEN_MATCH_WEIGHT = 0.05
REMOTE_MAX_TOKEN_SCORE = 0.25


class ExternalGraphProvider(Protocol):
    provider_name: str
    provider_mode: str

    def lookup_node(self, curie: str) -> ExternalNode | None: ...

    def edge_summary(self, curie: str) -> EdgeSummary: ...

    def incident_edges(
        self,
        curie: str,
        config: ExternalExpansionConfig | None = None,
    ) -> ExternalExpansionResult: ...


class RobokopProviderError(RuntimeError):
    pass


class RobokopProviderTimeout(RobokopProviderError):
    pass


class RobokopMalformedResponse(RobokopProviderError):
    pass


@dataclass(frozen=True)
class RobokopHttpConfig:
    base_url: str = DEFAULT_ROBOKOP_URL
    timeout_seconds: float = 20.0

    @classmethod
    def from_env(cls) -> RobokopHttpConfig:
        return cls(
            base_url=(os.getenv("ROBOKOP_URL") or DEFAULT_ROBOKOP_URL).rstrip("/"),
            timeout_seconds=float(os.getenv("ROBOKOP_TIMEOUT_SECONDS") or 20.0),
        )


class RobokopHttpProvider:
    provider_name = "robokop"
    provider_mode = "live"

    def __init__(self, config: RobokopHttpConfig | None = None, client: Any | None = None) -> None:
        self.config = config or RobokopHttpConfig.from_env()
        self.base_url = self.config.base_url.rstrip("/")
        self._client = client or httpx.Client(timeout=self.config.timeout_seconds)

    def close(self) -> None:
        close = getattr(self._client, "close", None)
        if callable(close):
            close()

    def lookup_node(self, curie: str) -> ExternalNode | None:
        payload = self._get_json(f"/node/{curie}")
        if not payload:
            return None
        if not isinstance(payload, dict) or not payload.get("id"):
            raise RobokopMalformedResponse("ROBOKOP node response must be an object with an id.")
        return _node_from_payload(payload)

    def edge_summary(self, curie: str) -> EdgeSummary:
        payload = self._get_json(f"/edge_summary/{curie}")
        if not isinstance(payload, dict):
            raise RobokopMalformedResponse("ROBOKOP edge summary response must be an object.")
        edge_types = payload.get("edge_types")
        if edge_types is None:
            edge_types = []
        if not isinstance(edge_types, list):
            raise RobokopMalformedResponse("ROBOKOP edge summary must contain an edge_types list.")
        items = tuple(
            EdgeSummaryItem(
                predicate=str(item.get("predicate") or ""),
                category=str(item.get("category") or ""),
                count=int(item.get("count") or 0),
            )
            for item in edge_types
            if isinstance(item, dict)
        )
        return EdgeSummary(
            query_curie=str(payload.get("query_curie") or curie),
            items=items,
            provider_mode=self.provider_mode,
        )

    def incident_edges(
        self,
        curie: str,
        config: ExternalExpansionConfig | None = None,
    ) -> ExternalExpansionResult:
        config = config or ExternalExpansionConfig()
        params: dict[str, Any] = {
            "limit": config.normalized_limit(),
            "offset": config.normalized_offset(),
        }
        if config.category:
            params["category"] = config.category
        if config.predicate:
            params["predicate"] = config.predicate

        payload = self._get_json(f"/edges/{curie}", params=params)
        result = _expansion_from_payload(
            payload,
            curie=curie,
            config=config,
            provider_mode=self.provider_mode,
            requested_params=params,
        )
        return rank_external_edges(result, query=config.query)

    def _get_json(self, path: str, params: dict[str, Any] | None = None) -> Any:
        try:
            response = self._client.get(f"{self.base_url}{path}", params=params)
            response.raise_for_status()
            return response.json()
        except httpx.TimeoutException as exc:
            raise RobokopProviderTimeout(f"ROBOKOP request timed out: {path}") from exc
        except httpx.HTTPError as exc:
            raise RobokopProviderError(f"ROBOKOP request failed: {exc}") from exc
        except ValueError as exc:
            raise RobokopMalformedResponse(f"ROBOKOP response was not valid JSON: {path}") from exc


class RobokopFixtureProvider:
    provider_name = "robokop"
    provider_mode = "fixture"

    def __init__(self, fixture_path: str | Path | None = None) -> None:
        self.fixture_path = Path(fixture_path or os.getenv("ROBOKOP_FIXTURE_PATH") or DEFAULT_FIXTURE_PATH)
        self.payload = json.loads(self.fixture_path.read_text())
        self.base_url = f"fixture://{self.fixture_path}"

    def lookup_node(self, curie: str) -> ExternalNode | None:
        payload = self.payload.get("nodes", {}).get(curie)
        return _node_from_payload(payload) if payload else None

    def edge_summary(self, curie: str) -> EdgeSummary:
        payload = self.payload.get("edge_summaries", {}).get(curie, {"query_curie": curie, "edge_types": []})
        items = tuple(
            EdgeSummaryItem(
                predicate=str(item.get("predicate") or ""),
                category=str(item.get("category") or ""),
                count=int(item.get("count") or 0),
            )
            for item in payload.get("edge_types", [])
        )
        return EdgeSummary(query_curie=curie, items=items, provider_mode=self.provider_mode)

    def incident_edges(
        self,
        curie: str,
        config: ExternalExpansionConfig | None = None,
    ) -> ExternalExpansionResult:
        config = config or ExternalExpansionConfig()
        all_edges = list(self.payload.get("edges", {}).get(curie, []))
        filtered = [
            edge
            for edge in all_edges
            if (not config.category or _edge_adjacent_category(edge) == config.category)
            and (not config.predicate or _edge_predicate(edge) == config.predicate)
        ]
        offset = config.normalized_offset()
        limit = config.normalized_limit()
        page = filtered[offset : offset + limit]
        payload = {
            "query_curie": curie,
            "edges": page,
            "pagination": {
                "count": len(page),
                "offset": offset,
                "limit": limit,
                "total": len(filtered),
            },
        }
        result = _expansion_from_payload(
            payload,
            curie=curie,
            config=config,
            provider_mode=self.provider_mode,
            requested_params={
                "category": config.category,
                "predicate": config.predicate,
                "limit": limit,
                "offset": offset,
            },
        )
        return rank_external_edges(result, query=config.query)


def robokop_provider_from_env() -> ExternalGraphProvider:
    mode = (os.getenv("ROBOKOP_MODE") or "live").strip().lower()
    if mode == "fixture":
        return RobokopFixtureProvider()
    if mode != "live":
        raise RobokopProviderError("ROBOKOP_MODE must be 'live' or 'fixture'.")
    return RobokopHttpProvider()


def rank_external_edges(
    result: ExternalExpansionResult,
    *,
    query: str,
) -> ExternalExpansionResult:
    intent = parse_query_intent(query)
    query_tokens = tokenize_keyword_query(query)
    requested_direction = result.config.normalized_direction()
    ranked: list[ExternalEdge] = []
    excluded_count = 0

    for original_rank, edge in enumerate(result.edges):
        if requested_direction != "either" and edge.direction != requested_direction:
            excluded_count += 1
            continue
        score, reasons, matched_tokens = _external_edge_score(edge, query_tokens=query_tokens, intent=intent)
        ranked.append(
            replace(
                edge,
                rank=original_rank,
                score=score,
                ranking_reasons=tuple(reasons),
                matched_query_tokens=tuple(matched_tokens),
            )
        )

    ranked.sort(key=lambda edge: (-(edge.score or 0.0), edge.rank if edge.rank is not None else 0, edge.edge_id))
    return ExternalExpansionResult(
        query_curie=result.query_curie,
        edges=tuple(ranked),
        config=result.config,
        provider_mode=result.provider_mode,
        total_returned=len(ranked),
        total_available=result.total_available,
        diagnostics={
            **result.diagnostics,
            "ranking_strategy": ROBOKOP_RANKING_STRATEGY,
            "query_intent": intent.to_dict(),
            "query_tokens": query_tokens,
            "direction_filter": requested_direction,
            "direction_filtered_count": excluded_count,
        },
    )


def _external_edge_score(edge: ExternalEdge, *, query_tokens: list[str], intent: Any) -> tuple[float, list[str], list[str]]:
    text_tokens = set(
        tokenize_keyword_query(
            " ".join(
                str(part)
                for part in (
                    edge.subject_curie,
                    edge.subject_name,
                    edge.predicate,
                    edge.object_curie,
                    edge.object_name,
                    " ".join(edge.publications),
                    " ".join(edge.supporting_sentences),
                    json.dumps(edge.qualifiers, sort_keys=True),
                )
                if part
            )
        )
    )
    matched_tokens = sorted(set(query_tokens) & text_tokens)
    score = min(len(matched_tokens) * REMOTE_TOKEN_MATCH_WEIGHT, REMOTE_MAX_TOKEN_SCORE)
    reasons: list[str] = []
    if matched_tokens:
        reasons.append("keyword match")

    requested_categories = {
        request.family
        for request in (
            *intent.requested_subject_categories,
            *intent.requested_object_categories,
            *intent.requested_endpoint_categories,
        )
    }
    observed_categories = category_families_from_labels(list(edge.subject_categories) + list(edge.object_categories))
    if requested_categories and requested_categories & observed_categories:
        score += NEIGHBOR_ENDPOINT_CATEGORY_MATCH
        reasons.append("endpoint category compatibility")

    requested_predicates = {request.family for request in intent.requested_predicate_families}
    if requested_predicates and _remote_predicate_family(edge.predicate) in requested_predicates:
        score += NEIGHBOR_PREDICATE_FAMILY_MATCH
        reasons.append("predicate family compatibility")

    return score, reasons or ["bounded ROBOKOP page fallback"], matched_tokens


def _node_from_payload(payload: dict[str, Any]) -> ExternalNode:
    properties = dict(payload.get("properties") or {})
    return ExternalNode(
        curie=str(payload.get("id")),
        name=str(payload.get("name") or payload.get("id") or ""),
        categories=tuple(_string_list(payload.get("category"))),
        properties=properties,
    )


def _expansion_from_payload(
    payload: Any,
    *,
    curie: str,
    config: ExternalExpansionConfig,
    provider_mode: str,
    requested_params: dict[str, Any],
) -> ExternalExpansionResult:
    if not isinstance(payload, dict):
        raise RobokopMalformedResponse("ROBOKOP edge response must be an object.")
    edges_payload = payload.get("edges")
    if not isinstance(edges_payload, list):
        raise RobokopMalformedResponse("ROBOKOP edge response must contain an edges list.")

    query_curie = str(payload.get("query_curie") or curie)
    edges = tuple(
        _edge_from_wrapper(edge_wrapper, query_curie=query_curie)
        for edge_wrapper in edges_payload
        if isinstance(edge_wrapper, dict)
    )
    pagination = payload.get("pagination") if isinstance(payload.get("pagination"), dict) else {}
    total_available = pagination.get("total")
    if total_available is None:
        total_available = pagination.get("count")
    return ExternalExpansionResult(
        query_curie=query_curie,
        edges=edges,
        config=config,
        provider_mode=provider_mode,
        total_returned=len(edges),
        total_available=int(total_available) if total_available is not None else None,
        diagnostics={
            "requested_params": {key: value for key, value in requested_params.items() if value is not None},
            "pagination": pagination,
            "direction_filter_applied_client_side": config.normalized_direction() != "either",
        },
    )


def _edge_from_wrapper(edge_wrapper: dict[str, Any], *, query_curie: str) -> ExternalEdge:
    edge = edge_wrapper.get("edge", edge_wrapper)
    adj_node = edge_wrapper.get("adj_node", {})
    if not isinstance(edge, dict) or not isinstance(adj_node, dict):
        raise RobokopMalformedResponse("ROBOKOP edge wrapper must contain edge and adj_node objects.")
    predicate = str(edge.get("predicate") or "")
    edge_direction = str(edge.get("direction") or ">")
    direction = "incoming" if edge_direction == "<" else "outgoing"
    adjacent_curie = str(adj_node.get("id") or "")
    adjacent_name = str(adj_node.get("name") or adjacent_curie)
    adjacent_categories = tuple(_string_list(adj_node.get("category")))
    properties = dict(edge.get("properties") or {})
    if direction == "incoming":
        subject_curie = adjacent_curie
        object_curie = query_curie
        subject_name = adjacent_name
        object_name = query_curie
        subject_categories = adjacent_categories
        object_categories = tuple(_categories_from_curie(query_curie))
    else:
        subject_curie = query_curie
        object_curie = adjacent_curie
        subject_name = query_curie
        object_name = adjacent_name
        subject_categories = tuple(_categories_from_curie(query_curie))
        object_categories = adjacent_categories

    publications = tuple(_string_list(properties.get("publications")))
    supporting_sentences = tuple(_sentences(properties.get("sentences")))
    qualifiers = {
        str(key): value
        for key, value in properties.items()
        if "qualifier" in str(key).lower()
    }
    primary_source = properties.get("primary_knowledge_source")
    edge_id = _stable_edge_id(
        query_curie=query_curie,
        adjacent_curie=adjacent_curie,
        predicate=predicate,
        direction=direction,
        properties=properties,
    )
    return ExternalEdge(
        edge_id=edge_id,
        subject_curie=subject_curie,
        object_curie=object_curie,
        predicate=predicate,
        direction=direction,
        adjacent_curie=adjacent_curie,
        query_curie=query_curie,
        subject_name=subject_name,
        object_name=object_name,
        subject_categories=subject_categories,
        object_categories=object_categories,
        qualifiers=qualifiers,
        primary_knowledge_source=str(primary_source) if primary_source else None,
        publications=publications,
        supporting_sentences=supporting_sentences,
        original_subject_curie=properties.get("original_subject"),
        original_object_curie=properties.get("original_object"),
        properties=properties,
    )


def _stable_edge_id(
    *,
    query_curie: str,
    adjacent_curie: str,
    predicate: str,
    direction: str,
    properties: dict[str, Any],
) -> str:
    primary_source = str(properties.get("primary_knowledge_source") or "")
    original_subject = str(properties.get("original_subject") or "")
    original_object = str(properties.get("original_object") or "")
    payload = json.dumps(
        {
            "query_curie": query_curie,
            "adjacent_curie": adjacent_curie,
            "predicate": predicate,
            "direction": direction,
            "primary_source": primary_source,
            "original_subject": original_subject,
            "original_object": original_object,
        },
        sort_keys=True,
    )
    digest = hashlib.sha1(payload.encode("utf-8")).hexdigest()[:16]
    return f"robokop:{digest}"


def _edge_adjacent_category(edge_wrapper: dict[str, Any]) -> str | None:
    categories = _string_list((edge_wrapper.get("adj_node") or {}).get("category"))
    return categories[0] if categories else None


def _edge_predicate(edge_wrapper: dict[str, Any]) -> str | None:
    return (edge_wrapper.get("edge") or edge_wrapper).get("predicate")


def _string_list(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, list):
        return [str(item) for item in value if item]
    if isinstance(value, tuple):
        return [str(item) for item in value if item]
    if isinstance(value, str):
        return [value] if value else []
    return [str(value)]


def _sentences(value: Any) -> list[str]:
    if isinstance(value, str):
        return [part.strip() for part in value.split("|") if part.strip() and part.strip() != "NA"]
    return _string_list(value)


def _categories_from_curie(curie: str) -> list[str]:
    prefix = curie.split(":", 1)[0].upper() if ":" in curie else ""
    if prefix in {"NCBIGENE", "HGNC", "ENSEMBL"}:
        return ["biolink:Gene"]
    if prefix in {"UNIPROTKB", "PR"}:
        return ["biolink:Protein"]
    if prefix in {"DRUGBANK", "CHEBI", "PUBCHEM.COMPOUND"}:
        return ["biolink:Drug"]
    if prefix in {"MONDO", "DOID"}:
        return ["biolink:Disease"]
    if prefix in {"CAID", "DBSNP", "CLINVAR.VARIATION"}:
        return ["biolink:SequenceVariant"]
    return []


def _remote_predicate_family(predicate: str) -> str:
    if predicate in {
        "biolink:affects_sensitivity_to",
        "biolink:increases_sensitivity_to",
        "biolink:decreases_sensitivity_to",
    }:
        return "drug_response"
    return predicate_family(predicate)
