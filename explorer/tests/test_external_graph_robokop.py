from __future__ import annotations

import httpx
import pytest

from explorer.backend.external_graph import (
    BridgeResolutionConfig,
    ExternalExpansionConfig,
    RobokopFixtureProvider,
    RobokopHttpConfig,
    RobokopHttpProvider,
    RobokopMalformedResponse,
    RobokopProviderTimeout,
    resolve_litcoin_bridge,
    robokop_provider_from_env,
)
from explorer.frontend.callbacks import (
    _add_robokop_edge_to_context,
    _bounded_nonnegative_int,
    _bounded_robokop_limit,
)
from explorer.frontend.graph import visible_subgraph


def test_normalized_identifier_maps_litcoin_node_to_robokop_lookup() -> None:
    resolution = resolve_litcoin_bridge({"id": "cyto-1", "curie": "NCBIGene:5728", "label": "PTEN"})

    assert resolution.is_resolved
    assert resolution.curie == "NCBIGene:5728"
    assert resolution.reason == "Resolved from explicit normalized CURIE."


def test_zero_and_ambiguous_mappings_require_explicit_handling() -> None:
    zero = resolve_litcoin_bridge({"id": "cyto-1", "label": "PTEN"})
    ambiguous = resolve_litcoin_bridge(
        {
            "id": "cyto-1",
            "label": "PTEN",
            "properties": {
                "equivalent_identifiers": ["NCBIGene:5728", "HGNC:9588"],
            },
        }
    )

    assert zero.status == "zero_mapping"
    assert "display-name equality" in zero.reason
    assert ambiguous.status == "ambiguous_mapping"
    assert ambiguous.candidates == ("NCBIGene:5728", "HGNC:9588")


def test_bridge_conflation_settings_affect_cache_identity() -> None:
    default = BridgeResolutionConfig()
    strict = BridgeResolutionConfig(conflate_gene_protein=False, conflate_drug_chemical=False)

    assert default.to_cache_dict() != strict.to_cache_dict()
    assert strict.to_cache_dict() == {
        "conflate_gene_protein": False,
        "conflate_drug_chemical": False,
    }


def test_http_provider_propagates_filters_and_retains_remote_provenance() -> None:
    client = FakeHttpClient(
        {
            "https://example.test/edges/NCBIGene:5728": {
                "query_curie": "NCBIGene:5728",
                "edges": [
                    edge_wrapper(
                        "biolink:increases_sensitivity_to",
                        direction=">",
                        adjacent_curie="DRUGBANK:DB00515",
                        adjacent_name="cisplatin",
                        adjacent_category="biolink:Drug",
                        properties={
                            "primary_knowledge_source": "infores:robokop-ctd",
                            "publications": ["PMID:12345"],
                            "object_aspect_qualifier": "response",
                            "sentences": "PTEN affects cancer drug response.",
                        },
                    ),
                    edge_wrapper(
                        "biolink:is_nearby_variant_of",
                        direction="<",
                        adjacent_curie="CAID:CA000103",
                        adjacent_name="rs786201041",
                        adjacent_category="biolink:SequenceVariant",
                        properties={"primary_knowledge_source": "infores:robokop-snpeff"},
                    ),
                ],
                "pagination": {"count": 2, "offset": 5, "limit": 2},
            }
        }
    )
    provider = RobokopHttpProvider(
        RobokopHttpConfig(base_url="https://example.test", timeout_seconds=1.0),
        client=client,
    )

    result = provider.incident_edges(
        "NCBIGene:5728",
        ExternalExpansionConfig(
            category="biolink:Drug",
            predicate="biolink:increases_sensitivity_to",
            direction="outgoing",
            limit=2,
            offset=5,
            query="genes involved in cancer drug sensitivity",
        ),
    )

    assert client.requests == [
        (
            "https://example.test/edges/NCBIGene:5728",
            {
                "category": "biolink:Drug",
                "predicate": "biolink:increases_sensitivity_to",
                "limit": 2,
                "offset": 5,
            },
        )
    ]
    assert [edge.adjacent_curie for edge in result.edges] == ["DRUGBANK:DB00515"]
    edge = result.edges[0]
    assert edge.source_graph == "robokop"
    assert edge.edge_id.startswith("robokop:")
    assert edge.primary_knowledge_source == "infores:robokop-ctd"
    assert edge.publications == ("PMID:12345",)
    assert edge.qualifiers == {"object_aspect_qualifier": "response"}
    assert edge.supporting_sentences == ("PTEN affects cancer drug response.",)
    assert edge.score and edge.score > 0
    assert "predicate family compatibility" in edge.ranking_reasons
    assert result.total_available is None
    assert result.diagnostics["requested_params"]["limit"] == 2
    assert result.diagnostics["page_count"] == 2
    assert result.diagnostics["direction_filter_applied_client_side"] is True


def test_http_provider_reports_malformed_and_timeout_responses() -> None:
    malformed = RobokopHttpProvider(
        RobokopHttpConfig(base_url="https://example.test"),
        client=FakeHttpClient({"https://example.test/edges/NCBIGene:5728": {"query_curie": "NCBIGene:5728"}}),
    )
    with pytest.raises(RobokopMalformedResponse, match="edges list"):
        malformed.incident_edges("NCBIGene:5728")

    timeout = RobokopHttpProvider(
        RobokopHttpConfig(base_url="https://example.test"),
        client=TimeoutClient(),
    )
    with pytest.raises(RobokopProviderTimeout):
        timeout.edge_summary("NCBIGene:5728")


def test_fixture_mode_and_live_mode_are_explicit(monkeypatch: pytest.MonkeyPatch) -> None:
    provider = RobokopFixtureProvider()

    node = provider.lookup_node("NCBIGene:5728")
    summary = provider.edge_summary("NCBIGene:5728")
    expansion = provider.incident_edges(
        "NCBIGene:5728",
        ExternalExpansionConfig(
            predicate="biolink:increases_sensitivity_to",
            query="PTEN cancer drug response",
        ),
    )

    assert node and node.source_graph == "robokop"
    assert summary.provider_mode == "fixture"
    assert expansion.provider_mode == "fixture"
    assert expansion.edges[0].primary_knowledge_source == "infores:robokop-ctd"

    monkeypatch.setenv("ROBOKOP_MODE", "fixture")
    assert robokop_provider_from_env().provider_mode == "fixture"


def test_bounded_controls_and_selected_remote_edge_merge_preserve_provenance() -> None:
    assert _bounded_robokop_limit(None) == 10
    assert _bounded_robokop_limit("100") == 25
    assert _bounded_nonnegative_int("-3") == 0

    local_edge_id = "r-local"
    remote_edge = {
        "edge_id": "robokop:r-local",
        "adjacent_curie": "DRUGBANK:DB00515",
        "direction": "outgoing",
        "object_name": "cisplatin",
        "object_categories": ["biolink:Drug"],
        "predicate": "biolink:increases_sensitivity_to",
        "primary_knowledge_source": "infores:robokop-ctd",
        "publications": ["PMID:12345"],
        "qualifiers": {"object_aspect_qualifier": "response"},
        "supporting_sentences": ["PTEN affects cancer drug response."],
        "original_subject_curie": "NCBIGene:5728",
        "original_object_curie": "DRUGBANK:DB00515",
    }
    path = {
        "nodes": [{"element_id": "n1"}, {"element_id": "n2"}],
        "edges": [{"element_id": local_edge_id}],
    }
    context = {
        "base_subgraph": {
            "nodes": [
                {"id": "n1", "curie": "NCBIGene:5728", "label": "PTEN"},
                {"id": "n2", "curie": "MONDO:0004992", "label": "cancer"},
            ],
            "edges": [{"id": local_edge_id, "source": "n1", "target": "n2", "label": "biolink:associated_with"}],
        },
        "semantic_subgraphs": {},
        "robokop_subgraphs": {},
        "robokop_state": {"n1": {"selected_edge_ids": []}},
        "hidden_ids": [],
    }

    _add_robokop_edge_to_context(context, "n1", remote_edge)
    visible = visible_subgraph(path, context)
    edge_ids = {edge["id"]: edge for edge in visible["edges"]}

    assert set(edge_ids) == {local_edge_id, "robokop:r-local"}
    assert edge_ids[local_edge_id].get("source_graph") is None
    assert edge_ids["robokop:r-local"]["source_graph"] == "robokop"
    assert edge_ids["robokop:r-local"]["properties"]["primary_knowledge_source"] == "infores:robokop-ctd"


def edge_wrapper(
    predicate: str,
    *,
    direction: str,
    adjacent_curie: str,
    adjacent_name: str,
    adjacent_category: str,
    properties: dict,
) -> dict:
    return {
        "edge": {
            "predicate": predicate,
            "direction": direction,
            "properties": properties,
        },
        "adj_node": {
            "id": adjacent_curie,
            "name": adjacent_name,
            "category": [adjacent_category],
        },
    }


class FakeResponse:
    def __init__(self, payload: dict) -> None:
        self.payload = payload

    def raise_for_status(self) -> None:
        return None

    def json(self) -> dict:
        return self.payload


class FakeHttpClient:
    def __init__(self, payloads: dict[str, dict]) -> None:
        self.payloads = payloads
        self.requests: list[tuple[str, dict | None]] = []

    def get(self, url: str, params: dict | None = None) -> FakeResponse:
        self.requests.append((url, params))
        return FakeResponse(self.payloads[url])


class TimeoutClient:
    def get(self, _url: str, params: dict | None = None) -> FakeResponse:
        raise httpx.TimeoutException("timeout")
