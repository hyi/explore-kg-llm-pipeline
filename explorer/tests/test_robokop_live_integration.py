from __future__ import annotations

import os

import pytest

from explorer.backend.external_graph import (
    ExternalExpansionConfig,
    RobokopHttpProvider,
)

pytestmark = pytest.mark.skipif(
    os.getenv("RUN_LIVE_ROBOKOP") != "1",
    reason="Set RUN_LIVE_ROBOKOP=1 to run the live ROBOKOP integration check.",
)


def test_live_robokop_provider_resolves_pten_summary_and_bounded_page() -> None:
    provider = RobokopHttpProvider()
    try:
        node = provider.lookup_node("NCBIGene:5728")
        summary = provider.edge_summary("NCBIGene:5728")
        page = provider.incident_edges(
            "NCBIGene:5728",
            ExternalExpansionConfig(
                predicate="biolink:increases_sensitivity_to",
                limit=1,
                offset=0,
                query="PTEN drug sensitivity in cancer",
            ),
        )
    finally:
        provider.close()

    assert node is not None
    assert node.curie == "NCBIGene:5728"
    assert summary.total_edges > 0
    assert page.provider_mode == "live"
    assert page.config.normalized_limit() == 1
