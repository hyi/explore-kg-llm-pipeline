from __future__ import annotations

import json

from explorer.backend.external_graph import RobokopFixtureProvider
from explorer.backend.models import Edge, Node, Path
from explorer.frontend.callbacks import (
    _add_robokop_edge_to_context,
    _bounded_expansion_limit,
    _bounded_nonnegative_int,
    _bounded_robokop_limit,
    _normalize_filter_values,
    _parse_comma_separated,
    _robokop_page_offset,
)
from explorer.frontend.components import (
    render_candidate_path_cards,
    render_path_details,
)
from explorer.frontend.constants import CANDIDATE_PREVIEW_LIMIT
from explorer.frontend.dash_app import create_app
from explorer.frontend.graph import (
    cytoscape_elements,
    node_action_panel,
    positions_from_elements,
    truncate_label,
    visible_subgraph,
)
from explorer.frontend.layout import build_layout
from explorer.frontend.state import path_from_dict, selected_node_id


def test_path_round_trip_from_dict() -> None:
    path = Path(
        id="path-1",
        nodes=[
            Node(element_id="n1", id="gene:1", name="Gene A", labels=["Gene"]),
            Node(element_id="n2", id="disease:1", name="Disease B", labels=["Disease"]),
        ],
        edges=[
            Edge(
                element_id="r1",
                type="biolink:associated_with",
                start_element_id="n1",
                end_element_id="n2",
                subject="Gene A",
                object="Disease B",
            )
        ],
        score=0.91,
        anchor_metadata={"anchor_score": 0.91, "semantic_score": 0.82},
    )

    restored = path_from_dict(path.to_dict())

    assert restored.id == "path-1"
    assert restored.summary() == path.summary()
    assert restored.score == 0.91
    assert restored.anchor_metadata == {"anchor_score": 0.91, "semantic_score": 0.82}


def test_visible_subgraph_keeps_initial_path_nodes_protected() -> None:
    path = {
        "nodes": [{"element_id": "n1"}, {"element_id": "n2"}],
        "edges": [{"element_id": "r1"}],
    }
    context = {
        "hidden_ids": ["n1", "n3"],
        "base_subgraph": {
            "nodes": [{"id": "n1"}, {"id": "n2"}, {"id": "n3"}],
            "edges": [
                {"id": "r1", "source": "n1", "target": "n2"},
                {"id": "r2", "source": "n2", "target": "n3"},
            ],
        },
        "semantic_subgraphs": {},
    }

    visible = visible_subgraph(path, context)

    assert [node["id"] for node in visible["nodes"]] == ["n1", "n2"]
    assert [edge["id"] for edge in visible["edges"]] == ["r1"]


def test_cytoscape_elements_mark_path_and_selected_node() -> None:
    path = {
        "id": "path-1",
        "nodes": [{"element_id": "n1"}, {"element_id": "n2"}],
        "edges": [{"element_id": "r1"}],
    }
    subgraph = {
        "nodes": [
            {"id": "n1", "label": "Gene A", "labels": ["Gene"]},
            {"id": "n2", "label": "Disease B", "labels": ["Disease"]},
        ],
        "edges": [{"id": "r1", "source": "n1", "target": "n2", "label": "biolink:associated_with"}],
    }
    context = {
        "positions": {"n1": {"x": 1, "y": 2}, "n2": {"x": 3, "y": 4}},
        "focus_ids": ["n1"],
        "selected_node_id": "n2",
    }

    elements = cytoscape_elements(path, subgraph, context)
    selected = [element for element in elements if element.get("selected")]

    assert selected[0]["data"]["id"] == "n2"
    assert any("path-edge" in element.get("classes", "") for element in elements)
    assert elements[0]["data"]["full_label"] == "Gene A"


def test_cytoscape_labels_are_truncated_for_display() -> None:
    label = "This is a very long biomedical entity label"

    assert truncate_label(label) == "This is a very long biomedica…"
    assert len(truncate_label(label)) == 30


def test_positions_and_selected_node_helpers() -> None:
    elements = [
        {"data": {"id": "n1"}, "position": {"x": 10, "y": 20}},
        {"data": {"id": "r1", "source": "n1", "target": "n2"}},
    ]

    assert positions_from_elements(elements, {}) == {"n1": {"x": 10.0, "y": 20.0}}
    assert selected_node_id([{"id": "n1"}]) == "n1"
    assert selected_node_id([]) is None


def test_path_action_buttons_include_path_id() -> None:
    path = Path(
        id="path-1",
        nodes=[
            Node(element_id="n1", id="gene:1", name="Gene A", labels=["Gene"]),
            Node(element_id="n2", id="disease:1", name="Disease B", labels=["Disease"]),
        ],
        edges=[
            Edge(
                element_id="r1",
                type="biolink:associated_with",
                start_element_id="n1",
                end_element_id="n2",
            )
        ],
    ).to_dict()

    details = render_path_details(path, context=None, session_store={})
    action_row = details[0].children[1]
    action_ids = [button.id for button in action_row.children]

    assert action_ids == [
        {"type": "path-action", "action": "accept", "path_id": "path-1"},
        {"type": "path-action", "action": "bookmark", "path_id": "path-1"},
        {"type": "path-action", "action": "reject", "path_id": "path-1"},
    ]


def test_graph_controls_render_above_graph() -> None:
    path = Path(
        id="path-1",
        nodes=[
            Node(element_id="n1", id="gene:1", name="Gene A", labels=["Gene"]),
            Node(element_id="n2", id="disease:1", name="Disease B", labels=["Disease"]),
        ],
        edges=[
            Edge(
                element_id="r1",
                type="biolink:associated_with",
                start_element_id="n1",
                end_element_id="n2",
            )
        ],
    ).to_dict()
    context = {
        "focus_ids": ["n1", "n2"],
        "base_subgraph": {
            "nodes": [{"id": "n1", "label": "Gene A"}, {"id": "n2", "label": "Disease B"}],
            "edges": [{"id": "r1", "source": "n1", "target": "n2"}],
        },
        "semantic_subgraphs": {},
        "hidden_ids": [],
        "positions": {"n1": {"x": 1, "y": 2}, "n2": {"x": 3, "y": 4}},
        "selected_node_id": None,
    }

    details = render_path_details(path, context=context, session_store={})
    graph_wrap = details[2]
    toolbar = graph_wrap.children[2]
    legend = toolbar.children[0]
    button_ids = [button.id for button in toolbar.children[1].children]

    assert toolbar.className == "graph-toolbar"
    assert legend.className == "graph-legend"
    assert button_ids == ["clear-selection-button", "reset-view-button", "zoom-in-button", "zoom-out-button"]


def test_context_graph_disables_wheel_zoom() -> None:
    path = Path(
        id="path-1",
        nodes=[
            Node(element_id="n1", id="gene:1", name="Gene A", labels=["Gene"]),
            Node(element_id="n2", id="disease:1", name="Disease B", labels=["Disease"]),
        ],
        edges=[
            Edge(
                element_id="r1",
                type="biolink:associated_with",
                start_element_id="n1",
                end_element_id="n2",
            )
        ],
    ).to_dict()
    context = {
        "focus_ids": ["n1", "n2"],
        "base_subgraph": {
            "nodes": [{"id": "n1", "label": "Gene A"}, {"id": "n2", "label": "Disease B"}],
            "edges": [{"id": "r1", "source": "n1", "target": "n2"}],
        },
        "semantic_subgraphs": {},
        "hidden_ids": [],
        "positions": {"n1": {"x": 1, "y": 2}, "n2": {"x": 3, "y": 4}},
        "selected_node_id": None,
        "zoom": 1.25,
    }

    details = render_path_details(path, context=context, session_store={})
    graph = details[2].children[3]

    assert graph.userZoomingEnabled is False
    assert graph.zoomingEnabled is True
    assert graph.userPanningEnabled is True
    assert graph.zoom == 1.25


def test_initial_and_reset_view_fit_graph_with_padding() -> None:
    path = Path(id="path-1", nodes=[Node(element_id="n1", id="n1", name="Node", labels=[])], edges=[]).to_dict()
    context = {
        "path-1": {
            "base_subgraph": {"nodes": [{"id": "n1", "label": "Node"}], "edges": []},
            "semantic_subgraphs": {},
            "hidden_ids": [],
            "graph_revision": 2,
            "zoom": 1.5,
            "pan": {"x": 20, "y": 30},
        },
    }
    app = create_app()
    callback_id = next(
        key for key, callback in app.callback_map.items()
        if any(item["id"] == "reset-view-button" for item in callback["inputs"])
    )
    response = app.server.test_client().post(
        "/_dash-update-component",
        json={
            "output": callback_id,
            "outputs": [
                {"id": "context-store", "property": "data"},
                {"id": "status-message", "property": "children"},
            ],
            "inputs": [{"id": "reset-view-button", "property": "n_clicks", "value": 1}],
            "state": [
                {"id": "selected-path-id-store", "property": "data", "value": "path-1"},
                {"id": "candidate-paths-store", "property": "data", "value": [path]},
                {"id": "context-store", "property": "data", "value": context},
            ],
            "changedPropIds": ["reset-view-button.n_clicks"],
        },
    )

    assert response.status_code == 200
    reset_context = response.get_json()["response"]["context-store"]["data"]["path-1"]
    assert reset_context["graph_revision"] == 3
    assert "zoom" not in reset_context and "pan" not in reset_context
    for graph_context in (reset_context, {"base_subgraph": context["path-1"]["base_subgraph"]}):
        graph = render_path_details(path, graph_context, session_store={})[2].children[3]
        props = graph.to_plotly_json()["props"]
        assert graph.layout == {"name": "preset", "fit": True, "padding": 90}
        assert "zoom" not in props and "pan" not in props


def test_node_action_panel_shows_no_new_connected_nodes_message() -> None:
    path = {"nodes": [{"element_id": "n1"}], "edges": []}
    context = {
        "base_subgraph": {"nodes": [{"id": "n1", "label": "Node 1"}], "edges": []},
        "semantic_subgraphs": {},
        "hidden_ids": [],
        "node_message": "No new connected nodes to expand for the selected node.",
        "node_message_node_id": "n1",
    }

    panel = node_action_panel(path, context, selected_node_id="n1")
    messages = [child for child in panel.children if getattr(child, "className", "") == "node-inline-message"]

    assert messages[0].children == "No new connected nodes to expand for the selected node."


def test_node_action_panel_hides_stale_node_message_for_new_selection() -> None:
    path = {"nodes": [{"element_id": "n1"}, {"element_id": "n2"}], "edges": []}
    context = {
        "base_subgraph": {
            "nodes": [{"id": "n1", "label": "Node 1"}, {"id": "n2", "label": "Node 2"}],
            "edges": [],
        },
        "semantic_subgraphs": {},
        "hidden_ids": [],
        "node_message": "No new connected nodes to expand for the selected node.",
        "node_message_node_id": "n1",
    }

    panel = node_action_panel(path, context, selected_node_id="n2")
    messages = [child for child in panel.children if getattr(child, "className", "") == "node-inline-message"]

    assert messages == []


def test_node_action_panel_includes_ranked_expansion_controls() -> None:
    path = {"nodes": [{"element_id": "n1"}], "edges": []}
    context = {
        "base_subgraph": {"nodes": [{"id": "n1", "label": "Node 1"}], "edges": []},
        "semantic_subgraphs": {},
        "hidden_ids": [],
        "active_query": "genes in cancer",
        "connected_expansion_limit": 7,
        "expansion_direction": "outgoing",
        "expansion_categories": ["biolink:Gene"],
        "expansion_predicates": ["biolink:affects_response_to"],
        "neighborhood_filter_options": {
            "node_categories": ["biolink:Disease", "biolink:Gene"],
            "predicates": ["biolink:affects_response_to", "biolink:associated_with"],
        },
    }

    panel = node_action_panel(path, context, selected_node_id="n1")
    by_id = {component.id: component for component in _walk_components(panel) if getattr(component, "id", None)}

    assert "expansion-query-input" not in by_id
    assert by_id["expansion-limit-input"].value == 7
    assert by_id["expansion-direction-dropdown"].value == "outgoing"
    assert by_id["expansion-category-filter-dropdown"].multi is True
    assert by_id["expansion-category-filter-dropdown"].value == ["biolink:Gene"]
    assert by_id["expansion-category-filter-dropdown"].options == [
        {"label": "biolink:Disease", "value": "biolink:Disease"},
        {"label": "biolink:Gene", "value": "biolink:Gene"},
    ]
    assert by_id["expansion-predicate-filter-dropdown"].multi is True
    assert by_id["expansion-predicate-filter-dropdown"].value == ["biolink:affects_response_to"]
    assert by_id["expansion-predicate-filter-dropdown"].options == [
        {"label": "biolink:affects_response_to", "value": "biolink:affects_response_to"},
        {"label": "biolink:associated_with", "value": "biolink:associated_with"},
    ]
    assert by_id["robokop-summary-button"].children == "Fetch ROBOKOP summary"
    assert not any(
        str(getattr(component, "children", "")).startswith("Mode:")
        for component in _walk_components(panel)
    )


def test_expansion_filter_helpers_parse_and_bound_values() -> None:
    assert _parse_comma_separated("biolink:Gene, biolink:Disease,,") == ["biolink:Gene", "biolink:Disease"]
    assert _normalize_filter_values(["biolink:Gene", " ", "biolink:Disease"]) == [
        "biolink:Gene",
        "biolink:Disease",
    ]
    assert _normalize_filter_values("biolink:Gene, biolink:Disease,,") == ["biolink:Gene", "biolink:Disease"]
    assert _bounded_expansion_limit(None) == 12
    assert _bounded_expansion_limit("200") == 50
    assert _bounded_expansion_limit("bad") == 12
    assert _bounded_robokop_limit(None) == 10
    assert _bounded_robokop_limit("200") == 25
    assert _bounded_nonnegative_int("-10") == 0
    assert _robokop_page_offset("10", limit=5, action="robokop-next-page-button") == 15
    assert _robokop_page_offset("10", limit=5, action="robokop-prev-page-button") == 5
    assert _robokop_page_offset("2", limit=5, action="robokop-prev-page-button") == 0
    assert _robokop_page_offset("10", limit=5, action="robokop-expand-button") == 10


def test_connected_expansion_uses_selected_path_query(monkeypatch) -> None:
    received_queries = []

    class FakeGraphAdapter:
        def has_unseen_neighbors(self, node_id, visible_node_ids):
            return True

        def context_subgraph(self, focus_ids, **kwargs):
            received_queries.append(kwargs["query"])
            return {"nodes": [{"id": "n1", "label": "PTEN"}], "edges": []}

        def close(self):
            pass

    monkeypatch.setattr("explorer.frontend.callbacks.Neo4jGraphAdapter", FakeGraphAdapter)
    app = create_app()
    callback_id = next(
        key for key, callback in app.callback_map.items()
        if any(item["id"] == "expand-connected-button" for item in callback["inputs"])
    )
    assert "expansion-query-input" not in {item["id"] for item in app.callback_map[callback_id]["state"] if isinstance(item["id"], str)}
    path = {"id": "path-1", "source_query": "genes in cancer", "nodes": [{"element_id": "n1"}], "edges": []}
    context = {
        "path-1": {
            "selected_node_id": "n1",
            "focus_ids": ["n1"],
            "hidden_ids": [],
            "base_subgraph": {"nodes": [{"id": "n1", "label": "PTEN"}], "edges": []},
            "active_query": "old refined query",
        }
    }
    response = app.server.test_client().post(
        "/_dash-update-component",
        json={
            "output": callback_id,
            "outputs": [
                {"id": "context-store", "property": "data"},
                {"id": "status-message", "property": "children"},
            ],
            "inputs": [
                {"id": "expand-connected-button", "property": "n_clicks", "value": 1},
                {"id": "expand-similar-button", "property": "n_clicks", "value": None},
                {"id": "collapse-node-button", "property": "n_clicks", "value": None},
                [],
            ],
            "state": [
                {"id": "selected-path-id-store", "property": "data", "value": "path-1"},
                {"id": "candidate-paths-store", "property": "data", "value": [path]},
                {"id": "context-store", "property": "data", "value": context},
                [],
                [],
                {"id": "query-input", "property": "value", "value": "other current query"},
                {"id": "expansion-direction-dropdown", "property": "value", "value": "either"},
                {"id": "expansion-limit-input", "property": "value", "value": 5},
                {"id": "expansion-category-filter-dropdown", "property": "value", "value": []},
                {"id": "expansion-predicate-filter-dropdown", "property": "value", "value": []},
            ],
            "changedPropIds": ["expand-connected-button.n_clicks"],
        },
    )

    assert response.status_code == 200
    assert received_queries == ["genes in cancer"]
    expanded = response.get_json()["response"]["context-store"]["data"]["path-1"]
    assert expanded["active_query"] == "genes in cancer"
    assert "expansion_query" not in expanded


def test_node_action_panel_renders_robokop_summary_and_edges() -> None:
    path = {"nodes": [{"element_id": "n1"}], "edges": []}
    context = {
        "base_subgraph": {
            "nodes": [{"id": "n1", "curie": "NCBIGene:5728", "label": "PTEN"}],
            "edges": [],
        },
        "semantic_subgraphs": {},
        "robokop_subgraphs": {},
        "hidden_ids": [],
        "robokop_state": {
            "n1": {
                "provider_mode": "fixture",
                "bridge": {"status": "resolved", "curie": "NCBIGene:5728"},
                "summary": {
                    "total_edges": 4,
                    "items": [{"predicate": "biolink:affects", "category": "biolink:Drug", "count": 4}],
                    "categories": ["biolink:Drug"],
                    "predicates": ["biolink:affects"],
                },
                "expansion": {
                    "config": {
                        "category": "biolink:Drug",
                        "predicate": "biolink:affects",
                        "direction": "outgoing",
                        "limit": 3,
                        "offset": 2,
                    },
                    "edges": [
                        {
                            "edge_id": "robokop:e1",
                            "subject_curie": "NCBIGene:5728",
                            "object_curie": "DRUGBANK:DB00515",
                            "predicate": "biolink:affects",
                            "primary_knowledge_source": "infores:test",
                            "score": 0.42,
                            "ranking_reasons": ["keyword match"],
                        }
                    ],
                },
                "selected_edge_ids": [],
            },
        },
    }

    panel = node_action_panel(path, context, selected_node_id="n1")
    by_id = {str(component.id): component for component in _walk_components(panel) if getattr(component, "id", None)}

    assert by_id["robokop-category-filter-dropdown"].options == [
        {"label": "biolink:Drug", "value": "biolink:Drug"}
    ]
    assert by_id["robokop-predicate-filter-dropdown"].options == [
        {"label": "biolink:affects", "value": "biolink:affects"}
    ]
    assert by_id["robokop-direction-dropdown"].value == "outgoing"
    assert by_id["robokop-limit-input"].value == 3
    assert by_id["robokop-offset-input"].value == 2
    assert by_id["robokop-expand-button"].children == "Fetch ROBOKOP edges"
    assert by_id["robokop-prev-page-button"].children == "Previous"
    assert by_id["robokop-next-page-button"].children == "Next"
    assert any(
        getattr(component, "children", None) == "Fixture data (not live)"
        for component in _walk_components(panel)
    )
    assert not any(
        str(getattr(component, "children", "")).startswith("Mode:")
        for component in _walk_components(panel)
    )
    controls = next(
        component for component in _walk_components(panel)
        if getattr(component, "className", None) == "robokop-controls"
    )
    fetch_index = next(
        index for index, child in enumerate(controls.children)
        if getattr(child, "className", None) == "button-row"
        and by_id["robokop-expand-button"] in child.children
    )
    results_index = next(
        index for index, child in enumerate(controls.children)
        if getattr(child, "className", None) == "robokop-results"
    )
    pagination = controls.children[-1]
    assert fetch_index < results_index < len(controls.children) - 1
    assert pagination.children == [
        by_id["robokop-prev-page-button"], by_id["robokop-next-page-button"]
    ]
    assert pagination.style == {}
    assert (
        "ROBOKOP returned 1 edge(s); displaying 1 candidate card(s) at offset 2 with limit 3 "
        "(category biolink:Drug; predicate biolink:affects; direction outgoing)."
    ) in [
        getattr(component, "children", None) for component in _walk_components(panel)
    ]
    assert any(
        getattr(component, "children", None) == "Add"
        for component in _walk_components(panel)
    )
    context["robokop_state"]["n1"]["selected_edge_ids"] = ["robokop:e1"]
    selected_panel = node_action_panel(path, context, selected_node_id="n1")
    toggle = next(
        component for component in _walk_components(selected_panel)
        if isinstance(getattr(component, "id", None), dict)
        and component.id.get("type") == "toggle-robokop-edge"
    )
    assert toggle.children == "Remove"
    assert toggle.disabled is False
    remote_results = [
        component for component in _walk_components(panel)
        if getattr(component, "className", None) == "robokop-results"
    ]
    assert len(remote_results) == 1
    assert remote_results[0].open is True


def test_remote_edge_button_toggles_graph_edge() -> None:
    app = create_app()
    callback_id = next(
        key for key, callback in app.callback_map.items()
        if any('"type":"toggle-robokop-edge"' in item["id"] for item in callback["inputs"])
    )
    remote_edge = {
        "edge_id": "robokop:e1",
        "adjacent_curie": "DRUGBANK:DB00515",
        "direction": "outgoing",
        "predicate": "biolink:affects",
    }
    path = {"id": "path-1", "nodes": [{"element_id": "n1"}], "edges": []}
    context_store = {
        "path-1": {
            "selected_node_id": "n1",
            "positions": {},
            "base_subgraph": {"nodes": [{"id": "n1"}], "edges": []},
            "robokop_subgraphs": {},
            "robokop_state": {"n1": {"expansion": {"edges": [remote_edge]}, "selected_edge_ids": []}},
        },
    }
    button_id = {"edge_id": "robokop:e1", "type": "toggle-robokop-edge"}

    def click(current_context: dict) -> tuple[dict, str]:
        response = app.server.test_client().post(
            "/_dash-update-component",
            json={
                "output": callback_id,
                "outputs": [
                    {"id": "context-store", "property": "data"},
                    {"id": "status-message", "property": "children"},
                ],
                "inputs": [
                    {"id": "expand-connected-button", "property": "n_clicks", "value": None},
                    {"id": "expand-similar-button", "property": "n_clicks", "value": None},
                    {"id": "collapse-node-button", "property": "n_clicks", "value": None},
                    [{"id": button_id, "property": "n_clicks", "value": 1}],
                ],
                "state": [
                    {"id": "selected-path-id-store", "property": "data", "value": "path-1"},
                    {"id": "candidate-paths-store", "property": "data", "value": [path]},
                    {"id": "context-store", "property": "data", "value": current_context},
                    [],
                    [],
                    {"id": "query-input", "property": "value", "value": None},
                    {"id": "expansion-direction-dropdown", "property": "value", "value": None},
                    {"id": "expansion-limit-input", "property": "value", "value": None},
                    {"id": "expansion-category-filter-dropdown", "property": "value", "value": None},
                    {"id": "expansion-predicate-filter-dropdown", "property": "value", "value": None},
                ],
                "changedPropIds": [f"{json.dumps(button_id, sort_keys=True, separators=(',', ':'))}.n_clicks"],
            },
        )
        assert response.status_code == 200
        result = response.get_json()["response"]
        return result["context-store"]["data"], result["status-message"]["children"]["props"]["children"]

    added_context, added_message = click(context_store)
    assert added_context["path-1"]["robokop_state"]["n1"]["selected_edge_ids"] == ["robokop:e1"]
    assert "Added selected" in added_message

    removed_context, removed_message = click(added_context)
    assert removed_context["path-1"]["robokop_state"]["n1"]["selected_edge_ids"] == []
    assert removed_context["path-1"]["robokop_subgraphs"] == {}
    assert "Removed selected" in removed_message

    readded_context, _ = click(removed_context)
    assert readded_context["path-1"]["robokop_state"]["n1"]["selected_edge_ids"] == ["robokop:e1"]


def test_robokop_summary_refresh_keeps_added_edge_removable(monkeypatch) -> None:
    monkeypatch.setattr(
        "explorer.frontend.callbacks.robokop_provider_from_env", RobokopFixtureProvider
    )
    app = create_app()
    callback_id = next(
        key for key, callback in app.callback_map.items()
        if any(item["id"] == "robokop-summary-button" for item in callback["inputs"])
    )
    path = {
        "id": "path-1",
        "nodes": [{"element_id": "n1", "id": "NCBIGene:5728", "name": "PTEN", "labels": ["biolink:Gene"]}],
        "edges": [],
    }
    current_context = {
        "path-1": {
            "selected_node_id": "n1",
            "base_subgraph": {"nodes": [{"id": "n1"}], "edges": []},
            "robokop_subgraphs": {"n1": {"nodes": [], "edges": [{"id": "robokop:e1"}]}},
            "robokop_state": {"n1": {"selected_edge_ids": ["robokop:e1"]}},
        },
    }
    response = app.server.test_client().post(
        "/_dash-update-component",
        json={
            "output": callback_id,
            "outputs": [
                {"id": "context-store", "property": "data"},
                {"id": "status-message", "property": "children"},
            ],
            "inputs": [{"id": "robokop-summary-button", "property": "n_clicks", "value": 1}],
            "state": [
                {"id": "selected-path-id-store", "property": "data", "value": "path-1"},
                {"id": "candidate-paths-store", "property": "data", "value": [path]},
                {"id": "context-store", "property": "data", "value": current_context},
                [],
            ],
            "changedPropIds": ["robokop-summary-button.n_clicks"],
        },
    )

    assert response.status_code == 200
    refreshed = response.get_json()["response"]["context-store"]["data"]["path-1"]
    assert refreshed["robokop_state"]["n1"]["selected_edge_ids"] == ["robokop:e1"]
    assert refreshed["robokop_subgraphs"] == current_context["path-1"]["robokop_subgraphs"]


def test_robokop_page_uses_selected_path_query(monkeypatch) -> None:
    provider = RobokopFixtureProvider()
    received_queries = []
    incident_edges = provider.incident_edges

    def record_query(curie, config):
        received_queries.append(config.query)
        return incident_edges(curie, config)

    monkeypatch.setattr(provider, "incident_edges", record_query)
    monkeypatch.setattr("explorer.frontend.callbacks.robokop_provider_from_env", lambda: provider)
    app = create_app()
    callback_id = next(
        key for key, callback in app.callback_map.items()
        if any(item["id"] == "robokop-expand-button" for item in callback["inputs"])
    )
    assert "expansion-query-input" not in {item["id"] for item in app.callback_map[callback_id]["state"] if isinstance(item["id"], str)}
    path = {
        "id": "path-1",
        "source_query": "genes in cancer",
        "nodes": [{"element_id": "n1", "id": "NCBIGene:5728", "name": "PTEN", "labels": ["biolink:Gene"]}],
        "edges": [],
    }
    context = {
        "path-1": {
            "selected_node_id": "n1",
            "active_query": "old refined query",
            "base_subgraph": {"nodes": [{"id": "n1", "curie": "NCBIGene:5728"}], "edges": []},
        }
    }
    response = app.server.test_client().post(
        "/_dash-update-component",
        json={
            "output": callback_id,
            "outputs": [
                {"id": "context-store", "property": "data"},
                {"id": "status-message", "property": "children"},
            ],
            "inputs": [
                {"id": "robokop-expand-button", "property": "n_clicks", "value": 1},
                {"id": "robokop-prev-page-button", "property": "n_clicks", "value": None},
                {"id": "robokop-next-page-button", "property": "n_clicks", "value": None},
            ],
            "state": [
                {"id": "selected-path-id-store", "property": "data", "value": "path-1"},
                {"id": "candidate-paths-store", "property": "data", "value": [path]},
                {"id": "context-store", "property": "data", "value": context},
                [],
                {"id": "query-input", "property": "value", "value": "other current query"},
                {"id": "robokop-category-filter-dropdown", "property": "value", "value": None},
                {"id": "robokop-predicate-filter-dropdown", "property": "value", "value": None},
                {"id": "robokop-direction-dropdown", "property": "value", "value": "either"},
                {"id": "robokop-limit-input", "property": "value", "value": 5},
                {"id": "robokop-offset-input", "property": "value", "value": 0},
            ],
            "changedPropIds": ["robokop-expand-button.n_clicks"],
        },
    )

    assert response.status_code == 200
    assert received_queries == ["genes in cancer"]
    expansion = response.get_json()["response"]["context-store"]["data"]["path-1"]["robokop_state"]["n1"]["expansion"]
    assert expansion["config"]["query"] == "genes in cancer"


def test_node_action_panel_shows_empty_robokop_summary() -> None:
    path = {"nodes": [{"element_id": "n1"}], "edges": []}
    context = {
        "base_subgraph": {
            "nodes": [{"id": "n1", "curie": "UMLS:C0596290", "label": "Cell Proliferation"}],
            "edges": [],
        },
        "semantic_subgraphs": {},
        "robokop_subgraphs": {},
        "hidden_ids": [],
        "robokop_state": {
            "n1": {
                "provider_mode": "live",
                "bridge": {"status": "resolved", "curie": "UMLS:C0596290"},
                "summary": {
                    "total_edges": 0,
                    "items": [],
                    "categories": [],
                    "predicates": [],
                },
            },
        },
    }

    panel = node_action_panel(path, context, selected_node_id="n1")
    text = [getattr(component, "children", None) for component in _walk_components(panel)]

    assert "Summary: 0 incident edges across 0 predicate/category groups." in text
    assert "No ROBOKOP incident edges were reported for this bridge CURIE." in text
    assert "Fixture data (not live)" not in text
    assert not any(isinstance(item, str) and item.startswith("Mode:") for item in text)
    assert not any(
        getattr(component, "id", None) == "robokop-prev-page-button"
        for component in _walk_components(panel)
    )


def test_node_action_panel_shows_empty_robokop_page() -> None:
    path = {"nodes": [{"element_id": "n1"}], "edges": []}
    context = {
        "base_subgraph": {
            "nodes": [{"id": "n1", "curie": "NCBIGene:5728", "label": "PTEN"}],
            "edges": [],
        },
        "semantic_subgraphs": {},
        "robokop_subgraphs": {},
        "hidden_ids": [],
        "robokop_state": {
            "n1": {
                "provider_mode": "live",
                "bridge": {"status": "resolved", "curie": "NCBIGene:5728"},
                "summary": {
                    "total_edges": 4,
                    "items": [{"predicate": "biolink:affects", "category": "biolink:Drug", "count": 4}],
                    "categories": ["biolink:Drug"],
                    "predicates": ["biolink:affects"],
                },
                "expansion": {
                    "config": {
                        "category": "biolink:Drug",
                        "predicate": "biolink:affects",
                        "direction": "incoming",
                        "limit": 3,
                        "offset": 0,
                    },
                    "edges": [],
                    "total_available": 3,
                    "diagnostics": {"direction_filtered_count": 3},
                },
            },
        },
    }

    panel = node_action_panel(path, context, selected_node_id="n1")
    text = [getattr(component, "children", None) for component in _walk_components(panel)]

    assert (
        "ROBOKOP returned 3 of 3 edge(s); displaying 0 candidate card(s) at offset 0 with limit 3 "
        "(category biolink:Drug; predicate biolink:affects; direction incoming). "
        "3 edge(s) were hidden by the direction filter; set direction to Either to inspect them. "
        "Try changing category, predicate, direction, or offset, then fetch again."
    ) in text
    controls = next(
        component for component in _walk_components(panel)
        if getattr(component, "className", None) == "robokop-controls"
    )
    assert controls.children[-1].style == {}
    assert [button.children for button in controls.children[-1].children] == ["Previous", "Next"]


def test_robokop_pagination_hidden_until_edges_are_fetched() -> None:
    path = {"nodes": [{"element_id": "n1"}], "edges": []}
    context = {
        "base_subgraph": {"nodes": [{"id": "n1", "curie": "NCBIGene:5728", "label": "PTEN"}], "edges": []},
        "semantic_subgraphs": {},
        "hidden_ids": [],
        "robokop_state": {
            "n1": {
                "summary": {
                    "total_edges": 4,
                    "items": [{"predicate": "biolink:affects", "category": "biolink:Drug", "count": 4}],
                    "categories": ["biolink:Drug"],
                    "predicates": ["biolink:affects"],
                },
            },
        },
    }

    panel = node_action_panel(path, context, selected_node_id="n1")
    controls = next(
        component for component in _walk_components(panel)
        if getattr(component, "className", None) == "robokop-controls"
    )
    buttons = {
        component.id: component for component in _walk_components(controls)
        if getattr(component, "id", None)
    }

    assert buttons["robokop-expand-button"].children == "Fetch ROBOKOP edges"
    assert controls.children[-1].style == {"display": "none"}
    assert [button.children for button in controls.children[-1].children] == ["Previous", "Next"]


def test_robokop_summary_callback_does_not_require_post_summary_controls() -> None:
    app = create_app()
    summary_callbacks = [
        metadata
        for metadata in app.callback_map.values()
        if any(item["id"] == "robokop-summary-button" for item in metadata.get("inputs", []))
    ]

    assert len(summary_callbacks) == 1
    state_ids = {item["id"] for item in summary_callbacks[0]["state"]}
    assert "robokop-category-filter-dropdown" not in state_ids
    assert "robokop-predicate-filter-dropdown" not in state_ids
    assert "robokop-direction-dropdown" not in state_ids
    assert "robokop-limit-input" not in state_ids
    assert "robokop-offset-input" not in state_ids


def test_robokop_subgraph_is_merged_and_rendered_as_remote() -> None:
    path = {
        "id": "path-1",
        "nodes": [{"element_id": "n1"}],
        "edges": [],
    }
    context = {
        "base_subgraph": {"nodes": [{"id": "n1", "label": "PTEN"}], "edges": []},
        "semantic_subgraphs": {},
        "robokop_subgraphs": {},
        "hidden_ids": [],
        "positions": {},
        "focus_ids": ["n1"],
    }
    _add_robokop_edge_to_context(
        context,
        "n1",
        {
            "edge_id": "robokop:e1",
            "adjacent_curie": "DRUGBANK:DB00515",
            "direction": "outgoing",
            "object_name": "cisplatin",
            "object_categories": ["biolink:Drug"],
            "predicate": "biolink:affects",
            "primary_knowledge_source": "infores:test",
        },
    )

    subgraph = visible_subgraph(path, context)
    elements = cytoscape_elements(path, subgraph, context)
    classes_by_id = {element["data"]["id"]: element.get("classes", "") for element in elements}

    assert "remote-node" in classes_by_id["robokop:DRUGBANK:DB00515"]
    assert "remote-edge" in classes_by_id["robokop:e1"]


def test_expanded_focus_nodes_are_green_unless_on_initial_path() -> None:
    path = {
        "id": "path-1",
        "nodes": [{"element_id": "n1"}, {"element_id": "n2"}],
        "edges": [{"element_id": "r1"}],
    }
    subgraph = {
        "nodes": [
            {"id": "n1", "label": "Initial path node"},
            {"id": "n2", "label": "Initial endpoint"},
            {"id": "n3", "label": "Expanded focus"},
        ],
        "edges": [
            {"id": "r1", "source": "n1", "target": "n2"},
            {"id": "r2", "source": "n2", "target": "n3"},
        ],
    }
    context = {
        "positions": {},
        "focus_ids": ["n1", "n2", "n3"],
        "selected_node_id": None,
    }

    elements = cytoscape_elements(path, subgraph, context)
    classes_by_id = {
        element["data"]["id"]: element.get("classes", "")
        for element in elements
        if "source" not in element["data"]
    }

    assert classes_by_id["n1"] == "path-node"
    assert classes_by_id["n3"] == "focus-node"


def test_exploration_stores_are_not_browser_persistent() -> None:
    layout = build_layout()
    stores = {component.id: component for component in layout.children if getattr(component, "id", None)}

    assert stores["candidate-paths-store"].storage_type == "memory"
    assert stores["candidate-list-expanded"].storage_type == "memory"
    assert stores["selected-path-id-store"].storage_type == "memory"
    assert stores["context-store"].storage_type == "memory"
    assert stores["session-store"].storage_type == "memory"


def test_candidate_preview_keeps_selected_path_visible() -> None:
    paths = [{"id": f"path-{index}", "summary": f"Path {index}"} for index in range(1, 8)]
    preview_ids = [f"path-{index}" for index in range(1, CANDIDATE_PREVIEW_LIMIT)] + ["path-7"]

    preview = render_candidate_path_cards(paths, "path-7")
    expanded = render_candidate_path_cards(paths, "path-7", expanded=True)

    assert [card.id["path_id"] for card in preview] == preview_ids
    assert preview[-1].children[0].children[0].children == "Path 7"
    assert preview[-1].className.endswith("selected")
    assert [card.id["path_id"] for card in expanded[:CANDIDATE_PREVIEW_LIMIT]] == preview_ids
    overflow = expanded[CANDIDATE_PREVIEW_LIMIT]
    assert overflow.className == "path-list-overflow"
    assert [card.id["path_id"] for card in overflow.children] == [
        path["id"] for path in paths if path["id"] not in preview_ids
    ]
    first_page = paths[:CANDIDATE_PREVIEW_LIMIT]
    assert [card.id["path_id"] for card in render_candidate_path_cards(first_page, first_page[-1]["id"])] == [
        path["id"] for path in first_page
    ]


def test_candidate_cards_show_endpoint_node_ids_only_as_metadata() -> None:
    path = Path(
        id="path-1",
        nodes=[
            Node(element_id="n1", id="NCBIGene:5728", name="PTEN"),
            Node(element_id="n2", id="MONDO:0007254", name="Breast cancer"),
        ],
        edges=[
            Edge(
                element_id="r1",
                type="biolink:associated_with",
                start_element_id="n1",
                end_element_id="n2",
            )
        ],
        seed_predicate="biolink:related_to",
        anchor_metadata={"ranking_reasons": ["keyword match", "anchor reason"]},
    ).to_dict()

    card = render_candidate_path_cards([path], path["id"])[0]

    assert card.children[1].children == path["summary"]
    assert card.children[2].children == "Subject ID: NCBIGene:5728 | Object ID: MONDO:0007254"
    assert len(card.children) == 3


def test_node_actions_live_in_sidebar_without_path_composition() -> None:
    layout = build_layout()
    panels = [component for component in _walk_components(layout) if getattr(component, "id", None) == "node-action-panel"]
    assert len(panels) == 1

    path = Path(id="path-1", nodes=[Node(element_id="n1", id="n1", name="Node", labels=[])], edges=[]).to_dict()
    details = render_path_details(path, context=None, session_store={})

    assert len(details) == 4
    assert details[1].children[0].children == "Semantic evidence"
    assert details[1].open is False
    assert not any(getattr(component, "children", None) == "Path composition" for component in _walk_components(details))
    assert not any(getattr(component, "id", None) == "node-action-panel" for component in _walk_components(details[2]))


def test_node_actions_render_from_saved_selection_after_graph_redraw() -> None:
    app = create_app()
    callback = app.callback_map["node-action-panel.children"]
    assert {item["id"] for item in callback["inputs"]} == {
        "context-store", "selected-path-id-store",
    }

    path = {"id": "path-1", "nodes": [{"element_id": "n1"}], "edges": []}
    context = {
        "path-1": {
            "selected_node_id": "n1",
            "base_subgraph": {"nodes": [{"id": "n1", "label": "Node 1"}], "edges": []},
        },
    }
    response = app.server.test_client().post(
        "/_dash-update-component",
        json={
            "output": "node-action-panel.children",
            "outputs": {"id": "node-action-panel", "property": "children"},
            "inputs": [
                {"id": "context-store", "property": "data", "value": context},
                {"id": "selected-path-id-store", "property": "data", "value": "path-1"},
            ],
            "state": [{"id": "candidate-paths-store", "property": "data", "value": [path]}],
            "changedPropIds": ["context-store.data"],
        },
    )

    assert response.status_code == 200
    assert "Node 1" in str(response.get_json()["response"]["node-action-panel"])


def test_unclicked_candidate_cards_do_not_reset_selection() -> None:
    app = create_app()
    output = next(key for key in app.callback_map if key.startswith("selected-path-id-store.data"))
    wildcard_id = json.dumps({"path_id": ["ALL"], "type": "select-path"}, sort_keys=True, separators=(",", ":"))
    clicked_id = json.dumps({"path_id": "path-1", "type": "select-path"}, sort_keys=True, separators=(",", ":"))

    def select(counts: list[int]):
        return app.server.test_client().post(
            "/_dash-update-component",
            json={
                "output": output,
                "outputs": {"id": "selected-path-id-store", "property": "data"},
                "inputs": [{"id": wildcard_id, "property": "n_clicks", "value": counts}],
                "state": [],
                "changedPropIds": [f"{clicked_id}.n_clicks"],
            },
        )

    assert select([0, 0]).status_code == 204
    clicked = select([1, 0])
    assert clicked.status_code == 200
    assert clicked.get_json()["response"]["selected-path-id-store"]["data"] == "path-1"


def test_candidate_list_toggle_resets_on_new_search() -> None:
    app = create_app()
    client = app.server.test_client()

    def toggle(changed: str, expanded: bool):
        return client.post(
            "/_dash-update-component",
            json={
                "output": "candidate-list-expanded.data",
                "outputs": {"id": "candidate-list-expanded", "property": "data"},
                "inputs": [
                    {"id": "toggle-candidate-paths", "property": "n_clicks", "value": 1},
                    {"id": "candidate-paths-store", "property": "data", "value": [{"id": "path-1"}]},
                ],
                "state": [{"id": "candidate-list-expanded", "property": "data", "value": expanded}],
                "changedPropIds": [changed],
            },
        )

    shown = toggle("toggle-candidate-paths.n_clicks", False)
    assert shown.status_code == 200
    assert shown.get_json()["response"]["candidate-list-expanded"]["data"] is True

    reset = toggle("candidate-paths-store.data", True)
    assert reset.status_code == 200
    assert reset.get_json()["response"]["candidate-list-expanded"]["data"] is False


def _walk_components(component):
    yield component
    children = getattr(component, "children", None)
    if children is None:
        return
    if not isinstance(children, list):
        children = [children]
    for child in children:
        if hasattr(child, "children") or hasattr(child, "id"):
            yield from _walk_components(child)
