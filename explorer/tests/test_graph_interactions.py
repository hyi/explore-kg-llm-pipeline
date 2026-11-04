from __future__ import annotations

import json
from copy import deepcopy

import httpx
import pytest

from explorer.backend.external_graph import (
    RobokopFixtureProvider,
    RobokopHttpConfig,
    RobokopHttpProvider,
    RobokopProviderError,
    RobokopProviderTimeout,
)
from explorer.frontend.components import render_path_details
from explorer.frontend.dash_app import create_app
from explorer.frontend.graph import (
    cytoscape_elements,
    node_action_panel,
    robokop_has_next_page,
    robokop_page_message,
    seed_new_positions,
)

GRAPH_ID = {"type": "context-graph", "path_id": "path-1", "revision": 0}


@pytest.fixture
def remote_provider(monkeypatch):
    class RecordingProvider(RobokopFixtureProvider):
        def __init__(self):
            super().__init__()
            self.calls = []

        def lookup_node(self, curie):
            self.calls.append("node")
            return super().lookup_node(curie)

        def edge_summary(self, curie):
            self.calls.append("summary")
            return super().edge_summary(curie)

        def incident_edges(self, curie, config):
            self.calls.append("edges")
            return super().incident_edges(curie, config)

        def close(self):
            self.calls.append("close")

    provider = RecordingProvider()
    monkeypatch.setattr("explorer.frontend.callbacks.robokop_provider_from_env", lambda: provider)
    return provider


@pytest.fixture
def graph_context() -> dict:
    return {
        "path-1": {
            "selected_node_id": "n1",
            "focus_ids": ["n1"],
            "positions": {"n1": {"x": 100, "y": 100}},
            "base_subgraph": {"nodes": [{"id": "n1", "label": "Apoptosis"}], "edges": []},
            "semantic_subgraphs": {},
            "robokop_subgraphs": {},
            "hidden_ids": [],
            "graph_revision": 0,
        },
    }


@pytest.fixture
def path() -> dict:
    return {"id": "path-1", "nodes": [{"element_id": "n1"}], "edges": []}


@pytest.fixture
def semantic_subgraph() -> dict:
    return {
        "nodes": [{"id": "n1"}, {"id": "n2"}],
        "edges": [{"id": "semantic:n1:n2", "source": "n1", "target": "n2", "kind": "semantic"}],
    }


def _graph_prop(property_name: str, value: object) -> list[dict]:
    return [{"id": GRAPH_ID, "property": property_name, "value": value}]


def _graph_prop_id(property_name: str, graph_id: dict | None = None) -> str:
    return f"{json.dumps(graph_id or GRAPH_ID, sort_keys=True, separators=(',', ':'))}.{property_name}"


def _remember_graph(app, context: dict, selected: list[dict], elements: list[dict], changed: str):
    callback_id = next(
        key for key, metadata in app.callback_map.items()
        if metadata.get("callback") and metadata["callback"].__name__ == "remember_selected_graph_node"
    )
    return app.server.test_client().post(
        "/_dash-update-component",
        json={
            "output": callback_id,
            "outputs": {"id": "context-store", "property": "data"},
            "inputs": [_graph_prop("selectedNodeData", selected), _graph_prop("elements", elements)],
            "state": [
                {"id": "selected-path-id-store", "property": "data", "value": "path-1"},
                {"id": "context-store", "property": "data", "value": context},
            ],
            "changedPropIds": [changed],
        },
    )


def _expand_semantically(app, path: dict, context: dict):
    callback_id = next(
        key for key, metadata in app.callback_map.items()
        if any(item["id"] == "expand-similar-button" for item in metadata["inputs"])
    )
    return app.server.test_client().post(
        "/_dash-update-component",
        json={
            "output": callback_id,
            "outputs": [
                {"id": "context-store", "property": "data"},
                {"id": "status-message", "property": "children"},
            ],
            "inputs": [
                {"id": "expand-connected-button", "property": "n_clicks", "value": None},
                {"id": "expand-similar-button", "property": "n_clicks", "value": 1},
                {"id": "collapse-node-button", "property": "n_clicks", "value": None},
                [],
            ],
            "state": [
                {"id": "selected-path-id-store", "property": "data", "value": "path-1"},
                {"id": "candidate-paths-store", "property": "data", "value": [path]},
                {"id": "context-store", "property": "data", "value": context},
                [],
                [],
                {"id": "query-input", "property": "value", "value": None},
                {"id": "expansion-direction-dropdown", "property": "value", "value": None},
                {"id": "expansion-limit-input", "property": "value", "value": None},
                {"id": "expansion-category-filter-dropdown", "property": "value", "value": None},
                {"id": "expansion-predicate-filter-dropdown", "property": "value", "value": None},
            ],
            "changedPropIds": ["expand-similar-button.n_clicks"],
        },
    )


def _fetch_remote(
    app, path: dict, context: dict, *, action="robokop-expand-button", curie="NCBIGene:5728", limit=2
):
    path = deepcopy(path)
    path["nodes"][0]["id"] = curie
    callback_id = next(
        key for key, metadata in app.callback_map.items()
        if any(item["id"] == "robokop-expand-button" for item in metadata["inputs"])
    )
    return app.server.test_client().post(
        "/_dash-update-component",
        json={
            "output": callback_id,
            "outputs": [
                {"id": "context-store", "property": "data"},
                {"id": "status-message", "property": "children"},
            ],
            "inputs": [
                {"id": button_id, "property": "n_clicks", "value": 1 if button_id == action else None}
                for button_id in ("robokop-expand-button", "robokop-prev-page-button", "robokop-next-page-button")
            ],
            "state": [
                {"id": "selected-path-id-store", "property": "data", "value": "path-1"},
                {"id": "candidate-paths-store", "property": "data", "value": [path]},
                {"id": "context-store", "property": "data", "value": context},
                [],
                {"id": "query-input", "property": "value", "value": "cancer drug response"},
                {"id": "robokop-limit-input", "property": "value", "value": limit},
            ],
            "changedPropIds": [f"{action}.n_clicks"],
        },
    )


def test_remote_fetch_requests_summary_once_and_one_page_per_click(
    remote_provider, graph_context, path
) -> None:
    app = create_app()
    response = _fetch_remote(app, path, graph_context)

    assert response.status_code == 200
    assert remote_provider.calls == ["summary", "edges", "close"]
    fetched = response.get_json()["response"]["context-store"]["data"]
    state = fetched["path-1"]["robokop_state"]["n1"]
    assert state["summary"]["total_edges"] > 0
    assert state["summary"]["categories"]
    assert state["summary"]["predicates"]
    assert len(state["expansion"]["edges"]) == 2
    assert state["expansion"]["config"]["offset"] == 0

    remote_provider.calls.clear()
    response = _fetch_remote(app, path, fetched, action="robokop-next-page-button")
    assert response.status_code == 200
    assert remote_provider.calls == ["edges", "close"]
    paged = response.get_json()["response"]["context-store"]["data"]
    assert paged["path-1"]["robokop_state"]["n1"]["summary"] == state["summary"]
    assert paged["path-1"]["robokop_state"]["n1"]["expansion"]["config"]["offset"] == 2

    remote_provider.calls.clear()
    response = _fetch_remote(app, path, paged)
    assert response.status_code == 200
    assert remote_provider.calls == ["edges", "close"]
    assert response.get_json()["response"]["context-store"]["data"]["path-1"]["robokop_state"]["n1"]["expansion"]["config"]["offset"] == 0


def test_cached_summary_does_not_require_another_summary_request(remote_provider, graph_context, path) -> None:
    graph_context["path-1"]["robokop_state"] = {"n1": {"summary": {"total_edges": 10000}}}
    response = _fetch_remote(create_app(), path, graph_context)

    assert response.status_code == 200
    assert remote_provider.calls == ["edges", "close"]
    state = response.get_json()["response"]["context-store"]["data"]["path-1"]["robokop_state"]["n1"]
    assert state["summary"]["total_edges"] == 10000
    assert len(state["expansion"]["edges"]) == 2
    assert state["provider_mode"] == "fixture"


def test_summary_timeout_does_not_block_pages_or_repeat_summary(remote_provider, graph_context, path, monkeypatch):
    def fail_summary(curie):
        remote_provider.calls.append("summary")
        raise RobokopProviderTimeout("Summary timed out")

    monkeypatch.setattr(remote_provider, "edge_summary", fail_summary)
    app = create_app()
    response = _fetch_remote(app, path, graph_context)
    assert response.status_code == 200
    fetched = response.get_json()["response"]["context-store"]["data"]
    state = fetched["path-1"]["robokop_state"]["n1"]
    assert state["summary_requested"] is True
    assert state["summary_error"] == "Summary timed out"
    assert len(state["expansion"]["edges"]) == 2

    response = _fetch_remote(app, path, fetched, action="robokop-next-page-button")
    assert response.status_code == 200
    assert remote_provider.calls == ["summary", "edges", "close", "edges", "close"]


def test_changing_robokop_page_size_restarts_at_first_page(remote_provider, graph_context, path):
    app = create_app()
    response = _fetch_remote(app, path, graph_context)
    fetched = response.get_json()["response"]["context-store"]["data"]
    response = _fetch_remote(app, path, fetched, action="robokop-next-page-button")
    paged = response.get_json()["response"]["context-store"]["data"]
    response = _fetch_remote(app, path, paged, action="robokop-next-page-button", limit=5)
    assert response.status_code == 200
    config = response.get_json()["response"]["context-store"]["data"]["path-1"]["robokop_state"]["n1"]["expansion"]["config"]
    assert config["offset"] == 0
    assert config["limit"] == 5
    assert remote_provider.calls.count("summary") == 1


def test_combined_remote_fetch_failure_preserves_added_edges(
    remote_provider, graph_context, path, monkeypatch
) -> None:
    def fail(*args):
        raise RobokopProviderError("Remote service unavailable")

    monkeypatch.setattr(remote_provider, "incident_edges", fail)
    graph_context["path-1"]["robokop_state"] = {"n1": {"selected_edge_ids": ["robokop:e1"]}}
    graph_context["path-1"]["robokop_subgraphs"] = {
        "n1": {"nodes": [{"id": "remote:1"}], "edges": [{"id": "robokop:e1"}]},
    }
    response = _fetch_remote(create_app(), path, graph_context)

    assert response.status_code == 200
    result = response.get_json()["response"]
    context = result["context-store"]["data"]["path-1"]
    assert context["robokop_subgraphs"] == graph_context["path-1"]["robokop_subgraphs"]
    assert context["robokop_state"]["n1"]["selected_edge_ids"] == ["robokop:e1"]
    assert context["robokop_state"]["n1"]["error"] == "Remote service unavailable"
    assert remote_provider.calls[-1] == "close"
    assert "error" in result["status-message"]["children"]["props"]["className"]


def test_high_degree_robokop_fetch_requests_summary_once_and_one_page_per_click(monkeypatch, graph_context, path) -> None:
    requests = []

    def respond(request):
        requests.append(request)
        if request.url.path == "/edge_summary/MONDO:0008903":
            return httpx.Response(200, json={"edge_types": [
                {"predicate": "biolink:affects", "category": "biolink:Gene", "count": 10000},
            ]})
        assert request.url.path == "/edges/MONDO:0008903"
        assert dict(request.url.params) == {"limit": "2", "offset": str((len(requests) - 2) * 2)}
        offset = int(request.url.params["offset"])
        category = "biolink:Gene" if offset == 0 else "biolink:Drug"
        predicate = "biolink:affects" if offset == 0 else "biolink:treats"
        return httpx.Response(
            200,
            json={
                "query_curie": "MONDO:0008903",
                "edges": [
                    {
                        "edge": {"predicate": predicate, "direction": ">", "properties": {}},
                        "adj_node": {"id": f"NCBIGene:{index}", "name": f"Neighbor {index}", "category": [category]},
                    }
                    for index in range(offset, offset + 2)
                ],
                "pagination": {"total": 10000, "count": 2, "limit": 2, "offset": offset},
            },
        )

    monkeypatch.setattr(
        "explorer.frontend.callbacks.robokop_provider_from_env",
        lambda: RobokopHttpProvider(
            RobokopHttpConfig(base_url="https://example.test"),
            client=httpx.Client(transport=httpx.MockTransport(respond)),
        ),
    )
    app = create_app()
    response = _fetch_remote(app, path, graph_context, curie="MONDO:0008903")
    assert response.status_code == 200
    assert len(requests) == 2
    assert requests[0].url.path == "/edge_summary/MONDO:0008903"
    assert requests[0].extensions["timeout"]["read"] == 3.0
    fetched = response.get_json()["response"]["context-store"]["data"]
    state = fetched["path-1"]["robokop_state"]["n1"]
    assert len(state["expansion"]["edges"]) == 2
    assert state["expansion"]["total_available"] == 10000
    assert robokop_has_next_page(state["expansion"])
    assert "ROBOKOP reports 10000 connections" in robokop_page_message(state["expansion"], summary=state["summary"])

    response = _fetch_remote(app, path, fetched, action="robokop-next-page-button", curie="MONDO:0008903")
    assert response.status_code == 200
    assert len(requests) == 3
    state = response.get_json()["response"]["context-store"]["data"]["path-1"]["robokop_state"]["n1"]
    assert len(state["expansion"]["edges"]) == 2
    assert state["expansion"]["config"]["offset"] == 2
    assert state["summary"]["total_edges"] == 10000
    assert state["expansion"]["config"]["category"] is None
    assert state["expansion"]["config"]["predicate"] is None
    assert state["expansion"]["config"]["direction"] == "either"


@pytest.mark.parametrize(
    ("page_count", "offset", "total_available", "expected"),
    [(10, 0, None, True), (3, 0, None, False), (0, 10, None, False), (10, 0, 10, False), (10, 0, 10000, True)],
)
def test_robokop_next_page_uses_unfiltered_page_size(page_count, offset, total_available, expected) -> None:
    expansion = {
        "config": {"limit": 10, "offset": offset},
        "edges": [],
        "total_available": total_available,
        "diagnostics": {"page_count": page_count, "direction_filtered_count": page_count},
    }

    assert robokop_has_next_page(expansion) is expected
    if expected:
        assert "Try Next" in robokop_page_message(expansion)
    else:
        assert "Try Next" not in robokop_page_message(expansion)


def test_semantic_expansion_notifies_when_no_results(monkeypatch, graph_context, path) -> None:
    class EmptyGraph:
        def semantic_similar_nodes(self, node_id):
            assert node_id == "n1"
            return {"nodes": [], "edges": []}

        def close(self):
            pass

    monkeypatch.setattr("explorer.frontend.callbacks.Neo4jGraphAdapter", EmptyGraph)
    response = _expand_semantically(create_app(), path, graph_context)

    assert response.status_code == 200
    result = response.get_json()["response"]
    context = result["context-store"]["data"]["path-1"]
    assert "No semantically similar nodes were returned" in context["node_message"]
    assert context["node_message_node_id"] == "n1"
    assert context["semantic_subgraphs"] == {}
    assert context["focus_ids"] == ["n1"]
    assert "warning" in result["status-message"]["children"]["props"]["className"]
    panel = node_action_panel(path, context, "n1")
    assert any(getattr(child, "children", None) == context["node_message"] for child in panel.children)


@pytest.mark.parametrize("already_visible", [False, True])
def test_semantic_expansion_reports_new_or_existing_matches(
    monkeypatch, graph_context, path, semantic_subgraph, already_visible
) -> None:
    class SimilarGraph:
        def semantic_similar_nodes(self, node_id):
            return semantic_subgraph

        def close(self):
            pass

    monkeypatch.setattr("explorer.frontend.callbacks.Neo4jGraphAdapter", SimilarGraph)
    if already_visible:
        graph_context["path-1"]["semantic_subgraphs"]["n1"] = semantic_subgraph
    else:
        graph_context["path-1"]["hidden_ids"] = ["n2"]
    response = _expand_semantically(create_app(), path, graph_context)

    assert response.status_code == 200
    context = response.get_json()["response"]["context-store"]["data"]["path-1"]
    assert context["semantic_subgraphs"]["n1"] == semantic_subgraph
    assert context["hidden_ids"] == []
    assert context["node_message"] == (
        "No new semantically similar nodes to add; the returned matches are already visible."
        if already_visible else "Added 1 semantically similar node(s) and 1 similarity link(s)."
    )


def test_semantic_expansion_errors_are_visible_beside_node(monkeypatch, graph_context, path) -> None:
    class BrokenGraph:
        def semantic_similar_nodes(self, node_id):
            raise RuntimeError("Embedding lookup unavailable")

        def close(self):
            pass

    monkeypatch.setattr("explorer.frontend.callbacks.Neo4jGraphAdapter", BrokenGraph)
    response = _expand_semantically(create_app(), path, graph_context)

    assert response.status_code == 200
    result = response.get_json()["response"]
    context = result["context-store"]["data"]["path-1"]
    assert context["node_message"] == "Semantic expansion failed: Embedding lookup unavailable"
    assert context["node_message_node_id"] == "n1"
    assert "error" in result["status-message"]["children"]["props"]["className"]


def test_remote_semantic_expansion_explains_local_only_support(monkeypatch, graph_context, path) -> None:
    def unexpected_graph_lookup():
        pytest.fail("Remote nodes must not be looked up by Neo4j element ID")

    monkeypatch.setattr("explorer.frontend.callbacks.Neo4jGraphAdapter", unexpected_graph_lookup)
    graph_context["path-1"]["selected_node_id"] = "robokop:NCBIGene:1"
    response = _expand_semantically(create_app(), path, graph_context)

    assert response.status_code == 200
    context = response.get_json()["response"]["context-store"]["data"]["path-1"]
    assert "Use ROBOKOP expansion for this node" in context["node_message"]
    assert context["node_message_node_id"] == "robokop:NCBIGene:1"


def test_sequential_expansion_positions_do_not_overlap(graph_context) -> None:
    context = graph_context["path-1"]
    for index in range(12):
        context["robokop_subgraphs"][str(index)] = {"nodes": [{"id": f"remote:{index}"}], "edges": []}
        previous_positions = deepcopy(context["positions"])
        seed_new_positions(context, "n1")
        assert all(context["positions"][node_id] == position for node_id, position in previous_positions.items())
        coordinates = [(position["x"], position["y"]) for position in context["positions"].values()]
        assert len(coordinates) == len(set(coordinates))


@pytest.mark.parametrize("changed_property", ["selectedNodeData", "elements"])
def test_graph_events_preserve_dragged_positions(graph_context, path, changed_property) -> None:
    graph_context["path-1"]["base_subgraph"]["nodes"].extend([{"id": "n2"}, {"id": "n3"}])
    elements = [
        {"data": {"id": "n1"}, "position": {"x": 100, "y": 100}},
        {"data": {"id": "n2"}, "position": {"x": 450, "y": 300}},
        {"data": {"id": "n3"}, "position": {"x": 500, "y": 400}},
    ]
    response = _remember_graph(create_app(), graph_context, [{"id": "n2"}], elements, _graph_prop_id(changed_property))

    assert response.status_code == 200
    context = response.get_json()["response"]["context-store"]["data"]["path-1"]
    assert context["positions"]["n3"] == {"x": 500.0, "y": 400.0}
    assert context["selected_node_id"] == ("n2" if changed_property == "selectedNodeData" else "n1")
    rendered = cytoscape_elements(path, context["base_subgraph"], context)
    assert next(element["position"] for element in rendered if element["data"]["id"] == "n3") == {"x": 500, "y": 400}


def test_graph_deselection_clears_node_actions(graph_context, path) -> None:
    response = _remember_graph(create_app(), graph_context, [], [], _graph_prop_id("selectedNodeData"))

    assert response.status_code == 200
    context = response.get_json()["response"]["context-store"]["data"]["path-1"]
    assert context["selected_node_id"] is None
    assert node_action_panel(path, context, None).className == "node-actions empty"


@pytest.mark.parametrize("selected_node", [None, "n1", "robokop:NCBIGene:2"])
def test_collapse_button_lives_in_graph_toolbar_and_requires_selection(graph_context, path, selected_node) -> None:
    context = graph_context["path-1"]
    context["selected_node_id"] = selected_node
    toolbar = render_path_details(path, context, {})[2].children[2]
    collapse = next(button for button in toolbar.children[1].children if button.id == "collapse-node-button")

    assert collapse.children == "Collapse expansions"
    assert collapse.disabled is (selected_node is None)
    assert collapse.title == "Collapse expansions of the selected node"
    assert toolbar.children[1].children[0].id == "clear-selection-button"


def test_context_graph_callback_accepts_missing_dynamic_node_controls() -> None:
    app = create_app()
    response = app.server.test_client().get("/_dash-dependencies")

    assert response.status_code == 200
    callback = next(
        item for item in response.get_json()
        if any(dependency["id"] == "collapse-node-button" for dependency in item["inputs"])
    )
    dependencies = {item["id"]: item for item in callback["inputs"] + callback["state"]}
    for component_id in (
        "expand-connected-button", "expand-similar-button", "collapse-node-button",
        "expansion-direction-dropdown", "expansion-limit-input",
        "expansion-category-filter-dropdown", "expansion-predicate-filter-dropdown",
    ):
        assert dependencies[component_id]["allow_optional"] is True
    assert not dependencies["context-store"].get("allow_optional", False)


def test_path_actions_accept_missing_note_field_when_details_panel_is_absent() -> None:
    app = create_app()
    response = app.server.test_client().get("/_dash-dependencies")

    assert response.status_code == 200
    callback = next(
        item for item in response.get_json()
        if any(dependency["id"] == "path-note" for dependency in item["state"])
    )
    dependencies = {item["id"]: item for item in callback["state"]}
    assert dependencies["path-note"]["allow_optional"] is True
    assert not dependencies["session-store"].get("allow_optional", False)


@pytest.mark.parametrize(
    ("node_id", "source_graph", "hidden"),
    [("n1", "litcoin", False), ("robokop:NCBIGene:2", "robokop", True), ("remote:2", "robokop", True)],
)
def test_remote_node_hides_local_expansion_controls_without_removing_callback_inputs(
    graph_context, path, node_id, source_graph, hidden
) -> None:
    context = graph_context["path-1"]
    context["base_subgraph"]["nodes"] = [{"id": node_id, "source_graph": source_graph}]
    panel = node_action_panel(path, context, node_id)
    sections = {
        getattr(child, "className", ""): child for child in panel.children
    }

    for class_name in ("expansion-section connected-expansion", "expansion-section semantic-expansion"):
        section = sections[class_name]
        assert section.style == ({"display": "none"} if hidden else {})
        assert section.children[1].children[0].disabled is hidden
    assert sections["robokop-controls"].className == "robokop-controls"
    assert all(getattr(child, "className", None) != "collapse-expansion-controls" for child in panel.children)


def test_unchanged_graph_event_does_not_cause_render_loop(graph_context) -> None:
    response = _remember_graph(create_app(), graph_context, [{"id": "n1"}], [], _graph_prop_id("selectedNodeData"))

    assert response.status_code == 204


def test_unset_initial_selection_does_not_clear_saved_node(graph_context) -> None:
    response = _remember_graph(create_app(), graph_context, None, [], _graph_prop_id("selectedNodeData"))

    assert response.status_code == 204


def test_old_graph_revision_event_is_ignored(graph_context) -> None:
    graph_context["path-1"]["graph_revision"] = 1
    response = _remember_graph(create_app(), graph_context, [{"id": "n2"}], [], _graph_prop_id("selectedNodeData"))

    assert response.status_code == 204


def test_clear_selection_cannot_be_restored_by_position_update(graph_context, path) -> None:
    app = create_app()
    callback_id = next(
        key for key, metadata in app.callback_map.items()
        if metadata.get("callback") and metadata["callback"].__name__ == "clear_graph_selection"
    )
    response = app.server.test_client().post(
        "/_dash-update-component",
        json={
            "output": callback_id,
            "outputs": [
                {"id": "context-store", "property": "data"},
                {"id": "status-message", "property": "children"},
            ],
            "inputs": [{"id": "clear-selection-button", "property": "n_clicks", "value": 1}],
            "state": [
                {"id": "selected-path-id-store", "property": "data", "value": "path-1"},
                {"id": "context-store", "property": "data", "value": graph_context},
            ],
            "changedPropIds": ["clear-selection-button.n_clicks"],
        },
    )
    assert response.status_code == 200
    cleared = response.get_json()["response"]["context-store"]["data"]
    response = _remember_graph(
        app, cleared, [{"id": "n1"}],
        [{"data": {"id": "n1"}, "position": {"x": 150, "y": 150}}],
        _graph_prop_id("elements"),
    )

    assert response.status_code == 200
    context = response.get_json()["response"]["context-store"]["data"]["path-1"]
    assert context["selected_node_id"] is None
    assert node_action_panel(path, context, None).className == "node-actions empty"
