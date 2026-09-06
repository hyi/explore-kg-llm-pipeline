from __future__ import annotations

from explorer.backend.models import Edge, Node, Path
from explorer.frontend.components import render_path_details
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
    graph_wrap = details[3]
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
    graph = details[3].children[4]

    assert graph.userZoomingEnabled is False
    assert graph.zoomingEnabled is True
    assert graph.userPanningEnabled is True
    assert graph.zoom == 1.25


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
    assert stores["selected-path-id-store"].storage_type == "memory"
    assert stores["context-store"].storage_type == "memory"
    assert stores["session-store"].storage_type == "memory"


def test_layout_includes_server_cache_clear_button() -> None:
    layout = build_layout()

    assert _find_component(layout, "clear-cache-button").children == "Clear server cache"


def _find_component(component, component_id: str):
    if getattr(component, "id", None) == component_id:
        return component
    children = getattr(component, "children", None)
    if children is None:
        return None
    if not isinstance(children, list):
        children = [children]
    for child in children:
        match = _find_component(child, component_id)
        if match is not None:
            return match
    return None
