from __future__ import annotations

import math
from copy import deepcopy
from typing import Any

from dash import dcc, html

from explorer.frontend.constants import NODE_LABEL_MAX_LENGTH


def visible_subgraph(path: dict[str, Any], context: dict[str, Any]) -> dict[str, list[dict[str, Any]]]:
    merged = merge_subgraphs(
        context.get("base_subgraph") or {"nodes": [], "edges": []},
        list((context.get("semantic_subgraphs") or {}).values())
        + list((context.get("robokop_subgraphs") or {}).values()),
    )
    protected = {node["element_id"] for node in path.get("nodes", [])}
    hidden = set(context.get("hidden_ids", [])) - protected
    if not hidden:
        return merged
    return {
        "nodes": [node for node in merged["nodes"] if node["id"] not in hidden],
        "edges": [
            edge for edge in merged["edges"]
            if edge["source"] not in hidden and edge["target"] not in hidden
        ],
    }


def merge_subgraphs(
    base: dict[str, list[dict[str, Any]]],
    additions: list[dict[str, list[dict[str, Any]]]],
) -> dict[str, list[dict[str, Any]]]:
    nodes_by_id = {node["id"]: node for node in base.get("nodes", [])}
    edges_by_id = {edge["id"]: edge for edge in base.get("edges", [])}
    for subgraph in additions:
        for node in subgraph.get("nodes", []):
            nodes_by_id[node["id"]] = node
        for edge in subgraph.get("edges", []):
            edges_by_id[edge["id"]] = edge
    return {"nodes": list(nodes_by_id.values()), "edges": list(edges_by_id.values())}


def cytoscape_elements(
    path: dict[str, Any],
    subgraph: dict[str, list[dict[str, Any]]],
    context: dict[str, Any],
) -> list[dict[str, Any]]:
    positions = context.get("positions", {})
    path_node_ids = {node["element_id"] for node in path.get("nodes", [])}
    path_edge_ids = {edge["element_id"] for edge in path.get("edges", [])}
    focus_ids = set(context.get("focus_ids", []))
    selected = context.get("selected_node_id")

    elements = []
    for node in subgraph["nodes"]:
        node_id = node["id"]
        full_label = node.get("label") or node_id
        position = positions.get(node_id) or {"x": 80 + len(elements) * 40, "y": 120}
        elements.append(
            {
                "data": {
                    "id": node_id,
                    "label": truncate_label(full_label),
                    "full_label": full_label,
                    "labels": ", ".join(node.get("labels", [])),
                    "is_path": node_id in path_node_ids,
                    "is_focus": node_id in focus_ids,
                    "source_graph": node.get("source_graph") or "litcoin",
                },
                "position": position,
                "selected": node_id == selected,
                "classes": (
                    "path-node"
                    if node_id in path_node_ids
                    else "focus-node"
                    if node_id in focus_ids
                    else "remote-node"
                    if node.get("source_graph") == "robokop"
                    else "context-node"
                ),
            }
        )
    for edge in subgraph["edges"]:
        classes = []
        if edge["id"] in path_edge_ids:
            classes.append("path-edge")
        if edge.get("kind") == "semantic":
            classes.append("semantic-edge")
        if edge.get("source_graph") == "robokop":
            classes.append("remote-edge")
        elements.append(
            {
                "data": {
                    "id": edge["id"],
                    "source": edge["source"],
                    "target": edge["target"],
                    "label": str(edge.get("label") or "").replace("biolink:", ""),
                    "source_graph": edge.get("source_graph") or "litcoin",
                },
                "classes": " ".join(classes),
            }
        )
    return elements


def truncate_label(label: str, max_length: int = NODE_LABEL_MAX_LENGTH) -> str:
    if len(label) <= max_length:
        return label
    return f"{label[: max_length - 1]}…"


def initial_positions(
    nodes: list[dict[str, Any]],
    edges: list[dict[str, Any]],
) -> dict[str, dict[str, float]]:
    if not nodes:
        return {}

    node_ids = [node["id"] for node in nodes]
    degree = {node_id: 0 for node_id in node_ids}
    for edge in edges:
        degree[edge["source"]] = degree.get(edge["source"], 0) + 1
        degree[edge["target"]] = degree.get(edge["target"], 0) + 1

    ordered = sorted(node_ids, key=lambda node_id: degree.get(node_id, 0), reverse=True)
    center_x, center_y = 460.0, 300.0
    positions = {ordered[0]: {"x": center_x, "y": center_y}}
    radius = 230.0
    for index, node_id in enumerate(ordered[1:], start=0):
        angle = (2 * math.pi * index) / max(1, len(ordered) - 1)
        positions[node_id] = {
            "x": center_x + radius * math.cos(angle),
            "y": center_y + radius * math.sin(angle),
        }
    return positions


def seed_new_positions(context: dict[str, Any], anchor_id: str) -> None:
    subgraph = merge_subgraphs(
        context.get("base_subgraph") or {"nodes": [], "edges": []},
        list((context.get("semantic_subgraphs") or {}).values())
        + list((context.get("robokop_subgraphs") or {}).values()),
    )
    positions = context.setdefault("positions", {})
    anchor = positions.get(anchor_id, {"x": 460.0, "y": 300.0})
    missing = [node["id"] for node in subgraph["nodes"] if node["id"] not in positions]
    slot = 0
    for node_id in missing:
        while True:
            position = {
                "x": anchor["x"] + 110 + (slot % 4) * 70,
                "y": anchor["y"] - 120 + (slot // 4) * 70,
            }
            slot += 1
            if all(
                math.hypot(position["x"] - existing["x"], position["y"] - existing["y"]) >= 60
                for existing in positions.values()
            ):
                positions[node_id] = position
                break


def positions_from_elements(
    elements: list[dict[str, Any]] | None,
    fallback: dict[str, dict[str, float]],
) -> dict[str, dict[str, float]]:
    positions = deepcopy(fallback or {})
    for element in elements or []:
        data = element.get("data") or {}
        position = element.get("position") or {}
        if data.get("source") or not data.get("id"):
            continue
        if "x" in position and "y" in position:
            positions[data["id"]] = {"x": float(position["x"]), "y": float(position["y"])}
    return positions


def node_action_panel(
    path: dict[str, Any] | None,
    context: dict[str, Any] | None,
    selected_node_id: str | None,
) -> Any:
    if not path or not context or not selected_node_id:
        return html.Div("Select a node in the graph to expand, compare, or collapse context.", className="node-actions empty")

    subgraph = visible_subgraph(path, context)
    node = next((item for item in subgraph["nodes"] if item["id"] == selected_node_id), None)
    if not node:
        return html.Div("Selected node is no longer visible.", className="node-actions empty")

    is_remote = node.get("source_graph") == "robokop" or selected_node_id.startswith("robokop:")
    incident_count = sum(
        1 for edge in subgraph["edges"] if edge["source"] == selected_node_id or edge["target"] == selected_node_id
    )
    node_message = (
        context.get("node_message")
        if context.get("node_message_node_id") == selected_node_id
        else None
    )
    children = [
            html.Div(
                [
                    html.Strong(node.get("label") or selected_node_id),
                    html.Span(", ".join(node.get("labels", [])), className="node-labels"),
                    html.Span(f"Bridge CURIE: {node.get('curie') or 'not available'}", className="node-labels"),
                    html.Span(f"Connected edges in view: {incident_count}", className="node-labels"),                    
                ],
                className="selected-node-copy",
            )
    ]
    if node_message:
        children.append(html.Div(node_message, className="node-inline-message"))
    filter_options = context.get("neighborhood_filter_options") or {}
    children.append(
        html.Div(
            [
                html.H4("Connected expansion"),
                html.Div(
                    [
                        html.Button(
                            "Expand connection",
                            id="expand-connected-button",
                            n_clicks=0,
                            disabled=is_remote,
                            className="secondary-button expand-connection-button",
                        ),
                    ],
                    className="button-row",
                ),
                html.Details(
                    [
                        html.Summary("Connection filters"),
                        html.Div(
                            [
                                html.Div(
                                    [
                                        html.Label("Limit", htmlFor="expansion-limit-input"),
                                        dcc.Input(
                                            id="expansion-limit-input",
                                            type="number",
                                            min=1,
                                            max=50,
                                            step=1,
                                            value=int(context.get("connected_expansion_limit") or 12),
                                            className="expansion-number",
                                        ),
                                    ],
                                    className="expansion-control-row",
                                ),
                                html.Div(
                                    [
                                        html.Label("Direction", htmlFor="expansion-direction-dropdown"),
                                        dcc.Dropdown(
                                            id="expansion-direction-dropdown",
                                            options=[
                                                {"label": "Either", "value": "either"},
                                                {"label": "Outgoing", "value": "outgoing"},
                                                {"label": "Incoming", "value": "incoming"},
                                            ],
                                            value=context.get("expansion_direction") or "either",
                                            clearable=False,
                                            className="expansion-select",
                                        ),
                                    ],
                                    className="expansion-control-row",
                                ),
                                html.Div(
                                    [
                                        html.Label("Neighbor categories", htmlFor="expansion-category-filter-dropdown"),
                                        dcc.Dropdown(
                                            id="expansion-category-filter-dropdown",
                                            options=_dropdown_options(filter_options.get("node_categories", [])),
                                            value=context.get("expansion_categories", []),
                                            multi=True,
                                            placeholder="Select one or more categories",
                                            className="expansion-select",
                                        ),
                                        html.Label("Predicates", htmlFor="expansion-predicate-filter-dropdown"),
                                        dcc.Dropdown(
                                            id="expansion-predicate-filter-dropdown",
                                            options=_dropdown_options(filter_options.get("predicates", [])),
                                            value=context.get("expansion_predicates", []),
                                            multi=True,
                                            placeholder="Select one or more predicates",
                                            className="expansion-select",
                                        ),
                                    ],
                                    className="expansion-control-row",
                                ),
                            ],
                            className="expansion-controls advanced",
                        ),
                    ],
                    className="expansion-filter-panel",
                    open=False,
                ),
            ],
            className="expansion-section connected-expansion",
            style={"display": "none"} if is_remote else {},
        )
    )
    children.append(
        html.Div(
            [
                html.H4("Semantic expansion"),
                html.Div(
                    [
                        html.Button(
                            "Expand semantically similar nodes",
                            id="expand-similar-button",
                            n_clicks=0,
                            disabled=is_remote,
                            className="secondary-button",
                        ),
                    ],
                    className="button-row",
                )
            ],
            className="expansion-section semantic-expansion",
            style={"display": "none"} if is_remote else {},
        )
    )
    children.append(_robokop_controls(context, selected_node_id, node))
    return html.Div(
        children,
        className="node-actions",
    )


def _dropdown_options(values: list[str] | tuple[str, ...]) -> list[dict[str, str]]:
    return [{"label": value, "value": value} for value in values]


def _robokop_controls(context: dict[str, Any], selected_node_id: str, node: dict[str, Any]) -> Any:
    robokop_state = (context.get("robokop_state") or {}).get(selected_node_id) or {}
    bridge = robokop_state.get("bridge") or {}
    summary = robokop_state.get("summary") or {}
    expansion = robokop_state.get("expansion") or {}
    config = expansion.get("config") or {}
    provider_mode = expansion.get("provider_mode") or summary.get("provider_mode") or robokop_state.get("provider_mode")
    error = robokop_state.get("error")
    category_options = _dropdown_options(summary.get("categories") or [])
    predicate_options = _dropdown_options(summary.get("predicates") or [])
    limit_value = int(config.get("limit") or context.get("robokop_limit") or 10)
    offset_value = int(config.get("offset") or context.get("robokop_offset") or 0)
    selected_edge_ids = set(robokop_state.get("selected_edge_ids") or [])

    children: list[Any] = [
        html.H4("ROBOKOP expansion"),
        html.Div(
            html.Span(f"Bridge: {bridge.get('curie') or node.get('curie') or 'unresolved'}", className="robokop-bridge"),
            className="robokop-meta-row",
        ),
    ]
    if error:
        children.append(html.Div(error, className="node-inline-message error"))
    if bridge.get("status") and bridge.get("status") != "resolved":
        children.append(html.Div(bridge.get("reason") or bridge["status"], className="node-inline-message"))
    if provider_mode == "fixture" and (summary or expansion):
        children.append(html.Span("Fixture data (not live)", className="source-badge fixture"))
    children.extend(
        [
            html.Details(
                [
                    html.Summary("ROBOKOP filters"),
                    html.Div(
                        [
                            html.Label("Neighbor category", htmlFor="robokop-category-filter-dropdown"),
                            dcc.Dropdown(
                                id="robokop-category-filter-dropdown",
                                options=category_options,
                                value=config.get("category"),
                                placeholder="Any category",
                                className="expansion-select",
                            ),
                            html.Label("Predicate", htmlFor="robokop-predicate-filter-dropdown"),
                            dcc.Dropdown(
                                id="robokop-predicate-filter-dropdown",
                                options=predicate_options,
                                value=config.get("predicate"),
                                placeholder="Any predicate",
                                className="expansion-select",
                            ),
                            html.Label("Direction", htmlFor="robokop-direction-dropdown"),
                            dcc.Dropdown(
                                id="robokop-direction-dropdown",
                                options=[
                                    {"label": "Either", "value": "either"},
                                    {"label": "Outgoing", "value": "outgoing"},
                                    {"label": "Incoming", "value": "incoming"},
                                ],
                                value=config.get("direction") or "either",
                                clearable=False,
                                className="expansion-select",
                            ),
                            html.Label("Page size", htmlFor="robokop-limit-input"),
                            dcc.Input(
                                id="robokop-limit-input",
                                type="number",
                                min=1,
                                max=25,
                                step=1,
                                value=limit_value,
                                className="expansion-number",
                            ),
                            html.Label("Start offset", htmlFor="robokop-offset-input"),
                            dcc.Input(
                                id="robokop-offset-input",
                                type="number",
                                min=0,
                                step=1,
                                value=offset_value,
                                className="expansion-number",
                            ),
                        ],
                        className="expansion-control-row",
                    ),
                ],
                className="expansion-filter-panel robokop-filter-panel",
                open=False,
            ),
            html.Div(
                [
                    html.Button(
                        "Fetch ROBOKOP edges",
                        id="robokop-expand-button",
                        n_clicks=0,
                        className="secondary-button",
                    ),
                ],
                className="button-row",
            ),
        ]
    )
    if summary or expansion:
        children.append(_robokop_page_status(expansion, summary=summary))
    if expansion.get("edges"):
        children.append(
            html.Div(
                "Add or Remove candidates to update the graph. The red endpoint is the bridge node.",
                className="hint",
            )
        )
        children.append(
            html.Details(
                [
                    html.Summary(f"Remote edge candidates ({len(expansion['edges'])})"),
                    html.Div(
                        [
                            _robokop_edge_row(
                                edge,
                                selected=edge.get("edge_id") in selected_edge_ids,
                                bridge_curie=bridge.get("curie") or node.get("curie"),
                            )
                            for edge in expansion["edges"]
                        ],
                        className="robokop-edge-list",
                    ),
                ],
                open=True,
                className="robokop-results",
            )
        )
    children.append(
        html.Div(
            [
                html.Button(
                    "Previous",
                    id="robokop-prev-page-button",
                    n_clicks=0,
                    className="secondary-button",
                    disabled=offset_value == 0,
                ),
                html.Span(f"Page {offset_value // limit_value + 1}", className="node-labels"),
                html.Button(
                    "Next",
                    id="robokop-next-page-button",
                    n_clicks=0,
                    className="secondary-button",
                ),
            ],
            className="button-row",
            style={} if expansion and summary.get("total_edges", 1) else {"display": "none"},
        )
    )
    return html.Div(children, className="robokop-controls")


def _robokop_page_status(expansion: dict[str, Any], *, summary: dict[str, Any] | None = None) -> Any:
    edges = expansion.get("edges") or []
    diagnostics = expansion.get("diagnostics") or {}
    filtered_count = int(diagnostics.get("direction_filtered_count") or 0)
    if summary and not summary.get("total_edges"):
        return html.Div("No ROBOKOP connections found for this bridge.", className="node-inline-message")
    message = (
        f"ROBOKOP reports {summary['total_edges']} connection{'s' if summary['total_edges'] != 1 else ''}"
        if summary else "ROBOKOP"
    )
    if expansion:
        candidate_text = f"fetched {len(edges)} candidate{'s' if len(edges) != 1 else ''}."
        message += f"; {candidate_text}" if summary else f" {candidate_text}"
    else:
        message += "."
    if filtered_count:
        message += f" {filtered_count} connections excluded by direction; choose Either to include them."
    if expansion and not edges:
        message += " No matching connections in this page. Change filters or try Next."
    return html.Div(message, className="node-inline-message" if not edges else "node-labels")


def _robokop_edge_row(edge: dict[str, Any], *, selected: bool, bridge_curie: str | None = None) -> Any:
    source = edge.get("primary_knowledge_source") or "unknown source"
    subject_curie = edge.get("subject_curie")
    object_curie = edge.get("object_curie")
    label = [
        html.Span(
            subject_curie,
            className="robokop-bridge" if bridge_curie and subject_curie == bridge_curie else "",
            title="Bridge node" if bridge_curie and subject_curie == bridge_curie else subject_curie,
        ),
        f" -[{edge.get('predicate', '').replace('biolink:', '')}]-> ",
        html.Span(
            object_curie,
            className="robokop-bridge" if bridge_curie and object_curie == bridge_curie else "",
            title="Bridge node" if bridge_curie and object_curie == bridge_curie else object_curie,
        ),
    ]
    reasons = ", ".join(
        reason for reason in edge.get("ranking_reasons") or []
        if reason != "bounded ROBOKOP page fallback"
    )
    return html.Div(
        [
            html.Div(label, className="robokop-edge-title"),
            html.Div(" | ".join(part for part in (source, reasons) if part), className="path-meta"),
            html.Button(
                "Remove" if selected else "Add",
                id={"type": "toggle-robokop-edge", "edge_id": edge.get("edge_id")},
                n_clicks=0,
                disabled=False,
                className="secondary-button",
            ),
        ],
        className="robokop-edge-card selected" if selected else "robokop-edge-card",
    )


def cytoscape_stylesheet() -> list[dict[str, Any]]:
    return [
        {
            "selector": "node",
            "style": {
                "label": "data(label)",
                "font-size": 11,
                "text-wrap": "wrap",
                "text-max-width": 120,
                "text-valign": "bottom",
                "text-margin-y": 8,
                "background-color": "#d8a03d",
                "border-color": "#20302a",
                "border-width": 1.4,
                "width": 24,
                "height": 24,
            },
        },
        {"selector": ".path-node", "style": {"background-color": "#c05a2b", "width": 36, "height": 36}},
        {"selector": ".focus-node", "style": {"background-color": "#2f6f63", "width": 31, "height": 31}},
        {"selector": ".remote-node", "style": {"background-color": "#547aa5", "shape": "diamond"}},
        {
            "selector": "node:selected",
            "style": {"border-color": "#111827", "border-width": 5, "overlay-opacity": 0.12},
        },
        {
            "selector": "edge",
            "style": {
                "line-color": "#9ba8a1",
                "target-arrow-shape": "triangle",
                "target-arrow-color": "#9ba8a1",
                "width": 1.5,
                "opacity": 0.68,
                "curve-style": "bezier",
                "label": "data(label)",
                "font-size": 9,
                "text-rotation": "autorotate",
                "text-background-color": "#fbfaf6",
                "text-background-opacity": 0.8,
                "text-background-padding": 2,
            },
        },
        {
            "selector": ".path-edge",
            "style": {"line-color": "#c05a2b", "target-arrow-color": "#c05a2b", "width": 4, "opacity": 0.95},
        },
        {
            "selector": ".semantic-edge",
            "style": {
                "line-color": "#5577b8", "line-style": "dashed", "width": 2, "opacity": 0.82,
                "target-arrow-shape": "none",
            },
        },
        {
            "selector": ".remote-edge",
            "style": {
                "line-color": "#547aa5", "line-style": "dotted", "width": 3, "opacity": 0.86,
                "target-arrow-color": "#547aa5",
            },
        },
    ]
