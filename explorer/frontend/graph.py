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
    for index, node_id in enumerate(missing):
        positions[node_id] = {
            "x": anchor["x"] + 110 + (index % 4) * 70,
            "y": anchor["y"] - 120 + (index // 4) * 70,
        }


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
                html.Label("Expansion query", htmlFor="expansion-query-input"),
                dcc.Input(
                    id="expansion-query-input",
                    type="text",
                    value=context.get("expansion_query") or context.get("active_query") or "",
                    placeholder="Optional refined query for connected neighbors",
                    debounce=True,
                    className="expansion-input",
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
            className="expansion-controls",
        )
    )
    children.append(
            html.Div(
                [
                    html.Button(
                        "Expand connected",
                        id="expand-connected-button",
                        n_clicks=0,
                        className="secondary-button",
                    ),
                    html.Button(
                        "Expand similar",
                        id="expand-similar-button",
                        n_clicks=0,
                        className="secondary-button",
                    ),
                    html.Button(
                        "Collapse expansions",
                        id="collapse-node-button",
                        n_clicks=0,
                        className="secondary-button",
                    ),
                ],
                className="button-row",
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
    provider_mode = robokop_state.get("provider_mode") or "not queried"
    error = robokop_state.get("error")
    summary_items = summary.get("items") or []
    category_options = _dropdown_options(summary.get("categories") or [])
    predicate_options = _dropdown_options(summary.get("predicates") or [])
    limit_value = int(config.get("limit") or context.get("robokop_limit") or 10)
    offset_value = int(config.get("offset") or context.get("robokop_offset") or 0)
    selected_edge_ids = set(robokop_state.get("selected_edge_ids") or [])

    children: list[Any] = [
        html.H4("ROBOKOP expansion"),
        html.Div(
            [
                html.Span(f"Mode: {provider_mode}", className=f"source-badge {provider_mode}"),
                html.Span(f"Bridge: {bridge.get('curie') or node.get('curie') or 'unresolved'}", className="node-labels"),
            ],
            className="robokop-meta-row",
        ),
        html.Button(
            "Fetch ROBOKOP summary",
            id="robokop-summary-button",
            n_clicks=0,
            className="secondary-button",
        ),
    ]
    if error:
        children.append(html.Div(error, className="node-inline-message error"))
    if bridge.get("status") and bridge.get("status") != "resolved":
        children.append(html.Div(bridge.get("reason") or bridge["status"], className="node-inline-message"))
    if summary:
        children.append(
            html.Div(
                f"Summary: {summary.get('total_edges', 0)} incident edges across "
                f"{len(summary_items)} predicate/category groups.",
                className="node-labels",
            )
        )
    if summary and not summary_items:
        children.append(
            html.Div(
                "No ROBOKOP incident edges were reported for this bridge CURIE.",
                className="node-inline-message",
            )
        )
    if summary_items:
        children.extend(
            [
                html.Div(
                    [
                        html.Label("ROBOKOP category", htmlFor="robokop-category-filter-dropdown"),
                        dcc.Dropdown(
                            id="robokop-category-filter-dropdown",
                            options=category_options,
                            value=config.get("category"),
                            placeholder="Optional adjacent category",
                            className="expansion-select",
                        ),
                        html.Label("ROBOKOP predicate", htmlFor="robokop-predicate-filter-dropdown"),
                        dcc.Dropdown(
                            id="robokop-predicate-filter-dropdown",
                            options=predicate_options,
                            value=config.get("predicate"),
                            placeholder="Optional predicate",
                            className="expansion-select",
                        ),
                    ],
                    className="expansion-control-row",
                ),
                html.Div(
                    [
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
                        html.Label("Limit", htmlFor="robokop-limit-input"),
                        dcc.Input(
                            id="robokop-limit-input",
                            type="number",
                            min=1,
                            max=25,
                            step=1,
                            value=limit_value,
                            className="expansion-number",
                        ),
                        html.Label("Offset", htmlFor="robokop-offset-input"),
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
                html.Div(
                    [
                        html.Button(
                            "Previous page",
                            id="robokop-prev-page-button",
                            n_clicks=0,
                            className="secondary-button",
                        ),
                        html.Button(
                            "Fetch ROBOKOP page",
                            id="robokop-expand-button",
                            n_clicks=0,
                            className="secondary-button",
                        ),
                        html.Button(
                            "Next page",
                            id="robokop-next-page-button",
                            n_clicks=0,
                            className="secondary-button",
                        ),
                    ],
                    className="button-row",
                ),
                html.Div(
                    "ROBOKOP pages are bounded. Fetched edges appear below as candidate cards; click Add remote edge to place one in the graph.",
                    className="hint",
                ),
            ]
        )
    if expansion:
        children.append(_robokop_page_status(expansion))
    if expansion.get("edges"):
        children.append(
            html.Div(
                [
                    _robokop_edge_row(edge, selected=edge.get("edge_id") in selected_edge_ids)
                    for edge in expansion["edges"]
                ],
                className="robokop-edge-list",
            )
        )
    return html.Div(children, className="robokop-controls")


def _robokop_page_status(expansion: dict[str, Any]) -> Any:
    edges = expansion.get("edges") or []
    config = expansion.get("config") or {}
    diagnostics = expansion.get("diagnostics") or {}
    filtered_count = int(diagnostics.get("direction_filtered_count") or 0)
    total_available = expansion.get("total_available")
    total_text = f" of {total_available}" if total_available is not None else ""
    filter_bits = [
        f"category {config.get('category')}" if config.get("category") else "any category",
        f"predicate {config.get('predicate')}" if config.get("predicate") else "any predicate",
        f"direction {config.get('direction') or 'either'}",
    ]
    message = (
        f"Fetched {len(edges)}{total_text} ROBOKOP edge(s) "
        f"at offset {config.get('offset', 0)} with limit {config.get('limit', 10)} "
        f"({'; '.join(filter_bits)})."
    )
    if filtered_count:
        message += f" {filtered_count} edge(s) were hidden by the direction filter."
    if not edges:
        message += " Try changing category, predicate, direction, or offset."
    return html.Div(message, className="node-inline-message" if not edges else "node-labels")


def _robokop_edge_row(edge: dict[str, Any], *, selected: bool) -> Any:
    source = edge.get("primary_knowledge_source") or "unknown source"
    score = edge.get("score")
    score_text = f"{float(score):.3f}" if score is not None else "n/a"
    label = (
        f"{edge.get('subject_curie')} -[{edge.get('predicate', '').replace('biolink:', '')}]-> "
        f"{edge.get('object_curie')}"
    )
    reasons = ", ".join(edge.get("ranking_reasons") or [])
    return html.Div(
        [
            html.Div(label, className="robokop-edge-title"),
            html.Div(f"Score {score_text} | {source} | {reasons}", className="path-meta"),
            html.Button(
                "Added" if selected else "Add remote edge",
                id={"type": "add-robokop-edge", "edge_id": edge.get("edge_id")},
                n_clicks=0,
                disabled=selected,
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
        {"selector": ".path-edge", "style": {"line-color": "#c05a2b", "width": 4, "opacity": 0.95}},
        {
            "selector": ".semantic-edge",
            "style": {"line-color": "#5577b8", "line-style": "dashed", "width": 2, "opacity": 0.82},
        },
        {
            "selector": ".remote-edge",
            "style": {"line-color": "#547aa5", "line-style": "dotted", "width": 3, "opacity": 0.86},
        },
    ]
