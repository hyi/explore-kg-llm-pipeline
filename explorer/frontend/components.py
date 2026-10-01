from __future__ import annotations

from typing import Any

import dash_cytoscape as cyto
from dash import dcc, html

from explorer.frontend.constants import CANDIDATE_PREVIEW_LIMIT
from explorer.frontend.graph import (
    cytoscape_elements,
    cytoscape_stylesheet,
    visible_subgraph,
)


def status(children: Any, level: str) -> Any:
    return html.Div(children, className=f"status-message {level}")


def render_candidate_path_cards(
    paths: list[dict[str, Any]] | None,
    selected_path_id: str | None,
    expanded: bool = False,
) -> list[Any] | Any:
    if not paths:
        return html.Div("No paths to display yet.", className="empty")

    indexed_paths = list(enumerate(paths, start=1))
    if len(indexed_paths) > CANDIDATE_PREVIEW_LIMIT:
        preview = indexed_paths[:CANDIDATE_PREVIEW_LIMIT]
        if selected_path_id not in {path["id"] for _, path in preview}:
            selected = next(
                ((index, path) for index, path in indexed_paths[CANDIDATE_PREVIEW_LIMIT:] if path["id"] == selected_path_id),
                None,
            )
            if selected:
                preview[-1] = selected
        if expanded:
            preview_ids = {path["id"] for _, path in preview}
            indexed_paths = preview + [item for item in indexed_paths if item[1]["id"] not in preview_ids]
        else:
            indexed_paths = preview

    cards = []
    for index, path in indexed_paths:
        selected = path["id"] == selected_path_id
        nodes = path.get("nodes") or []
        endpoint_ids = (
            f"Subject ID: {nodes[0]['id']} | Object ID: {nodes[-1]['id']}"
            if nodes and nodes[0].get("id") and nodes[-1].get("id")
            else None
        )
        cards.append(
            html.Button(
                [
                    html.Div(
                        [
                            html.Span(f"Path {index}", className="path-index"),
                            html.Span(f"{float(path.get('score') or 0):.4f}", className="score-pill"),
                        ],
                        className="path-card-top",
                    ),
                    html.Div(path.get("summary") or "(empty path)", className="path-summary"),
                    *([html.Div(endpoint_ids, className="path-meta")] if endpoint_ids else []),
                ],
                id={"type": "select-path", "path_id": path["id"]},
                n_clicks=0,
                className=f"path-card{' selected' if selected else ''}",
            )
        )
    if expanded and len(cards) > CANDIDATE_PREVIEW_LIMIT:
        return [
            *cards[:CANDIDATE_PREVIEW_LIMIT],
            html.Div(cards[CANDIDATE_PREVIEW_LIMIT:], className="path-list-overflow"),
        ]
    return cards


def render_path_details(
    path: dict[str, Any] | None,
    context: dict[str, Any] | None,
    session_store: dict[str, Any] | None,
) -> Any:
    if not path:
        return html.Div("Select a candidate path to inspect its details.", className="empty")

    graph_section = html.Div("Loading path context graph...", className="empty")
    if context:
        subgraph = visible_subgraph(path, context)
        viewport = {}
        if context.get("zoom") is not None:
            viewport["zoom"] = context["zoom"]
        if context.get("pan") is not None:
            viewport["pan"] = context["pan"]
        graph_section = cyto.Cytoscape(
            id={
                "type": "context-graph",
                "path_id": path["id"],
                "revision": context.get("graph_revision", 0),
            },
            elements=cytoscape_elements(path, subgraph, context),
            layout={"name": "preset", "fit": True, "padding": 90},
            style={"width": "100%", "height": "640px"},
            minZoom=0.15,
            maxZoom=3,
            **viewport,
            zoomingEnabled=True,
            userZoomingEnabled=False,
            userPanningEnabled=True,
            boxSelectionEnabled=False,
            autoungrabify=False,
            stylesheet=cytoscape_stylesheet(),
        )

    note_value = (session_store or {}).get("user_notes", {}).get(path["id"], "")
    return [
        html.Div(
            [
                html.Div(
                    [
                        html.H2("Path Details"),
                        html.P(path.get("summary") or "(empty path)", className="detail-summary"),
                    ],
                    className="details-heading",
                ),
                html.Div(
                    [
                        html.Button(
                            "Accept",
                            id={"type": "path-action", "action": "accept", "path_id": path["id"]},
                            n_clicks=0,
                            className="secondary-button",
                        ),
                        html.Button(
                            "Bookmark",
                            id={"type": "path-action", "action": "bookmark", "path_id": path["id"]},
                            n_clicks=0,
                            className="secondary-button",
                        ),
                        html.Button(
                            "Reject",
                            id={"type": "path-action", "action": "reject", "path_id": path["id"]},
                            n_clicks=0,
                            className="danger-button",
                        ),
                    ],
                    className="button-row",
                ),
            ],
            className="details-title-row",
        ),
        html.Details(
            [
                html.Summary("Semantic evidence"),
                html.P(path.get("evidence_text") or "No semantic evidence text available."),
            ],
            open=False,
            className="evidence-block",
        ),
        html.Div(
            [
                html.H3("Path Context Subgraph"),
                html.P(
                    "Initial path context loads nearby connected nodes. Click a node for actions, drag nodes to reposition, drag the background to pan, and use toolbar buttons to zoom.",
                    className="hint",
                ),
                html.Div(
                    [
                        html.Div(
                            [
                                html.Span([html.Span(className="legend-dot legend-path"), "Initial path"]),
                                html.Span([html.Span(className="legend-dot legend-focus"), "Expanded focus"]),
                                html.Span([html.Span(className="legend-dot legend-context"), "Context node"]),
                                html.Span([html.Span(className="legend-line legend-semantic"), "Semantic similarity"]),
                                html.Span([html.Span(className="legend-line legend-remote"), "ROBOKOP edge"]),
                            ],
                            className="graph-legend",
                        ),
                        html.Div(
                            [
                                html.Button(
                                    "Clear selection",
                                    id="clear-selection-button",
                                    n_clicks=0,
                                    className="secondary-button",
                                ),
                                html.Button(
                                    "Reset view",
                                    id="reset-view-button",
                                    n_clicks=0,
                                    className="secondary-button",
                                ),
                                html.Button(
                                    "Zoom in",
                                    id="zoom-in-button",
                                    n_clicks=0,
                                    className="secondary-button",
                                ),
                                html.Button(
                                    "Zoom out",
                                    id="zoom-out-button",
                                    n_clicks=0,
                                    className="secondary-button",
                                ),
                            ],
                            className="button-row graph-control-buttons",
                        ),
                    ],
                    className="graph-toolbar",
                ),
                graph_section,
            ],
            className="graph-wrap",
        ),
        html.Div(
            [
                html.Label("Session note", htmlFor="path-note"),
                dcc.Textarea(id="path-note", value=note_value, className="note-input"),
                html.Button(
                    "Save note",
                    id={"type": "path-action", "action": "save_note", "path_id": path["id"]},
                    n_clicks=0,
                    className="secondary-button",
                ),
            ],
            className="note-wrap",
        ),
    ]


def saved_session_view(session_store: dict[str, Any]) -> Any:
    blocks = []
    for key, title in [
        ("accepted_paths", "Accepted"),
        ("bookmarked_paths", "Bookmarked"),
        ("rejected_paths", "Rejected"),
    ]:
        paths = session_store.get(key, [])
        blocks.append(
            html.Div(
                [
                    html.H3(f"{title} ({len(paths)})"),
                    html.Ul([html.Li(path.get("summary") or path.get("id")) for path in paths])
                    if paths
                    else html.Div("None yet.", className="empty"),
                ],
                className="saved-block",
            )
        )
    history = session_store.get("search_history", [])
    blocks.append(
        html.Div(
            [
                html.H3("Search History"),
                html.Ul(
                    [
                        html.Li(f"{item.get('query')} ({item.get('result_count')} results)")
                        for item in history
                    ]
                )
                if history
                else html.Div("No searches yet.", className="empty"),
            ],
            className="saved-block",
        )
    )
    return html.Div(blocks, className="saved-grid")
