from __future__ import annotations

import traceback
from copy import deepcopy
from typing import Any

from dash import (
    ALL,
    ClientsideFunction,
    Dash,
    Input,
    Output,
    State,
    ctx,
    html,
    no_update,
)
from dash.exceptions import PreventUpdate

from explorer.backend.external_graph import (
    BridgeResolutionConfig,
    ExternalExpansionConfig,
    RobokopProviderError,
    resolve_litcoin_bridge,
    robokop_provider_from_env,
)
from explorer.backend.graph_adapter.neo4j import Neo4jGraphAdapter
from explorer.backend.path_search import PathSearchService
from explorer.frontend.components import (
    render_candidate_path_cards,
    render_path_details,
    saved_session_view,
    status,
)
from explorer.frontend.constants import (
    DEFAULT_PATH_HITS,
    PATHS_PER_SEMANTIC_HIT,
    SEMANTIC_FETCH_BUFFER,
)
from explorer.frontend.graph import (
    initial_positions,
    node_action_panel,
    positions_from_elements,
    seed_new_positions,
    visible_subgraph,
)
from explorer.frontend.state import (
    empty_session_store,
    find_path,
    path_from_dict,
    selected_node_id,
    session_from_store,
)

CONNECTED_EXPANSION_LIMIT = 12


def register_callbacks(app: Dash) -> None:
    app.clientside_callback(
        ClientsideFunction(namespace="kgExplorer", function_name="zoomGraph"),
        Output("graph-viewport-signal", "data"),
        Input("zoom-in-button", "n_clicks"),
        Input("zoom-out-button", "n_clicks"),
        prevent_initial_call=True,
        )

    @app.callback(
        Output("candidate-paths-store", "data"),
        Output("selected-path-id-store", "data"),
        Output("context-store", "data"),
        Output("session-store", "data"),
        Output("status-message", "children"),
        Input("search-button", "n_clicks"),
        Input("query-input", "n_submit"),
        State("query-input", "value"),
        State("path-hit-slider", "value"),
        State("session-store", "data"),
        prevent_initial_call=True,
        running=[
            (Output("search-button", "disabled"), True, False),
            (Output("search-button", "children"), "Searching...", "Search paths"),
        ],
    )
    def run_search(
        _n_clicks: int | None,
        _n_submit: int | None,
        query: str | None,
        relationship_k: int | None,
        session_store: dict[str, Any] | None,
    ) -> tuple[list[dict[str, Any]], str | None, dict[str, Any], dict[str, Any], Any]:
        query = (query or "").strip()
        if not query:
            return no_update, no_update, no_update, no_update, status("Enter a query before searching.", "warning")

        relationship_k = int(relationship_k or DEFAULT_PATH_HITS)
        session = session_from_store(session_store)

        try:
            semantic_fetch_k = min(relationship_k + SEMANTIC_FETCH_BUFFER, 50)
            search_result = PathSearchService().search(
                query=query,
                relationship_k=relationship_k,
                semantic_fetch_k=semantic_fetch_k,
                paths_per_hit=PATHS_PER_SEMANTIC_HIT,
            )
        except Exception as exc:  # noqa: BLE001 - surface any search failure in the Dash UI.
            details = traceback.format_exc()
            return (
                no_update,
                no_update,
                no_update,
                no_update,
                status([html.Div(f"Search failed: {exc}"), html.Pre(details)], "error"),
            )

        path_dicts = [dict(path, source_query=query) for path in search_result.paths]
        session.add_search(query, len(path_dicts))
        selected_path_id = path_dicts[0]["id"] if path_dicts else None
        cache_label = " from cache" if search_result.cache_hit else ""
        return (
            path_dicts,
            selected_path_id,
            {},
            session.to_dict(),
            status(f"Found {len(path_dicts)} candidate paths{cache_label}.", "success"),
        )

    @app.callback(
        Output("candidate-paths", "children"),
        Input("client-state-store", "data"),
        Input("candidate-paths-store", "data"),
        Input("selected-path-id-store", "data"),
    )
    def render_candidate_paths(
        _client_state: dict[str, Any] | None,
        paths: list[dict[str, Any]] | None,
        selected_path_id: str | None,
    ) -> list[Any] | Any:
        return render_candidate_path_cards(paths, selected_path_id)

    @app.callback(
        Output("selected-path-id-store", "data", allow_duplicate=True),
        Input({"type": "select-path", "path_id": ALL}, "n_clicks"),
        prevent_initial_call=True,
    )
    def select_path(_clicks: list[int]) -> str:
        if not ctx.triggered_id:
            raise PreventUpdate
        return ctx.triggered_id["path_id"]

    @app.callback(
        Output("context-store", "data", allow_duplicate=True),
        Input("selected-path-id-store", "data"),
        State("candidate-paths-store", "data"),
        State("context-store", "data"),
        State("query-input", "value"),
        prevent_initial_call=True,
    )
    def ensure_context_for_selected_path(
        selected_path_id: str | None,
        paths: list[dict[str, Any]] | None,
        context_store: dict[str, Any] | None,
        current_query: str | None,
    ) -> dict[str, Any]:
        if not selected_path_id:
            raise PreventUpdate

        context_store = deepcopy(context_store or {})
        if selected_path_id in context_store:
            return context_store

        path = find_path(paths or [], selected_path_id)
        if not path:
            raise PreventUpdate

        graph = Neo4jGraphAdapter()
        try:
            focus_ids = [node["element_id"] for node in path["nodes"]]
            subgraph = graph.context_subgraph(focus_ids)
            filter_options = graph.neighborhood_filter_options()
        finally:
            graph.close()

        context_store[selected_path_id] = {
            "focus_ids": focus_ids,
            "base_subgraph": subgraph,
            "semantic_subgraphs": {},
            "robokop_subgraphs": {},
            "robokop_state": {},
            "hidden_ids": [],
            "positions": initial_positions(subgraph["nodes"], subgraph["edges"]),
            "selected_node_id": None,
            "active_query": path.get("source_query") or (current_query or "").strip(),
            "connected_expansion_limit": CONNECTED_EXPANSION_LIMIT,
            "neighborhood_filter_options": filter_options,
            "graph_revision": 0,
            "zoom": 1,
            "pan": {"x": 0, "y": 0},
        }
        return context_store

    @app.callback(
        Output("path-details", "children"),
        Input("client-state-store", "data"),
        Input("selected-path-id-store", "data"),
        Input("candidate-paths-store", "data"),
        Input("context-store", "data"),
        Input("session-store", "data"),
    )
    def render_selected_path_details(
        _client_state: dict[str, Any] | None,
        selected_path_id: str | None,
        paths: list[dict[str, Any]] | None,
        context_store: dict[str, Any] | None,
        session_store: dict[str, Any] | None,
    ) -> Any:
        path = find_path(paths or [], selected_path_id)
        context = (context_store or {}).get(path["id"]) if path else None
        return render_path_details(path, context, session_store)

    @app.callback(
        Output("node-action-panel", "children"),
        Input({"type": "context-graph", "path_id": ALL, "revision": ALL}, "selectedNodeData"),
        Input("context-store", "data"),
        State("selected-path-id-store", "data"),
        State("candidate-paths-store", "data"),
        prevent_initial_call=True,
    )
    def render_selected_node_actions(
        selected_node_data_values: list[list[dict[str, Any]] | None] | None,
        context_store: dict[str, Any] | None,
        selected_path_id: str | None,
        paths: list[dict[str, Any]] | None,
    ) -> Any:
        path = find_path(paths or [], selected_path_id)
        context = (context_store or {}).get(selected_path_id or "")
        if ctx.triggered_id == "context-store":
            node_id = context.get("selected_node_id") if context else None
        else:
            node_id = _active_selected_node_id(selected_node_data_values)
        return node_action_panel(path, context, node_id)

    @app.callback(
        Output("context-store", "data", allow_duplicate=True),
        Input({"type": "context-graph", "path_id": ALL, "revision": ALL}, "selectedNodeData"),
        State("selected-path-id-store", "data"),
        State("context-store", "data"),
        prevent_initial_call=True,
    )
    def remember_selected_graph_node(
        selected_node_data_values: list[list[dict[str, Any]] | None] | None,
        selected_path_id: str | None,
        context_store: dict[str, Any] | None,
    ) -> dict[str, Any]:
        node_id = _active_selected_node_id(selected_node_data_values)
        if not node_id or not selected_path_id or selected_path_id not in (context_store or {}):
            raise PreventUpdate

        context_store = deepcopy(context_store)
        context_store[selected_path_id]["selected_node_id"] = node_id
        return context_store

    @app.callback(
        Output("context-store", "data", allow_duplicate=True),
        Output("status-message", "children", allow_duplicate=True),
        Input("expand-connected-button", "n_clicks"),
        Input("expand-similar-button", "n_clicks"),
        Input("collapse-node-button", "n_clicks"),
        Input({"type": "add-robokop-edge", "edge_id": ALL}, "n_clicks"),
        State("selected-path-id-store", "data"),
        State("candidate-paths-store", "data"),
        State("context-store", "data"),
        State({"type": "context-graph", "path_id": ALL, "revision": ALL}, "selectedNodeData"),
        State({"type": "context-graph", "path_id": ALL, "revision": ALL}, "elements"),
        State("query-input", "value"),
        State("expansion-query-input", "value"),
        State("expansion-direction-dropdown", "value"),
        State("expansion-limit-input", "value"),
        State("expansion-category-filter-dropdown", "value"),
        State("expansion-predicate-filter-dropdown", "value"),
        prevent_initial_call=True,
    )
    def update_context_graph(
        _expand_connected: int | None,
        _expand_similar: int | None,
        _collapse_node: int | None,
        _robokop_add_clicks: list[int | None] | None,
        selected_path_id: str | None,
        paths: list[dict[str, Any]] | None,
        context_store: dict[str, Any] | None,
        selected_node_data_values: list[list[dict[str, Any]] | None] | None,
        elements_values: list[list[dict[str, Any]] | None] | None,
        current_query: str | None,
        expansion_query: str | None,
        expansion_direction: str | None,
        expansion_limit: float | str | None,
        expansion_categories: list[str] | str | None,
        expansion_predicates: list[str] | str | None,
    ) -> tuple[dict[str, Any], Any]:
        action = ctx.triggered_id
        if not action or not selected_path_id:
            raise PreventUpdate
        clicked = ctx.triggered[0]["value"] if ctx.triggered else None
        if not clicked:
            raise PreventUpdate

        path = find_path(paths or [], selected_path_id)
        context = deepcopy((context_store or {}).get(selected_path_id))
        node_id = _active_selected_node_id(selected_node_data_values) or (context or {}).get("selected_node_id")
        if not path or not context or not node_id:
            return no_update, status("Select a node before expanding or collapsing.", "warning")

        context["selected_node_id"] = node_id
        context["positions"] = positions_from_elements(_active_elements(elements_values), context.get("positions", {}))
        path_node_ids = {node["element_id"] for node in path["nodes"]}
        if isinstance(action, dict) and action.get("type") == "add-robokop-edge":
            edge_id = str(action.get("edge_id") or "")
            edge = _robokop_edge_from_context(context, node_id, edge_id)
            if not edge:
                return no_update, status("Selected ROBOKOP edge is no longer available.", "warning")
            _add_robokop_edge_to_context(context, node_id, edge)
            seed_new_positions(context, node_id)
            context_store = deepcopy(context_store or {})
            context_store[selected_path_id] = context
            return context_store, status("Added selected ROBOKOP edge to the visible graph.", "success")

        try:
            graph = Neo4jGraphAdapter()
            try:
                if action == "expand-connected-button":
                    visible_node_ids = {node["id"] for node in visible_subgraph(path, context)["nodes"]}
                    if not graph.has_unseen_neighbors(node_id, visible_node_ids):
                        context["selected_node_id"] = node_id
                        context["node_message"] = "No new connected nodes to expand for the selected node."
                        context["node_message_node_id"] = node_id
                        context_store = deepcopy(context_store or {})
                        context_store[selected_path_id] = context
                        return (
                            context_store,
                            status("No new connected nodes to expand for the selected node.", "warning"),
                        )
                    if node_id not in context["focus_ids"]:
                        context["focus_ids"].append(node_id)
                    context["hidden_ids"] = [hidden_id for hidden_id in context["hidden_ids"] if hidden_id != node_id]
                    context.pop("node_message", None)
                    context.pop("node_message_node_id", None)
                    active_query = (
                        expansion_query
                        or context.get("active_query")
                        or path.get("source_query")
                        or current_query
                        or ""
                    ).strip()
                    limit = _bounded_expansion_limit(expansion_limit)
                    categories = _normalize_filter_values(expansion_categories)
                    predicates = _normalize_filter_values(expansion_predicates)
                    context["active_query"] = active_query
                    context["expansion_query"] = active_query
                    context["connected_expansion_limit"] = limit
                    context["expansion_direction"] = expansion_direction or "either"
                    context["expansion_categories"] = categories
                    context["expansion_predicates"] = predicates
                    context["base_subgraph"] = graph.context_subgraph(
                        context["focus_ids"],
                        query=active_query,
                        limit_per_focus=limit,
                        node_categories=categories,
                        predicates=predicates,
                        direction=expansion_direction or "either",
                    )
                    seed_new_positions(context, node_id)
                    message = f"Expanded top {limit} connected neighbors per focus node."
                elif action == "expand-similar-button":
                    context["semantic_subgraphs"][node_id] = graph.semantic_similar_nodes(node_id)
                    if node_id not in context["focus_ids"]:
                        context["focus_ids"].append(node_id)
                    context["hidden_ids"] = [hidden_id for hidden_id in context["hidden_ids"] if hidden_id != node_id]
                    context.pop("node_message", None)
                    context.pop("node_message_node_id", None)
                    seed_new_positions(context, node_id)
                    message = "Expanded semantically similar nodes."
                elif action == "collapse-node-button":
                    context["semantic_subgraphs"].pop(node_id, None)
                    context.get("robokop_subgraphs", {}).pop(node_id, None)
                    context.get("robokop_state", {}).pop(node_id, None)
                    if node_id not in path_node_ids:
                        context["focus_ids"] = [focus_id for focus_id in context["focus_ids"] if focus_id != node_id]
                        active_query = (context.get("active_query") or path.get("source_query") or current_query or "").strip()
                        if active_query:
                            context["base_subgraph"] = graph.context_subgraph(
                                context["focus_ids"],
                                query=active_query,
                                limit_per_focus=int(context.get("connected_expansion_limit") or CONNECTED_EXPANSION_LIMIT),
                                node_categories=context.get("expansion_categories") or [],
                                predicates=context.get("expansion_predicates") or [],
                                direction=context.get("expansion_direction") or "either",
                            )
                        else:
                            context["base_subgraph"] = graph.context_subgraph(context["focus_ids"])
                    context["hidden_ids"] = [hidden_id for hidden_id in context["hidden_ids"] if hidden_id != node_id]
                    context.pop("node_message", None)
                    context.pop("node_message_node_id", None)
                    message = "Collapsed selected node context."
                else:
                    raise PreventUpdate
            finally:
                graph.close()
        except RobokopProviderError as exc:
            state = context.setdefault("robokop_state", {}).setdefault(node_id, {})
            state["error"] = str(exc)
            state.setdefault("provider_mode", "error")
            context_store = deepcopy(context_store or {})
            context_store[selected_path_id] = context
            return context_store, status(f"ROBOKOP expansion failed: {exc}", "error")
        except Exception as exc:  # noqa: BLE001 - graph adapter errors should become UI status.
            return no_update, status(f"Graph update failed: {exc}", "error")

        context_store = deepcopy(context_store or {})
        context_store[selected_path_id] = context
        return context_store, status(message, "success")

    @app.callback(
        Output("context-store", "data", allow_duplicate=True),
        Output("status-message", "children", allow_duplicate=True),
        Input("robokop-summary-button", "n_clicks"),
        State("selected-path-id-store", "data"),
        State("candidate-paths-store", "data"),
        State("context-store", "data"),
        State({"type": "context-graph", "path_id": ALL, "revision": ALL}, "selectedNodeData"),
        prevent_initial_call=True,
    )
    def fetch_robokop_summary(
        n_clicks: int | None,
        selected_path_id: str | None,
        paths: list[dict[str, Any]] | None,
        context_store: dict[str, Any] | None,
        selected_node_data_values: list[list[dict[str, Any]] | None] | None,
    ) -> tuple[dict[str, Any], Any]:
        if not n_clicks or not selected_path_id:
            raise PreventUpdate

        path = find_path(paths or [], selected_path_id)
        context = deepcopy((context_store or {}).get(selected_path_id))
        node_id = _active_selected_node_id(selected_node_data_values) or (context or {}).get("selected_node_id")
        if not path or not context or not node_id:
            return no_update, status("Select a node before fetching a ROBOKOP summary.", "warning")

        context["selected_node_id"] = node_id
        bridge_node = _node_from_context(path, context, node_id)
        bridge = resolve_litcoin_bridge(bridge_node, BridgeResolutionConfig())
        context.setdefault("robokop_state", {})[node_id] = {
            "bridge": bridge.to_dict(),
            "provider_mode": "not queried",
        }
        if not bridge.is_resolved:
            context_store = deepcopy(context_store or {})
            context_store[selected_path_id] = context
            return context_store, status(bridge.reason, "warning")

        try:
            provider = robokop_provider_from_env()
            try:
                remote_node = provider.lookup_node(bridge.curie)
                summary = provider.edge_summary(bridge.curie)
            finally:
                close = getattr(provider, "close", None)
                if callable(close):
                    close()
        except RobokopProviderError as exc:
            state = context.setdefault("robokop_state", {}).setdefault(node_id, {})
            state["error"] = str(exc)
            state.setdefault("provider_mode", "error")
            context_store = deepcopy(context_store or {})
            context_store[selected_path_id] = context
            return context_store, status(f"ROBOKOP summary failed: {exc}", "error")

        context["robokop_state"][node_id] = {
            "bridge": bridge.to_dict(),
            "provider_mode": summary.provider_mode,
            "provider_base_url": getattr(provider, "base_url", None),
            "node": remote_node.to_dict() if remote_node else None,
            "summary": summary.to_dict(),
            "selected_edge_ids": [],
        }
        context_store = deepcopy(context_store or {})
        context_store[selected_path_id] = context
        return (
            context_store,
            status(
                f"Fetched ROBOKOP {summary.provider_mode} summary for {bridge.curie}: "
                f"{summary.total_edges} incident edges reported.",
                "success",
            ),
        )

    @app.callback(
        Output("context-store", "data", allow_duplicate=True),
        Output("status-message", "children", allow_duplicate=True),
        Input("robokop-expand-button", "n_clicks"),
        Input("robokop-prev-page-button", "n_clicks"),
        Input("robokop-next-page-button", "n_clicks"),
        State("selected-path-id-store", "data"),
        State("candidate-paths-store", "data"),
        State("context-store", "data"),
        State({"type": "context-graph", "path_id": ALL, "revision": ALL}, "selectedNodeData"),
        State("query-input", "value"),
        State("expansion-query-input", "value"),
        State("robokop-category-filter-dropdown", "value"),
        State("robokop-predicate-filter-dropdown", "value"),
        State("robokop-direction-dropdown", "value"),
        State("robokop-limit-input", "value"),
        State("robokop-offset-input", "value"),
        prevent_initial_call=True,
    )
    def fetch_robokop_page(
        n_clicks: int | None,
        prev_clicks: int | None,
        next_clicks: int | None,
        selected_path_id: str | None,
        paths: list[dict[str, Any]] | None,
        context_store: dict[str, Any] | None,
        selected_node_data_values: list[list[dict[str, Any]] | None] | None,
        current_query: str | None,
        expansion_query: str | None,
        robokop_category: str | None,
        robokop_predicate: str | None,
        robokop_direction: str | None,
        robokop_limit: float | str | None,
        robokop_offset: float | str | None,
    ) -> tuple[dict[str, Any], Any]:
        action = ctx.triggered_id
        clicked = ctx.triggered[0]["value"] if ctx.triggered else None
        if not clicked or not selected_path_id:
            raise PreventUpdate

        path = find_path(paths or [], selected_path_id)
        context = deepcopy((context_store or {}).get(selected_path_id))
        node_id = _active_selected_node_id(selected_node_data_values) or (context or {}).get("selected_node_id")
        if not path or not context or not node_id:
            return no_update, status("Select a node before fetching a ROBOKOP page.", "warning")

        context["selected_node_id"] = node_id
        bridge_node = _node_from_context(path, context, node_id)
        bridge = resolve_litcoin_bridge(bridge_node, BridgeResolutionConfig())
        if not bridge.is_resolved:
            context.setdefault("robokop_state", {})[node_id] = {
                "bridge": bridge.to_dict(),
                "provider_mode": "not queried",
            }
            context_store = deepcopy(context_store or {})
            context_store[selected_path_id] = context
            return context_store, status(bridge.reason, "warning")

        active_query = (
            expansion_query
            or context.get("active_query")
            or path.get("source_query")
            or current_query
            or ""
        ).strip()
        try:
            provider = robokop_provider_from_env()
            try:
                limit = _bounded_robokop_limit(robokop_limit)
                offset = _robokop_page_offset(
                    robokop_offset,
                    limit=limit,
                    action=str(action),
                )
                expansion = provider.incident_edges(
                    bridge.curie,
                    ExternalExpansionConfig(
                        category=robokop_category or None,
                        predicate=robokop_predicate or None,
                        direction=robokop_direction or "either",
                        limit=limit,
                        offset=offset,
                        query=active_query,
                    ),
                )
            finally:
                close = getattr(provider, "close", None)
                if callable(close):
                    close()
        except RobokopProviderError as exc:
            state = context.setdefault("robokop_state", {}).setdefault(node_id, {})
            state["error"] = str(exc)
            state.setdefault("provider_mode", "error")
            context_store = deepcopy(context_store or {})
            context_store[selected_path_id] = context
            return context_store, status(f"ROBOKOP page fetch failed: {exc}", "error")

        state = context.setdefault("robokop_state", {}).setdefault(node_id, {})
        state["bridge"] = bridge.to_dict()
        state["provider_mode"] = expansion.provider_mode
        state["provider_base_url"] = getattr(provider, "base_url", None)
        state["expansion"] = expansion.to_dict()
        state.setdefault("selected_edge_ids", [])
        state.pop("error", None)
        context_store = deepcopy(context_store or {})
        context_store[selected_path_id] = context
        return (
            context_store,
            status(
                f"Fetched {expansion.total_returned} bounded ROBOKOP {expansion.provider_mode} "
                f"edge(s) for {bridge.curie}.",
                "success",
            ),
        )

    @app.callback(
        Output("context-store", "data", allow_duplicate=True),
        Output("status-message", "children", allow_duplicate=True),
        Input("clear-selection-button", "n_clicks"),
        State("selected-path-id-store", "data"),
        State("context-store", "data"),
        prevent_initial_call=True,
    )
    def clear_graph_selection(
        n_clicks: int | None,
        selected_path_id: str | None,
        context_store: dict[str, Any] | None,
    ) -> tuple[dict[str, Any], Any]:
        if not n_clicks or not selected_path_id or selected_path_id not in (context_store or {}):
            raise PreventUpdate

        context_store = deepcopy(context_store)
        context_store[selected_path_id]["selected_node_id"] = None
        context_store[selected_path_id].pop("node_message", None)
        context_store[selected_path_id].pop("node_message_node_id", None)
        return context_store, status("Selection cleared.", "success")

    @app.callback(
        Output("context-store", "data", allow_duplicate=True),
        Output("status-message", "children", allow_duplicate=True),
        Input("reset-view-button", "n_clicks"),
        State("selected-path-id-store", "data"),
        State("candidate-paths-store", "data"),
        State("context-store", "data"),
        prevent_initial_call=True,
    )
    def reset_graph_view(
        n_clicks: int | None,
        selected_path_id: str | None,
        paths: list[dict[str, Any]] | None,
        context_store: dict[str, Any] | None,
    ) -> tuple[dict[str, Any], Any]:
        if not n_clicks or not selected_path_id or selected_path_id not in (context_store or {}):
            raise PreventUpdate

        path = find_path(paths or [], selected_path_id)
        if not path:
            raise PreventUpdate

        context_store = deepcopy(context_store)
        context = context_store[selected_path_id]
        subgraph = visible_subgraph(path, context)
        context["positions"] = initial_positions(subgraph["nodes"], subgraph["edges"])
        context["zoom"] = 1
        context["pan"] = {"x": 0, "y": 0}
        context["graph_revision"] = int(context.get("graph_revision", 0)) + 1
        context.pop("node_message", None)
        context.pop("node_message_node_id", None)
        return context_store, status("Graph view reset.", "success")

    @app.callback(
        Output("session-store", "data", allow_duplicate=True),
        Output("status-message", "children", allow_duplicate=True),
        Input({"type": "path-action", "action": ALL, "path_id": ALL}, "n_clicks"),
        State("candidate-paths-store", "data"),
        State("session-store", "data"),
        State("path-note", "value"),
        prevent_initial_call=True,
    )
    def update_session(
        _clicks: list[int | None],
        paths: list[dict[str, Any]] | None,
        session_store: dict[str, Any] | None,
        note: str | None,
    ) -> tuple[dict[str, Any], Any]:
        triggered = ctx.triggered_id
        if not isinstance(triggered, dict):
            raise PreventUpdate

        clicked = ctx.triggered[0]["value"] if ctx.triggered else None
        if not clicked:
            raise PreventUpdate

        action = triggered.get("action")
        path_dict = find_path(paths or [], triggered.get("path_id"))
        if not action or not path_dict:
            raise PreventUpdate

        session = session_from_store(session_store)
        path = path_from_dict(path_dict)
        summary = path.summary()
        if action == "accept":
            session.accept(path)
            message = f"Accepted path: {summary}"
        elif action == "bookmark":
            session.bookmark(path)
            message = f"Bookmarked path: {summary}"
        elif action == "reject":
            session.reject(path)
            message = f"Rejected path: {summary}"
        elif action == "save_note":
            session.save_note(path.id, note or "")
            message = f"Saved note for path: {summary}"
        else:
            raise PreventUpdate

        if action in {"accept", "bookmark"} and note is not None:
            session.save_note(path.id, note)
        return session.to_dict(), status(message, "success")

    @app.callback(
        Output("session-metrics", "children"),
        Output("saved-session", "children"),
        Input("client-state-store", "data"),
        Input("session-store", "data"),
    )
    def render_session(
        _client_state: dict[str, Any] | None,
        session_store: dict[str, Any] | None,
    ) -> tuple[list[Any], Any]:
        session_store = session_store or empty_session_store()
        metrics = [
            html.Span(f"Accepted {len(session_store.get('accepted_paths', []))}"),
            html.Span(f"Bookmarked {len(session_store.get('bookmarked_paths', []))}"),
            html.Span(f"Rejected {len(session_store.get('rejected_paths', []))}"),
        ]
        return metrics, saved_session_view(session_store)


def _active_selected_node_id(
    selected_node_data_values: list[list[dict[str, Any]] | None] | None,
) -> str | None:
    for selected_node_data in selected_node_data_values or []:
        node_id = selected_node_id(selected_node_data)
        if node_id:
            return node_id
    return None


def _parse_comma_separated(value: str | None) -> list[str]:
    return [part.strip() for part in (value or "").split(",") if part.strip()]


def _normalize_filter_values(value: list[str] | tuple[str, ...] | str | None) -> list[str]:
    if isinstance(value, (list, tuple)):
        return [str(part).strip() for part in value if str(part).strip()]
    return _parse_comma_separated(value)


def _bounded_expansion_limit(value: float | str | None) -> int:
    if value in (None, ""):
        return CONNECTED_EXPANSION_LIMIT
    try:
        parsed = int(value)
    except (TypeError, ValueError):
        return CONNECTED_EXPANSION_LIMIT
    return max(1, min(parsed, 50))


def _bounded_robokop_limit(value: float | str | None) -> int:
    if value in (None, ""):
        return 10
    try:
        parsed = int(value)
    except (TypeError, ValueError):
        return 10
    return max(1, min(parsed, 25))


def _bounded_nonnegative_int(value: float | str | None) -> int:
    if value in (None, ""):
        return 0
    try:
        parsed = int(value)
    except (TypeError, ValueError):
        return 0
    return max(0, parsed)


def _robokop_page_offset(value: float | str | None, *, limit: int, action: str) -> int:
    current = _bounded_nonnegative_int(value)
    if action == "robokop-next-page-button":
        return current + max(1, int(limit))
    if action == "robokop-prev-page-button":
        return max(0, current - max(1, int(limit)))
    return current


def _active_elements(
    elements_values: list[list[dict[str, Any]] | None] | None,
) -> list[dict[str, Any]] | None:
    for elements in elements_values or []:
        if elements:
            return elements
    return None


def _node_from_context(path: dict[str, Any], context: dict[str, Any], node_id: str) -> dict[str, Any] | None:
    for node in path.get("nodes", []):
        if node.get("element_id") == node_id:
            return {
                "id": node.get("id"),
                "curie": node.get("id"),
                "label": node.get("name"),
                "labels": node.get("labels", []),
                "properties": node.get("properties", {}),
            }
    subgraphs = [
        context.get("base_subgraph") or {},
        *(context.get("semantic_subgraphs") or {}).values(),
        *(context.get("robokop_subgraphs") or {}).values(),
    ]
    for subgraph in subgraphs:
        for node in subgraph.get("nodes", []):
            if node.get("id") == node_id:
                return node
    return None


def _robokop_edge_from_context(context: dict[str, Any], node_id: str, edge_id: str) -> dict[str, Any] | None:
    expansion = ((context.get("robokop_state") or {}).get(node_id) or {}).get("expansion") or {}
    return next((edge for edge in expansion.get("edges", []) if edge.get("edge_id") == edge_id), None)


def _add_robokop_edge_to_context(context: dict[str, Any], node_id: str, edge: dict[str, Any]) -> None:
    subgraphs = context.setdefault("robokop_subgraphs", {})
    subgraph = subgraphs.setdefault(node_id, {"nodes": [], "edges": []})
    state = context.setdefault("robokop_state", {}).setdefault(node_id, {})
    selected_edge_ids = state.setdefault("selected_edge_ids", [])
    if edge["edge_id"] not in selected_edge_ids:
        selected_edge_ids.append(edge["edge_id"])

    remote_node_id = f"robokop:{edge['adjacent_curie']}"
    if edge.get("direction") == "incoming":
        source, target = remote_node_id, node_id
        remote_name = edge.get("subject_name") or edge["adjacent_curie"]
        remote_categories = edge.get("subject_categories") or []
    else:
        source, target = node_id, remote_node_id
        remote_name = edge.get("object_name") or edge["adjacent_curie"]
        remote_categories = edge.get("object_categories") or []

    nodes_by_id = {item["id"]: item for item in subgraph.get("nodes", [])}
    nodes_by_id[remote_node_id] = {
        "id": remote_node_id,
        "curie": edge["adjacent_curie"],
        "label": remote_name,
        "labels": remote_categories,
        "source_graph": "robokop",
        "properties": {
            "curie": edge["adjacent_curie"],
            "source_graph": "robokop",
        },
    }
    edges_by_id = {item["id"]: item for item in subgraph.get("edges", [])}
    edges_by_id[edge["edge_id"]] = {
        "id": edge["edge_id"],
        "source": source,
        "target": target,
        "label": edge.get("predicate") or "",
        "source_graph": "robokop",
        "properties": {
            "source_graph": "robokop",
            "predicate": edge.get("predicate"),
            "primary_knowledge_source": edge.get("primary_knowledge_source"),
            "publications": edge.get("publications") or [],
            "qualifiers": edge.get("qualifiers") or {},
            "supporting_sentences": edge.get("supporting_sentences") or [],
            "original_subject_curie": edge.get("original_subject_curie"),
            "original_object_curie": edge.get("original_object_curie"),
        },
    }
    subgraph["nodes"] = list(nodes_by_id.values())
    subgraph["edges"] = list(edges_by_id.values())
