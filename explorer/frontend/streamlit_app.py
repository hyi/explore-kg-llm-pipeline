from __future__ import annotations

import json
import math
import sys
import traceback
from pathlib import Path

import streamlit as st
import streamlit.components.v1 as components

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

SUBGRAPH_COMPONENT = components.declare_component(
    "kg_subgraph",
    path=str(Path(__file__).resolve().parent / "components" / "subgraph"),
)

from explorer.backend.exploration_session import ExplorationSession
from explorer.backend.graph_adapter.neo4j import Neo4jGraphAdapter
from explorer.backend.models import Path
from explorer.backend.path_discovery import PathDiscoveryService
from explorer.backend.path_ranking import PathRankingService
from explorer.backend.semantic_search import SemanticSearchService


st.set_page_config(page_title="Semantic Graph Explorer", layout="wide")


def main() -> None:
    st.title("Human-Guided Semantic Graph Explorer")
    st.caption("Search -> Candidate Paths -> Human Selection -> Path Expansion -> Save Session")

    session = _get_exploration_session()
    _init_ui_state()

    with st.sidebar:
        st.header("Search Controls")
        relationship_k = st.slider("Semantic relationship hits", 1, 25, 10)
        max_hops = st.slider("Candidate path max hops", 1, 5, 3)
        paths_per_hit = st.slider("Paths per semantic hit", 1, 10, 3)
        expansion_limit = st.slider("Expansion candidates", 1, 25, 10)
        context_neighbors = st.slider("Context neighbors per node", 1, 25, 8)
        st.divider()
        st.metric("Accepted", len(session.accepted_paths))
        st.metric("Bookmarked", len(session.bookmarked_paths))
        st.metric("Rejected", len(session.rejected_paths))

    query = st.text_input(
        "Natural language query",
        placeholder="Example: genes involved in drug resistance in cancer",
    )
    search_clicked = st.button("Search paths", type="primary", disabled=not query.strip())

    if search_clicked:
        _run_search(
            query=query.strip(),
            relationship_k=relationship_k,
            max_hops=max_hops,
            paths_per_hit=paths_per_hit,
            session=session,
        )

    candidate_paths: list[Path] = st.session_state.candidate_paths
    expanded_paths: list[Path] = st.session_state.expanded_paths

    tab_candidates, tab_expanded, tab_saved, tab_history = st.tabs(
        ["Candidate Paths", "Expanded Paths", "Saved Session", "Search History"]
    )

    with tab_candidates:
        _render_path_list(candidate_paths, "candidate", expansion_limit, context_neighbors)

    with tab_expanded:
        _render_path_list(expanded_paths, "expanded", expansion_limit, context_neighbors)

    with tab_saved:
        _render_saved_session(session)

    with tab_history:
        st.dataframe(session.search_history, use_container_width=True)


def _run_search(
    query: str,
    relationship_k: int,
    max_hops: int,
    paths_per_hit: int,
    session: ExplorationSession,
) -> None:
    try:
        with st.spinner("Running semantic search and discovering candidate paths..."):
            graph = Neo4jGraphAdapter()
            try:
                semantic_results = SemanticSearchService().search(query, relationship_k=relationship_k)
                discovery = PathDiscoveryService(graph)
                ranking = PathRankingService()
                paths = discovery.discover_from_semantic_results(
                    semantic_results,
                    max_hops=max_hops,
                    paths_per_hit=paths_per_hit,
                )
                st.session_state.candidate_paths = ranking.rank(paths)
                st.session_state.expanded_paths = []
                session.add_search(query, len(paths))
            finally:
                graph.close()
        st.success(f"Found {len(st.session_state.candidate_paths)} candidate paths.")
    except Exception as exc:
        st.error(f"Search failed: {exc}")
        with st.expander("Error details"):
            st.code(traceback.format_exc())


def _render_path_list(
    paths: list[Path],
    namespace: str,
    expansion_limit: int,
    context_neighbors: int,
) -> None:
    if not paths:
        st.info("No paths to display yet.")
        return

    for index, path in enumerate(paths, start=1):
        with st.container(border=True):
            col_title, col_score = st.columns([4, 1])
            with col_title:
                st.subheader(f"Path {index}: {path.summary()}")
            with col_score:
                st.metric("Score", f"{path.score:.4f}")

            st.caption(
                f"Length: {path.length} | Source: {path.source}"
                + (f" | Seed predicate: {path.seed_predicate}" if path.seed_predicate else "")
            )

            with st.expander("Path details"):
                st.write("Nodes")
                st.dataframe([_node_row(node) for node in path.nodes], use_container_width=True)
                st.write("Edges")
                st.dataframe([_edge_row(edge) for edge in path.edges], use_container_width=True)
                if path.evidence_text:
                    st.write("Semantic evidence")
                    st.write(path.evidence_text)

                st.write("Path context subgraph")
                _render_context_subgraph(path, namespace, context_neighbors)

            note_key = f"note_{namespace}_{path.id}"
            current_note = st.session_state.exploration_session.user_notes.get(path.id, "")
            note = st.text_area("Note", value=current_note, key=note_key, height=80)

            action_cols = st.columns(5)
            if action_cols[0].button("Accept", key=f"accept_{namespace}_{path.id}"):
                _get_exploration_session().accept(path)
                _get_exploration_session().save_note(path.id, note)
                st.rerun()
            if action_cols[1].button("Bookmark", key=f"bookmark_{namespace}_{path.id}"):
                _get_exploration_session().bookmark(path)
                _get_exploration_session().save_note(path.id, note)
                st.rerun()
            if action_cols[2].button("Reject", key=f"reject_{namespace}_{path.id}"):
                _get_exploration_session().reject(path)
                st.rerun()
            if action_cols[3].button("Save note", key=f"save_note_{namespace}_{path.id}"):
                _get_exploration_session().save_note(path.id, note)
                st.rerun()
            if action_cols[4].button("Expand", key=f"expand_{namespace}_{path.id}"):
                _expand_path(path, expansion_limit)


def _expand_path(path: Path, expansion_limit: int) -> None:
    try:
        with st.spinner("Expanding path endpoints..."):
            graph = Neo4jGraphAdapter()
            try:
                expanded = PathDiscoveryService(graph).expand(path, limit=expansion_limit)
                ranked = PathRankingService().rank(expanded)
                existing_signatures = {_path_signature(existing) for existing in st.session_state.expanded_paths}
                st.session_state.expanded_paths.extend(
                    candidate for candidate in ranked if _path_signature(candidate) not in existing_signatures
                )
            finally:
                graph.close()
        st.success(f"Added {len(expanded)} expanded paths.")
    except Exception as exc:
        st.error(f"Expansion failed: {exc}")
        with st.expander("Error details"):
            st.code(traceback.format_exc())


def _render_context_subgraph(path: Path, namespace: str, context_neighbors: int) -> None:
    focus_key = f"context_focus_{namespace}_{path.id}"
    subgraph_key = f"context_subgraph_{namespace}_{path.id}"
    positions_key = f"context_positions_{namespace}_{path.id}"
    expanded_from_key = f"context_expanded_from_{namespace}_{path.id}"
    clear_key = f"context_clear_selection_{namespace}_{path.id}"
    st.session_state.setdefault(focus_key, path.node_element_ids)
    st.session_state.setdefault(clear_key, 0)

    cols = st.columns([3, 1])
    with cols[1]:
        if st.button("Clear selection", key=f"clear_context_selection_{namespace}_{path.id}"):
            st.session_state[clear_key] += 1

    if subgraph_key not in st.session_state:
        try:
            graph = Neo4jGraphAdapter()
            try:
                st.session_state[subgraph_key] = graph.context_subgraph(
                    st.session_state[focus_key],
                    neighbors_per_node=context_neighbors,
                )
            finally:
                graph.close()
        except Exception as exc:
            st.warning(f"Could not load context subgraph: {exc}")
            return

    subgraph = st.session_state[subgraph_key]
    if not subgraph["nodes"]:
        st.caption("No context subgraph available.")
        return

    component_nodes = _component_nodes(subgraph)
    component_edges = _component_edges(subgraph)
    positions = _merge_layout_positions(
        nodes=component_nodes,
        edges=component_edges,
        previous_positions=st.session_state.get(positions_key, {}),
        expanded_from=st.session_state.get(expanded_from_key),
    )
    st.session_state[positions_key] = positions

    component_event = SUBGRAPH_COMPONENT(
        nodes=component_nodes,
        edges=component_edges,
        positions=positions,
        clear_nonce=st.session_state[clear_key],
        key=f"subgraph_component_{namespace}_{path.id}",
        default=None,
    )

    event_key = f"context_event_{namespace}_{path.id}"
    if not isinstance(component_event, dict):
        return
    if component_event.get("nonce") == st.session_state.get(event_key):
        return

    st.session_state[event_key] = component_event.get("nonce")
    event_positions = _valid_positions(component_event.get("positions"))
    if event_positions:
        st.session_state[positions_key] = event_positions

    if component_event.get("action") == "layout":
        st.rerun()

    if component_event.get("action") == "expand":
        selected_id = component_event.get("node_id")
        if selected_id and selected_id not in st.session_state[focus_key]:
            st.session_state[focus_key].append(selected_id)
        st.session_state[expanded_from_key] = selected_id
        st.session_state.pop(subgraph_key, None)
        st.rerun()


def _component_nodes(subgraph: dict[str, list[dict[str, object]]]) -> list[dict[str, object]]:
    return [
        {
            "id": node["id"],
            "label": node["label"],
            "labels": node.get("labels", []),
            "is_focus": node.get("is_focus", False),
        }
        for node in subgraph["nodes"]
    ]


def _component_edges(subgraph: dict[str, list[dict[str, object]]]) -> list[dict[str, object]]:
    return [
        {
            "id": edge["id"],
            "source": edge["source"],
            "target": edge["target"],
            "label": edge["label"],
        }
        for edge in subgraph["edges"]
    ]


def _layout_positions(nodes: list[dict[str, object]]) -> dict[str, dict[str, float]]:
    focus_nodes = [node for node in nodes if node.get("is_focus")]
    context_nodes = [node for node in nodes if not node.get("is_focus")]
    positions = {}
    positions.update(_ring_positions(focus_nodes, center_x=500, center_y=320, radius=105))
    positions.update(_ring_positions(context_nodes, center_x=500, center_y=320, radius=245))
    return positions


def _merge_layout_positions(
    nodes: list[dict[str, object]],
    edges: list[dict[str, object]],
    previous_positions: dict[str, dict[str, float]],
    expanded_from: str | None,
) -> dict[str, dict[str, float]]:
    node_ids = {str(node["id"]) for node in nodes}
    positions = {
        node_id: {"x": float(position["x"]), "y": float(position["y"])}
        for node_id, position in previous_positions.items()
        if node_id in node_ids and "x" in position and "y" in position
    }

    missing_nodes = [node for node in nodes if node["id"] not in positions]
    if not positions:
        return _layout_positions(nodes)
    if not missing_nodes:
        return positions

    positions.update(
        _positions_for_new_nodes(
            missing_nodes=missing_nodes,
            existing_positions=positions,
            edges=edges,
            expanded_from=expanded_from,
        )
    )
    return positions


def _positions_for_new_nodes(
    missing_nodes: list[dict[str, object]],
    existing_positions: dict[str, dict[str, float]],
    edges: list[dict[str, object]],
    expanded_from: str | None,
) -> dict[str, dict[str, float]]:
    anchor_id = expanded_from if expanded_from in existing_positions else None
    if anchor_id is None:
        for edge in edges:
            source = str(edge["source"])
            target = str(edge["target"])
            if source in existing_positions and any(node["id"] == target for node in missing_nodes):
                anchor_id = source
                break
            if target in existing_positions and any(node["id"] == source for node in missing_nodes):
                anchor_id = target
                break

    anchor = existing_positions.get(anchor_id or "", {"x": 500.0, "y": 320.0})
    radius = 95.0
    positions = {}
    for index, node in enumerate(missing_nodes):
        angle = (2 * math.pi * index / max(1, len(missing_nodes))) - (math.pi / 2)
        positions[str(node["id"])] = {
            "x": _clamp(anchor["x"] + radius * math.cos(angle), 24.0, 976.0),
            "y": _clamp(anchor["y"] + radius * math.sin(angle), 24.0, 430.0),
        }
    return positions


def _valid_positions(value: object) -> dict[str, dict[str, float]]:
    if not isinstance(value, dict):
        return {}

    positions = {}
    for node_id, position in value.items():
        if not isinstance(position, dict):
            continue
        try:
            positions[str(node_id)] = {
                "x": _clamp(float(position["x"]), 24.0, 976.0),
                "y": _clamp(float(position["y"]), 24.0, 430.0),
            }
        except (KeyError, TypeError, ValueError):
            continue
    return positions


def _clamp(value: float, minimum: float, maximum: float) -> float:
    return max(minimum, min(maximum, value))


def _ring_positions(
    nodes: list[dict[str, object]],
    center_x: float,
    center_y: float,
    radius: float,
) -> dict[str, dict[str, float]]:
    if not nodes:
        return {}
    if len(nodes) == 1:
        return {nodes[0]["id"]: {"x": center_x, "y": center_y}}

    positions = {}
    for index, node in enumerate(nodes):
        angle = (2 * math.pi * index / len(nodes)) - (math.pi / 2)
        positions[node["id"]] = {
            "x": center_x + radius * math.cos(angle),
            "y": center_y + radius * math.sin(angle),
        }
    return positions


def _render_saved_session(session: ExplorationSession) -> None:
    st.subheader("Accepted Paths")
    _render_saved_paths(session.accepted_paths.values(), session)

    st.subheader("Bookmarked Paths")
    _render_saved_paths(session.bookmarked_paths.values(), session)

    st.subheader("Rejected Paths")
    _render_saved_paths(session.rejected_paths.values(), session)

    st.download_button(
        "Download session JSON",
        data=json.dumps(session.to_dict(), indent=2),
        file_name=f"exploration-session-{session.id}.json",
        mime="application/json",
    )


def _render_saved_paths(paths: object, session: ExplorationSession) -> None:
    rows = []
    for path in paths:
        rows.append(
            {
                "id": path.id,
                "score": path.score,
                "length": path.length,
                "summary": path.summary(),
                "note": session.user_notes.get(path.id, ""),
            }
        )
    if rows:
        st.dataframe(rows, use_container_width=True)
    else:
        st.caption("None yet.")


def _node_row(node: object) -> dict[str, object]:
    return {
        "name": node.name,
        "id": node.id,
        "labels": ", ".join(node.labels),
        "element_id": node.element_id,
    }


def _edge_row(edge: object) -> dict[str, object]:
    return {
        "type": edge.type,
        "subject": edge.subject,
        "object": edge.object,
        "element_id": edge.element_id,
    }


def _path_signature(path: Path) -> tuple[tuple[str, ...], tuple[str, ...]]:
    return (
        tuple(node.element_id for node in path.nodes),
        tuple(edge.element_id for edge in path.edges),
    )


def _get_exploration_session() -> ExplorationSession:
    if "exploration_session" not in st.session_state:
        st.session_state.exploration_session = ExplorationSession()
    return st.session_state.exploration_session


def _init_ui_state() -> None:
    st.session_state.setdefault("candidate_paths", [])
    st.session_state.setdefault("expanded_paths", [])


if __name__ == "__main__":
    main()
