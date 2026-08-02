from __future__ import annotations

from typing import Any
from uuid import uuid4

from explorer.backend.exploration_session import ExplorationSession
from explorer.backend.models import Edge, Node, Path


def new_client_state() -> dict[str, Any]:
    return {"key": str(uuid4())}


def empty_session_store() -> dict[str, Any]:
    return ExplorationSession().to_dict()


def session_from_store(data: dict[str, Any] | None) -> ExplorationSession:
    data = data or {}
    session = ExplorationSession(id=data["id"]) if data.get("id") else ExplorationSession()
    session.accepted_paths = {
        path["id"]: path_from_dict(path) for path in data.get("accepted_paths", [])
    }
    session.rejected_paths = {
        path["id"]: path_from_dict(path) for path in data.get("rejected_paths", [])
    }
    session.bookmarked_paths = {
        path["id"]: path_from_dict(path) for path in data.get("bookmarked_paths", [])
    }
    session.search_history = list(data.get("search_history", []))
    session.user_notes = dict(data.get("user_notes", {}))
    return session


def path_from_dict(data: dict[str, Any]) -> Path:
    return Path(
        id=data["id"],
        score=float(data.get("score") or 0.0),
        source=data.get("source") or "semantic",
        seed_subject=data.get("seed_subject"),
        seed_object=data.get("seed_object"),
        seed_predicate=data.get("seed_predicate"),
        evidence_text=data.get("evidence_text"),
        nodes=[
            Node(
                element_id=node["element_id"],
                id=node.get("id") or node["element_id"],
                name=node.get("name") or node.get("id") or node["element_id"],
                labels=list(node.get("labels", [])),
                properties=dict(node.get("properties", {})),
            )
            for node in data.get("nodes", [])
        ],
        edges=[
            Edge(
                element_id=edge["element_id"],
                type=edge["type"],
                start_element_id=edge["start_element_id"],
                end_element_id=edge["end_element_id"],
                subject=edge.get("subject", ""),
                object=edge.get("object", ""),
                properties=dict(edge.get("properties", {})),
            )
            for edge in data.get("edges", [])
        ],
    )


def find_path(paths: list[dict[str, Any]], path_id: str | None) -> dict[str, Any] | None:
    if not path_id:
        return None
    return next((path for path in paths if path.get("id") == path_id), None)


def selected_node_id(selected_node_data: list[dict[str, Any]] | None) -> str | None:
    if not selected_node_data:
        return None
    return selected_node_data[0].get("id")
