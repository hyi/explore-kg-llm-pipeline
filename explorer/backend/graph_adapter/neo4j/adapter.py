from __future__ import annotations

from typing import Any

from neo4j import GraphDatabase

from src.config import NEO4J_PASSWORD, NEO4J_URI, NEO4J_USERNAME
from explorer.backend.models import Edge, Node, Path


class Neo4jGraphAdapter:
    """Neo4j-only graph access layer for V1 path discovery."""

    def __init__(
        self,
        uri: str | None = None,
        username: str | None = None,
        password: str | None = None,
    ) -> None:
        self.uri = uri or NEO4J_URI
        self.username = username or NEO4J_USERNAME
        self.password = password or NEO4J_PASSWORD
        if not self.uri or not self.username or not self.password:
            raise ValueError("NEO4J_URI, NEO4J_USERNAME, and NEO4J_PASSWORD must be configured.")
        self._driver = GraphDatabase.driver(self.uri, auth=(self.username, self.password))

    def close(self) -> None:
        self._driver.close()

    def candidate_paths_for_semantic_hit(
        self,
        metadata: dict[str, Any],
        score: float,
        evidence_text: str | None = None,
        max_hops: int = 3,
        limit: int = 3,
    ) -> list[Path]:
        subject = _first_present(metadata, "original_subject", "subject", "llm_subject")
        obj = _first_present(metadata, "original_object", "object", "llm_object")
        if not subject or not obj:
            return []

        max_hops = _bounded_int(max_hops, minimum=1, maximum=5)
        limit = _bounded_int(limit, minimum=1, maximum=25)
        cypher = f"""
        MATCH (start), (end)
        WHERE start <> end
          AND (
            start.id = $subject OR start.identifier = $subject OR start.curie = $subject
            OR toLower(coalesce(start.name, "")) = toLower($subject)
          )
          AND (
            end.id = $object OR end.identifier = $object OR end.curie = $object
            OR toLower(coalesce(end.name, "")) = toLower($object)
          )
        MATCH p = shortestPath((start)-[*..{max_hops}]-(end))
        RETURN
          [n IN nodes(p) | n {{
            .*,
            element_id: elementId(n),
            labels: labels(n),
            id: coalesce(n.id, n.identifier, n.curie, n.name, elementId(n)),
            display_name: coalesce(n.name, n.id, n.identifier, n.curie, elementId(n))
          }}] AS nodes,
          [r IN relationships(p) | r {{
            .*,
            element_id: elementId(r),
            type: type(r),
            start_element_id: elementId(startNode(r)),
            end_element_id: elementId(endNode(r)),
            subject: coalesce(startNode(r).name, startNode(r).id, startNode(r).identifier, startNode(r).curie, elementId(startNode(r))),
            object: coalesce(endNode(r).name, endNode(r).id, endNode(r).identifier, endNode(r).curie, elementId(endNode(r)))
          }}] AS relationships
        LIMIT $limit
        """

        paths = []
        with self._driver.session() as session:
            records = session.run(cypher, subject=subject, object=obj, limit=limit)
            for record in records:
                path = self._record_to_path(
                    record,
                    score=score,
                    source="semantic_path",
                    seed_subject=metadata.get("llm_subject") or subject,
                    seed_object=metadata.get("llm_object") or obj,
                    seed_predicate=metadata.get("predicate"),
                    evidence_text=evidence_text,
                )
                if path:
                    paths.append(path)
        return paths

    def expand_path(self, path: Path, limit: int = 10) -> list[Path]:
        if not path.nodes:
            return []

        limit = _bounded_int(limit, minimum=1, maximum=50)
        cypher = """
        MATCH (boundary)
        WHERE elementId(boundary) IN $endpoint_ids
        MATCH (boundary)-[r]-(neighbor)
        WHERE NOT elementId(neighbor) IN $existing_ids
        RETURN
          boundary {
            .*,
            element_id: elementId(boundary),
            labels: labels(boundary),
            id: coalesce(boundary.id, boundary.identifier, boundary.curie, boundary.name, elementId(boundary)),
            display_name: coalesce(boundary.name, boundary.id, boundary.identifier, boundary.curie, elementId(boundary))
          } AS boundary,
          neighbor {
            .*,
            element_id: elementId(neighbor),
            labels: labels(neighbor),
            id: coalesce(neighbor.id, neighbor.identifier, neighbor.curie, neighbor.name, elementId(neighbor)),
            display_name: coalesce(neighbor.name, neighbor.id, neighbor.identifier, neighbor.curie, elementId(neighbor))
          } AS neighbor,
          r {
            .*,
            element_id: elementId(r),
            type: type(r),
            start_element_id: elementId(startNode(r)),
            end_element_id: elementId(endNode(r)),
            subject: coalesce(startNode(r).name, startNode(r).id, startNode(r).identifier, startNode(r).curie, elementId(startNode(r))),
            object: coalesce(endNode(r).name, endNode(r).id, endNode(r).identifier, endNode(r).curie, elementId(endNode(r)))
          } AS relationship
        LIMIT $limit
        """

        expansions = []
        with self._driver.session() as session:
            records = session.run(
                cypher,
                endpoint_ids=path.endpoint_element_ids,
                existing_ids=path.node_element_ids,
                limit=limit,
            )
            for record in records:
                expanded = self._append_expansion(path, record["boundary"], record["neighbor"], record["relationship"])
                if expanded:
                    expansions.append(expanded)
        return expansions

    def context_subgraph(
        self,
        focus_element_ids: list[str],
        neighbors_per_node: int = 8,
    ) -> dict[str, list[dict[str, Any]]]:
        if not focus_element_ids:
            return {"nodes": [], "edges": []}

        neighbors_per_node = _bounded_int(neighbors_per_node, minimum=1, maximum=50)
        cypher = """
        MATCH (focus)
        WHERE elementId(focus) IN $focus_ids
        WITH collect(DISTINCT focus) AS focus_nodes
        UNWIND focus_nodes AS focus
        CALL {
          WITH focus
          MATCH (focus)-[r]-(neighbor)
          RETURN r, neighbor
          LIMIT $neighbors_per_node
        }
        WITH focus_nodes, collect(DISTINCT neighbor) AS neighbor_nodes, collect(DISTINCT r) AS relationships
        WITH focus_nodes + neighbor_nodes AS all_nodes, relationships
        RETURN
          [n IN all_nodes | n {
            .*,
            element_id: elementId(n),
            labels: labels(n),
            id: coalesce(n.id, n.identifier, n.curie, n.name, elementId(n)),
            display_name: coalesce(n.name, n.id, n.identifier, n.curie, elementId(n))
          }] AS nodes,
          [r IN relationships | r {
            .*,
            element_id: elementId(r),
            type: type(r),
            start_element_id: elementId(startNode(r)),
            end_element_id: elementId(endNode(r)),
            subject: coalesce(startNode(r).name, startNode(r).id, startNode(r).identifier, startNode(r).curie, elementId(startNode(r))),
            object: coalesce(endNode(r).name, endNode(r).id, endNode(r).identifier, endNode(r).curie, elementId(endNode(r)))
          }] AS relationships
        """

        with self._driver.session() as session:
            record = session.run(
                cypher,
                focus_ids=focus_element_ids,
                neighbors_per_node=neighbors_per_node,
            ).single()

        if not record:
            return {"nodes": [], "edges": []}

        nodes_by_id = {}
        for node_projection in record["nodes"]:
            node = _node_from_projection(node_projection)
            nodes_by_id[node.element_id] = {
                "id": node.element_id,
                "label": node.name,
                "labels": node.labels,
                "properties": node.properties,
                "is_focus": node.element_id in focus_element_ids,
            }

        edges_by_id = {}
        for edge_projection in record["relationships"]:
            edge = _edge_from_projection(edge_projection)
            edges_by_id[edge.element_id] = {
                "id": edge.element_id,
                "source": edge.start_element_id,
                "target": edge.end_element_id,
                "label": edge.type,
                "properties": edge.properties,
            }

        return {
            "nodes": list(nodes_by_id.values()),
            "edges": list(edges_by_id.values()),
        }

    def _record_to_path(
        self,
        record: Any,
        score: float,
        source: str,
        seed_subject: str | None,
        seed_object: str | None,
        seed_predicate: str | None,
        evidence_text: str | None,
    ) -> Path | None:
        nodes = [_node_from_projection(node) for node in record["nodes"]]
        edges = [_edge_from_projection(edge) for edge in record["relationships"]]
        if not nodes:
            return None
        return Path(
            nodes=nodes,
            edges=edges,
            score=float(score or 0.0),
            source=source,
            seed_subject=seed_subject,
            seed_object=seed_object,
            seed_predicate=seed_predicate,
            evidence_text=evidence_text,
        )

    def _append_expansion(
        self,
        path: Path,
        boundary_projection: dict[str, Any],
        neighbor_projection: dict[str, Any],
        edge_projection: dict[str, Any],
    ) -> Path | None:
        boundary = _node_from_projection(boundary_projection)
        neighbor = _node_from_projection(neighbor_projection)
        edge = _edge_from_projection(edge_projection)

        if boundary.element_id == path.nodes[0].element_id:
            nodes = [neighbor, *path.nodes]
            edges = [edge, *path.edges]
        elif boundary.element_id == path.nodes[-1].element_id:
            nodes = [*path.nodes, neighbor]
            edges = [*path.edges, edge]
        else:
            return None

        return Path(
            nodes=nodes,
            edges=edges,
            score=path.score,
            source="expanded_path",
            seed_subject=path.seed_subject,
            seed_object=path.seed_object,
            seed_predicate=path.seed_predicate,
            evidence_text=path.evidence_text,
        )


def _node_from_projection(projection: dict[str, Any]) -> Node:
    properties = dict(projection)
    element_id = str(properties.pop("element_id"))
    labels = list(properties.pop("labels", []))
    node_id = str(properties.pop("id", element_id))
    name = str(properties.pop("display_name", node_id))
    properties.pop("embedding", None)
    return Node(element_id=element_id, id=node_id, name=name, labels=labels, properties=properties)


def _edge_from_projection(projection: dict[str, Any]) -> Edge:
    properties = dict(projection)
    element_id = str(properties.pop("element_id"))
    edge_type = str(properties.pop("type"))
    start_element_id = str(properties.pop("start_element_id"))
    end_element_id = str(properties.pop("end_element_id"))
    subject = str(properties.pop("subject", ""))
    obj = str(properties.pop("object", ""))
    properties.pop("embedding", None)
    return Edge(
        element_id=element_id,
        type=edge_type,
        start_element_id=start_element_id,
        end_element_id=end_element_id,
        subject=subject,
        object=obj,
        properties=properties,
    )


def _first_present(metadata: dict[str, Any], *keys: str) -> str | None:
    for key in keys:
        value = metadata.get(key)
        if value:
            return str(value)
    return None


def _bounded_int(value: int, minimum: int, maximum: int) -> int:
    return max(minimum, min(maximum, int(value)))
