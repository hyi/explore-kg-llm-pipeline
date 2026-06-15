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
        limit: int = 3,
    ) -> list[Path]:
        subject = _first_present(metadata, "original_subject", "subject", "llm_subject")
        obj = _first_present(metadata, "original_object", "object", "llm_object")
        rel_id = _first_present(metadata, "id", "rel_id", "relationship_id")
        predicate = metadata.get("predicate")
        if not rel_id and (not subject or not obj):
            return []

        limit = _bounded_int(limit, minimum=1, maximum=25)
        cypher = """
        MATCH (matched_start)-[r]-(matched_end)
        WHERE (
            ($rel_id IS NOT NULL AND r.id = $rel_id)
            OR (
              $subject IS NOT NULL
              AND $object IS NOT NULL
              AND matched_start <> matched_end
              AND (
                matched_start.id = $subject
                OR toLower(coalesce(matched_start.name, "")) = toLower($subject)
              )
              AND (
                matched_end.id = $object
                OR toLower(coalesce(matched_end.name, "")) = toLower($object)
              )
            )
          )
          AND ($predicate IS NULL OR type(r) = $predicate)
        WITH DISTINCT r
        WITH startNode(r) AS start, endNode(r) AS end, r
        RETURN
          [n IN [start, end] | n {
            .*,
            element_id: elementId(n),
            labels: labels(n),
            id: coalesce(n.id, n.name, elementId(n)),
            display_name: coalesce(n.name, n.id, elementId(n))
          }] AS nodes,
          [r {
            .*,
            element_id: elementId(r),
            type: type(r),
            start_element_id: elementId(startNode(r)),
            end_element_id: elementId(endNode(r)),
            subject: coalesce(startNode(r).name, startNode(r).id, elementId(startNode(r))),
            object: coalesce(endNode(r).name, endNode(r).id, elementId(endNode(r)))
          }] AS relationships
        LIMIT $limit
        """

        paths = []
        with self._driver.session() as session:
            records = session.run(
                cypher,
                rel_id=rel_id,
                subject=subject,
                object=obj,
                predicate=predicate,
                limit=limit,
            )
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

    def context_subgraph(
        self,
        focus_element_ids: list[str],
    ) -> dict[str, list[dict[str, Any]]]:
        if not focus_element_ids:
            return {"nodes": [], "edges": []}

        cypher = """
        MATCH (focus)
        WHERE elementId(focus) IN $focus_ids
        WITH collect(DISTINCT focus) AS focus_nodes
        UNWIND focus_nodes AS focus
        CALL (focus) {
          MATCH (focus)-[r]-(neighbor)
          RETURN r, neighbor
        }
        WITH focus_nodes, collect(DISTINCT neighbor) AS neighbor_nodes, collect(DISTINCT r) AS relationships
        WITH focus_nodes + neighbor_nodes AS all_nodes, relationships
        RETURN
          [n IN all_nodes | n {
            .*,
            element_id: elementId(n),
            labels: labels(n),
            id: coalesce(n.id, n.name, elementId(n)),
            display_name: coalesce(n.name, n.id, elementId(n))
          }] AS nodes,
          [r IN relationships | r {
            .*,
            element_id: elementId(r),
            type: type(r),
            start_element_id: elementId(startNode(r)),
            end_element_id: elementId(endNode(r)),
            subject: coalesce(startNode(r).name, startNode(r).id, elementId(startNode(r))),
            object: coalesce(endNode(r).name, endNode(r).id, elementId(endNode(r)))
          }] AS relationships
        """

        with self._driver.session() as session:
            record = session.run(
                cypher,
                focus_ids=focus_element_ids,
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

    def has_unseen_neighbors(self, node_element_id: str, visible_node_ids: set[str]) -> bool:
        cypher = """
        MATCH (node)-[]-(neighbor)
        WHERE elementId(node) = $node_id
          AND NOT elementId(neighbor) IN $visible_node_ids
        RETURN count(neighbor) > 0 AS has_unseen
        """
        with self._driver.session() as session:
            record = session.run(
                cypher,
                node_id=node_element_id,
                visible_node_ids=list(visible_node_ids),
            ).single()
        return bool(record and record["has_unseen"])

    def unseen_neighbor_counts(self, node_element_ids: set[str]) -> dict[str, int]:
        if not node_element_ids:
            return {}

        visible_node_ids = list(node_element_ids)
        cypher = """
        MATCH (node)
        WHERE elementId(node) IN $visible_node_ids
        CALL (node) {
          MATCH (node)--(neighbor)
          WHERE NOT elementId(neighbor) IN $visible_node_ids
          RETURN count(DISTINCT neighbor) AS unseen_count
        }
        RETURN elementId(node) AS node_id, unseen_count
        """
        with self._driver.session() as session:
            records = session.run(cypher, visible_node_ids=visible_node_ids)
            return {record["node_id"]: int(record["unseen_count"]) for record in records}

    def semantic_similar_nodes(
        self,
        anchor_element_id: str,
        limit: int = 8,
    ) -> dict[str, list[dict[str, Any]]]:
        limit = _bounded_int(limit, minimum=1, maximum=25)
        cypher = """
        MATCH (anchor)
        WHERE elementId(anchor) = $anchor_id
          AND anchor.embedding IS NOT NULL
        MATCH (similar)
        WHERE similar <> anchor
          AND similar.embedding IS NOT NULL
        RETURN
          anchor {
            .*,
            element_id: elementId(anchor),
            labels: labels(anchor),
            id: coalesce(anchor.id, anchor.name, elementId(anchor)),
            display_name: coalesce(anchor.name, anchor.id, elementId(anchor))
          } AS anchor,
          similar {
            .*,
            element_id: elementId(similar),
            labels: labels(similar),
            id: coalesce(similar.id, similar.name, elementId(similar)),
            display_name: coalesce(similar.name, similar.id, elementId(similar))
          } AS similar,
          vector.similarity.cosine(anchor.embedding, similar.embedding) AS score
        ORDER BY score DESC
        LIMIT $limit
        """

        with self._driver.session() as session:
            records = list(session.run(cypher, anchor_id=anchor_element_id, limit=limit))

        nodes_by_id = {}
        edges = []
        for record in records:
            anchor = _node_from_projection(record["anchor"])
            similar = _node_from_projection(record["similar"])
            nodes_by_id[anchor.element_id] = {
                "id": anchor.element_id,
                "label": anchor.name,
                "labels": anchor.labels,
                "properties": anchor.properties,
                "is_focus": True,
            }
            nodes_by_id[similar.element_id] = {
                "id": similar.element_id,
                "label": similar.name,
                "labels": similar.labels,
                "properties": similar.properties,
                "is_focus": False,
            }
            score = float(record["score"] or 0.0)
            edges.append(
                {
                    "id": f"semantic:{anchor.element_id}:{similar.element_id}",
                    "source": anchor.element_id,
                    "target": similar.element_id,
                    "label": f"semantic similarity {score:.3f}",
                    "kind": "semantic",
                    "similarity": score,
                    "properties": {"similarity": score},
                }
            )

        return {
            "nodes": list(nodes_by_id.values()),
            "edges": edges,
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
