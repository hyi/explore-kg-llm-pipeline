from __future__ import annotations

from typing import Any

from langchain_core.documents import Document
from neo4j import GraphDatabase

from explorer.backend.models import Edge, Node, Path
from explorer.backend.semantic_search.neighborhood import (
    NeighborhoodExpansionConfig,
    rank_neighborhood_candidates,
)
from explorer.backend.semantic_search.ranking import compact_anchor_metadata
from explorer.backend.semantic_search.retrieval import tokenize_keyword_query
from src.config import NEO4J_PASSWORD, NEO4J_URI, NEO4J_USERNAME
from src.embeddings.embedding_utils import (
    cypher_escape_identifier,
    get_embedding_property,
)


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

    def enrich_relationship_hits(self, candidates: list[Any]) -> list[Any]:
        candidate_keys = []
        for position, candidate in enumerate(candidates):
            metadata = dict(getattr(candidate, "metadata", {}) or {})
            candidate_keys.append(
                {
                    "position": position,
                    "rel_id": _first_present(metadata, "id", "rel_id", "relationship_id"),
                    "subject": _first_present(metadata, "original_subject", "subject", "llm_subject"),
                    "object": _first_present(metadata, "original_object", "object", "llm_object"),
                    "predicate": metadata.get("predicate"),
                }
            )

        if not candidate_keys:
            return []

        cypher = """
        UNWIND $candidates AS candidate
        MATCH (start)-[r]->(end)
        WHERE (
            candidate.rel_id IS NOT NULL
            AND r.id = candidate.rel_id
          )
          OR (
            candidate.subject IS NOT NULL
            AND candidate.object IS NOT NULL
            AND (
              start.id = candidate.subject
              OR toLower(coalesce(start.name, "")) = toLower(candidate.subject)
            )
            AND (
              end.id = candidate.object
              OR toLower(coalesce(end.name, "")) = toLower(candidate.object)
            )
            AND (candidate.predicate IS NULL OR type(r) = candidate.predicate)
          )
        WITH candidate, r, start, end, candidate.rel_id IS NOT NULL AND r.id = candidate.rel_id AS exact_rel_id
        ORDER BY candidate.position, exact_rel_id DESC, elementId(r)
        WITH candidate, collect({
          relationship_id: coalesce(r.id, elementId(r)),
          relationship_element_id: elementId(r),
          predicate: type(r),
          subject_id: coalesce(start.id, start.name, elementId(start)),
          object_id: coalesce(end.id, end.name, elementId(end)),
          subject_name: coalesce(start.name, start.id, elementId(start)),
          object_name: coalesce(end.name, end.id, elementId(end)),
          subject_labels: labels(start),
          object_labels: labels(end),
          publications: r.publications,
          llm_abstract_id: r.llm_abstract_id,
          abstract_title: r.abstract_title,
          publication_id: CASE
            WHEN r.publications IS NOT NULL AND size(r.publications) > 0 THEN r.publications[0]
            WHEN r.llm_abstract_id IS NOT NULL THEN toString(r.llm_abstract_id)
            ELSE r.abstract_title
          END
        }) AS matches
        RETURN candidate.position AS position, head(matches) AS metadata
        """

        with self._driver.session() as session:
            records = list(session.run(cypher, candidates=candidate_keys))

        enrichment_by_position = {
            int(record["position"]): dict(record["metadata"] or {})
            for record in records
            if record["metadata"]
        }
        enriched = []
        for position, candidate in enumerate(candidates):
            metadata = dict(getattr(candidate, "metadata", {}) or {})
            metadata.update(enrichment_by_position.get(position, {}))
            enriched.append(_candidate_with_metadata(candidate, metadata))
        return enriched

    def keyword_relationship_search(self, query: str, k: int = 25) -> list[Document]:
        tokens = tokenize_keyword_query(query)
        if not tokens or k <= 0:
            return []

        limit = _bounded_int(k, minimum=1, maximum=100)
        cypher = """
        MATCH (start)-[r]->(end)
        WITH start, r, end, properties(r) AS rel_props
        WITH start, r, end,
          toLower(
            coalesce(rel_props.edge_text, "") + " " +
            coalesce(rel_props.llm_subject, "") + " " +
            coalesce(rel_props.llm_subject_qualifier, "") + " " +
            type(r) + " " +
            coalesce(rel_props.llm_relationship, "") + " " +
            coalesce(rel_props.llm_object, "") + " " +
            coalesce(rel_props.llm_object_qualifier, "") + " " +
            coalesce(rel_props.llm_statement_qualifier, "") + " " +
            coalesce(start.name, start.id, "") + " " +
            coalesce(end.name, end.id, "")
          ) AS edge_text,
          toLower(coalesce(rel_props.title_text, rel_props.abstract_title, "")) AS title_text,
          toLower(
            coalesce(rel_props.context_text, "") + " " +
            coalesce(rel_props.supporting_sentence, "") + " " +
            coalesce(rel_props.abstract_text, "")
          ) AS context_text
        WITH start, r, end, edge_text, title_text, context_text,
          [token IN $tokens WHERE edge_text CONTAINS token] AS edge_matches,
          [token IN $tokens WHERE title_text CONTAINS token] AS title_matches,
          [token IN $tokens WHERE context_text CONTAINS token] AS context_matches
        WITH start, r, end, edge_text, title_text, context_text,
          edge_matches, title_matches, context_matches,
          3.0 * size(edge_matches) + 2.0 * size(title_matches) + 1.0 * size(context_matches) AS score
        WHERE score > 0
        RETURN
          coalesce(r.id, elementId(r)) AS relationship_id,
          elementId(r) AS relationship_element_id,
          type(r) AS predicate,
          coalesce(start.id, start.name, elementId(start)) AS subject_id,
          coalesce(end.id, end.name, elementId(end)) AS object_id,
          coalesce(start.name, start.id, elementId(start)) AS subject_name,
          coalesce(end.name, end.id, elementId(end)) AS object_name,
          labels(start) AS subject_labels,
          labels(end) AS object_labels,
          r.publications AS publications,
          r.llm_abstract_id AS llm_abstract_id,
          r.abstract_title AS abstract_title,
          CASE
            WHEN r.publications IS NOT NULL AND size(r.publications) > 0 THEN r.publications[0]
            WHEN r.llm_abstract_id IS NOT NULL THEN toString(r.llm_abstract_id)
            ELSE r.abstract_title
          END AS publication_id,
          coalesce(r.llm_subject, start.name, start.id) AS llm_subject,
          coalesce(r.llm_object, end.name, end.id) AS llm_object,
          r.llm_relationship AS llm_relationship,
          r.semantic_text AS semantic_text,
          edge_matches,
          title_matches,
          context_matches,
          score
        ORDER BY score DESC, elementId(r)
        LIMIT $limit
        """

        with self._driver.session() as session:
            records = list(session.run(cypher, tokens=tokens, limit=limit))

        results: list[Document] = []
        for record in records:
            metadata = {
                "id": record["relationship_id"],
                "relationship_id": record["relationship_id"],
                "element_id": record["relationship_element_id"],
                "relationship_element_id": record["relationship_element_id"],
                "predicate": record["predicate"],
                "original_subject": record["subject_id"],
                "original_object": record["object_id"],
                "subject": record["subject_id"],
                "object": record["object_id"],
                "subject_name": record["subject_name"],
                "object_name": record["object_name"],
                "llm_subject": record["llm_subject"],
                "llm_object": record["llm_object"],
                "llm_relationship": record["llm_relationship"],
                "subject_labels": list(record["subject_labels"] or []),
                "object_labels": list(record["object_labels"] or []),
                "publications": record["publications"],
                "llm_abstract_id": record["llm_abstract_id"],
                "abstract_title": record["abstract_title"],
                "publication_id": record["publication_id"],
                "semantic_text": record["semantic_text"],
                "score": float(record["score"] or 0.0),
                "keyword_score": float(record["score"] or 0.0),
                "keyword_edge_matches": list(record["edge_matches"] or []),
                "keyword_title_matches": list(record["title_matches"] or []),
                "keyword_context_matches": list(record["context_matches"] or []),
                "retrieval_method": "neo4j_keyword_scan",
            }
            content_parts = [
                metadata.get("llm_subject"),
                metadata.get("predicate"),
                metadata.get("llm_object"),
                metadata.get("abstract_title"),
            ]
            results.append(
                Document(
                    page_content=" ".join(str(part) for part in content_parts if part),
                    metadata=metadata,
                )
            )
        return results

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
                    anchor_metadata=compact_anchor_metadata(metadata),
                )
                if path:
                    paths.append(path)
        return paths

    def context_subgraph(
        self,
        focus_element_ids: list[str],
        *,
        query: str | None = None,
        limit_per_focus: int | None = None,
        node_categories: list[str] | tuple[str, ...] | None = None,
        predicates: list[str] | tuple[str, ...] | None = None,
        direction: str = "either",
    ) -> dict[str, list[dict[str, Any]]]:
        if not focus_element_ids:
            return {"nodes": [], "edges": []}
        if query or limit_per_focus is not None or node_categories or predicates or direction != "either":
            return self._ranked_context_subgraph(
                focus_element_ids,
                query=query or "",
                limit_per_focus=limit_per_focus,
                node_categories=node_categories,
                predicates=predicates,
                direction=direction,
            )

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

    def _ranked_context_subgraph(
        self,
        focus_element_ids: list[str],
        *,
        query: str,
        limit_per_focus: int | None,
        node_categories: list[str] | tuple[str, ...] | None,
        predicates: list[str] | tuple[str, ...] | None,
        direction: str,
    ) -> dict[str, list[dict[str, Any]]]:
        limit = limit_per_focus if limit_per_focus is not None else 12
        config = NeighborhoodExpansionConfig(
            limit=limit,
            node_categories=tuple(node_categories or ()),
            predicates=tuple(predicates or ()),
            direction=direction,
        )
        cypher = """
        MATCH (focus)
        WHERE elementId(focus) IN $focus_ids
        WITH collect(DISTINCT focus) AS focus_nodes
        UNWIND focus_nodes AS focus
        CALL (focus) {
          MATCH (focus)-[r]-(neighbor)
          WITH focus, r, neighbor,
            CASE WHEN elementId(startNode(r)) = elementId(focus) THEN "outgoing" ELSE "incoming" END AS direction
          RETURN {
            focus_id: elementId(focus),
            edge_id: elementId(r),
            neighbor_id: elementId(neighbor),
            predicate: type(r),
            direction: direction,
            focus: focus {
              .*,
              element_id: elementId(focus),
              labels: labels(focus),
              id: coalesce(focus.id, focus.name, elementId(focus)),
              display_name: coalesce(focus.name, focus.id, elementId(focus))
            },
            neighbor: neighbor {
              .*,
              element_id: elementId(neighbor),
              labels: labels(neighbor),
              id: coalesce(neighbor.id, neighbor.name, elementId(neighbor)),
              display_name: coalesce(neighbor.name, neighbor.id, elementId(neighbor))
            },
            edge: r {
              .*,
              element_id: elementId(r),
              type: type(r),
              start_element_id: elementId(startNode(r)),
              end_element_id: elementId(endNode(r)),
              subject: coalesce(startNode(r).name, startNode(r).id, elementId(startNode(r))),
              object: coalesce(endNode(r).name, endNode(r).id, elementId(endNode(r)))
            }
          } AS candidate
        }
        WITH focus_nodes, candidate
        ORDER BY candidate.focus_id, candidate.edge_id, candidate.neighbor_id
        RETURN
          [n IN focus_nodes | n {
            .*,
            element_id: elementId(n),
            labels: labels(n),
            id: coalesce(n.id, n.name, elementId(n)),
            display_name: coalesce(n.name, n.id, elementId(n))
          }] AS focus_nodes,
          collect(candidate) AS candidates
        """

        with self._driver.session() as session:
            record = session.run(
                cypher,
                focus_ids=focus_element_ids,
            ).single()

        if not record:
            return {"nodes": [], "edges": [], "diagnostics": {"ranked_neighborhoods": {}}}

        nodes_by_id = {}
        for node_projection in record["focus_nodes"]:
            node = _node_from_projection(node_projection)
            nodes_by_id[node.element_id] = {
                "id": node.element_id,
                "label": node.name,
                "labels": node.labels,
                "properties": node.properties,
                "is_focus": True,
            }

        candidate_groups: dict[str, list[dict[str, Any]]] = {focus_id: [] for focus_id in focus_element_ids}
        for candidate in record["candidates"]:
            focus = _node_from_projection(candidate["focus"])
            neighbor = _node_from_projection(candidate["neighbor"])
            edge = _edge_from_projection(candidate["edge"])
            row = {
                "predicate": candidate["predicate"],
                "direction": candidate["direction"],
                "focus": {
                    "id": focus.element_id,
                    "label": focus.name,
                    "labels": focus.labels,
                },
                "neighbor": {
                    "id": neighbor.element_id,
                    "label": neighbor.name,
                    "labels": neighbor.labels,
                    "properties": neighbor.properties,
                },
                "edge": {
                    "id": edge.element_id,
                    "source": edge.start_element_id,
                    "target": edge.end_element_id,
                    "label": edge.type,
                    "type": edge.type,
                    "subject": edge.subject,
                    "object": edge.object,
                    "semantic_text": edge.properties.get("semantic_text"),
                    "abstract_title": edge.properties.get("abstract_title"),
                    "supporting_sentence": edge.properties.get("supporting_sentence"),
                    "properties": edge.properties,
                },
            }
            candidate_groups.setdefault(focus.element_id, []).append(row)

        edges_by_id = {}
        diagnostics_by_focus: dict[str, Any] = {}
        for focus_id, candidates in candidate_groups.items():
            ranking_result = rank_neighborhood_candidates(
                candidates,
                query=query,
                config=config,
            )
            diagnostics_by_focus[focus_id] = ranking_result.diagnostics
            for candidate in ranking_result.candidates:
                neighbor = candidate["neighbor"]
                edge = candidate["edge"]
                nodes_by_id[neighbor["id"]] = {
                    "id": neighbor["id"],
                    "label": neighbor["label"],
                    "labels": neighbor["labels"],
                    "properties": neighbor["properties"],
                    "is_focus": False,
                    "ranked_from_focus": focus_id,
                    "neighborhood_score": candidate["neighborhood_score"],
                }
                edge_properties = dict(edge["properties"])
                edge_properties.update(
                    {
                        "neighborhood_score": candidate["neighborhood_score"],
                        "neighborhood_ranking_components": candidate["ranking_components"],
                        "neighborhood_ranking_reasons": candidate["ranking_reasons"],
                        "matched_query_tokens": candidate["matched_query_tokens"],
                        "matched_query_facets": candidate["matched_query_facets"],
                        "expansion_focus_id": focus_id,
                    }
                )
                edges_by_id[edge["id"]] = {
                    "id": edge["id"],
                    "source": edge["source"],
                    "target": edge["target"],
                    "label": edge["label"],
                    "properties": edge_properties,
                    "neighborhood_score": candidate["neighborhood_score"],
                    "ranking_reasons": candidate["ranking_reasons"],
                }

        return {
            "nodes": list(nodes_by_id.values()),
            "edges": list(edges_by_id.values()),
            "diagnostics": {"ranked_neighborhoods": diagnostics_by_focus},
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
        embedding_property = cypher_escape_identifier(get_embedding_property())
        cypher = f"""
        MATCH (anchor)
        WHERE elementId(anchor) = $anchor_id
          AND anchor.`{embedding_property}` IS NOT NULL
        MATCH (similar)
        WHERE similar <> anchor
          AND similar.`{embedding_property}` IS NOT NULL
        RETURN
          anchor {{
            .*,
            element_id: elementId(anchor),
            labels: labels(anchor),
            id: coalesce(anchor.id, anchor.name, elementId(anchor)),
            display_name: coalesce(anchor.name, anchor.id, elementId(anchor))
          }} AS anchor,
          similar {{
            .*,
            element_id: elementId(similar),
            labels: labels(similar),
            id: coalesce(similar.id, similar.name, elementId(similar)),
            display_name: coalesce(similar.name, similar.id, elementId(similar))
          }} AS similar,
          vector.similarity.cosine(anchor.`{embedding_property}`, similar.`{embedding_property}`) AS score
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
        anchor_metadata: dict[str, Any] | None = None,
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
            anchor_metadata=anchor_metadata or {},
        )

def _node_from_projection(projection: dict[str, Any]) -> Node:
    properties = dict(projection)
    element_id = str(properties.pop("element_id"))
    labels = list(properties.pop("labels", []))
    node_id = str(properties.pop("id", element_id))
    name = str(properties.pop("display_name", node_id))
    properties.pop("embedding", None)
    properties.pop("sapbert_embedding", None)
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
    properties.pop("sapbert_embedding", None)
    return Edge(
        element_id=element_id,
        type=edge_type,
        start_element_id=start_element_id,
        end_element_id=end_element_id,
        subject=subject,
        object=obj,
        properties=properties,
    )


def _candidate_with_metadata(candidate: Any, metadata: dict[str, Any]) -> Any:
    if hasattr(candidate, "model_copy"):
        return candidate.model_copy(update={"metadata": metadata})

    import copy

    cloned = copy.copy(candidate)
    cloned.metadata = metadata
    return cloned


def _first_present(metadata: dict[str, Any], *keys: str) -> str | None:
    for key in keys:
        value = metadata.get(key)
        if value:
            return str(value)
    return None


def _bounded_int(value: int, minimum: int, maximum: int) -> int:
    return max(minimum, min(maximum, int(value)))
