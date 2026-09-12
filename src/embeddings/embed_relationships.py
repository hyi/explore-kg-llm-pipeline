# src/explore_kg_llm/embeddings/embed_relationships.py
import math
from functools import lru_cache
from pathlib import Path

from langchain_community.vectorstores import Neo4jVector
from langchain_core.documents import Document
from neo4j import GraphDatabase

from src.config import NEO4J_PASSWORD, NEO4J_URI, NEO4J_USERNAME
from src.embeddings.embedding_utils import (
    cypher_escape_identifier,
    embedding_index_name,
    get_embedding_client,
    get_embedding_dimensions,
    get_embedding_index_suffix,
    get_embedding_property,
)

SEMANTIC_TEXT_CYPHER_PATH = (
    Path(__file__).resolve().parents[1]
    / "cypher"
    / "create_semantic_text.cypher"
)

RELATIONSHIP_TYPES = [
    "biolink:related_to",
    "biolink:associated_with",
    "biolink:correlated_with",
    "biolink:positively_correlated_with",
    "biolink:genetically_associated_with",
    "biolink:contributes_to",
    "biolink:causes",
    "biolink:affects",
    "biolink:regulates",

    "biolink:treats",
    "biolink:studied_to_treat",
    "biolink:treats_or_applied_or_studied_to_treat",
    "biolink:ameliorates_condition",
    "biolink:exacerbates_condition",
    "biolink:preventative_for_condition",

    "biolink:affects_response_to",
    "biolink:increases_response_to",
    "biolink:decreases_response_to",
    "biolink:associated_with_resistance_to"
]

def relationship_similarity_search(query, k=5, model: str | None = None):
    rel_stores = _relationship_stores(model)
    if not rel_stores:
        _relationship_stores.cache_clear()
        return _relationship_similarity_search_scan(query, k=k, model=model)

    embedding_property = get_embedding_property(model)
    embedding_dimensions = get_embedding_dimensions(model=model)
    results = []
    for rel_type, store_info in rel_stores.items():
        store = store_info["store"]
        hits = store.similarity_search_with_score(query, k=k)
        for doc, score in hits:
            doc.metadata["predicate"] = doc.metadata.get("predicate") or rel_type
            doc.metadata["score"] = score
            doc.metadata["retrieval_model"] = model or "configured"
            doc.metadata["retrieval_embedding_property"] = embedding_property
            doc.metadata["retrieval_expected_dimensions"] = embedding_dimensions
            doc.metadata["retrieval_method"] = "neo4j_vector_index"
            doc.metadata["retrieval_index_name"] = store_info["index_name"]
            results.append(doc)

    return sorted(
        results,
        key=lambda x: x.metadata.get("score", 0),
        reverse=True
    )[:k]


@lru_cache(maxsize=1)
def _relationship_stores(model: str | None = None):
    rel_stores = {}
    available_indexes = _relationship_vector_indexes_by_type()
    embedding_client = get_embedding_client(model)
    embedding_property = get_embedding_property(model)
    retrieval_query = _relationship_retrieval_query(embedding_property)

    for rel_type in sorted(available_indexes):
        for index_name in _relationship_index_candidates(
            rel_type,
            available_indexes,
            model,
        ):
            try:
                rel_stores[rel_type] = {
                    "store": Neo4jVector.from_existing_relationship_index(
                        embedding=embedding_client,
                        url=NEO4J_URI,
                        username=NEO4J_USERNAME,
                        password=NEO4J_PASSWORD,
                        index_name=index_name,
                        text_node_property="semantic_text",
                        retrieval_query=retrieval_query,
                    ),
                    "index_name": index_name,
                }
                break
            except ValueError as exc:
                if "does not exist" not in str(exc).lower():
                    raise

    return rel_stores


def _relationship_retrieval_query(embedding_property: str) -> str:
    escaped_embedding_property = cypher_escape_identifier(embedding_property)
    return f"""
    RETURN relationship.semantic_text AS text, score,
           relationship {{
             .*,
             semantic_text: Null,
             `{escaped_embedding_property}`: Null,
             id: coalesce(relationship.id, elementId(relationship)),
             relationship_id: coalesce(relationship.id, elementId(relationship)),
             element_id: elementId(relationship),
             relationship_element_id: elementId(relationship),
             predicate: type(relationship),
             original_subject: coalesce(startNode(relationship).id, startNode(relationship).name, elementId(startNode(relationship))),
             original_object: coalesce(endNode(relationship).id, endNode(relationship).name, elementId(endNode(relationship))),
             subject: coalesce(startNode(relationship).id, startNode(relationship).name, elementId(startNode(relationship))),
             object: coalesce(endNode(relationship).id, endNode(relationship).name, elementId(endNode(relationship))),
             subject_name: coalesce(startNode(relationship).name, startNode(relationship).id, elementId(startNode(relationship))),
             object_name: coalesce(endNode(relationship).name, endNode(relationship).id, elementId(endNode(relationship))),
             subject_labels: labels(startNode(relationship)),
             object_labels: labels(endNode(relationship)),
             publication_id: CASE
               WHEN relationship.publications IS NOT NULL AND size(relationship.publications) > 0 THEN relationship.publications[0]
               WHEN relationship.llm_abstract_id IS NOT NULL THEN toString(relationship.llm_abstract_id)
               ELSE relationship.abstract_title
             END
           }} AS metadata
    """


def _relationship_similarity_search_scan(query, k=5, model: str | None = None):
    embedding_property = get_embedding_property(model)
    escaped_embedding_property = cypher_escape_identifier(embedding_property)
    embedding_client = get_embedding_client(model)
    query_embedding = embedding_client.embed_query(query)
    query_embedding_dimensions = len(query_embedding)
    driver = GraphDatabase.driver(
        NEO4J_URI,
        auth=(NEO4J_USERNAME, NEO4J_PASSWORD),
    )
    try:
        with driver.session() as session:
            rows = session.run(
                f"""
                MATCH ()-[r]->()
                WHERE r.`{escaped_embedding_property}` IS NOT NULL
                  AND r.semantic_text IS NOT NULL
                RETURN type(r) AS predicate,
                       r {{ .* }} AS metadata,
                       r.semantic_text AS text,
                       r.`{escaped_embedding_property}` AS embedding
                """
            )
            results = []
            for row in rows:
                metadata = dict(row["metadata"])
                metadata.pop(embedding_property, None)
                embedding = row["embedding"]
                if not embedding:
                    continue
                metadata["predicate"] = row["predicate"]
                metadata["score"] = _cosine_similarity(query_embedding, embedding)
                metadata["retrieval_model"] = model or "configured"
                metadata["retrieval_embedding_property"] = embedding_property
                metadata["retrieval_expected_dimensions"] = get_embedding_dimensions(
                    embedding_client,
                    model=model,
                )
                metadata["retrieval_query_embedding_dimensions"] = query_embedding_dimensions
                metadata["retrieval_method"] = "neo4j_scan"
                results.append(Document(page_content=row["text"], metadata=metadata))
    finally:
        driver.close()

    return sorted(
        results,
        key=lambda x: x.metadata.get("score", 0),
        reverse=True
    )[:k]


def _relationship_vector_indexes_by_type():
    driver = GraphDatabase.driver(
        NEO4J_URI,
        auth=(NEO4J_USERNAME, NEO4J_PASSWORD),
    )
    try:
        with driver.session() as session:
            rows = session.run(
                """
                SHOW VECTOR INDEXES
                YIELD name, entityType, labelsOrTypes
                WHERE entityType = 'RELATIONSHIP'
                RETURN name, labelsOrTypes
                """
            )
            indexes = {}
            for row in rows:
                for rel_type in row["labelsOrTypes"]:
                    indexes.setdefault(rel_type, []).append(row["name"])
            return indexes
    except Exception:  # noqa: BLE001 - absence/misconfiguration of optional vector indexes triggers scan fallback.
        return {}
    finally:
        driver.close()


def _relationship_index_candidates(
    rel_type,
    available_indexes=None,
    model: str | None = None,
):
    candidates = [_relationship_index_name(rel_type, model)]

    if available_indexes and rel_type in available_indexes:
        candidates.extend(
            index_name
            for index_name in available_indexes[rel_type]
            if _index_name_matches_model(index_name, model)
        )

    if not get_embedding_index_suffix(model):
        candidates.extend(
            [
                f"{rel_type.lower()}_vector_idx",
                f"{rel_type.replace(':', '_').lower()}_vector_idx",
                f"{rel_type.replace(':', '_')}_vector_idx",
            ]
        )

    deduplicated = []
    for index_name in candidates:
        if not _index_name_matches_model(index_name, model):
            continue
        if index_name not in deduplicated:
            deduplicated.append(index_name)
    return deduplicated


def _index_name_matches_model(index_name: str, model: str | None = None) -> bool:
    suffix = get_embedding_index_suffix(model)
    index_name = index_name.lower()
    if suffix:
        return suffix.lower() in index_name
    return "_sapbert" not in index_name


def _relationship_index_name(rel_type, model: str | None = None):
    return embedding_index_name(f"{rel_type.lower()}_vector_idx", model=model)


def _cosine_similarity(left, right):
    numerator = sum(a * b for a, b in zip(left, right))
    left_norm = math.sqrt(sum(a * a for a in left))
    right_norm = math.sqrt(sum(b * b for b in right))
    if not left_norm or not right_norm:
        return 0.0
    return numerator / (left_norm * right_norm)


def create_relationship_semantic_text(session):
    query = SEMANTIC_TEXT_CYPHER_PATH.read_text().strip().removesuffix(";")
    session.run(query).consume()
    print("Relationship semantic_text created")


def embed_relationships():
    embedding_client = get_embedding_client()
    embedding_property = get_embedding_property()
    embedding_dimensions = get_embedding_dimensions(embedding_client)
    escaped_embedding_property = cypher_escape_identifier(embedding_property)

    driver = GraphDatabase.driver(
        NEO4J_URI,
        auth=(NEO4J_USERNAME, NEO4J_PASSWORD),
    )

    with driver.session() as session:
        create_relationship_semantic_text(session)

        results = session.run(
            f"""
            MATCH ()-[r]->()
            WHERE r.semantic_text IS NOT NULL
              AND r.`{escaped_embedding_property}` IS NULL
            RETURN elementId(r) AS rid, 
            type(r) as rel_type, r.semantic_text AS text
            """
        )
        for record in results:
            rid = record["rid"]
            text = record["text"]
            vector = embedding_client.embed_query(text)

            session.run(
                f"""
                MATCH ()-[r]->() 
                WHERE elementId(r) = $rid
                SET r.`{escaped_embedding_property}` = $embedding
                """,
                rid=rid,
                embedding=vector,
            )

        rel_index_list = []
        for rel_type in _relationship_types_with_semantic_text(session):
            # create relationship vector index
            index_name = _relationship_index_name(rel_type)
            escaped_index_name = cypher_escape_identifier(index_name)
            session.run(
                f"""
                CREATE VECTOR INDEX `{escaped_index_name}` IF NOT EXISTS 
                FOR ()-[r:`{rel_type}`]-()
                ON (r.`{escaped_embedding_property}`)
                OPTIONS {{
                  indexConfig: {{
                    `vector.dimensions`: {embedding_dimensions},
                    `vector.similarity_function`: "cosine"
                  }}
                }}
                """
            )
            rel_index_list.append(index_name)

        print("Relationship embeddings created")

        # query = "drug resistance in cancer"
        # search_results = relationship_similarity_search(query)
        # print_search_result(search_results)


def _relationship_types_with_semantic_text(session) -> list[str]:
    rows = session.run(
        """
        MATCH ()-[r]->()
        WHERE r.semantic_text IS NOT NULL
        RETURN DISTINCT type(r) AS rel_type
        ORDER BY rel_type
        """
    )
    return [row["rel_type"] for row in rows]

if __name__ == "__main__":
    embed_relationships()
