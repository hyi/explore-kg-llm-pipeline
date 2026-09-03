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
    print_search_result,
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

    results = []
    for rel_type, store in rel_stores.items():
        hits = store.similarity_search_with_score(query, k=k)
        for doc, score in hits:
            doc.metadata["predicate"] = rel_type
            doc.metadata["score"] = score
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

    for rel_type in RELATIONSHIP_TYPES:
        for index_name in _relationship_index_candidates(
            rel_type,
            available_indexes,
            model,
        ):
            try:
                rel_stores[rel_type] = Neo4jVector.from_existing_relationship_index(
                    embedding=embedding_client,
                    url=NEO4J_URI,
                    username=NEO4J_USERNAME,
                    password=NEO4J_PASSWORD,
                    index_name=index_name,
                    text_node_property="semantic_text"
                )
                break
            except ValueError as exc:
                if "does not exist" not in str(exc).lower():
                    raise

    return rel_stores


def _relationship_similarity_search_scan(query, k=5, model: str | None = None):
    embedding_property = get_embedding_property(model)
    escaped_embedding_property = cypher_escape_identifier(embedding_property)
    query_embedding = get_embedding_client(model).embed_query(query)
    driver = GraphDatabase.driver(
        NEO4J_URI,
        auth=(NEO4J_USERNAME, NEO4J_PASSWORD),
    )
    try:
        with driver.session() as session:
            rows = session.run(
                f"""
                MATCH ()-[r]->()
                WHERE type(r) IN $relationship_types
                  AND r.`{escaped_embedding_property}` IS NOT NULL
                  AND r.semantic_text IS NOT NULL
                RETURN type(r) AS predicate,
                       r {{ .* }} AS metadata,
                       r.semantic_text AS text,
                       r.`{escaped_embedding_property}` AS embedding
                """,
                relationship_types=RELATIONSHIP_TYPES,
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
    except Exception:
        return {}
    finally:
        driver.close()


def _relationship_index_candidates(
    rel_type,
    available_indexes=None,
    model: str | None = None,
):
    candidates = [_relationship_index_name(rel_type, model)]

    if (
        not get_embedding_index_suffix(model)
        and available_indexes
        and rel_type in available_indexes
    ):
        candidates.extend(available_indexes[rel_type])

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
        if index_name not in deduplicated:
            deduplicated.append(index_name)
    return deduplicated


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
            rel_type = record["rel_type"]
            if rel_type not in RELATIONSHIP_TYPES:
                continue
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
        for rel_type in RELATIONSHIP_TYPES:
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

        query = "drug resistance in cancer"
        search_results = relationship_similarity_search(query)
        print_search_result(search_results)

if __name__ == "__main__":
    embed_relationships()
