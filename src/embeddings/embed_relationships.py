# src/explore_kg_llm/embeddings/embed_relationships.py
import math
from functools import lru_cache

from langchain_core.documents import Document
from langchain_community.vectorstores import Neo4jVector
from neo4j import GraphDatabase
from src.embeddings.embedding_utils import get_embedding_client, print_search_result
from src.config import NEO4J_URI, NEO4J_USERNAME, NEO4J_PASSWORD


EMBEDDING = get_embedding_client()

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

def relationship_similarity_search(query, k=5):
    rel_stores = _relationship_stores()
    if not rel_stores:
        _relationship_stores.cache_clear()
        return _relationship_similarity_search_scan(query, k=k)

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
def _relationship_stores():
    rel_stores = {}
    available_indexes = _relationship_vector_indexes_by_type()

    for rel_type in RELATIONSHIP_TYPES:
        for index_name in _relationship_index_candidates(rel_type, available_indexes):
            try:
                rel_stores[rel_type] = Neo4jVector.from_existing_relationship_index(
                    embedding=EMBEDDING,
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


def _relationship_similarity_search_scan(query, k=5):
    query_embedding = EMBEDDING.embed_query(query)
    driver = GraphDatabase.driver(
        NEO4J_URI,
        auth=(NEO4J_USERNAME, NEO4J_PASSWORD),
    )
    try:
        with driver.session() as session:
            rows = session.run(
                """
                MATCH ()-[r]->()
                WHERE type(r) IN $relationship_types
                  AND r.embedding IS NOT NULL
                  AND r.semantic_text IS NOT NULL
                RETURN type(r) AS predicate, r { .* } AS metadata, r.semantic_text AS text
                """,
                relationship_types=RELATIONSHIP_TYPES,
            )
            results = []
            for row in rows:
                metadata = dict(row["metadata"])
                embedding = metadata.pop("embedding", None)
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


def _relationship_index_candidates(rel_type, available_indexes=None):
    candidates = []
    if available_indexes and rel_type in available_indexes:
        candidates.extend(available_indexes[rel_type])

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


def _cosine_similarity(left, right):
    numerator = sum(a * b for a, b in zip(left, right))
    left_norm = math.sqrt(sum(a * a for a in left))
    right_norm = math.sqrt(sum(b * b for b in right))
    if not left_norm or not right_norm:
        return 0.0
    return numerator / (left_norm * right_norm)


def embed_relationships():

    driver = GraphDatabase.driver(
        NEO4J_URI,
        auth=(NEO4J_USERNAME, NEO4J_PASSWORD),
    )

    with driver.session() as session:
        results = session.run(
            """
            MATCH ()-[r]->()
            WHERE r.semantic_text IS NOT NULL
              AND r.embedding IS NULL
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
            vector = EMBEDDING.embed_query(text)

            session.run(
                """
                MATCH ()-[r]->() 
                WHERE elementId(r) = $rid
                SET r.embedding = $embedding
                """,
                rid=rid,
                embedding=vector,
            )

        rel_index_list = []
        for rel_type in RELATIONSHIP_TYPES:
            # create relationship vector index
            index_name = f"{rel_type.lower()}_vector_idx"
            session.run(
                f"""
                CREATE VECTOR INDEX `{index_name}` IF NOT EXISTS 
                FOR ()-[r:`{rel_type}`]-()
                ON (r.embedding)
                OPTIONS {{
                  indexConfig: {{
                    `vector.dimensions`: 1536,
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
