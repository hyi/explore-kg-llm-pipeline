# src/explore_kg_llm/embeddings/embed_nodes.py
import math
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
    get_embedding_property,
)

NODE_TEXT_CYPHER_PATH = (
    Path(__file__).resolve().parents[1]
    / "cypher"
    / "create_node_text.cypher"
)

NODE_LABELS = [
    "biolink:Disease",
    "biolink:Drug",
    "biolink:PhenotypicFeature",
    "biolink:Gene",
    "biolink:Protein",
    "biolink:ChemicalOrDrugOrTreatment",
]

def node_index_name(label: str, model: str | None = None) -> str:
    return embedding_index_name(f"{label.replace(':', '_')}_idx", model=model)

def ensure_node_vector_indexes(model: str | None = None):
    embedding_client = get_embedding_client(model)
    embedding_property = get_embedding_property(model)
    embedding_dimensions = get_embedding_dimensions(embedding_client, model)
    cypher = """
    CREATE VECTOR INDEX $index_name IF NOT EXISTS
    FOR (n:`%s`)
    ON (n.`%s`)
    OPTIONS {
      indexConfig: {
        `vector.dimensions`: $dims,
        `vector.similarity_function`: 'cosine'
      }
    }
    """
    driver = GraphDatabase.driver(
        NEO4J_URI,
        auth=(NEO4J_USERNAME, NEO4J_PASSWORD),
    )
    with driver.session() as session:
        for label in NODE_LABELS:
            session.run(
                cypher % (label, cypher_escape_identifier(embedding_property)),
                index_name=node_index_name(label, model),
                dims=embedding_dimensions,
            )

def get_node_stores(model: str | None = None):
    stores = {}
    embedding_client = get_embedding_client(model)

    try:
        ensure_node_vector_indexes(model)
        for label in NODE_LABELS:
            stores[label] = Neo4jVector.from_existing_index(
                embedding=embedding_client,
                url=NEO4J_URI,
                username=NEO4J_USERNAME,
                password=NEO4J_PASSWORD,
                index_name=node_index_name(label, model),
                text_node_property="node_text"
            )
    except Exception:
        return {}
    return stores


def node_similarity_search(
    node_stores: dict,
    query: str,
    k_per_index: int = 2,
    max_total: int = 8,
    model: str | None = None,
):
    if not node_stores:
        return _node_similarity_search_scan(
            query,
            k_per_index=k_per_index,
            max_total=max_total,
            model=model,
        )

    results = []
    names_in_results = []
    for label, store in node_stores.items():
        hits = store.similarity_search_with_score(query, k=k_per_index)
        for doc, score in hits:
            if doc.metadata['name'] in names_in_results:
                continue
            doc.metadata["node_label"] = label
            doc.metadata["score"] = score
            names_in_results.append(doc.metadata['name'])
            results.append(doc)

    # Sort globally by similarity score
    return sorted(
        results,
        key=lambda x: x.metadata.get("score", 0),
        reverse=True
    )[:max_total]


def _node_similarity_search_scan(
    query: str,
    k_per_index: int = 2,
    max_total: int = 8,
    model: str | None = None,
):
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
                MATCH (n)
                WHERE any(label IN labels(n) WHERE label IN $node_labels)
                  AND n.`{escaped_embedding_property}` IS NOT NULL
                  AND n.node_text IS NOT NULL
                RETURN labels(n) AS labels,
                       n {{ .* }} AS metadata,
                       n.node_text AS text,
                       n.`{escaped_embedding_property}` AS embedding
                """,
                node_labels=NODE_LABELS,
            )
            results = []
            names_in_results = []
            label_counts = {label: 0 for label in NODE_LABELS}
            for row in rows:
                labels = row["labels"]
                label = next((label for label in NODE_LABELS if label in labels), None)
                if label is None or label_counts[label] >= k_per_index:
                    continue

                metadata = dict(row["metadata"])
                metadata.pop(embedding_property, None)
                embedding = row["embedding"]
                name = metadata.get("name") or metadata.get("id")
                if not embedding or name in names_in_results:
                    continue

                metadata["node_label"] = label
                metadata["score"] = _cosine_similarity(query_embedding, embedding)
                results.append(Document(page_content=row["text"], metadata=metadata))
                names_in_results.append(name)
                label_counts[label] += 1
    finally:
        driver.close()

    return sorted(
        results,
        key=lambda x: x.metadata.get("score", 0),
        reverse=True
    )[:max_total]


def _cosine_similarity(left, right):
    numerator = sum(a * b for a, b in zip(left, right))
    left_norm = math.sqrt(sum(a * a for a in left))
    right_norm = math.sqrt(sum(b * b for b in right))
    if not left_norm or not right_norm:
        return 0.0
    return numerator / (left_norm * right_norm)


def create_node_text():
    query = NODE_TEXT_CYPHER_PATH.read_text().strip().removesuffix(";")

    driver = GraphDatabase.driver(
        NEO4J_URI,
        auth=(NEO4J_USERNAME, NEO4J_PASSWORD),
    )
    with driver.session() as session:
        session.run(query).consume()
    
    print("Node node_text created")


def embed_nodes():
    embedding_client = get_embedding_client()
    embedding_property = get_embedding_property()

    create_node_text()
    
    for label in NODE_LABELS:
        Neo4jVector.from_existing_graph(
            embedding=embedding_client,
            url=NEO4J_URI,
            username=NEO4J_USERNAME,
            password=NEO4J_PASSWORD,
            index_name=node_index_name(label),
            node_label=label,
            text_node_properties=["name", "description"],
            embedding_node_property=embedding_property,
        )

        print(f"Embedded nodes for {label}")

if __name__ == "__main__":
    embed_nodes()
