# src/explore_kg_llm/embeddings/embed_nodes.py
import math

from langchain_core.documents import Document
from langchain_community.vectorstores import Neo4jVector
from src.embeddings.embedding_utils import get_embedding_client
from src.config import NEO4J_URI, NEO4J_USERNAME, NEO4J_PASSWORD
from neo4j import GraphDatabase


NODE_LABELS = [
    "biolink:Disease",
    "biolink:Drug",
    "biolink:PhenotypicFeature",
    "biolink:Gene",
    "biolink:Protein",
    "biolink:ChemicalOrDrugOrTreatment",
]

def node_index_name(label: str) -> str:
    return f"{label.replace(':', '_')}_idx"

def ensure_node_vector_indexes():
    cypher = """
    CREATE VECTOR INDEX $index_name IF NOT EXISTS
    FOR (n:`%s`)
    ON (n.%s)
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
                cypher % (label, 'embedding'),
                index_name=node_index_name(label),
                dims=1536,
            )

def get_node_stores():
    stores = {}

    try:
        ensure_node_vector_indexes()
        for label in NODE_LABELS:
            stores[label] = Neo4jVector.from_existing_index(
                embedding=get_embedding_client(),
                url=NEO4J_URI,
                username=NEO4J_USERNAME,
                password=NEO4J_PASSWORD,
                index_name=node_index_name(label),
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
):
    if not node_stores:
        return _node_similarity_search_scan(
            query,
            k_per_index=k_per_index,
            max_total=max_total,
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
):
    query_embedding = get_embedding_client().embed_query(query)
    driver = GraphDatabase.driver(
        NEO4J_URI,
        auth=(NEO4J_USERNAME, NEO4J_PASSWORD),
    )
    try:
        with driver.session() as session:
            rows = session.run(
                """
                MATCH (n)
                WHERE any(label IN labels(n) WHERE label IN $node_labels)
                  AND n.embedding IS NOT NULL
                  AND n.node_text IS NOT NULL
                RETURN labels(n) AS labels, n { .* } AS metadata, n.node_text AS text
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
                embedding = metadata.pop("embedding", None)
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


def embed_nodes():
    for label in NODE_LABELS:
        Neo4jVector.from_existing_graph(
            embedding=get_embedding_client(),
            url=NEO4J_URI,
            username=NEO4J_USERNAME,
            password=NEO4J_PASSWORD,
            index_name=f"{label.replace(':','_')}_idx",
            node_label=label,
            text_node_properties=["name", "description"],
            embedding_node_property="embedding",
        )

        print(f"Embedded nodes for {label}")

if __name__ == "__main__":
    embed_nodes()
