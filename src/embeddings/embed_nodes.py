# src/explore_kg_llm/embeddings/embed_nodes.py
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

    ensure_node_vector_indexes()
    for label in NODE_LABELS:
        stores[label] = Neo4jVector.from_existing_index(
            embedding=get_embedding_client(),
            url=NEO4J_URI,
            username=NEO4J_USERNAME,
            password=NEO4J_PASSWORD,
            index_name=node_index_name(label),
        )
    return stores


def node_similarity_search(
    node_stores: dict,
    query: str,
    k_per_index: int = 2,
    max_total: int = 8,
):
    results = []

    for label, store in node_stores.items():
        hits = store.similarity_search_with_score(query, k=k_per_index)
        for doc, score in hits:
            doc.metadata["node_label"] = label
            doc.metadata["score"] = score
            results.append(doc)

    # Sort globally by similarity score
    return sorted(
        results,
        key=lambda x: x.metadata.get("score", 0),
        reverse=True
    )[:max_total]


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
