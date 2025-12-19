# src/explore_kg_llm/embeddings/embed_nodes.py
from langchain_community.vectorstores import Neo4jVector
from src.embeddings.embedding_utils import get_embedding_client
from src.config import NEO4J_URI, NEO4J_USERNAME, NEO4J_PASSWORD

NODE_LABELS = [
    "biolink:Disease",
    "biolink:Drug",
    "biolink:PhenotypicFeature",
    "biolink:Gene",
    "biolink:Protein",
    "biolink:ChemicalOrDrugOrTreatment",
]

def get_node_stores():
    stores = {}

    for label in NODE_LABELS:
        index_name = f"{label.replace(':','_')}_idx"

        stores[label] = Neo4jVector.from_existing_index(
            embedding=get_embedding_client(),
            url=NEO4J_URI,
            username=NEO4J_USERNAME,
            password=NEO4J_PASSWORD,
            index_name=index_name,
        )
    return stores


def node_similarity_search(
    node_stores: dict,
    query: str,
    k_per_index: int = 2,
    max_total: int = 10,
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
