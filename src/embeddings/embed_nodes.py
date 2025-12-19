# src/explore_kg_llm/embeddings/embed_nodes.py
from langchain_community.vectorstores import Neo4jVector
from src.embeddings.embedding_client import get_embedding_client
from src.config import NEO4J_URI, NEO4J_USERNAME, NEO4J_PASSWORD

NODE_LABELS = [
    "biolink:Disease",
    "biolink:Drug",
    "biolink:PhenotypicFeature",
    "biolink:Gene",
    "biolink:Protein",
    "biolink:ChemicalOrDrugOrTreatment",
]

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
