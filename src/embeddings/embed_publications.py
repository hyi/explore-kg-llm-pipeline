# src/explore_kg_llm/embeddings/embed_publications.py
from langchain_community.vectorstores import Neo4jVector
from src.embeddings.embedding_utils import (
    embedding_index_name,
    get_embedding_client,
    get_embedding_property,
)
from src.config import NEO4J_URI, NEO4J_USERNAME, NEO4J_PASSWORD

def embed_publications():
    embedding_client = get_embedding_client()
    embedding_property = get_embedding_property()

    Neo4jVector.from_existing_graph(
        embedding=embedding_client,
        url=NEO4J_URI,
        username=NEO4J_USERNAME,
        password=NEO4J_PASSWORD,
        index_name=embedding_index_name("publication_abstract_idx"),
        node_label="biolink:Publication",
        text_node_properties=["name", "abstract_text"],
        embedding_node_property=embedding_property,
    )

    print("Publication embeddings created")

if __name__ == "__main__":
    embed_publications()
