# src/explore_kg_llm/embeddings/embed_publications.py
from langchain_community.vectorstores import Neo4jVector
from src.embeddings.embedding_utils import get_embedding_client
from src.config import NEO4J_URI, NEO4J_USERNAME, NEO4J_PASSWORD

def embed_publications():
    Neo4jVector.from_existing_graph(
        embedding=get_embedding_client(),
        url=NEO4J_URI,
        username=NEO4J_USERNAME,
        password=NEO4J_PASSWORD,
        index_name="publication_abstract_idx",
        node_label="biolink:Publication",
        text_node_properties=["name", "abstract_text"],
        embedding_node_property="embedding",
    )

    print("Publication embeddings created")

if __name__ == "__main__":
    embed_publications()
