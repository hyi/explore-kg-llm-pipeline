import json
from neo4j import GraphDatabase
from src.config import NEO4J_URI, NEO4J_USERNAME, NEO4J_PASSWORD


driver = GraphDatabase.driver(
    NEO4J_URI, auth=(NEO4J_USERNAME, NEO4J_PASSWORD)
)


def restore_nodes(path="data/node_embeddings.jsonl"):
    query = """
    MATCH (n {id: $node_id})
    SET n.embedding = $embedding
    """
    with driver.session() as session, open(path) as f:
        for line in f:
            row = json.loads(line)
            session.run(query, **row)


def restore_relationships(path="data/relationship_embeddings.jsonl"):
    query = """
    MATCH ()-[r]-()
    WHERE r.id = $rel_id
    SET
      r.embedding = $embedding,
      r.semantic_text = $semantic_text
    """
    with driver.session() as session, open(path) as f:
        for line in f:
            row = json.loads(line)
            session.run(query, **row)


if __name__ == "__main__":
    restore_nodes()
    restore_relationships()
    print("Embeddings restored successfully")
