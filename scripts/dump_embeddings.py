import json
from neo4j import GraphDatabase
from src.config import NEO4J_URI, NEO4J_USERNAME, NEO4J_PASSWORD


driver = GraphDatabase.driver(
    NEO4J_URI, auth=(NEO4J_USERNAME, NEO4J_PASSWORD)
)


def dump_nodes(path="data/node_embeddings.jsonl"):
    query = """
    MATCH (n)
    WHERE n.embedding IS NOT NULL
    RETURN
      n.id AS node_id,
      labels(n) AS labels,
      n.name AS name,
      n.embedding AS embedding
    """
    with driver.session() as session, open(path, "w") as f:
        for r in session.run(query):
            f.write(json.dumps(r.data()) + "\n")


def dump_relationships(path="data/relationship_embeddings.jsonl"):
    query = """
    MATCH ()-[r]-()
    WHERE r.embedding IS NOT NULL
    RETURN
      r.id AS rel_id,
      type(r) AS predicate,
      r.original_subject AS subject,
      r.original_object AS object,
      r.semantic_text AS semantic_text,
      r.embedding AS embedding
    """
    with driver.session() as session, open(path, "w") as f:
        for r in session.run(query):
            f.write(json.dumps(r.data()) + "\n")


if __name__ == "__main__":
    dump_nodes()
    dump_relationships()
    print("Embeddings dumped successfully")
