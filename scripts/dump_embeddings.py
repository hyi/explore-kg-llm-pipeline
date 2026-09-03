import argparse
import json

from neo4j import GraphDatabase

from src.config import NEO4J_PASSWORD, NEO4J_URI, NEO4J_USERNAME
from src.embeddings.embedding_utils import cypher_escape_identifier

driver = GraphDatabase.driver(
    NEO4J_URI, auth=(NEO4J_USERNAME, NEO4J_PASSWORD)
)


EMBEDDING_CONFIGS = {
    "openai": {
        "embedding_key": "embedding",
        "embedding_property": "embedding",
        "node_path": "data/node_embeddings.jsonl",
        "relationship_path": "data/relationship_embeddings.jsonl",
    },
    "sapbert": {
        "embedding_key": "sapbert_embedding",
        "embedding_property": "sapbert_embedding",
        "node_path": "data/sapbert_node_embeddings.jsonl",
        "relationship_path": "data/sapbert_relationship_embeddings.jsonl",
    },
}


def embedding_config(model: str):
    return EMBEDDING_CONFIGS[model]


def dump_nodes(path=None, model="openai"):
    config = embedding_config(model)
    path = path or config["node_path"]
    embedding_key = config["embedding_key"]
    embedding_property = cypher_escape_identifier(config["embedding_property"])
    query = f"""
    MATCH (n)
    WHERE n.`{embedding_property}` IS NOT NULL
    RETURN
      n.id AS node_id,
      labels(n) AS labels,
      n.name AS name,
      n.node_text AS node_text,
      n.`{embedding_property}` AS {embedding_key}
    """
    with driver.session() as session, open(path, "w") as f:
        f.writelines(json.dumps(r.data()) + "\n" for r in session.run(query))


def dump_relationships(path=None, model="openai"):
    config = embedding_config(model)
    path = path or config["relationship_path"]
    embedding_key = config["embedding_key"]
    embedding_property = cypher_escape_identifier(config["embedding_property"])
    query = f"""
    MATCH ()-[r]-()
    WHERE r.`{embedding_property}` IS NOT NULL
    RETURN
      r.id AS rel_id,
      type(r) AS predicate,
      r.original_subject AS subject,
      r.original_object AS object,
      r.semantic_text AS semantic_text,
      r.`{embedding_property}` AS {embedding_key}
    """
    with driver.session() as session, open(path, "w") as f:
        f.writelines(json.dumps(r.data()) + "\n" for r in session.run(query))


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model",
        choices=sorted(EMBEDDING_CONFIGS),
        default="openai",
        help="Embedding model property set to dump.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    dump_nodes(model=args.model)
    dump_relationships(model=args.model)
    driver.close()
    print(f"{args.model} embeddings dumped successfully")
