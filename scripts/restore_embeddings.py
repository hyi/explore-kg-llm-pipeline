import argparse
import json
from pathlib import Path

from neo4j import GraphDatabase

from src.config import NEO4J_PASSWORD, NEO4J_URI, NEO4J_USERNAME
from src.embeddings.embedding_utils import cypher_escape_identifier

driver = GraphDatabase.driver(
    NEO4J_URI, auth=(NEO4J_USERNAME, NEO4J_PASSWORD)
)

SCRIPT_DIR = Path(__file__).resolve().parent
DATA_DIR = SCRIPT_DIR / "data"

EMBEDDING_CONFIGS = {
    "openai": {
        "dimensions": 1536,
        "embedding_key": "embedding",
        "embedding_property": "embedding",
        "index_suffix": "",
        "node_path": DATA_DIR / "node_embeddings.jsonl",
        "relationship_path": DATA_DIR / "relationship_embeddings.jsonl",
    },
    "sapbert": {
        "dimensions": 768,
        "embedding_key": "sapbert_embedding",
        "embedding_property": "sapbert_embedding",
        "index_suffix": "_sapbert",
        "node_path": DATA_DIR / "sapbert_node_embeddings.jsonl",
        "relationship_path": DATA_DIR / "sapbert_relationship_embeddings.jsonl",
    },
}

NODE_LABELS = [
    "biolink:Disease",
    "biolink:Drug",
    "biolink:PhenotypicFeature",
    "biolink:Gene",
    "biolink:Protein",
    "biolink:ChemicalOrDrugOrTreatment",
]

RELATIONSHIP_TYPES = [
    "biolink:related_to",
    "biolink:associated_with",
    "biolink:correlated_with",
    "biolink:positively_correlated_with",
    "biolink:genetically_associated_with",
    "biolink:contributes_to",
    "biolink:causes",
    "biolink:affects",
    "biolink:regulates",
    "biolink:treats",
    "biolink:studied_to_treat",
    "biolink:treats_or_applied_or_studied_to_treat",
    "biolink:ameliorates_condition",
    "biolink:exacerbates_condition",
    "biolink:preventative_for_condition",
    "biolink:affects_response_to",
    "biolink:increases_response_to",
    "biolink:decreases_response_to",
    "biolink:associated_with_resistance_to",
]


def embedding_config(model: str):
    return EMBEDDING_CONFIGS[model]


def restore_nodes(path=None, model="openai"):
    config = embedding_config(model)
    path = path or config["node_path"]
    embedding_key = config["embedding_key"]
    embedding_property = cypher_escape_identifier(config["embedding_property"])
    query = f"""
    MATCH (n {{id: $node_id}})
    SET n.`{embedding_property}` = $embedding,
        n.node_text = $node_text
    """
    with driver.session() as session, open(path) as f:
        for line in f:
            row = json.loads(line)
            session.run(
                query,
                node_id=row["node_id"],
                embedding=row[embedding_key],
                node_text=row.get("node_text"),
            )


def restore_relationships(path=None, model="openai"):
    config = embedding_config(model)
    path = path or config["relationship_path"]
    embedding_key = config["embedding_key"]
    embedding_property = cypher_escape_identifier(config["embedding_property"])
    query = f"""
    MATCH ()-[r]-()
    WHERE r.id = $rel_id
    SET
      r.`{embedding_property}` = $embedding,
      r.semantic_text = $semantic_text
    """
    with driver.session() as session, open(path) as f:
        for line in f:
            row = json.loads(line)
            session.run(
                query,
                rel_id=row["rel_id"],
                embedding=row[embedding_key],
                semantic_text=row.get("semantic_text"),
            )


def create_vector_indexes(model="openai"):
    config = embedding_config(model)
    embedding_property = cypher_escape_identifier(config["embedding_property"])
    with driver.session() as session:
        for label in NODE_LABELS:
            session.run(
                f"""
                CREATE VECTOR INDEX `{node_index_name(label, model)}` IF NOT EXISTS
                FOR (n:`{label}`)
                ON (n.`{embedding_property}`)
                OPTIONS {{
                  indexConfig: {{
                    `vector.dimensions`: $dims,
                    `vector.similarity_function`: 'cosine'
                  }}
                }}
                """,
                dims=config["dimensions"],
            )

        for rel_type in RELATIONSHIP_TYPES:
            session.run(
                f"""
                CREATE VECTOR INDEX `{relationship_index_name(rel_type, model)}` IF NOT EXISTS
                FOR ()-[r:`{rel_type}`]-()
                ON (r.`{embedding_property}`)
                OPTIONS {{
                  indexConfig: {{
                    `vector.dimensions`: $dims,
                    `vector.similarity_function`: 'cosine'
                  }}
                }}
                """,
                dims=config["dimensions"],
            )

        session.run("CALL db.awaitIndexes($timeout_seconds)", timeout_seconds=300)


def node_index_name(label: str, model="openai") -> str:
    config = embedding_config(model)
    return f"{label.replace(':', '_')}{config['index_suffix']}_idx"


def relationship_index_name(rel_type: str, model="openai") -> str:
    config = embedding_config(model)
    if model == "sapbert":
        return f"{rel_type.lower()}{config['index_suffix']}_vector_idx"
    return f"{rel_type.replace(':', '_').lower()}_vector_idx"


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model",
        choices=sorted(EMBEDDING_CONFIGS),
        default="openai",
        help="Embedding model property set to restore.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    restore_nodes(model=args.model)
    restore_relationships(model=args.model)
    create_vector_indexes(model=args.model)
    driver.close()
    print(f"{args.model} embeddings restored and vector indexes created successfully")
