import json
from pathlib import Path

from neo4j import GraphDatabase
from src.config import NEO4J_URI, NEO4J_USERNAME, NEO4J_PASSWORD


driver = GraphDatabase.driver(
    NEO4J_URI, auth=(NEO4J_USERNAME, NEO4J_PASSWORD)
)

SCRIPT_DIR = Path(__file__).resolve().parent
DATA_DIR = SCRIPT_DIR / "data"
EMBEDDING_DIMS = 1536

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


def restore_nodes(path=DATA_DIR / "node_embeddings.jsonl"):
    query = """
    MATCH (n {id: $node_id})
    SET n.embedding = $embedding,
        n.node_text = $node_text
    """
    with driver.session() as session, open(path) as f:
        for line in f:
            row = json.loads(line)
            session.run(query, **row)


def restore_relationships(path=DATA_DIR / "relationship_embeddings.jsonl"):
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


def create_vector_indexes():
    with driver.session() as session:
        for label in NODE_LABELS:
            session.run(
                f"""
                CREATE VECTOR INDEX `{node_index_name(label)}` IF NOT EXISTS
                FOR (n:`{label}`)
                ON (n.embedding)
                OPTIONS {{
                  indexConfig: {{
                    `vector.dimensions`: $dims,
                    `vector.similarity_function`: 'cosine'
                  }}
                }}
                """,
                dims=EMBEDDING_DIMS,
            )

        for rel_type in RELATIONSHIP_TYPES:
            session.run(
                f"""
                CREATE VECTOR INDEX `{relationship_index_name(rel_type)}` IF NOT EXISTS
                FOR ()-[r:`{rel_type}`]-()
                ON (r.embedding)
                OPTIONS {{
                  indexConfig: {{
                    `vector.dimensions`: $dims,
                    `vector.similarity_function`: 'cosine'
                  }}
                }}
                """,
                dims=EMBEDDING_DIMS,
            )

        session.run("CALL db.awaitIndexes($timeout_seconds)", timeout_seconds=300)


def node_index_name(label: str) -> str:
    return f"{label.replace(':', '_')}_idx"


def relationship_index_name(rel_type: str) -> str:
    return f"{rel_type.replace(':', '_').lower()}_vector_idx"


if __name__ == "__main__":
    restore_nodes()
    restore_relationships()
    create_vector_indexes()
    driver.close()
    print("Embeddings restored and vector indexes created successfully")
