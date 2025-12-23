# src/explore_kg_llm/embeddings/embed_relationships.py
from langchain_community.vectorstores import Neo4jVector
from neo4j import GraphDatabase
from src.embeddings.embedding_utils import get_embedding_client, print_search_result
from src.config import NEO4J_URI, NEO4J_USERNAME, NEO4J_PASSWORD


EMBEDDING = get_embedding_client()

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
    "biolink:associated_with_resistance_to"
]

def relationship_similarity_search(query, k=5):
    rel_stores = {}
    rel_index_list = [f"{rel_type.lower()}_vector_idx" for rel_type in RELATIONSHIP_TYPES]
    for rel_type, index_name in zip(RELATIONSHIP_TYPES, rel_index_list):
        rel_stores[rel_type] = Neo4jVector.from_existing_relationship_index(
            embedding=EMBEDDING,
            url=NEO4J_URI,
            username=NEO4J_USERNAME,
            password=NEO4J_PASSWORD,
            index_name=index_name,
            text_node_property="semantic_text"
        )
    results = []
    for rel_type, store in rel_stores.items():
        hits = store.similarity_search_with_score(query, k=k)
        for doc, score in hits:
            doc.metadata["predicate"] = rel_type
            doc.metadata["score"] = score
            results.append(doc)

    return sorted(
        results,
        key=lambda x: x.metadata.get("score", 0),
        reverse=True
    )[:k]

def embed_relationships():

    driver = GraphDatabase.driver(
        NEO4J_URI,
        auth=(NEO4J_USERNAME, NEO4J_PASSWORD),
    )

    with driver.session() as session:
        results = session.run(
            """
            MATCH ()-[r]->()
            WHERE r.semantic_text IS NOT NULL
              AND r.embedding IS NULL
            RETURN elementId(r) AS rid, 
            type(r) as rel_type, r.semantic_text AS text
            """
        )
        for record in results:
            rid = record["rid"]
            text = record["text"]
            rel_type = record["rel_type"]
            if rel_type not in RELATIONSHIP_TYPES:
                continue
            vector = EMBEDDING.embed_query(text)

            session.run(
                """
                MATCH ()-[r]->() 
                WHERE elementId(r) = $rid
                SET r.embedding = $embedding
                """,
                rid=rid,
                embedding=vector,
            )

        rel_index_list = []
        for rel_type in RELATIONSHIP_TYPES:
            # create relationship vector index
            index_name = f"{rel_type.lower()}_vector_idx"
            session.run(
                f"""
                CREATE VECTOR INDEX `{index_name}` IF NOT EXISTS 
                FOR ()-[r:`{rel_type}`]-()
                ON (r.embedding)
                OPTIONS {{
                  indexConfig: {{
                    `vector.dimensions`: 1536,
                    `vector.similarity_function`: "cosine"
                  }}
                }}
                """
            )
            rel_index_list.append(index_name)

        print("Relationship embeddings created")

        query = "drug resistance mechanisms in cancer"
        search_results = relationship_similarity_search(query)
        print_search_result(search_results)

if __name__ == "__main__":
    embed_relationships()
