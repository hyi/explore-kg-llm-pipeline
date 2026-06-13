import argparse
from src.embeddings.embed_relationships import relationship_similarity_search
from src.embeddings.embedding_utils import extract_entities_from_relationships
from src.embeddings.embed_nodes import get_node_stores, node_similarity_search
from src.search.evidence_graph import EvidenceGraph


def run_semantic_search(
    query: str,
    relationship_k: int = 10,
    node_k_per_entity: int = 2,
    max_nodes_per_entity: int = 8,
) -> EvidenceGraph:
    """Run the existing semantic relationship + node expansion workflow."""
    evid_graph = EvidenceGraph(query)
    evid_graph.relationships = relationship_similarity_search(query, k=relationship_k)

    entities = extract_entities_from_relationships(evid_graph.relationships)
    nodes = {}
    node_stores = get_node_stores()
    for entity in entities:
        nodes[entity] = node_similarity_search(
            node_stores,
            entity,
            k_per_index=node_k_per_entity,
            max_total=max_nodes_per_entity,
        )
    evid_graph.nodes = nodes
    return evid_graph


def main():
    parser = argparse.ArgumentParser(description="Semantic search based on node and relationship embeddings")
    parser.add_argument(
        "--query",
        type=str,
        # default='genes involved in chemoresistance in cancer',
        default='drug resistance in cancer',
        required=False,
        help="query string for semantic search",
    )

    args = parser.parse_args()
    query = args.query
    evid_graph = run_semantic_search(query, relationship_k=5)
    evid_graph.save_tables()
    evid_graph.save_evidence_html()
    evid_graph.export_as_cypher()

if __name__ == "__main__":
    main()
    print('done')
    exit()
