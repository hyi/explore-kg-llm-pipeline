import argparse
from src.embeddings.embed_relationships import relationship_similarity_search
from src.embeddings.embedding_utils import extract_entities_from_relationships
from src.embeddings.embed_nodes import get_node_stores, node_similarity_search
from src.search.evidence_graph import EvidenceGraph


def main():
    parser = argparse.ArgumentParser(description="Semantic search based on node and relationship embeddings")
    parser.add_argument(
        "--query",
        type=str,
        default='drug resistance in cancer',
        required=False,
        help="query string for semantic search",
    )

    args = parser.parse_args()
    query = args.query
    evid_graph = EvidenceGraph(query)
    evid_graph.relationships = relationship_similarity_search(query)

    entities = extract_entities_from_relationships(evid_graph.relationships)
    nodes = {}
    node_stores = get_node_stores()
    for entity in entities:
        nodes[entity] = node_similarity_search(node_stores, entity)
    evid_graph.nodes = nodes
    evid_graph.save_tables()
    evid_graph.save_evidence_html()
    evid_graph.export_as_cypher()

if __name__ == "__main__":
    main()
    print('done')
    exit()