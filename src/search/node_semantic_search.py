import argparse
from src.embeddings.embed_relationships import relationship_similarity_search
from src.embeddings.embedding_utils import extract_entities_from_relationships
from src.embeddings.embed_nodes import get_node_stores, node_similarity_search
from src.search.evidence_graph import EvidenceGraph


def main():
    parser = argparse.ArgumentParser(description="Node semantic search based on node embeddings")
    parser.add_argument(
        "--query",
        type=str,
        default='condition that causes airways to swell, narrow, and fill with mucus',
        required=False,
        help="query string for node semantic search",
    )

    args = parser.parse_args()
    query = args.query
    evid_graph = EvidenceGraph(query)

    nodes = {}
    node_stores = get_node_stores()
    nodes[query] = node_similarity_search(node_stores, query)
    evid_graph.nodes = nodes
    evid_graph.save_tables()
    evid_graph.save_evidence_html()

if __name__ == "__main__":
    main()
    print('done')
    exit()
