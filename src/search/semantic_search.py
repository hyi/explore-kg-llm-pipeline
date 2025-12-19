import argparse
from src.embeddings.embed_relationships import relationship_similarity_search
from src.embeddings.embedding_utils import print_search_result
from src.embeddings.embed_nodes import get_node_stores, node_similarity_search


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

    search_results = relationship_similarity_search(query)
    print('relationship semantic search results:')
    print_search_result(search_results)
    node_stores = get_node_stores()
    expanded_nodes = node_similarity_search(node_stores, 'ADAM12 Chemoresistance cancer')
    print('expanded_nodes semantic search results:')
    print_search_result(expanded_nodes)

if __name__ == "__main__":
    main()
