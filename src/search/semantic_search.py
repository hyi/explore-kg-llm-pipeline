import argparse
from src.embeddings.embed_relationships import relationship_similarity_search
from src.embeddings.embedding_utils import print_search_result, extract_entities_from_relationships
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

    search_results = relationship_similarity_search(query)
    print('relationship semantic search results:')
    print_search_result(search_results)
    entities = extract_entities_from_relationships(search_results)
    node_stores = get_node_stores()
    for entity in entities:
        expanded_nodes = node_similarity_search(node_stores, entity)
        print(f'expanded_nodes semantic search results for {entity}:')
        print_search_result(expanded_nodes)

if __name__ == "__main__":
    main()
