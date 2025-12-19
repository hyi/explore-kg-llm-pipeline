import argparse
from langchain_community.vectorstores import Neo4jVector
from src.embeddings.embed_relationships import relationship_similarity_search
from src.embeddings.embedding_utils import print_search_result


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
    print_search_result(search_results)


if __name__ == "__main__":
    main()
