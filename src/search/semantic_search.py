import argparse
from src.embeddings.embed_relationships import relationship_similarity_search


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
    for res in search_results:
        print("-" * 80)
        print(f"Similarity Score: {res.metadata['score']:.4f}")
        print(f"Document Content: {res.page_content[:200]}...")
        print(f"Metadata: {res.metadata}")
        print("-" * 80)


if __name__ == "__main__":
    main()
