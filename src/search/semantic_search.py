import argparse
import sys

from src.embeddings.embed_nodes import get_node_stores, node_similarity_search
from src.embeddings.embed_relationships import relationship_similarity_search
from src.embeddings.embedding_utils import extract_entities_from_relationships
from src.search.evidence_graph import EvidenceGraph


def run_semantic_search(
    query: str,
    relationship_k: int = 10,
    node_k_per_entity: int = 2,
    max_nodes_per_entity: int = 8,
    model: str | None = None,
) -> EvidenceGraph:
    """Run the existing semantic relationship + node expansion workflow."""
    evid_graph = EvidenceGraph(query)
    evid_graph.relationships = relationship_similarity_search(
        query,
        k=relationship_k,
        model=model,
    )

    entities = extract_entities_from_relationships(evid_graph.relationships)
    nodes = {}
    node_stores = get_node_stores(model)
    for entity in entities:
        nodes[entity] = node_similarity_search(
            node_stores,
            entity,
            k_per_index=node_k_per_entity,
            max_total=max_nodes_per_entity,
            model=model,
        )
    evid_graph.nodes = nodes
    return evid_graph


def main():
    parser = argparse.ArgumentParser(description="Semantic search based on node and relationship embeddings")
    parser.add_argument(
        "--query",
        type=str,
        default='genes involved in drug resistance in cancer',
        required=False,
        help="query string for semantic search",
    )
    parser.add_argument(
        "--model",
        choices=["openai", "sapbert"],
        default="openai",
        help="embedding model to use for semantic search",
    )
    parser.add_argument(
        "--out-dir",
        default=None,
        help="directory for result files; defaults to results for openai and results/sapbert for sapbert",
    )

    args = parser.parse_args()
    query = args.query
    evid_graph = run_semantic_search(query, relationship_k=5, model=args.model)
    out_dir = args.out_dir or ("results" if args.model == "openai" else "results/sapbert")
    evid_graph.save_tables(out_dir=out_dir)
    evid_graph.save_evidence_html(out_file=f"{out_dir}/evidence_graph.html")
    evid_graph.export_as_cypher(out_file=f"{out_dir}/evidence_graph.cypher")

if __name__ == "__main__":
    main()
    print('done')
    sys.exit()
