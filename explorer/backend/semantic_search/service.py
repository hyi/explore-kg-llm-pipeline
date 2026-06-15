from __future__ import annotations

from explorer.backend.models import SemanticSearchResult
from src.embeddings.embed_relationships import relationship_similarity_search


class SemanticSearchService:
    """Thin wrapper around existing semantic search utilities.

    The explorer's path search only needs relationship hits. Node semantic
    expansion remains available for callers that explicitly request it.
    """

    def search(
        self,
        query: str,
        relationship_k: int = 10,
        node_k_per_entity: int = 2,
        max_nodes_per_entity: int = 8,
        include_nodes: bool = False,
    ) -> SemanticSearchResult:
        if not include_nodes:
            return SemanticSearchResult(
                relationships=relationship_similarity_search(query, k=relationship_k),
                nodes={},
            )

        from src.search.semantic_search import run_semantic_search

        evidence_graph = run_semantic_search(
            query=query,
            relationship_k=relationship_k,
            node_k_per_entity=node_k_per_entity,
            max_nodes_per_entity=max_nodes_per_entity,
        )
        return SemanticSearchResult(
            relationships=evidence_graph.relationships,
            nodes=evidence_graph.nodes,
        )
