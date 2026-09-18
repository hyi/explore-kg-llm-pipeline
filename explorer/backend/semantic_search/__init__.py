from explorer.backend.semantic_search.neighborhood import (
    NeighborhoodExpansionConfig,
    rank_neighborhood_candidates,
)
from explorer.backend.semantic_search.query_intent import QueryIntent
from explorer.backend.semantic_search.ranking import AnchorRankingConfig
from explorer.backend.semantic_search.retrieval import RetrievalConfig

__all__ = [
    "AnchorRankingConfig",
    "NeighborhoodExpansionConfig",
    "QueryIntent",
    "RetrievalConfig",
    "SemanticSearchService",
    "rank_neighborhood_candidates",
]


def __getattr__(name: str):
    if name == "SemanticSearchService":
        from explorer.backend.semantic_search.service import SemanticSearchService

        return SemanticSearchService
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
