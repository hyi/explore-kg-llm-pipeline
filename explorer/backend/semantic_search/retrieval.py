from __future__ import annotations

import copy
import re
from collections.abc import Callable
from dataclasses import asdict, dataclass
from typing import Any

from explorer.backend.semantic_search.query_intent import build_bm25_content_query
from explorer.backend.semantic_search.ranking import relationship_identity

DEFAULT_RRF_K = 60
DEFAULT_DENSE_WEIGHT = 1.0
DEFAULT_KEYWORD_WEIGHT = 0.8
SUPPORTED_RETRIEVAL_MODES = frozenset({"dense", "keyword", "hybrid"})
KEYWORD_STOPWORDS = frozenset(
    {
        "a",
        "an",
        "and",
        "are",
        "for",
        "in",
        "into",
        "is",
        "of",
        "or",
        "the",
        "to",
        "with",
    }
)


@dataclass(frozen=True)
class RetrievalConfig:
    mode: str = "dense"
    dense_weight: float = DEFAULT_DENSE_WEIGHT
    keyword_weight: float = DEFAULT_KEYWORD_WEIGHT
    rrf_k: int = DEFAULT_RRF_K

    def normalized_mode(self) -> str:
        mode = (self.mode or "dense").strip().lower()
        if mode not in SUPPORTED_RETRIEVAL_MODES:
            raise ValueError(f"Unsupported retrieval mode '{self.mode}'. Use dense, keyword, or hybrid.")
        return mode

    def to_cache_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["mode"] = self.normalized_mode()
        return payload


@dataclass(frozen=True)
class RelationshipRetrievalResult:
    candidates: list[Any]
    diagnostics: dict[str, Any]


def tokenize_keyword_query(query: str) -> list[str]:
    tokens = re.findall(r"[a-z0-9]+", (query or "").casefold())
    return [token for token in tokens if token not in KEYWORD_STOPWORDS]


def retrieve_relationship_candidates(
    query: str,
    k: int,
    *,
    dense_retriever: Callable[..., list[Any]],
    keyword_retriever: Callable[..., list[Any]] | None = None,
    model: str | None = None,
    config: RetrievalConfig | None = None,
) -> RelationshipRetrievalResult:
    config = config or RetrievalConfig()
    mode = config.normalized_mode()
    bm25_query = build_bm25_content_query(query)
    diagnostics: dict[str, Any] = {
        "retrieval_mode": mode,
        "requested_k": k,
        "original_query": query,
        "keyword_query": bm25_query.to_dict(),
        "keyword_tokens": list(bm25_query.tokens),
        "channels": {},
        "degraded": False,
        "degraded_reasons": [],
    }
    if k <= 0:
        return RelationshipRetrievalResult(candidates=[], diagnostics=diagnostics)

    dense_hits: list[Any] = []
    keyword_hits: list[Any] = []

    if mode in {"dense", "hybrid"}:
        dense_hits = dense_retriever(query, k=k, model=model)
        dense_hits = _annotate_channel_hits(dense_hits, channel="dense")
        diagnostics["channels"]["dense"] = _channel_diagnostics(dense_hits, "dense")

    if mode in {"keyword", "hybrid"}:
        if keyword_retriever is None:
            diagnostics["degraded"] = True
            diagnostics["degraded_reasons"].append("keyword_retriever_unavailable")
        else:
            try:
                keyword_hits = keyword_retriever(query, k=k)
            except (RuntimeError, TypeError, ValueError) as exc:
                diagnostics["degraded"] = True
                diagnostics["degraded_reasons"].append(f"keyword_retriever_failed:{type(exc).__name__}")
                keyword_hits = []
            keyword_hits = _annotate_channel_hits(keyword_hits, channel="keyword")
            diagnostics["channels"]["keyword"] = _channel_diagnostics(keyword_hits, "keyword")

    if mode == "dense":
        candidates = dense_hits[:k]
    elif mode == "keyword":
        candidates = reciprocal_rank_fusion(
            dense_hits=[],
            keyword_hits=keyword_hits,
            limit=k,
            config=config,
        )
    else:
        candidates = reciprocal_rank_fusion(
            dense_hits=dense_hits,
            keyword_hits=keyword_hits,
            limit=k,
            config=config,
        )

    diagnostics["candidate_count"] = len(candidates)
    diagnostics["candidates"] = _candidate_diagnostics(candidates)
    return RelationshipRetrievalResult(candidates=candidates, diagnostics=diagnostics)


def reciprocal_rank_fusion(
    *,
    dense_hits: list[Any],
    keyword_hits: list[Any],
    limit: int,
    config: RetrievalConfig | None = None,
) -> list[Any]:
    config = config or RetrievalConfig()
    fused: dict[str, dict[str, Any]] = {}

    _add_channel_to_fusion(
        fused=fused,
        hits=dense_hits,
        channel="dense",
        weight=float(config.dense_weight),
        rrf_k=int(config.rrf_k),
    )
    _add_channel_to_fusion(
        fused=fused,
        hits=keyword_hits,
        channel="keyword",
        weight=float(config.keyword_weight),
        rrf_k=int(config.rrf_k),
    )

    max_score = max((item["fusion_score"] for item in fused.values()), default=0.0)
    ranked = sorted(
        fused.values(),
        key=lambda item: (
            -item["fusion_score"],
            min(item["ranks"]),
            item["identity"],
        ),
    )

    candidates = []
    for item in ranked[: max(0, int(limit))]:
        metadata = dict(getattr(item["candidate"], "metadata", {}) or {})
        normalized_score = item["fusion_score"] / max_score if max_score else 0.0
        metadata.update(
            {
                "retrieval_score": normalized_score,
                "score": normalized_score,
                "fusion_score": item["fusion_score"],
                "retrieval_channels": sorted(item["channels"]),
            }
        )
        candidates.append(_with_metadata(item["candidate"], metadata))
    return candidates


def _add_channel_to_fusion(
    *,
    fused: dict[str, dict[str, Any]],
    hits: list[Any],
    channel: str,
    weight: float,
    rrf_k: int,
) -> None:
    for rank, hit in enumerate(hits):
        metadata = dict(getattr(hit, "metadata", {}) or {})
        raw_score = float(metadata.get("score", 0.0) or 0.0)
        metadata.setdefault(f"{channel}_rank", rank)
        metadata.setdefault(f"{channel}_score", raw_score)
        if channel == "dense":
            metadata.setdefault("semantic_score", raw_score)
        elif "semantic_score" not in metadata:
            metadata["semantic_score"] = 0.0
        hit = _with_metadata(hit, metadata)
        identity = relationship_identity(metadata, fallback=f"{channel}:{rank}")
        item = fused.setdefault(
            identity,
            {
                "identity": identity,
                "candidate": hit,
                "fusion_score": 0.0,
                "ranks": [],
                "channels": set(),
            },
        )
        item["candidate"] = _merge_channel_metadata(item["candidate"], hit, channel)
        item["fusion_score"] += weight / (rrf_k + rank + 1)
        item["ranks"].append(rank)
        item["channels"].add(channel)


def _annotate_channel_hits(hits: list[Any], channel: str) -> list[Any]:
    annotated = []
    for rank, hit in enumerate(hits):
        metadata = dict(getattr(hit, "metadata", {}) or {})
        raw_score = float(metadata.get("score", 0.0) or 0.0)
        metadata[f"{channel}_rank"] = rank
        metadata[f"{channel}_score"] = raw_score
        if channel == "dense":
            metadata["semantic_score"] = float(metadata.get("semantic_score", raw_score) or 0.0)
        elif "semantic_score" not in metadata:
            metadata["semantic_score"] = 0.0
        annotated.append(_with_metadata(hit, metadata))
    return annotated


def _merge_channel_metadata(preferred: Any, update_from: Any, channel: str) -> Any:
    metadata = dict(getattr(preferred, "metadata", {}) or {})
    update_metadata = dict(getattr(update_from, "metadata", {}) or {})
    for key, value in update_metadata.items():
        if key.startswith(f"{channel}_"):
            metadata[key] = value
    if channel == "dense" and "semantic_score" in update_metadata:
        metadata["semantic_score"] = update_metadata["semantic_score"]
    return _with_metadata(preferred, metadata)


def _channel_diagnostics(hits: list[Any], channel: str) -> list[dict[str, Any]]:
    rows = []
    for hit in hits:
        metadata = dict(getattr(hit, "metadata", {}) or {})
        rows.append(
            {
                "relationship_identity": relationship_identity(metadata),
                "rank": metadata.get(f"{channel}_rank"),
                "score": metadata.get(f"{channel}_score"),
                "predicate": metadata.get("predicate"),
                "publication_id": metadata.get("publication_id"),
            }
        )
    return rows


def _candidate_diagnostics(candidates: list[Any]) -> list[dict[str, Any]]:
    rows = []
    for rank, candidate in enumerate(candidates):
        metadata = dict(getattr(candidate, "metadata", {}) or {})
        rows.append(
            {
                "rank": rank,
                "relationship_identity": relationship_identity(metadata, fallback=f"candidate:{rank}"),
                "retrieval_score": metadata.get("retrieval_score"),
                "fusion_score": metadata.get("fusion_score"),
                "retrieval_channels": metadata.get("retrieval_channels"),
                "dense_rank": metadata.get("dense_rank"),
                "dense_score": metadata.get("dense_score"),
                "keyword_rank": metadata.get("keyword_rank"),
                "keyword_score": metadata.get("keyword_score"),
            }
        )
    return rows


def _with_metadata(candidate: Any, metadata: dict[str, Any]) -> Any:
    if hasattr(candidate, "model_copy"):
        return candidate.model_copy(update={"metadata": metadata})

    cloned = copy.copy(candidate)
    cloned.metadata = metadata
    return cloned
