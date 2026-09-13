from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from typing import Any

from explorer.backend.semantic_search.retrieval import tokenize_keyword_query

KEYWORD_SCORING_METHOD = "edge_claim_bm25_v1"
BM25_K1 = 1.2
BM25_B = 0.75
EDGE_FIELD_WEIGHT = 4.0


@dataclass(frozen=True)
class KeywordFieldWeights:
    edge: float = EDGE_FIELD_WEIGHT

    def to_dict(self) -> dict[str, float]:
        return asdict(self)


def rank_keyword_records(
    records: list[dict[str, Any]],
    query_tokens: list[str],
    *,
    limit: int,
    weights: KeywordFieldWeights | None = None,
) -> list[dict[str, Any]]:
    weights = weights or KeywordFieldWeights()
    tokens = list(dict.fromkeys(query_tokens))
    if not tokens or limit <= 0:
        return []

    prepared = [_prepare_record(record) for record in records]
    corpus_size = len(prepared)
    if not corpus_size:
        return []

    average_edge_length = sum(len(row["edge_tokens"]) for row in prepared) / corpus_size
    document_frequencies = {
        token: sum(1 for row in prepared if token in set(row["edge_tokens"]))
        for token in tokens
    }

    scored: list[dict[str, Any]] = []
    for row in prepared:
        edge_score, edge_matches = _field_score(
            row["edge_tokens"],
            tokens,
            document_frequencies,
            average_edge_length,
            corpus_size,
        )
        weighted_score = weights.edge * edge_score
        if weighted_score <= 0:
            continue
        scored.append(
            {
                **row["record"],
                "keyword_score": weighted_score,
                "keyword_scoring_method": KEYWORD_SCORING_METHOD,
                "keyword_field_weights": weights.to_dict(),
                "keyword_components": {
                    "edge_bm25": edge_score,
                    "weighted_edge": weights.edge * edge_score,
                },
                "keyword_edge_matches": edge_matches,
                "keyword_title_matches": [],
                "keyword_context_matches": [],
            }
        )

    return sorted(
        scored,
        key=lambda row: (
            -float(row["keyword_score"]),
            str(row.get("relationship_id") or ""),
            str(row.get("predicate") or ""),
        ),
    )[:limit]


def _prepare_record(record: dict[str, Any]) -> dict[str, Any]:
    return {
        "record": record,
        "edge_tokens": tokenize_keyword_query(str(record.get("edge_text") or "")),
    }


def _field_score(
    field_tokens: list[str],
    query_tokens: list[str],
    document_frequency: dict[str, int],
    average_length: float,
    corpus_size: int,
) -> tuple[float, list[str]]:
    if not field_tokens:
        return 0.0, []
    field_length = len(field_tokens)
    average_length = average_length or 1.0
    score = 0.0
    matches: list[str] = []
    for token in query_tokens:
        term_frequency = field_tokens.count(token)
        if not term_frequency:
            continue
        matches.append(token)
        idf = _bm25_idf(corpus_size, document_frequency.get(token, 0))
        denominator = term_frequency + BM25_K1 * (1 - BM25_B + BM25_B * field_length / average_length)
        score += idf * (term_frequency * (BM25_K1 + 1)) / denominator
    return score, matches


def _bm25_idf(corpus_size: int, document_frequency: int) -> float:
    return math.log(1 + (corpus_size - document_frequency + 0.5) / (document_frequency + 0.5))
