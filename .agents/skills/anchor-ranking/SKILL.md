---
name: anchor-ranking
description: Query-aware, diversified semantic anchor ranking design for LitCoin relationship retrieval (Workstream 1) — scoring formula, named constants, endpoint-label boosts, predicate-family compatibility, publication diversification, cache-identity rules, and required test fixtures. Load before touching anchor scoring/reranking in explorer/backend/semantic_search/, or before reopening ranking work that this workstream already closed.
---

# Workstream 1: Query-Aware, Diverse Semantic Anchors

Implemented and closed unless the user explicitly reopens ranking work. Treat the rules below as
binding on any future change to this code, not just historical design notes.

## Motivation

Raw edge-vector similarity is a candidate generator, not the final anchor ranking. In the LitCoin
graph, broad document context can dominate the edge claim. For example, a query about genes
involved in cancer chemoresistance can return several edges from one chemotherapy-neuropathy
abstract because the shared abstract contains words such as cancer, chemotherapy, treatment, and
response.

Improve the initial anchors used for path discovery by combining:

* raw semantic similarity
* query-to-entity-type compatibility
* query-to-predicate compatibility
* lightweight query/context compatibility
* result diversity across publications and seed edges

This is retrieval and ranking functionality, not autonomous scientific reasoning. Keep the scores
and explanations transparent to the user.

## Scope and Location

Implemented under `explorer/backend/semantic_search/`. Reuse
`src.embeddings.embed_relationships.relationship_similarity_search` only as the raw candidate
retriever. The ranking policy belongs to the explorer application, not
`src/search/semantic_search.py`.

Integrated into the existing flow: `PathSearchService` -> `SemanticSearchService` ->
`PathDiscoveryService`. `PathRankingService` (which ranks paths after anchors are already
selected) is a separate concern and must not be replaced by this workstream.

Constraints that still apply to any future change here:

* do not regenerate or change stored embeddings
* do not add an LLM call for query parsing or reranking
* do not introduce an external search service or new database
* do not hard-filter all nonmatching edges; this small KG needs graceful fallback when exact
  matches do not exist

## Required Design

1. Over-fetch raw candidates before reranking. Candidate-pool size is configurable and bounded:
   default `max(requested_k * 5, 25)`, upper bound 100.

2. Parse only explicit, testable query hints. At minimum recognize a gene intent from terms such
   as `gene`, `genes`, `genetic`, `mutation`, and `mutations`; recognize resistance/response intent
   from terms such as `resistance`, `resistant`, `chemoresistance`, `sensitivity`, and `response`.
   Keep this logic in a small pure function and keep it easy to extend. Do not pretend the
   heuristic fully understands the query.

3. Preserve the raw vector score as `semantic_score`. Compute a separate `anchor_score`; never
   overwrite the original score without retaining it. The ranking must remain interpretable.
   Record component scores or a compact list of ranking reasons in result metadata.

4. Use endpoint node labels when available. For a gene-intent query, strongly boost a relationship
   whose subject or object has a gene-like Neo4j label. Use lexical fallback only when endpoint
   labels are absent. Do not maintain a hand-written list of gene symbols.

5. Boost predicate compatibility for resistance/response queries. The compatible predicate set:

   * `biolink:affects_response_to`
   * `biolink:increases_response_to`
   * `biolink:decreases_response_to`
   * `biolink:associated_with_resistance_to`

   Broader predicates such as `affects`, `regulates`, `contributes_to`, `causes`, and association
   predicates receive a smaller boost, not the same boost as explicit response/resistance
   predicates.

6. Prefer soft boosts and penalties over hard filters. If no edge satisfies a requested entity
   type or predicate family, return the best semantic fallbacks and mark them as fallbacks in
   metadata.

7. Diversify after relevance reranking. Avoid filling the displayed anchors with near-duplicate
   edges from one abstract. Publication identity uses the best stable metadata field available,
   falling back to `abstract_title`. Default: at most two selected anchors per publication. Also
   discourage exact repetition of the same subject-object neighborhood. A simple greedy selector
   is sufficient; full MMR is optional.

8. Enrich raw relationship hits with endpoint labels and stable relationship and publication
   identifiers when not already returned. Keep Neo4j-specific enrichment inside the Neo4j adapter
   or existing Neo4j retrieval layer. Avoid one database query per candidate; enrich candidates in
   one batch where practical.

9. `PathDiscoveryService` must receive anchors in final anchor-rank order. It stops once
   `max_paths` is reached, so order affects which neighborhoods can become candidate paths.

10. Cache identity/versioning must distinguish results from the old raw-similarity strategy from
    results from the anchor-ranking strategy. Every user-visible ranking option belongs in the
    cache key.

## Suggested Interfaces

```python
@dataclass(frozen=True)
class QueryIntent:
    requested_subject_categories: tuple[Any, ...] = ()
    requested_object_categories: tuple[Any, ...] = ()
    requested_endpoint_categories: tuple[Any, ...] = ()
    requested_predicate_families: tuple[Any, ...] = ()
    requested_direction: str | None = None
    content_terms: tuple[str, ...] = ()


@dataclass(frozen=True)
class AnchorRankingConfig:
    candidate_multiplier: int = 5
    minimum_candidate_pool: int = 25
    maximum_candidate_pool: int = 100
    max_per_publication: int = 2


def parse_query_intent(query: str) -> QueryIntent: ...


def rerank_and_diversify_relationships(
    query: str,
    candidates: list[Any],
    requested_k: int,
    config: AnchorRankingConfig,
) -> list[Any]: ...
```

Keep scoring constants named and centralized. Document their role as initial heuristics that
require evaluation rather than empirically validated weights. Do not add generic query-text-overlap
as a graph-aware ranking score; dense and keyword retrieval already measure textual relevance.
Matched claim terms may be kept as diagnostics.

For drug-response predicates, do not treat the normalized Biolink predicate family alone as a
complete structural match. Validate the predicate family against endpoint categories, original
relationship text, endpoint roles, and claim-level subject/object/statement qualifiers. A gene or
variant paired with a drug/chemical, a drug/chemical paired with a response-bearing phenotype or
process, or explicit qualifier evidence for treatment resistance, sensitivity, efficacy, or response
may produce a complete match. Disease or phenotype endpoints alone should remain partial or unknown
unless claim-level evidence establishes treatment-response semantics. Drug, treatment, therapy, or
chemotherapy context alone may support partial compatibility but should not validate complete
drug-response compatibility without explicit response evidence.

Parse query-intent expressions longest-match-first using non-overlapping spans unless an overlap is
explicitly intentional (e.g. prefer `gene products` over `gene`, `associated with resistance to`
over its shorter subphrases). Treat broad `biolink:GenomicEntity` labels as genomic context, not as
exact sequence-variant evidence; exact sequence-variant matching should use specific categories such
as `biolink:SequenceVariant`, `biolink:Allele`, and `biolink:Haplotype`.

For queries with structural intent, anchor ordering respects generic compatibility tiers before
retrieval score: validated `complete_match`, then `partial_match`, then `retrieval_only`, with
contextual `biolink:mentions` available only through labeled fallback passes. Retrieval score ranks
candidates within the same tier. Queries with no structural intent keep retrieval ordering except
for query-independent relationship quality and diversity handling.

## Output and UI Compatibility

Maintain compatibility with code that reads `metadata["score"]`. After reranking, it may represent
`anchor_score`, provided `metadata["semantic_score"]` retains the raw similarity. Also expose:

* `anchor_score`
* `semantic_score`
* `ranking_reasons`
* `is_semantic_fallback`
* `publication_id` when available

Do not expose embedding arrays in API or UI payloads. If candidate path cards are changed, show the
overall anchor/path score and a short explanation such as `gene endpoint match`, `resistance
predicate match`, or `semantic fallback`. Avoid presenting heuristic scores as scientific
confidence.

## Tests and Acceptance Criteria

Pytest unit tests, no live Neo4j instance or API keys required. Use small synthetic `Document`
objects or equivalent fixtures.

Required tests:

* Raw candidates are over-fetched, then exactly the requested number is returned when enough
  candidates exist.
* A gene-labeled cancer edge outranks a higher raw-similarity non-gene edge for `genes involved in
  chemoresistance in cancer` when the configured boost is sufficient.
* An explicit resistance/response predicate outranks an otherwise comparable generic predicate.
* Results gracefully fall back to semantic similarity when no structural hint matches.
* No more than the configured number of anchors from the same publication are selected when
  alternatives exist.
* Diversity selection is deterministic, including score ties.
* Raw `semantic_score`, final `anchor_score`, and ranking reasons are preserved.
* Empty candidate lists and `requested_k <= 0` are handled explicitly.
* Cache keys distinguish ranking strategy/configuration changes.
* Existing path search, session, and frontend tests continue to pass.

Regression fixture modeled on observed LitCoin behavior: four high-similarity edges from one
chemotherapy-induced peripheral neuropathy paper, one PTEN/cancer edge with a slightly lower raw
similarity, and one unrelated genetics/opioid-use edge. For a gene/chemoresistance/cancer query,
the PTEN/cancer candidate must be selected ahead of the chemotherapy-neuropathy duplicates and the
opioid-use candidate, while unmatched results may still appear as clearly marked fallbacks if too
few better candidates exist.
