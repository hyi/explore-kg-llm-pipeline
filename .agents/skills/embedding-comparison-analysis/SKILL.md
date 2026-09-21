---
name: embedding-comparison-analysis
description: Offline OpenAI-vs-SapBERT relationship-embedding investigation (Workstream 2) — reusable analysis module vs notebook split, data-matching rules, separate-projection rules (no OpenAI/SapBERT coordinate mixing), required analyses (projections, rank comparison, neighborhood agreement, anchor relevance, exploration utility), and test fixtures. Load before touching analysis/, notebooks/, or tests/test_embedding_comparison.py.
---

# Workstream 2: Comparative Embedding and Retrieval Investigation

Implemented. Treat as closed unless the user explicitly reopens this investigation.

## Purpose

A reproducible offline investigation of how OpenAI and PubMedBERT SapBERT relationship embeddings
organize the same LitCoin edges, how their search results differ, and whether keyword and hybrid
retrieval improve the anchors used for graph exploration. This work supports model comparison,
retrieval diagnosis, and presentation of the investigation. It is not a new primary workflow in the
explorer web app — see the `presentation-demo-readiness` skill for the "do not add this to the Dash
app" constraint, which remains a standing rule.

## Reference Notes

* The PubMedBERT SapBERT model lives at `sapbert_model/SapBERT-from-PubMedBERT-fulltext`.
* `EMBEDDING_PROVIDER` and `EMBEDDING_MODEL` env vars (set in `.env`) select sapbert vs openai.
  Model-switching via these env vars or a `model` argument is implemented throughout `src/`, but
  **not** in `explorer/` code.

## Scope and Location

* `analysis/embedding_comparison.py`: reusable loading, validation, metrics, projection, and
  plotting functions
* `notebooks/embedding_comparison.ipynb`: narrative, configuration, displayed results,
  interpretation
* `tests/test_embedding_comparison.py`: tests for the reusable analysis functions

Keep substantial logic out of notebook cells; the notebook calls tested functions from the analysis
module so results are reproducible without manually executing undocumented cell state.

## Required Data Handling

* Match the two models' vectors using a stable relationship identifier.
* Verify and report whether both models cover the same edge population.
* Fail clearly on duplicate relationship identifiers or embedding-dimension inconsistencies within
  one model.
* Preserve useful metadata: subject, predicate, object, endpoint labels, publication identifier,
  abstract title, semantic text.
* Do not place database credentials, API keys, or embedding arrays in committed notebook output.
* Prefer a reusable export from Neo4j so the notebook can be rerun without a live database when
  practical.
* Record model names, embedding dimensions, extraction date, projection parameters, and random
  seeds in the output.

## Projection Rules

OpenAI and SapBERT embeddings occupy different vector spaces and may have different dimensions. Do
not treat their raw coordinates as comparable. Do not concatenate the two vector matrices into one
ordinary projection or interpret the orientation and axis positions of separately fitted
projections.

Create separate projections for the same edge set. Use consistent edge IDs, colors, symbols,
labels, and hover fields across panels. Compare neighborhood membership, clustering, and
highlighted query results rather than absolute coordinates, axis direction, rotation, or visual
distance between panels.

Use a fixed random seed. UMAP is appropriate for exploratory local-neighborhood views; include PCA
as a deterministic reference when useful. Treat t-SNE as optional and never as the only projection.

## Required Analyses

1. **Side-by-side embedding projections** — one point per matched relationship edge, separate
   panels for OpenAI and SapBERT, consistent coloring by publication/predicate family/entity type,
   configurable top-k query highlighting, informative hover text without embedding arrays.

2. **Query-result rank comparison** across all retrieval configurations: raw dense OpenAI, raw
   dense SapBERT, keyword/BM25, hybrid keyword+OpenAI, hybrid keyword+SapBERT, and query-aware
   diversified final anchors for each applicable mode. Query set includes at least `drug resistance
   in cancer`, `genes involved in chemoresistance in cancer`, one biomedical entity/synonym-focused
   query, and one broader relational query — and stays configurable rather than hard-coded.

3. **Neighborhood agreement**: top-k retrieval overlap between models, per-edge nearest-neighbor
   Jaccard overlap, rank correlation for shared candidates, same-publication fraction among nearest
   neighbors, same-predicate/predicate-family fraction among nearest neighbors.

4. **Anchor relevance and diversity** per model/query: unique publications in top k, unique
   predicates and endpoint neighborhoods, requested entity-type matches, query-compatible predicate
   matches, raw semantic results vs. reranked/diversified anchors, optional human judgments
   (`relevant`/`adjacent`/`irrelevant`).

5. **Exploration utility**: whether a useful graph neighborhood or path can be reached from each
   anchor, number of user-approved expansions required, whether global re-anchoring was needed. An
   anchor is useful when it provides an efficient entry into an informative region, even if it is
   not a complete answer by itself.

## Outputs

Presentation-ready interactive HTML figures when practical, plus a straightforward path to export
static figures for slides without the Dash app. Notebook tables are exportable as CSV.

Include a concise conclusions section suitable for group-meeting slides, separating direct
observations, interpretations, and future hypotheses. Every numerical or visual claim must be
reproducible from notebook inputs. Every figure states the model, projection method, important
parameters, and what can/cannot be inferred from the view. Never describe a 2D projection as proof
that one embedding model is globally superior.

## Tests and Acceptance Criteria

Tests must not require Neo4j, OpenAI access, or downloading SapBERT — use small synthetic vectors
and metadata fixtures. Required:

* edge matching is stable regardless of input row order
* missing and duplicate relationship IDs are reported clearly
* mismatched model coverage is reported rather than silently discarded
* nearest-neighbor and top-k-overlap metrics return known values on fixtures
* rank-fusion and retrieval-comparison tables retain the retrieval-mode label
* same-publication fractions are calculated correctly
* projection and sampling are reproducible with a fixed seed
* query-result highlighting uses stable relationship IDs
* raw and reranked result sets remain distinguishable
* embedding arrays are excluded from exported hover and table payloads
