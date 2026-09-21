# Human-Guided Semantic Graph Exploration

## Purpose

This project provides a toolkit for interactive knowledge graph exploration. It combines semantic
retrieval, graph structure, path discovery, path ranking, and human-guided exploration to help
users discover, compare, refine, and save meaningful paths through a knowledge graph.

LitCoin serves as the reference implementation. The architecture should remain compatible with any
Cypher-compatible knowledge graph.

## Design Principles

This project is NOT a chatbot, a KGQA system, an autonomous agent, or an MCP-first application.

This project IS a graph exploration tool, a path-centric discovery tool, and a human-guided
exploration system. Human guidance is more important than automated reasoning.

## Primary Workflow

Search → Candidate Paths → Human Selection → Path Expansion → Path Refinement → Save Session.
Design features around this workflow.

## Core Domain Objects

Treat Graph, Node, Edge, Path, and ExplorationSession as first-class concepts. Path is the primary
object — most interactions should revolve around paths rather than individual nodes, though node
semantic search can support path exploration under human guidance when useful.

ExplorationSession must support accepted paths, rejected paths, bookmarked paths, search history,
and user notes. Preserving exploration state is more important than user authentication.

## Implemented Baseline

The system is a Plotly Dash web application backed by Neo4j, extended through four workstreams
(see "Active skills" below — all four are implemented and closed). Existing capabilities include
semantic relationship/node retrieval, hybrid dense+keyword anchor ranking, semantic-hit-driven path
discovery, candidate path ranking/selection, human-guided ranked one-hop neighborhood expansion,
bounded LitCoin-to-ROBOKOP cross-graph expansion, interactive graph/path expansion, exploration
session state, path-search caching, a Neo4j graph adapter, an offline embedding-comparison
analysis notebook, and pytest coverage across backend and frontend helpers.

Treat these capabilities as the working baseline. Extend the existing implementation rather than
rebuilding it.

## Repository Structure

New application functionality belongs under `explorer/`:

* `explorer/backend/semantic_search/`: application-level retrieval and anchor ranking
* `explorer/backend/path_discovery/`: shortest, semantic, and constrained paths; path expansion
* `explorer/backend/path_ranking/`: path scoring, ranking strategies, path comparison
* `explorer/backend/path_search/`: search orchestration and caching
* `explorer/backend/graph_adapter/`: graph access, isolated by source (Neo4j under `neo4j/`; the
  external ROBOKOP provider boundary alongside it)
* `explorer/backend/exploration_session/`: user exploration state, saved progress, bookmarks, notes
* `explorer/frontend/`: Plotly Dash interface — keep it path-centric; avoid node-centric graph
  browser workflows that quickly become difficult to navigate
* `explorer/tests/`: application tests

`src/` contains earlier embedding, Cypher, and semantic-search utilities — reuse where practical.
`analysis/` and `notebooks/` contain the offline embedding-comparison investigation. Keep
Neo4j-specific code isolated in `graph_adapter`; keep remote-graph transport details isolated
behind the same provider boundary rather than leaking into session, path, ranking, or frontend
code. Avoid duplicating working services or spreading application logic across unrelated parts of
the repository.

## Implementation Rules

* Before implementing a major feature or architectural change: propose the design, explain
  tradeoffs, and obtain approval. Routine fixes and tests that follow an approved design do not
  require another approval pause.
* Keep implementations simple and preserve the working baseline.
* Write all new and modified tests using pytest. Run the full pytest suite (and `ruff check` where
  applicable) before reporting a task complete.
* Do not add UMAP, t-SNE, PCA, or a general embedding-space browser to the Dash application — this
  is a deferred product decision. A future diagnostic panel (e.g. an "why this anchor?" view) is
  possible but needs its own proposal covering user value, interaction design, and performance
  before implementation. The offline embedding-comparison investigation does not by itself
  authorize adding it to the app.

## Active Skills

Detailed, task-specific design specs, scoring constants, interfaces, and acceptance criteria for
each workstream live in skills, loaded on demand, rather than in this file. Load the relevant skill
before touching that area — especially before reopening or re-tuning already-implemented behavior:

* `anchor-ranking` — query-aware, diversified semantic anchor ranking. Load before touching anchor
  scoring/reranking in `explorer/backend/semantic_search/`.
* `hybrid-retrieval-expansion` — BM25/hybrid global retrieval, ranked one-hop neighborhood
  expansion, and re-anchoring. Load before touching retrieval modes, neighborhood expansion, or
  re-anchoring UI/backend.
* `embedding-comparison-analysis` — the offline OpenAI-vs-SapBERT retrieval investigation. Load
  before touching `analysis/`, `notebooks/`, or `tests/test_embedding_comparison.py`.
* `robokop-cross-graph` — bounded LitCoin-to-ROBOKOP cross-graph expansion. Load before touching
  the external graph-provider boundary or ROBOKOP integration.
* `presentation-demo-readiness` — group-presentation and live-demo preflight checklist. Load before
  preparing a presentation or demo handoff.

All four workstreams above are implemented and should be treated as baselines. New work requests should be built on top of these baselines. Only refine or update these baselines if the requested new work warrants it and if users approve it.
