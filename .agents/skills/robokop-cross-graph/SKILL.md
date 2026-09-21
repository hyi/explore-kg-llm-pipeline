---
name: robokop-cross-graph
description: Bounded, human-guided LitCoin-to-ROBOKOP cross-graph expansion (Workstream 3) — RoboMCP integration constraints, the external graph-provider boundary, CURIE bridge normalization, one-hop expansion flow, demo fixture rules, and required tests. Load before touching the external graph-provider boundary, ROBOKOP integration code, or bridge/CURIE normalization logic.
---

# Workstream 3: LitCoin-to-ROBOKOP Cross-Graph Expansion

Implemented. Treat as closed unless the user explicitly reopens cross-graph work.

## Goal

A human-guided cross-graph workflow: (1) find and select useful anchors in the LitCoin graph,
(2) explore LitCoin paths and neighborhoods, (3) choose a normalized LitCoin entity as a bridge,
(4) expand that entity into the much larger ROBOKOP knowledge graph, (5) optionally continue
bounded one-hop exploration in either graph. Demonstrates how a small, literature-focused KG can
provide interpretable anchors into a large integrated biomedical KG. Do not ingest or replicate the
complete ROBOKOP graph into the LitCoin Neo4j database.

## Integration Reference and Constraints

RoboMCP (`https://github.com/cbizon/RoboMCP`, `robokop-mcp` server) provides `get_node(curie)`,
`get_edges(curie, category, predicate, limit, offset, count_only)`, `get_edge_summary(curie)`, and
`get_edges_between(curie1, curie2)`. These support known-CURIE lookup and adjacency expansion; they
do not provide global semantic or keyword search over all ROBOKOP nodes. LitCoin retrieval
establishes the initial anchor; ROBOKOP is used only after a bridge CURIE is known.

The inspected MCP implementation returns human-formatted strings — do not parse those strings in
production application code. The explorer's own provider interface hides transport details (typed
adapter to the ROBOKOP HTTP API, or an upstream-compatible RoboMCP enhancement) from session, path,
ranking, and frontend code.

Do not use the current `get_edges_between` implementation for arbitrary high-degree nodes without
review: it may request up to 10,000 edges from one endpoint and filter client-side.

## Identifier Normalization and Bridge Resolution

LitCoin and ROBOKOP are expected to already share Biolink categories/predicates and canonical
CURIEs for an entity, but verify before external expansion rather than assuming it.

## Provider Boundary

A small external graph-provider abstraction sits apart from the Neo4j adapter (the Neo4j adapter
must not itself call remote services). It supports: node lookup by normalized CURIE, edge-summary
retrieval, bounded incident-edge retrieval with category/predicate/direction/limit/pagination
controls where supported, and optional direct-edge lookup between two known CURIEs.

Remote results become explorer domain objects only after retaining: source graph (`litcoin` or
`robokop`), stable/reproducible edge identity, original and normalized endpoint CURIEs, Biolink
categories and predicates, edge direction, qualifiers, primary knowledge source, and
publications/supporting sentences when available. Never merge two edges merely because their
subject, predicate, and object look similar — preserve source-specific evidence and provenance.

## Bounded Cross-Graph Interaction

User-initiated, one-hop ROBOKOP expansion only. For a selected bridge entity: (1) fetch and display
an edge summary first, (2) let the user choose or confirm category and predicate filters, (3) fetch
a small page of incident edges, (4) rank the returned page against the active or refined query,
(5) let the user select which remote edges to add to the visible exploration.

ROBOKOP has no stored embeddings available through RoboMCP — ranking starts with keyword relevance,
query-intent compatibility, and explicit type/predicate filters. On-demand embedding reranking of a
small fetched candidate set is future work; it must be cached, bounded, labeled, and evaluated
against the simpler lexical baseline before being added.

Never imply that a limited page is the complete ROBOKOP neighborhood. Show counts, limits,
pagination state, loading state, source badges, and remote errors clearly. A ROBOKOP timeout or
outage must not corrupt the LitCoin session or remove locally explored paths.

## Demo Scope

Optimize for one reliable, explainable cross-graph story over broad feature coverage. The scripted
example demonstrates: semantic/keyword/hybrid discovery of a LitCoin anchor, query-aware local
exploration, CURIE normalization and bridge confirmation, bounded ROBOKOP expansion with visible
provenance, and user selection of a remote edge or branch.

Keep a local fixture or recorded, non-sensitive response for deterministic UI testing and
presentation fallback. Clearly label fixture data in development and never present it as a live
response. Do not silently fall back from live to recorded data during the demo.

## Cross-Graph Tests and Acceptance Criteria

Tests must not depend on live ROBOKOP or Translator services — use committed, small response
fixtures that preserve realistic response structure and provenance. Required:

* one normalized identifier maps a LitCoin node to the intended ROBOKOP lookup
* zero and ambiguous mappings require explicit handling
* display-name equality alone cannot create a bridge
* gene/protein and drug/chemical conflation settings affect cache identity
* category, predicate, limit, offset, and direction controls are propagated
* remote results retain source graph, identifiers, qualifiers, publications, and primary knowledge
  source
* duplicate-looking LitCoin and ROBOKOP edges retain separate provenance
* high-degree expansion is bounded and paginated
* timeout, malformed response, and unavailable-provider states are visible and leave the LitCoin
  session intact
* fixture mode and live mode cannot be confused in the UI

A separate opt-in integration check runs against the configured live endpoint. Live network access
is never part of the default pytest suite.
