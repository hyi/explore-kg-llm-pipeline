---
name: hybrid-retrieval-expansion
description: BM25/hybrid global relationship retrieval, separated edge/title/context text fields, rank-fusion rules, human-guided ranked one-hop neighborhood expansion (category/predicate/direction/limit filters), and session-preserving re-anchoring (Workstream 1B). Load before touching retrieval modes (dense/keyword/hybrid), neighborhood expansion UI or backend, or re-anchoring behavior.
---

# Workstream 1B: Hybrid Retrieval and Ranked Neighborhood Expansion

Implemented. Backend ranked one-hop expansion and frontend node-expansion controls exist; baseline
session-preserving re-anchoring exists via re-running global search. Treat this as closed unless
the user explicitly reopens retrieval or expansion work.

## Design Basis

Uses ideas from Polonuer et al., "Autonomous Knowledge Graph Exploration with Adaptive Breadth-
Depth Retrieval" (ACL 2026, see `paper/2026.acl-long.714.pdf`) as design motivation only, not as an
instruction to reproduce ARK:

* retain global retrieval for broad discovery and later re-anchoring
* use bounded one-hop expansion for relational depth
* rank local neighbors against the active query or a user-refined subquery
* filter heterogeneous neighborhoods by node type and predicate when useful
* stop expansion selectively rather than traversing indiscriminately
* evaluate the latency and quality effects of increasing retrieval depth

ARK is autonomous and node-retrieval oriented. This project remains human-guided and path-centric.
Do not add parallel agents, autonomous trajectories, trajectory voting, or model distillation in
this workstream.

## Separate Searchable Text Fields

Do not rely on one undifferentiated text field for every retrieval method. Preserve or create
logically separate representations for:

* `edge_text`: subject, qualifiers, predicate, object, and qualifiers
* `title_text`: publication title
* `context_text`: supporting sentence when available, otherwise abstract text

Neo4j property names may be adjusted after inspecting the data, but the edge claim must remain
distinguishable from publication context. Avoid copying a complete abstract into API responses or
UI state unnecessarily.

## Keyword and Hybrid Global Retrieval

BM25 (or the nearest practical Neo4j full-text equivalent) is a second global relationship-
candidate channel. Primary keyword retrieval is edge-claim centered. Publication title and broad
context search is a separate publication-context retrieval layer or explicit user action, not part
of initial anchor generation except as a labeled fallback.

BM25 keyword queries are built from content-bearing terms separately from parsed structural intent.
Generic endpoint-type expressions such as genes, proteins, variants, drugs, diseases, and pathways
primarily drive graph-aware category intent rather than BM25 scoring. Broad relational scaffolding
such as `involved in` and `related to` is de-emphasized for BM25. Biomedical content terms and
relation-specific concepts (e.g. `chemoresistance`, `cancer`, treatment, response, sensitivity,
named entities) remain available to keyword retrieval. The dense query text stays unchanged.

Independently testable retrieval modes: `dense`, `keyword`, `hybrid`.

Raw cosine and BM25 scores are never combined directly (their scales are unrelated). Use a
rank-based fusion method such as weighted reciprocal rank fusion, with named, configurable
constants. Preserve each candidate's source ranks and raw scores for diagnosis.

Hybrid candidate flow:

1. retrieve bounded dense and keyword candidate pools
2. deduplicate by stable relationship identity
3. fuse channel ranks
4. apply query-to-entity-type and query-to-predicate compatibility
5. diversify by publication and endpoint neighborhood
6. pass ordered anchors to path discovery

If a channel is unavailable, return a clearly identified degraded result rather than silently
labeling a single-channel result as hybrid.

## Human-Guided Neighborhood Retrieval

The UI lets the user select a node, provide an optional refined expansion query, set direction, set
a limit, and apply adjacent-category and predicate filters, then expand connected neighbors.
Adjacent category and predicate filters are KG-populated multi-select dropdowns backed by Neo4j
node labels and relationship types; selected values are OR filters. Ranking uses query intent,
endpoint category compatibility, predicate-family compatibility, lexical matched-token diagnostics,
and relationship-quality handling. Bounded expansion by limit is the available fallback exploration
mechanism — there is no separate "show all unranked neighbors" button.

When a user expands a LitCoin node or path, keep the active query available and allow an optional
user-refined expansion query. Bounded controls: adjacent node category; relationship predicate or
predicate family; incoming/outgoing/either direction; number of returned neighbors.

Rank the filtered neighborhood against the active or refined query using the available lexical,
semantic, and structural signals. Provide an explicit way to reveal additional or unranked
neighbors so heuristic ranking does not hide the graph.

Do not automatically execute an unbounded multi-hop traversal. Preserve human selection between
meaningful expansions. Show unexplored-neighbor counts when available and make retrieval limits
visible rather than presenting them as complete neighborhoods.

## Re-Anchoring

Baseline session-preserving re-anchoring is available: a user can run another global search without
clearing accepted paths, bookmarks, rejected paths, notes, or search history. A dedicated UI action
that merges alternate anchors into the currently displayed candidate branch is future product work,
not required for Workstream 2.

Global search must remain available after exploration begins. Preserve a workflow that lets the
user find alternate anchors and add a new branch without discarding accepted paths, bookmarks,
notes, or the current session.

Treat re-anchoring as recovery from an incomplete or misleading local region, not as failure of the
user. Keep the provenance of the query and retrieval mode that introduced each anchor.

## Hybrid Retrieval Tests and Evaluation

Tests that do not require a live Neo4j instance cover:

* deterministic keyword tokenization and query handling
* stable rank fusion with ties and missing channels
* relationship deduplication across retrieval channels
* preservation of raw channel scores and ranks
* explicit degraded-mode metadata when one channel fails or is unavailable
* query-conditioned local-neighbor ranking
* node-category, predicate, direction, and limit filtering
* session-preserving re-anchoring

The offline evaluation compares dense OpenAI, dense SapBERT, keyword, and both hybrid variants. Do
not claim the ACL paper establishes the best retriever for LitCoin; its benchmarks are much larger,
retrieve nodes rather than LitCoin paths, and use autonomous agent-generated subqueries.
