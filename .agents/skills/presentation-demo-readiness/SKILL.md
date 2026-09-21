---
name: presentation-demo-readiness
description: Group-presentation and live-demo preflight checklist tying the Workstream 2 analysis notebook to a live explorer demo — primary/backup query selection, fixture vs. live routes, provenance requirements, and the "presentation" output folder convention. Load before preparing a group presentation, demo script, or demo handoff.
---

# Group Presentation and Demo Readiness

The near-term milestone is a group presentation that uses evidence from the analysis notebook (see
the `embedding-comparison-analysis` skill) and a brief live demonstration of the explorer (using
capabilities from the `anchor-ranking`, `hybrid-retrieval-expansion`, and `robokop-cross-graph`
skills as relevant). The goal is to communicate practical design insights that may improve an
existing path finder application or motivate future grant proposals.

Prioritize a coherent evidence-to-demo story over feature count. The notebook is the source of
quantitative and visual claims; the app demonstrates the human-guided workflow.

Before presentation handoff:

* identify one primary and one backup query
* preflight the expected LitCoin anchors and local expansions
* preflight the bounded ROBOKOP example if that workstream's demo is in scope and ready
* provide a fixture-backed test route for development and a clearly separate live-demo route
* document startup commands, required environment variables, expected latency, and recovery steps
* ensure every visible edge has source and evidence provenance
* avoid claims of model superiority based only on 2D projections or two example queries
* prepare a short table of implemented findings, observed limitations, and grant-worthy future
  directions
* place all generated artifacts useful for the group presentation and demo in the `presentation`
  subfolder

Do not add slide-generation or presentation-authoring dependencies to the explorer unless
explicitly requested. Produce reusable figures, tables, and concise conclusions that can be
transferred into slides.
