# Group Presentation Package

Generated from `analysis/outputs/embedding_comparison` at `2026-09-18T19:56:38+00:00`.

## What Is Included

- `artifacts/`: copied interactive HTML figures and source CSV/JSON evidence from the embedding comparison run.
- `tables/retrieval_summary.csv`: compact retrieval-mode summary for slide tables.
- `tables/embedding_coverage.csv`: model coverage and embedding dimensions.
- `tables/demo_readiness_matrix.csv`: preflight status for the notebook and live demo.
- `demo_runbook.md`: commands, environment variables, demo path, and recovery steps.
- `conclusions.md`: slide-ready observations, interpretations, caveats, and future directions.

## Evidence Snapshot

- openai: 2003 edges, 1536 dimensions
- sapbert: 2003 edges, 768 dimensions

- Same edge population: `True`.
- Mean top-10 nearest-neighbor Jaccard: `0.542`.
- Query overlap for `genes involved in chemoresistance in cancer`: `7` shared top-15 edges, Jaccard `0.304`.
- Missing source artifacts: `none`.

## Primary Demo Query

`genes involved in chemoresistance in cancer`

## Backup Demo Query

`drug resistance in cancer`

## Interpretation Guardrails

- Compare OpenAI and SapBERT by matched relationship IDs, retrieved neighborhoods, and rankings, not by absolute 2D projection coordinates.
- Treat projection figures as exploratory diagnostics, not proof that one embedding model is globally better.
- Treat bounded ROBOKOP pages as partial neighborhoods; pagination, filters, and mode badges are part of the demo story.
