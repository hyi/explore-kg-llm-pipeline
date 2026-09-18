# Running The Embedding Comparison Notebook

Use the base Anaconda Jupyter server and the repo-local `uv` kernel. This avoids
installing Jupyter Server into `.venv`, while notebook code still executes with
the project dependencies.

From the repository root:

```bash
export JUPYTER_PATH="$PWD/.jupyter${JUPYTER_PATH:+:$JUPYTER_PATH}"
/home/hongyi/anaconda3/bin/jupyter lab --notebook-dir "$PWD"
```

Open `notebooks/embedding_comparison.ipynb` and select the
`KG Explorer (uv)` kernel.

The kernel command uses:

```bash
uv --directory /home/hongyi/explore-kg-llm-pipeline run --frozen --with ipykernel --with nbformat --with umap-learn python -m ipykernel_launcher
```

This installs only kernel-side packages needed by `ipykernel`, Plotly notebook
rendering, and UMAP in the `uv` environment. Do not install `jupyter`,
`notebook`, or `jupyterlab` into the project `.venv` unless there is a separate
reason to debug Jupyter Server itself.

To verify that Jupyter can see the kernel:

```bash
export JUPYTER_PATH="$PWD/.jupyter${JUPYTER_PATH:+:$JUPYTER_PATH}"
/home/hongyi/anaconda3/bin/jupyter kernelspec list
```

For non-interactive artifact generation without opening Jupyter:

```bash
uv run python scripts/generate_embedding_comparison_artifacts.py
```
