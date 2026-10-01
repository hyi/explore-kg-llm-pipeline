# Explorer Helm Chart

Deploy the Dash Human-Guided Semantic Graph Explorer.

```bash
helm install explorer deploy/helm/explorer \
    --set env.existingSecret=explorer-env \
    --set env.createSecret=false
```

Or create local-values.yaml file that contains environment variables containing secrets, 
and deploy with the command `helm install -n <namespace> explorer . -f local-values.yaml`

For production, prefer an existing Kubernetes secret:

```bash
kubectl create secret generic explorer-env \
  --from-literal=NEO4J_URI='bolt://neo4j:7687' \
  --from-literal=NEO4J_USERNAME='<username>' \
  --from-literal=NEO4J_PASSWORD='<password>' \
  --from-literal=OPENAI_API_KEY='<api-key>'

helm install explorer deploy/helm/explorer \
  --set image.repository=<registry>/explore-kg-llm-pipeline \
  --set image.tag=<tag> \
  --set env.existingSecret=explorer-env \
  --set env.createSecret=false
```

Path-search cache is local to each pod and defaults to `/tmp/kg_explorer/path_search_cache.json`.
Override it with:

```bash
helm upgrade --install explorer deploy/helm/explorer \
  --set cache.path=/tmp/kg_explorer/path_search_cache.json
```

When `EMBEDDING_PROVIDER=sapbert`, the app loads
`cambridgeltl/SapBERT-from-PubMedBERT-fulltext` from Hugging Face on the first
embedding request. The model is not bundled in the image. Pods need outbound
access to Hugging Face and writable cache space (by default under `/tmp/.cache`
because the image sets `HOME=/tmp`). The chart requests 512Mi of ephemeral
storage and limits usage to 2Gi to allow room for the downloaded model. Adjust both for
your namespace's quota and observed cache usage, or
mount persistent storage for the Hugging Face cache if repeated cold downloads
are undesirable. An offline deployment needs a pre-populated cache or a locally 
available model instead.

Specify non-secret application settings with `extraEnv` in your values file,
not under `env` (which only configures the chart's connection secret):

```yaml
extraEnv:
  - name: EMBEDDING_PROVIDER
    value: sapbert
  - name: KG_EXPLORER_RETRIEVAL_MODE
    value: hybrid
```
