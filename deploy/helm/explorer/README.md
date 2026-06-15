# Explorer Helm Chart

Deploy the Streamlit Human-Guided Semantic Graph Explorer.

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
