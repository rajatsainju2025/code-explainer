# Configuration

Code Explainer can be configured via YAML files and environment variables.

## YAML Config

Create `config.yaml`:

```yaml
model:
  name: "Salesforce/codet5-small"
  max_length: 512
  device: auto  # cpu | cuda | auto

retrieval:
  enabled: true
  top_k: 5
  use_reranker: true
  mmr_lambda: 0.5

api:
  host: 0.0.0.0
  port: 8000
  workers: 2

logging:
  level: INFO
  json: true
```

## Environment Variables

```bash
export CODE_EXPLAINER_MODEL="Salesforce/codet5-small"
export CODE_EXPLAINER_MAX_LENGTH=512
export CODE_EXPLAINER_DEVICE="auto"
export CODE_EXPLAINER_METRICS_ENABLED=true
export CODE_EXPLAINER_LOG_LEVEL="INFO"
```

## CLI Overrides

```bash
code-explainer explain --strategy enhanced_rag --max-length 512
```
