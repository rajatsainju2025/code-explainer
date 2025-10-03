# Configuration Guide

Code Explainer supports comprehensive configuration through YAML files, environment variables, and runtime overrides. The configuration system is hierarchical and supports validation, defaults, and hot-reloading.

## 📁 Configuration Files

### Default Configuration
The system loads configuration from multiple sources in order of priority:

1. **Environment Variables** (highest priority)
2. **User Config File** (`config.yaml` or specified path)
3. **Default Config** (`configs/default.yaml`)
4. **Built-in Defaults** (lowest priority)

### Configuration Structure

```yaml
# Main configuration file (config.yaml)
model:
  name: "microsoft/CodeGPT-small-py"
  arch: "causal"
  max_length: 512
  temperature: 0.7
  top_p: 0.9
  top_k: 50
  torch_dtype: "auto"
  load_in_8bit: false
  device: "auto"

training:
  output_dir: "results"
  num_train_epochs: 3
  per_device_train_batch_size: 8
  per_device_eval_batch_size: 8
  learning_rate: 5e-5
  weight_decay: 0.01
  save_steps: 500
  eval_steps: 500
  logging_steps: 100

cache:
  enabled: true
  advanced_cache_enabled: true
  directory: ".cache"
  max_size: 10000
  ttl: 3600
  strategy: "lru"
  persistence: true
  compression: true

retrieval:
  enabled: true
  index_path: "data/code_retrieval_index.faiss"
  corpus_path: "data/code_corpus.json"
  top_k: 5
  use_reranker: true
  mmr_lambda: 0.5
  reranker_model: "cross-encoder/ms-marco-MiniLM-L-6-v2"

api:
  host: "0.0.0.0"
  port: 8000
  workers: 2
  rate_limit: "60/minute"
  max_request_size: "10MB"
  cors_origins: ["*"]
  enable_metrics: true
  enable_health_checks: true

security:
  strict_mode: true
  rate_limiting_enabled: true
  input_validation_enabled: true
  security_monitoring_enabled: true
  max_code_length: 10000
  audit_log_enabled: true
  audit_log_path: "logs/security.log"
  allowed_imports: ["typing", "dataclasses", "enum"]

performance:
  enable_quantization: false
  quantization_bits: 8
  enable_gradient_checkpointing: false
  optimize_for_inference: false
  optimize_tokenizer: false
  batch_size: 32
  max_workers: 4
  memory_monitoring_enabled: true
  performance_logging_enabled: true

logging:
  level: "INFO"
  format: "json"
  log_file: "logs/code_explainer.log"
  max_file_size: "100MB"
  backup_count: 5
  json_format: true

prompt:
  strategy: "vanilla"
  template: "Explain this code:\n{code}\n"
  max_new_tokens: 150
  repetition_penalty: 1.1
  length_penalty: 1.0
  no_repeat_ngram_size: 3
```

## 🔧 Environment Variables

### Model Configuration
```bash
# Model selection and architecture
export CODE_EXPLAINER_MODEL="microsoft/CodeGPT-small-py"
export CODE_EXPLAINER_MODEL_ARCH="causal"
export CODE_EXPLAINER_MAX_LENGTH=512
export CODE_EXPLAINER_TEMPERATURE=0.7
export CODE_EXPLAINER_TOP_P=0.9
export CODE_EXPLAINER_TOP_K=50
export CODE_EXPLAINER_TORCH_DTYPE="auto"
export CODE_EXPLAINER_LOAD_IN_8BIT=false
export CODE_EXPLAINER_DEVICE="auto"
export CODE_EXPLAINER_PRECISION="fp32"
```

### API Configuration
```bash
# Server settings
export CODE_EXPLAINER_API_HOST="0.0.0.0"
export CODE_EXPLAINER_API_PORT=8000
export CODE_EXPLAINER_API_WORKERS=2

# Security and limits
export CODE_EXPLAINER_RATE_LIMIT="60/minute"
export CODE_EXPLAINER_MAX_REQUEST_SIZE="10MB"
export ALLOWED_ORIGINS="http://localhost:3000,http://localhost:8080"

# Features
export CODE_EXPLAINER_METRICS_ENABLED=true
export CODE_EXPLAINER_HEALTH_CHECKS_ENABLED=true
```

### Security Configuration
```bash
# Security settings
export CODE_EXPLAINER_SECURITY_STRICT=true
export CODE_EXPLAINER_RATE_LIMITING_ENABLED=true
export CODE_EXPLAINER_INPUT_VALIDATION_ENABLED=true
export CODE_EXPLAINER_SECURITY_MONITORING_ENABLED=true
export CODE_EXPLAINER_MAX_CODE_LENGTH=10000
export CODE_EXPLAINER_AUDIT_LOG_ENABLED=true
export CODE_EXPLAINER_AUDIT_LOG_PATH="logs/security.log"
```

### Performance Configuration
```bash
# Performance optimizations
export CODE_EXPLAINER_ENABLE_QUANTIZATION=false
export CODE_EXPLAINER_QUANTIZATION_BITS=8
export CODE_EXPLAINER_ENABLE_GRADIENT_CHECKPOINTING=false
export CODE_EXPLAINER_OPTIMIZE_FOR_INFERENCE=false
export CODE_EXPLAINER_OPTIMIZE_TOKENIZER=false

# Processing settings
export CODE_EXPLAINER_BATCH_SIZE=32
export CODE_EXPLAINER_MAX_WORKERS=4
export CODE_EXPLAINER_MEMORY_MONITORING_ENABLED=true
export CODE_EXPLAINER_PERFORMANCE_LOGGING_ENABLED=true
```

### Cache Configuration
```bash
# Cache settings
export CODE_EXPLAINER_CACHE_ENABLED=true
export CODE_EXPLAINER_ADVANCED_CACHE_ENABLED=true
export CODE_EXPLAINER_CACHE_DIR=".cache"
export CODE_EXPLAINER_CACHE_MAX_SIZE=10000
export CODE_EXPLAINER_CACHE_TTL=3600
export CODE_EXPLAINER_CACHE_STRATEGY="lru"
export CODE_EXPLAINER_CACHE_PERSISTENCE=true
export CODE_EXPLAINER_CACHE_COMPRESSION=true
```

### Logging Configuration
```bash
# Logging settings
export CODE_EXPLAINER_LOG_LEVEL="INFO"
export CODE_EXPLAINER_LOG_FORMAT="json"
export CODE_EXPLAINER_LOG_FILE="logs/code_explainer.log"
export CODE_EXPLAINER_LOG_MAX_SIZE="100MB"
export CODE_EXPLAINER_LOG_BACKUP_COUNT=5
export CODE_EXPLAINER_JSON_LOGGING=true
```

## 🎯 Configuration Profiles

### Development Profile
```yaml
# configs/development.yaml
model:
  load_in_8bit: false  # Full precision for debugging
  max_length: 256      # Shorter for faster iteration

cache:
  max_size: 1000       # Smaller cache for development
  persistence: false   # No persistence to avoid conflicts

security:
  strict_mode: false   # Relaxed security for development
  audit_log_enabled: false

logging:
  level: "DEBUG"
  json_format: false   # Human-readable logs

api:
  rate_limit: "1000/minute"  # Higher limits for testing
```

### Production Profile
```yaml
# configs/production.yaml
model:
  load_in_8bit: true   # Memory efficient
  device: "cuda"       # GPU acceleration

cache:
  max_size: 50000      # Large cache for performance
  persistence: true    # Persistent across restarts
  compression: true    # Save disk space

security:
  strict_mode: true    # Maximum security
  audit_log_enabled: true
  rate_limiting_enabled: true

performance:
  enable_quantization: true
  optimize_for_inference: true
  memory_monitoring_enabled: true

api:
  workers: 4           # Multiple workers for load
  rate_limit: "60/minute"
  enable_metrics: true

logging:
  level: "WARNING"
  json_format: true    # Structured logging
```

### High-Performance Profile
```yaml
# configs/high-performance.yaml
model:
  torch_dtype: "fp16"  # Mixed precision
  load_in_8bit: true   # Quantization

performance:
  enable_quantization: true
  quantization_bits: 4
  enable_gradient_checkpointing: true
  optimize_for_inference: true
  optimize_tokenizer: true
  batch_size: 64

cache:
  max_size: 100000
  strategy: "lfu"      # Frequency-based eviction

api:
  workers: 8
  rate_limit: "200/minute"
```

## 🔄 Runtime Configuration

### Dynamic Reconfiguration
```python
from code_explainer import CodeExplainer

# Load with specific config
explainer = CodeExplainer(config_path="configs/production.yaml")

# Runtime optimization
explainer.enable_quantization(bits=8)
explainer.optimize_for_inference()

# Check current configuration
config = explainer.get_config()
print(config)
```

### CLI Configuration
```bash
# Override config with CLI flags
code-explainer explain \
  --config configs/custom.yaml \
  --model "Salesforce/codet5-base" \
  --max-length 1024 \
  --strategy enhanced_rag

# Environment override
CODE_EXPLAINER_MODEL="microsoft/CodeGPT-small-py" \
code-explainer serve
```

## ✅ Configuration Validation

### Schema Validation
The configuration system validates:
- **Type Safety**: Ensures correct data types
- **Range Validation**: Checks numeric ranges
- **Enum Validation**: Validates string enums
- **Path Validation**: Checks file/directory existence
- **Dependency Validation**: Ensures required components

### Validation Errors
```python
# Invalid configuration will raise errors
from code_explainer.config import ConfigError

try:
    explainer = CodeExplainer(config_path="invalid.yaml")
except ConfigError as e:
    print(f"Configuration error: {e}")
```

### Validation Rules
```python
# Example validation rules
model_config = {
    "max_length": {"type": int, "min": 1, "max": 4096},
    "temperature": {"type": float, "min": 0.0, "max": 2.0},
    "strategy": {"type": str, "enum": ["vanilla", "ast_augmented", "enhanced_rag"]}
}
```

## 🔍 Configuration Debugging

### Configuration Inspection
```python
# View current configuration
explainer = CodeExplainer()
config = explainer.get_config()
print(yaml.dump(config, default_flow_style=False))

# Validate configuration
is_valid, errors = explainer.validate_config()
if not is_valid:
    for error in errors:
        print(f"Config error: {error}")
```

### Configuration Sources
```python
# Check which config sources are loaded
sources = explainer.get_config_sources()
for source, path in sources.items():
    print(f"{source}: {path}")
```

### Hot Reloading
```python
# Reload configuration without restart
explainer.reload_config("new_config.yaml")

# Check if reload was successful
status = explainer.get_reload_status()
print(f"Reload status: {status}")
```

## 🚀 Advanced Configuration

### Custom Config Classes
```python
from code_explainer.config import Config, register_config

@register_config
class CustomConfig(Config):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.custom_setting = kwargs.get("custom_setting", "default")

    def validate(self):
        super().validate()
        # Custom validation logic
        if self.custom_setting not in ["option1", "option2"]:
            raise ValueError("Invalid custom setting")
```

### Configuration Plugins
```python
# Plugin-based configuration extension
from code_explainer.config import ConfigPlugin

class DatabaseConfig(ConfigPlugin):
    def apply(self, config):
        config.database = {
            "host": os.getenv("DB_HOST", "localhost"),
            "port": int(os.getenv("DB_PORT", 5432))
        }
```

### Environment-Specific Configs
```python
# Load config based on environment
import os

env = os.getenv("ENVIRONMENT", "development")
config_path = f"configs/{env}.yaml"

explainer = CodeExplainer(config_path=config_path)
```

## 📊 Configuration Monitoring

### Configuration Metrics
```
# HELP config_load_total Total configuration loads
# TYPE config_load_total counter
config_load_total 42

# HELP config_validation_errors_total Configuration validation errors
# TYPE config_validation_errors_total counter
config_validation_errors_total 3
```

### Configuration Dashboard
```python
# Configuration health check
health = explainer.get_config_health()
print(f"Config health: {health['status']}")
print(f"Validation errors: {health['errors']}")
print(f"Last modified: {health['last_modified']}")
```

## 🐛 Troubleshooting

### Common Configuration Issues

#### Config File Not Found
```bash
# Check if file exists
ls -la configs/default.yaml

# Use absolute path
explainer = CodeExplainer(config_path="/full/path/to/config.yaml")
```

#### Invalid Configuration
```python
# Validate before loading
from code_explainer.config import validate_config_file

errors = validate_config_file("config.yaml")
if errors:
    for error in errors:
        print(f"Validation error: {error}")
```

#### Environment Variable Issues
```bash
# Check environment variables
env | grep CODE_EXPLAINER

# Debug variable loading
explainer = CodeExplainer(debug_config=True)
```

#### Permission Issues
```bash
# Check file permissions
ls -la config.yaml

# Fix permissions
chmod 644 config.yaml
```

## 📚 Best Practices

### Configuration Management
- Use version control for configuration files
- Document all configuration options
- Use environment-specific configs
- Validate configurations in CI/CD
- Monitor configuration changes

### Security Considerations
- Don't commit secrets to version control
- Use environment variables for sensitive data
- Validate configuration sources
- Audit configuration changes
- Use principle of least privilege

### Performance Optimization
- Cache frequently-used configurations
- Use efficient data structures
- Minimize configuration lookups
- Profile configuration loading
- Optimize for your deployment environment
