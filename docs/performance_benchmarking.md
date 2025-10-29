# Performance Benchmarking Guide

This guide explains how to benchmark and profile the Code Explainer system.

## Quick Benchmarks

### Basic Inference Speed

```bash
# Run basic benchmark
python benchmarks/benchmark_inference.py --model codet5-small --samples 10

# Compare multiple models
python benchmarks/benchmark_inference.py --compare
```

### API Performance

```bash
# Start API server
make api-dev

# In another terminal, run load test
python scripts/load_test_api.py --requests 100 --concurrent 10
```

## Detailed Profiling

### Memory Profiling

```python
from memory_profiler import profile

@profile
def explain_code(code: str):
    explainer = CodeExplainer()
    return explainer.explain(code)

# Run with: python -m memory_profiler script.py
```

### CPU Profiling

```bash
# Using cProfile
python -m cProfile -o profile.stats train.py

# Analyze results
python -c "import pstats; p=pstats.Stats('profile.stats'); p.sort_stats('cumulative'); p.print_stats(20)"

# Or use snakeviz for visualization
pip install snakeviz
snakeviz profile.stats
```

### Line Profiling

```bash
# Install line_profiler
pip install line_profiler

# Add @profile decorator to functions
# Run with: kernprof -l -v script.py
```

## Benchmarking Strategies

### Explanation Quality vs Speed

```python
import time
from code_explainer import CodeExplainer

strategies = ["basic", "detailed", "advanced"]
code = "def factorial(n): return 1 if n <= 1 else n * factorial(n-1)"

for strategy in strategies:
    explainer = CodeExplainer(strategy=strategy)
    
    start = time.time()
    result = explainer.explain(code)
    elapsed = time.time() - start
    
    print(f"{strategy}: {elapsed:.2f}s, {len(result)} chars")
```

### Batch Processing

```python
from code_explainer import CodeExplainer
import time

explainer = CodeExplainer()
codes = [f"print({i})" for i in range(100)]

# Sequential
start = time.time()
for code in codes:
    explainer.explain(code)
seq_time = time.time() - start

# Batch
start = time.time()
explainer.explain_batch(codes)
batch_time = time.time() - start

print(f"Sequential: {seq_time:.2f}s")
print(f"Batch: {batch_time:.2f}s")
print(f"Speedup: {seq_time/batch_time:.2f}x")
```

## Performance Metrics

### Key Metrics to Track

1. **Inference Time**: Time to generate explanation
2. **Model Load Time**: Time to load model into memory
3. **Memory Usage**: Peak memory during inference
4. **Throughput**: Explanations per second
5. **Cache Hit Rate**: Percentage of cached responses

### Using Prometheus Metrics

```bash
# Start API with monitoring
docker-compose --profile monitoring up

# View metrics
curl http://localhost:8000/prometheus

# Grafana dashboard
open http://localhost:3000
```

## Optimization Tips

### 1. Model Selection

```yaml
# Fast but less detailed
model: codet5-small

# Balanced
model: codet5-base

# High quality but slower
model: codellama-instruct
```

### 2. Batch Processing

```python
# Good for multiple explanations
explainer.explain_batch(codes, batch_size=8)
```

### 3. Caching

```python
# Enable Redis caching
export CODE_EXPLAINER_CACHE_ENABLED=true
export CODE_EXPLAINER_CACHE_TYPE=redis
export CODE_EXPLAINER_CACHE_URL=redis://localhost:6379
```

### 4. Device Selection

```python
# Use GPU if available
explainer = CodeExplainer(device="cuda")

# Use Apple Silicon GPU
explainer = CodeExplainer(device="mps")
```

## Continuous Benchmarking

### Add to CI/CD

```yaml
# .github/workflows/benchmark.yml
name: Benchmark
on: [push]
jobs:
  benchmark:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - name: Run benchmarks
        run: |
          python benchmarks/benchmark_inference.py --output results.json
      - name: Compare with baseline
        run: |
          python scripts/compare_benchmarks.py results.json baseline.json
```

## Troubleshooting Performance

### Slow Inference

1. Check device: `torch.cuda.is_available()`
2. Reduce batch size if OOM
3. Use smaller model
4. Enable caching

### High Memory Usage

1. Use `torch.no_grad()` during inference
2. Clear cache regularly: `torch.cuda.empty_cache()`
3. Reduce model size
4. Use gradient checkpointing

### Cold Start Issues

1. Pre-load models at startup
2. Use model serving (e.g., TorchServe)
3. Keep warm instances in production

## Resources

- [PyTorch Performance Tuning Guide](https://pytorch.org/tutorials/recipes/recipes/tuning_guide.html)
- [Python Profiling Tools](https://docs.python.org/3/library/profile.html)
- [Grafana Dashboards](monitoring/grafana-dashboard.json)
