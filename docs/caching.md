# Caching

The Code Explainer system provides comprehensive caching capabilities to improve performance and reduce redundant computations.

## Overview

Caching is implemented at multiple levels:

- **Explanation Cache**: Stores generated explanations for code snippets
- **Embedding Cache**: Caches vector embeddings for retrieval-augmented generation
- **Advanced Cache**: Multi-strategy cache with eviction policies, invalidation, and persistence

## Explanation Cache

The basic explanation cache stores explanations keyed by code, strategy, and model name.

```python
from src.code_explainer.cache import ExplanationCache

cache = ExplanationCache(cache_dir=".cache/explanations", max_size=1000)
cache.put("def add(a, b): return a + b", "vanilla", "codet5-base", "Adds two numbers")
result = cache.get("def add(a, b): return a + b", "vanilla", "codet5-base")
```

## Advanced Cache

The advanced cache provides sophisticated caching with multiple strategies:

### Cache Strategies

- **LRU (Least Recently Used)**: Evicts least recently accessed items
- **LFU (Least Frequently Used)**: Evicts least frequently accessed items
- **FIFO (First In, First Out)**: Evicts oldest items
- **Size-based**: Evicts largest items first
- **Adaptive**: Adapts strategy based on access patterns

### Features

- **Persistence**: Cache survives across application restarts
- **Invalidation**: Tag-based, time-based, version-based, and content-based invalidation
- **Metrics**: Hit rates, access times, eviction counts
- **Background cleanup**: Automatic expired entry removal

```python
from src.code_explainer.advanced_cache import AdvancedCacheManager, CacheStrategy

cache = AdvancedCacheManager(
    cache_dir=".cache/advanced",
    max_memory_entries=1000,
    strategy=CacheStrategy.LRU
)

# Store with metadata
cache.put("key", "value", tags={"group_a"}, version="1.0")

# Retrieve
result = cache.get("key")

# Invalidate by tag
invalidated = cache.invalidate_by_tag("group_a")

# Get metrics
stats = cache.get_metrics()
```

## Configuration

Caching can be configured via the main configuration file:

```yaml
cache:
  enabled: true
  type: advanced  # basic, advanced, or none
  directory: .cache
  max_size: 1000
  strategy: lru
  ttl_hours: 24
```

## Performance Tips

1. **Choose appropriate strategy**: LRU for general use, LFU for frequently accessed data
2. **Set reasonable limits**: Balance memory usage with hit rates
3. **Use tags for invalidation**: Group related entries for efficient cleanup
4. **Monitor metrics**: Track hit rates and adjust configuration as needed
5. **Enable persistence**: For expensive computations that should survive restarts

## Troubleshooting

### Low Hit Rates

- Check if cache keys are consistent
- Verify TTL settings aren't too aggressive
- Consider increasing cache size

### Memory Issues

- Reduce max_memory_entries
- Use disk-only caching for large datasets
- Monitor cache size metrics

### Invalidation Problems

- Ensure tags are applied correctly
- Check version/content hashes for accuracy
- Verify invalidation timing