Branch: performance-tweaks

This branch contains small, conservative performance improvements and testing fixes:

- LRU caching for fast hashing
- Compiled ContentFilter regexes cached at class level
- Lazy import for get_device
- Faster pickle protocol for model cache
- Minor benchmarks and documentation notes

Commits: 20 (small, incremental)
