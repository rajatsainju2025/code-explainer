# Performance Tuning Guide

This guide collects practical tips to run Code Explainer faster in development and production.

## Server Settings

- Use `orjson` for JSON responses (already enabled)
- Enable gzip compression (already enabled)
- Prefer `uvloop` event loop on Linux/macOS
- Tune Uvicorn:
  - `--workers 4` (or `uvicorn_config.py` for auto)
  - `--limit-concurrency 100`
  - `--backlog 2048`
  - `--timeout-keep-alive 20`
  - `--no-access-log`

## API Usage

- Use `/explain/batch` for multiple snippets to amortize overhead. The endpoint serves cached results and computes misses concurrently.
- Reuse HTTP connections (keep-alive). Most HTTP clients do this by default.

## Caching

- Explanations are cached by `(code, strategy, model)` with a disk+memory hybrid cache.
- Configure cache directory and TTL via config.
- Warm up cache by sending common snippets at startup.

## Model

- `torch.inference_mode()` is used for generation where available.
- Prefer smaller models for latency-sensitive scenarios.
- Use `mps` on Apple Silicon or `cuda` on NVIDIA GPUs when available.

## Data Loading

- Use `DataLoader.iter_dataset()` to stream large datasets (supports `.jsonl` and ijson fallback).

## Profiling

- Use `scripts/profile_api.py` for quick cProfile sessions.
- Measure at realistic concurrency with `ab`, `wrk`, or `hey`.

## Observability

- Prometheus metrics available at `/api/v1/prometheus`. Not cached.
- `Server-Timing` and `X-Response-Time` headers are included for quick checks.
