Performance improvements applied during this session:

- LRU caching added to fast hashing utilities to speed up repeated keys.
- ContentFilter regexes compiled once per class to reduce instantiation overhead.
- Model cache serialization uses the highest pickle protocol for faster IO.
- Lazy import of device manager in `get_device()` to reduce startup and circular import risk.

These changes are conservative and focused on hot paths identified by quick test runs and code inspection.
