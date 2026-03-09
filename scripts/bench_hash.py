import time
from code_explainer.utils.hashing import fast_hash_str

keys = [f"key-{i}" for i in range(1000)] * 100

start = time.time()
for k in keys:
    fast_hash_str(k)
end = time.time()
print(f"Computed {len(keys)} hashes in {end-start:.4f}s")
