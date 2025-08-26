"""Example: Query the /retrieve endpoint to fetch similar code snippets.

Run the API first:
  make api-serve

Then run:
  python examples/api_retrieval_example.py
"""

import requests

payload = {
    "code": "def fib(n): return n if n < 2 else fib(n-1)+fib(n-2)",
    "index_path": "data/code_retrieval_index.faiss",
    "top_k": 3,
}

resp = requests.post("http://localhost:8000/retrieve", json=payload, timeout=30)
print(resp.status_code)
print(resp.json())
