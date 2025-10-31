"""Profile the /explain endpoint with cProfile.

Usage:
  python scripts/profile_api.py --url http://localhost:8000/api/v1/explain --iters 20
"""

import argparse
import cProfile
import io
import json
import pstats
import time
from urllib import request as urlrequest


def hit_endpoint(url: str, payload: dict) -> None:
    data = json.dumps(payload).encode("utf-8")
    req = urlrequest.Request(url, data=data, headers={"Content-Type": "application/json"})
    with urlrequest.urlopen(req, timeout=10) as resp:
        _ = resp.read()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", default="http://localhost:8000/api/v1/explain")
    parser.add_argument("--iters", type=int, default=20)
    args = parser.parse_args()

    payload = {"code": "print('hello')", "strategy": "vanilla"}

    pr = cProfile.Profile()
    pr.enable()
    start = time.time()
    for _ in range(args.iters):
        hit_endpoint(args.url, payload)
    elapsed = time.time() - start
    pr.disable()

    print(f"Completed {args.iters} requests in {elapsed:.2f}s ({args.iters/elapsed:.2f} rps)")

    s = io.StringIO()
    ps = pstats.Stats(pr, stream=s).sort_stats("cumulative")
    ps.print_stats(30)
    print(s.getvalue())


if __name__ == "__main__":
    main()
