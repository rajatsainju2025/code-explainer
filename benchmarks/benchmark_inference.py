import time
from code_explainer import CodeExplainer


def run_benchmark(n: int = 3):
    ex = CodeExplainer()
    code = "def add(a,b): return a+b"
    times = []
    for _ in range(n):
        t0 = time.time()
        _ = ex.explain_code(code)
        times.append(time.time() - t0)
    print({"avg_sec": sum(times)/len(times), "runs": n})


if __name__ == "__main__":
    run_benchmark()
