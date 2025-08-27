import numpy as np
from code_explainer.retrieval import CodeRetriever

class FakeModel:
    def encode(self, texts, show_progress_bar=False):
        # Simple bag-of-char hash to 16-d vector for determinism
        def embed(t):
            v = np.zeros(16, dtype=np.float32)
            for ch in t:
                v[ord(ch) % 16] += 1.0
            return v
        if isinstance(texts, (list, tuple)):
            return np.stack([embed(t) for t in texts], axis=0)
        return embed(texts)

def test_retrieval_methods_return_k_results():
    corpus = [
        "def add(a,b): return a+b",
        "def sub(a,b): return a-b",
        "def mul(a,b): return a*b",
        "def div(a,b): return a/b if b else None",
    ]
    retriever = CodeRetriever(model=FakeModel())
    retriever.build_index(corpus)

    q = "def add(x,y): return x+y"
    methods = ["faiss"]
    try:
        import rank_bm25  # noqa: F401
        methods.extend(["bm25", "hybrid"])
    except Exception:
        pass

    for method in methods:
        matches = retriever.retrieve_similar_code(q, k=2, method=method, alpha=0.5)
        assert len(matches) == 2
        assert any("add" in m for m in matches)
