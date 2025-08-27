# Quickstart

- Install: `pip install -e .` (use provided requirements)
- Serve API: `uvicorn code_explainer.api.server:app --host 0.0.0.0 --port 8000`
- Build Index: `cx build-index --config configs/default.yaml --output-path data/code_retrieval_index.faiss`
- Retrieve: POST /retrieve with code and index_path

See API page for payloads and environment variables.
