# Provenance Cards

Use provenance cards to review attribution quality for RAG explanations.

Generate cards from predictions:
```bash
python scripts/provenance_card.py --preds out/preds.jsonl --out out/cards/
```

What’s shown
- Code, prediction, optional reference
- Source IDs (if provided)

Planned
- Highlight cited spans and compute coverage metrics.
