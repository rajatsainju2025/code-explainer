# Model Card: <Model Name>

- Base model: <hf id>
- Finetuning data: <dataset source>
- Task: Code explanation
- License: <license>

## Intended Use

- Educational explanations, onboarding docs, code reviews support.

## Metrics

- BLEU-4: 
- ROUGE-L: 
- BERTScore: 
- CodeBLEU: 

## Limitations

- May hallucinate if code is ambiguous or incomplete.
- Retrieval coverage affects grounding quality.

## Safety

- Security redaction enabled by default; avoid sensitive inputs.

## Reproducibility

- Config: `configs/<model>.yaml`
- Seed: 1337
- Hardware: <gpu/cpu>