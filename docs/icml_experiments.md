# ICML Experiments: How to Run

This guide shows how to prepare datasets, run experiments, generate results, and produce paper-ready artifacts.

## 1) Prepare Datasets

```bash
python scripts/prepare_icml_datasets.py --config configs/icml_experiment_full.yaml --output-dir data
```

Outputs:
- data/<dataset>/train.jsonl, val.jsonl, test.jsonl
- data/dataset_summary.json

## 2) Run Full Experimental Pipeline

```bash
python scripts/run_icml_experiments.py --config configs/icml_experiment_full.yaml --phase all
```

Phases:
- datasets: prepare datasets
- experiments: train/evaluate primary + baselines + ablations
- analysis: statistical tests and reports
- outputs: LaTeX tables, plots, and paper sections

## 3) Generate Analysis Only (using saved results)

```bash
python scripts/run_icml_experiments.py --config configs/icml_experiment_full.yaml --phase analysis
```

## 4) Generate Paper Outputs Only

```bash
python scripts/run_icml_experiments.py --config configs/icml_experiment_full.yaml --phase outputs
```

## 5) Where to Find Outputs

All outputs are written under `results/icml_experiment/`:
- main_results_table.tex, ablation_results_table.tex
- learning_curves.pdf, performance_comparison.pdf, error_analysis.pdf
- results_section.md, discussion_section.md
- significance_report.md, validation_report.json

## Notes
- The framework will create dummy results if a baseline fails, to keep the pipeline running
- GPU is optional but recommended; CPU will be slow
- Ensure required Python deps are installed per `requirements.txt`
