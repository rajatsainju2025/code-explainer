# Code Explainer: Advanced Evaluation Tutorial

This comprehensive tutorial demonstrates the state-of-the-art evaluation capabilities of Code Explainer, incorporating the latest research in open evaluations and LLM assessment methodologies.

## Table of Contents

1. [Installation & Setup](#installation--setup)
2. [Basic Evaluation](#basic-evaluation)
3. [LLM-as-a-Judge Evaluation](#llm-as-a-judge-evaluation)
4. [Preference-Based Evaluation](#preference-based-evaluation)
5. [Multi-Dimensional Assessment](#multi-dimensional-assessment)
6. [Robustness & Adversarial Testing](#robustness--adversarial-testing)
7. [Contamination Detection](#contamination-detection)
8. [Advanced Metrics & Analysis](#advanced-metrics--analysis)
9. [Reproducibility & Artifacts](#reproducibility--artifacts)
10. [Production Monitoring](#production-monitoring)

## CLI Commands

The project provides comprehensive CLI commands for all evaluation methods:

```bash
# LLM-as-a-Judge evaluation (requires API keys)
code-explainer eval-llm-judge 
  --test-data test.jsonl 
  --predictions predictions.jsonl 
  --judges gpt-4 claude-3-sonnet 
  --criteria accuracy clarity completeness

# Preference-based evaluation  
code-explainer eval-preference 
  --test-data test.jsonl 
  --predictions-a model_a.jsonl 
  --predictions-b model_b.jsonl 
  --use-bradley-terry

# Contamination detection
code-explainer eval-contamination 
  --train-data train.jsonl 
  --test-data test.jsonl 
  --methods exact ngram substring 
  --include-semantic

# Robustness testing
code-explainer eval-robustness 
  --test-data test.jsonl 
  --model-path ./results 
  --test-types typo case whitespace punctuation 
  --severity-levels 0.05 0.1 0.2

# Traditional metrics evaluation
code-explainer evaluate \
  --test-data test.jsonl \
  --predictions predictions.jsonl \
  --metrics bleu rouge bertscore codebleu
```

## Complete Evaluation Pipeline

Here's a comprehensive Python example demonstrating all evaluation capabilities:

```python
import json
from pathlib import Path

from code_explainer.evaluation.metrics import calculate_all_metrics
from code_explainer.evaluation.contamination import run_contamination_detection  
from code_explainer.evaluation.robustness import run_robustness_tests
from code_explainer.model import CodeExplainer

def run_comprehensive_evaluation(
    model_path: str,
    train_data: str,
    test_data: str,
    predictions: str, 
    output_dir: str
):
    """Run complete evaluation pipeline following open-eval best practices."""
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True)
    
    print("🚀 Starting comprehensive evaluation...")
    
    # 1. Traditional metrics
    print("📊 Running traditional metrics...")
    metrics = calculate_all_metrics(
        predictions_file=predictions,
        references_file=test_data
    )
    
    with open(output_dir / "traditional_metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)
    
    # 2. Contamination detection
    print("🔍 Running contamination detection...")
    contamination_report = run_contamination_detection(
        train_file=train_data,
        test_file=test_data,
        output_file=output_dir / "contamination_report.json",
        methods=["exact", "ngram", "substring"]
    )
    
    # 3. Robustness testing
    print("🛡️ Running robustness tests...")
    with open(test_data) as f:
        test_examples = [json.loads(line) for line in f]
    
    explainer = CodeExplainer(model_path=model_path)
    
    def predict_func(example):
        return explainer.explain_code(example['code'])
    
    robustness_report = run_robustness_tests(
        examples=test_examples[:100],  # Limit for demonstration
        predict_func=predict_func,
        output_file=output_dir / "robustness_report.json",
        test_types=["typo", "case", "whitespace", "punctuation"],
        severity_levels=[0.05, 0.1, 0.2],
        random_seed=42
    )
    
    # 4. Generate summary report
    summary = {
        "evaluation_metadata": {
            "timestamp": "2024-01-15T10:30:00Z",
            "model_path": model_path,
            "test_examples": len(test_examples),
            "evaluation_methods": ["traditional", "contamination", "robustness"]
        },
        "traditional_metrics": metrics,
        "contamination": {
            "rate": contamination_report.contamination_rate,
            "detected_examples": len(contamination_report.contaminated_examples),
            "methods_used": contamination_report.detection_methods
        },
        "robustness": {
            "overall_score": robustness_report.overall_robustness_score,
            "test_summaries": robustness_report.test_summaries,
            "total_tests": robustness_report.total_tests
        },
        "recommendations": generate_recommendations(metrics, contamination_report, robustness_report)
    }
    
    with open(output_dir / "evaluation_summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    
    print(f"✅ Evaluation complete! Results saved to {output_dir}")
    print_summary_table(summary)
    return summary

def generate_recommendations(metrics, contamination_report, robustness_report):
    """Generate actionable recommendations based on evaluation results."""
    recommendations = []
    
    # Traditional metrics recommendations
    if metrics.get("bleu", 0) < 0.3:
        recommendations.append("Consider improving model training data quality or training duration")
    
    # Contamination recommendations  
    if contamination_report.contamination_rate > 0.05:
        recommendations.append("⚠️ High contamination detected - review data preparation pipeline")
    
    # Robustness recommendations
    if robustness_report.overall_robustness_score < 0.7:
        recommendations.append("Consider robustness training or data augmentation")
    
    return recommendations

def print_summary_table(summary):
    """Print a nice summary table of results."""
    from rich.console import Console
    from rich.table import Table
    
    console = Console()
    table = Table(title="Evaluation Summary")
    
    table.add_column("Metric", style="cyan")
    table.add_column("Score", style="magenta") 
    table.add_column("Status", style="green")
    
    # Traditional metrics
    bleu_score = summary["traditional_metrics"].get("bleu", 0)
    table.add_row("BLEU", f"{bleu_score:.3f}", "✅" if bleu_score > 0.3 else "⚠️")
    
    # Contamination
    contam_rate = summary["contamination"]["rate"]
    table.add_row("Contamination Rate", f"{contam_rate:.1%}", "✅" if contam_rate < 0.05 else "⚠️")
    
    # Robustness
    robust_score = summary["robustness"]["overall_score"]
    table.add_row("Robustness", f"{robust_score:.3f}", "✅" if robust_score > 0.7 else "⚠️")
    
    console.print(table)

# Example usage
if __name__ == "__main__":
    summary = run_comprehensive_evaluation(
        model_path="./results", 
        train_data="data/train.jsonl",
        test_data="data/test.jsonl",
        predictions="predictions.jsonl",
        output_dir="evaluation_results"
    )
```

## Best Practices for Open Evaluation

1. **Reproducibility**: Always set random seeds and document versions
2. **Contamination Checking**: Run before any evaluation
3. **Multi-Method Validation**: Use traditional + judge-based + preference metrics
4. **Robustness Testing**: Essential for production readiness
5. **Documentation**: Save all evaluation artifacts and metadata
6. **Version Control**: Track evaluation configs and results
7. **Confidence Intervals**: Report uncertainty in metrics
8. **Human Validation**: Include human evaluation for critical applications

## Integration with Popular Frameworks

The evaluation system integrates with common ML workflows:

```python
# Weights & Biases integration
import wandb

wandb.init(project="code-explainer-eval")
wandb.log(summary["traditional_metrics"])
wandb.log({"contamination_rate": summary["contamination"]["rate"]})
wandb.log({"robustness_score": summary["robustness"]["overall_score"]})

# MLflow integration  
import mlflow

with mlflow.start_run():
    mlflow.log_metrics(summary["traditional_metrics"])
    mlflow.log_artifact("evaluation_results/evaluation_summary.json")
```

This completes the advanced evaluation tutorial. For questions or contributions, see our [GitHub repository](https://github.com/rajatsainju2025/code-explainer).

