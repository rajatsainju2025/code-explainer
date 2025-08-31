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

## 1. Installation & Setup

### Quick Start
```bash
# Clone and setup
git clone https://github.com/rajatsainju2025/code-explainer.git
cd code-explainer

# Create environment
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate

# Install with all evaluation dependencies
pip install -e ".[dev,eval,judge]"

# Verify installation
code-explainer --version
code-explainer eval --help
```

### Environment Configuration
```bash
# Set up API keys for LLM judges (optional)
export OPENAI_API_KEY="your-key-here"
export ANTHROPIC_API_KEY="your-key-here"

# Configure evaluation cache
export CODE_EXPLAINER_CACHE_DIR="./eval_cache"
export CODE_EXPLAINER_JUDGE_MODEL="gpt-4-turbo"
```

## 2. Basic Evaluation

### Standard Metrics Evaluation
```bash
# Evaluate on HumanEval with basic metrics
code-explainer eval \
  --dataset humaneval \
  --max-samples 50 \
  --config configs/default.yaml \
  --output results/basic_eval.jsonl \
  --report results/basic_report.md

# JSONL evaluation with provenance
code-explainer eval \
  --test-file data/examples/tiny_eval.jsonl \
  --self-consistency 3 \
  --config configs/default.yaml \
  --output results/provenance_eval.jsonl
```

### Multi-Model Comparison
```bash
# Compare multiple strategies
code-explainer eval \
  --dataset humaneval \
  --strategies vanilla ast_augmented enhanced_rag \
  --max-samples 25 \
  --output results/strategy_comparison.jsonl \
  --report results/strategy_report.md
```

## 3. LLM-as-a-Judge Evaluation

### Single Judge Evaluation
```bash
# Use GPT-4 as judge for explanation quality
code-explainer judge-eval \
  --test-file results/basic_eval.jsonl \
  --judge-model gpt-4-turbo \
  --criteria accuracy clarity completeness \
  --output results/judge_eval.jsonl \
  --report results/judge_report.md
```

### Multi-Judge Consensus
```bash
# Use multiple judges for reliability
code-explainer judge-eval \
  --test-file results/basic_eval.jsonl \
  --judge-models gpt-4-turbo claude-3-sonnet gemini-pro \
  --criteria accuracy clarity completeness \
  --consensus-method majority \
  --output results/multi_judge.jsonl
```

### Constitutional AI Evaluation
```bash
# Evaluate against specific principles
code-explainer constitutional-eval \
  --test-file results/basic_eval.jsonl \
  --constitution configs/explanation_constitution.yaml \
  --output results/constitutional.jsonl
```

## 4. Preference-Based Evaluation

### Pairwise Comparison
```bash
# Generate pairwise preferences
code-explainer preference-eval \
  --predictions-a results/vanilla_preds.jsonl \
  --predictions-b results/enhanced_rag_preds.jsonl \
  --judge-model gpt-4-turbo \
  --criteria overall_quality \
  --output results/pairwise_prefs.jsonl
```

### Bradley-Terry Ranking
```bash
# Rank multiple systems using Bradley-Terry model
code-explainer ranking-eval \
  --predictions-files results/vanilla_preds.jsonl results/ast_preds.jsonl results/rag_preds.jsonl \
  --judge-model gpt-4-turbo \
  --output results/bt_ranking.json \
  --report results/ranking_report.md
```

### Best-of-N Analysis
```bash
# Analyze best-of-N performance
code-explainer best-of-n \
  --test-file data/examples/tiny_eval.jsonl \
  --n-values 1 3 5 10 \
  --strategy enhanced_rag \
  --output results/best_of_n.json
```

## 5. Multi-Dimensional Assessment

### Rubric-Based Evaluation
```bash
# Use detailed rubric
code-explainer rubric-eval \
  --test-file results/basic_eval.jsonl \
  --rubric configs/explanation_rubric.yaml \
  --judge-model gpt-4-turbo \
  --output results/rubric_scores.jsonl \
  --report results/rubric_report.md
```

### Aspect-Specific Analysis
```bash
# Evaluate specific aspects
code-explainer aspect-eval \
  --test-file results/basic_eval.jsonl \
  --aspects technical_accuracy readability completeness pedagogical_value \
  --judge-model gpt-4-turbo \
  --output results/aspect_scores.jsonl
```

### Human-AI Agreement Analysis
```bash
# Compare human and AI judgments
code-explainer agreement-analysis \
  --human-scores data/human_evaluations.csv \
  --ai-scores results/judge_eval.jsonl \
  --output results/agreement_analysis.json \
  --report results/agreement_report.md
```

## 6. Robustness & Adversarial Testing

### Counterfactual Evaluation
```bash
# Test on semantically equivalent code variations
code-explainer counterfactual-eval \
  --test-file data/examples/tiny_eval.jsonl \
  --mutation-types variable_rename function_rename comment_change \
  --num-mutations 3 \
  --output results/counterfactual.jsonl
```

### Adversarial Prompt Testing
```bash
# Test robustness to adversarial prompts
code-explainer adversarial-eval \
  --test-file data/examples/tiny_eval.jsonl \
  --attack-types prompt_injection jailbreak misleading_context \
  --output results/adversarial.jsonl \
  --report results/adversarial_report.md
```

### Stress Testing
```bash
# Test on edge cases and extreme inputs
code-explainer stress-test \
  --test-file data/stress_test_cases.jsonl \
  --config configs/stress_test.yaml \
  --output results/stress_test.jsonl
```

## 7. Contamination Detection

### Data Contamination Analysis
```bash
# Detect potential data contamination
code-explainer detect-contamination \
  --eval-file data/examples/tiny_eval.jsonl \
  --train-data data/training_data.jsonl \
  --similarity-threshold 0.8 \
  --output results/contamination_report.json
```

### N-gram Overlap Analysis
```bash
# Analyze n-gram overlap between train/test
code-explainer ngram-contamination \
  --eval-file data/examples/tiny_eval.jsonl \
  --train-data data/training_data.jsonl \
  --ngram-sizes 4 8 16 \
  --output results/ngram_contamination.json
```

### Embedding-Based Detection
```bash
# Use embeddings to detect similar examples
code-explainer embedding-contamination \
  --eval-file data/examples/tiny_eval.jsonl \
  --train-data data/training_data.jsonl \
  --model sentence-transformers/all-MiniLM-L6-v2 \
  --threshold 0.9 \
  --output results/embedding_contamination.json
```

## 8. Advanced Metrics & Analysis

### Instruction Following Evaluation
```bash
# Evaluate instruction following capability
code-explainer instruction-eval \
  --test-file data/instruction_following.jsonl \
  --judge-model gpt-4-turbo \
  --output results/instruction_eval.jsonl
```

### Factual Consistency Check
```bash
# Check factual consistency of explanations
code-explainer factual-eval \
  --test-file results/basic_eval.jsonl \
  --fact-checker configs/fact_checker.yaml \
  --output results/factual_consistency.jsonl
```

### Semantic Coherence Analysis
```bash
# Analyze semantic coherence
code-explainer coherence-eval \
  --test-file results/basic_eval.jsonl \
  --coherence-model sentence-transformers/all-MiniLM-L6-v2 \
  --output results/coherence_scores.json
```

### Diversity & Coverage Metrics
```bash
# Analyze explanation diversity
code-explainer diversity-eval \
  --test-file results/basic_eval.jsonl \
  --diversity-metrics lexical semantic structural \
  --output results/diversity_analysis.json
```

## 9. Reproducibility & Artifacts

### Experiment Tracking
```bash
# Run evaluation with full artifact saving
code-explainer eval \
  --dataset humaneval \
  --max-samples 100 \
  --config configs/default.yaml \
  --save-artifacts \
  --experiment-name "enhanced_rag_v1.0" \
  --output results/experiments/enhanced_rag_v1.0/
```

### Reproducibility Report
```bash
# Generate reproducibility report
code-explainer repro-report \
  --experiment-dir results/experiments/enhanced_rag_v1.0/ \
  --output results/reproducibility_report.md
```

### Evaluation Card Generation
```bash
# Generate evaluation card
code-explainer eval-card \
  --experiment-dir results/experiments/enhanced_rag_v1.0/ \
  --output results/evaluation_card.md
```

## 10. Production Monitoring

### Continuous Evaluation
```bash
# Set up continuous evaluation
code-explainer monitor \
  --eval-schedule "0 2 * * *"  # Daily at 2 AM \
  --test-file data/monitor_set.jsonl \
  --alert-thresholds configs/alert_thresholds.yaml \
  --output-dir results/monitoring/
```

### Performance Dashboard
```bash
# Start evaluation dashboard
code-explainer dashboard \
  --results-dir results/ \
  --port 8080 \
  --refresh-interval 300
```

### A/B Testing Framework
```bash
# Run A/B test between two models
code-explainer ab-test \
  --model-a configs/model_a.yaml \
  --model-b configs/model_b.yaml \
  --test-file data/ab_test_set.jsonl \
  --significance-level 0.05 \
  --output results/ab_test_results.json
```

## Best Practices & Tips

### 1. Evaluation Design
- **Stratified Sampling**: Ensure balanced test sets across difficulty levels
- **Multiple Metrics**: Use complementary metrics (lexical + semantic + human)
- **Statistical Power**: Calculate required sample sizes for significance testing
- **Blind Evaluation**: Use anonymized predictions when possible

### 2. Judge Selection
- **Model Diversity**: Use judges from different model families
- **Calibration**: Regularly calibrate judges against human evaluations
- **Bias Detection**: Monitor for systematic biases in judge outputs
- **Cost Optimization**: Balance judge quality with evaluation cost

### 3. Reproducibility
- **Version Control**: Pin all dependency versions
- **Seed Management**: Set and document random seeds
- **Environment Capture**: Save complete environment specifications
- **Data Lineage**: Track data transformations and filtering

### 4. Interpretation
- **Effect Sizes**: Report practical significance, not just statistical
- **Confidence Intervals**: Provide uncertainty estimates
- **Failure Analysis**: Analyze failure modes and edge cases
- **Generalization**: Test on out-of-distribution examples

### 5. Ethical Considerations
- **Bias Testing**: Evaluate for demographic and cultural biases
- **Fairness Metrics**: Use appropriate fairness definitions
- **Privacy**: Ensure evaluation data doesn't leak sensitive information
- **Transparency**: Document evaluation procedures and limitations

## Advanced Configuration Examples

### Judge Configuration (configs/judge_config.yaml)
```yaml
judges:
  gpt-4-turbo:
    model: "gpt-4-turbo"
    temperature: 0.0
    max_tokens: 1000
    system_prompt: "You are an expert code reviewer..."
  
  claude-3-sonnet:
    model: "claude-3-sonnet"
    temperature: 0.0
    max_tokens: 1000
    system_prompt: "You are an expert code reviewer..."

evaluation:
  criteria:
    accuracy:
      weight: 0.4
      description: "Technical correctness of the explanation"
    clarity:
      weight: 0.3
      description: "Clarity and readability"
    completeness:
      weight: 0.3
      description: "Coverage of important aspects"
```

### Rubric Configuration (configs/explanation_rubric.yaml)
```yaml
rubric:
  technical_accuracy:
    scale: [1, 2, 3, 4, 5]
    descriptions:
      1: "Contains major technical errors"
      2: "Contains minor technical errors"
      3: "Mostly accurate with some ambiguities"
      4: "Accurate with minor omissions"
      5: "Completely accurate and precise"
  
  pedagogical_value:
    scale: [1, 2, 3, 4, 5]
    descriptions:
      1: "Not helpful for learning"
      2: "Somewhat helpful for learning"
      3: "Moderately helpful for learning"
      4: "Very helpful for learning"
      5: "Exceptionally helpful for learning"
```

## Troubleshooting

### Common Issues
1. **Judge API Limits**: Use rate limiting and retry logic
2. **Memory Issues**: Process evaluations in batches
3. **Reproducibility**: Ensure deterministic evaluation order
4. **Cost Management**: Monitor API usage and set budgets

### Performance Optimization
1. **Caching**: Cache judge responses for identical inputs
2. **Parallelization**: Run independent evaluations in parallel
3. **Batching**: Batch API requests when possible
4. **Sampling**: Use representative samples for large datasets

## Next Steps

1. **Custom Evaluations**: Implement domain-specific evaluation metrics
2. **Human-in-the-Loop**: Set up human evaluation workflows
3. **Continual Learning**: Use evaluation feedback for model improvement
4. **Benchmark Submission**: Prepare results for community benchmarks

For more advanced usage and customization, see:
- [Evaluation API Reference](../docs/api/evaluation.md)
- [Custom Judge Implementation](../docs/guides/custom_judges.md)
- [Benchmark Submission Guide](../docs/guides/benchmark_submission.md)
