#!/usr/bin/env python3
"""
Command-line interface for the evaluation system.

Usage:
    python -m evals.cli run --config configs/default.yaml
    python -m evals.cli ablation --config configs/default.yaml --components retrieval reranking
    python -m evals.cli compare --configs configs/strategy_a.yaml configs/strategy_b.yaml
"""

import argparse
import sys
from pathlib import Path
from typing import List, Optional, Dict, Any

from .config import load_config, EvalConfig
from .runner import EvalRunner


def run_evaluation(config_path: str, overrides: Optional[Dict[str, Any]] = None) -> None:
    """Run a single evaluation."""
    try:
        config = load_config(config_path, overrides)
        runner = EvalRunner(config)
        results = runner.run_evaluation()
        
        print(f"\n✅ Evaluation completed: {config.name}")
        print(f"📊 BLEU Score: {results.bleu_score:.4f}")
        print(f"📊 ROUGE-L: {results.rouge_l:.4f}")
        print(f"⚡ Avg Latency: {results.avg_latency:.3f}s")
        print(f"💰 Total Cost: ${results.total_cost:.4f}")
        print(f"📁 Results saved to: {config.output_dir}")
        
    except Exception as e:
        print(f"❌ Evaluation failed: {e}")
        sys.exit(1)


def run_ablation_study(config_path: str, components: List[str]) -> None:
    """Run ablation study."""
    try:
        config = load_config(config_path)
        runner = EvalRunner(config)
        results = runner.run_ablation_study(components)
        
        print(f"\n✅ Ablation study completed: {len(results)} configurations")
        
        baseline = results.get('baseline')
        if baseline:
            print(f"\n📊 Baseline Results:")
            print(f"   BLEU Score: {baseline.bleu_score:.4f}")
            print(f"   Avg Latency: {baseline.avg_latency:.3f}s")
        
        print(f"\n📈 Ablation Results:")
        for name, result in results.items():
            if name != 'baseline':
                delta_bleu = result.bleu_score - baseline.bleu_score if baseline else 0
                delta_latency = result.avg_latency - baseline.avg_latency if baseline else 0
                print(f"   {name}: BLEU Δ{delta_bleu:+.4f}, Latency Δ{delta_latency:+.3f}s")
        
    except Exception as e:
        print(f"❌ Ablation study failed: {e}")
        sys.exit(1)


def compare_strategies(config_paths: List[str]) -> None:
    """Compare multiple strategies."""
    try:
        configs = [load_config(path) for path in config_paths]
        runner = EvalRunner(configs[0])  # Use first config as base
        results = runner.compare_strategies(configs)
        
        print(f"\n✅ Strategy comparison completed: {len(results)} strategies")
        print(f"\n📊 Results (sorted by BLEU score):")
        
        sorted_results = sorted(results.items(), key=lambda x: x[1].bleu_score, reverse=True)
        for name, result in sorted_results:
            print(f"   {name}: BLEU {result.bleu_score:.4f}, "
                  f"Latency {result.avg_latency:.3f}s, Cost ${result.total_cost:.4f}")
        
    except Exception as e:
        print(f"❌ Strategy comparison failed: {e}")
        sys.exit(1)


def main():
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(description="Code Explainer Evaluation System")
    subparsers = parser.add_subparsers(dest='command', help='Available commands')
    
    # Run evaluation command
    run_parser = subparsers.add_parser('run', help='Run a single evaluation')
    run_parser.add_argument('--config', required=True, help='Configuration file path')
    run_parser.add_argument('--output-dir', help='Override output directory')
    run_parser.add_argument('--seed', type=int, help='Override random seed')
    run_parser.add_argument('--max-samples', type=int, help='Override max samples')
    
    # Ablation study command
    ablation_parser = subparsers.add_parser('ablation', help='Run ablation study')
    ablation_parser.add_argument('--config', required=True, help='Configuration file path')
    ablation_parser.add_argument('--components', nargs='+', required=True, 
                                help='Components to ablate (e.g., retrieval reranking)')
    
    # Strategy comparison command
    compare_parser = subparsers.add_parser('compare', help='Compare multiple strategies')
    compare_parser.add_argument('--configs', nargs='+', required=True,
                               help='Configuration file paths to compare')
    
    # Benchmark command
    benchmark_parser = subparsers.add_parser('benchmark', help='Run standard benchmarks')
    benchmark_parser.add_argument('--suite', choices=['minimal', 'standard', 'comprehensive'],
                                 default='standard', help='Benchmark suite to run')
    
    args = parser.parse_args()
    
    if not args.command:
        parser.print_help()
        return
    
    if args.command == 'run':
        overrides = {}
        if args.output_dir:
            overrides['output_dir'] = args.output_dir
        if args.seed is not None:
            overrides['seed'] = args.seed
        if args.max_samples is not None:
            overrides['dataset'] = {'max_samples': args.max_samples}
            
        run_evaluation(args.config, overrides)
        
    elif args.command == 'ablation':
        run_ablation_study(args.config, args.components)
        
    elif args.command == 'compare':
        compare_strategies(args.configs)
        
    elif args.command == 'benchmark':
        print(f"🚀 Running {args.suite} benchmark suite...")
        # This would run predefined benchmark configurations
        benchmark_configs = {
            'minimal': ['configs/minimal.yaml'],
            'standard': ['configs/codet5-base.yaml', 'configs/enhanced.yaml'],
            'comprehensive': ['configs/codet5-base.yaml', 'configs/enhanced.yaml', 
                            'configs/codellama-instruct.yaml']
        }
        
        configs = benchmark_configs.get(args.suite, [])
        if configs:
            compare_strategies(configs)
        else:
            print(f"❌ No configurations found for {args.suite} benchmark")


if __name__ == '__main__':
    main()
