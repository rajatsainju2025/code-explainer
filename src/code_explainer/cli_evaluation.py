#!/usr/bin/env python3
"""Command-line interface for running research-driven code explanation evaluation."""

import argparse
import asyncio
import json
import logging
import sys
from pathlib import Path
from typing import Optional, Dict, Any

from .research_evaluation_orchestrator import (
    ResearchEvaluationOrchestrator,
    ResearchEvaluationConfig
)

logger = logging.getLogger(__name__)


def setup_logging(verbose: bool = False) -> None:
    """Setup logging configuration.
    
    Args:
        verbose: Enable verbose logging
    """
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler('evaluation.log')
        ]
    )


def load_model_function(model_path: str) -> Any:
    """Load model function from path.
    
    Args:
        model_path: Path to model or model configuration
        
    Returns:
        Model function
    """
    # Mock implementation - in practice would load actual model
    def mock_model(prompt: str) -> str:
        return f"Mock response to: {prompt[:100]}..."
    
    logger.info(f"Loaded mock model from {model_path}")
    return mock_model


def load_test_prompts(prompts_file: Optional[str]) -> Optional[list]:
    """Load test prompts from file.
    
    Args:
        prompts_file: Path to prompts file
        
    Returns:
        List of prompts or None
    """
    if not prompts_file:
        return None
    
    try:
        with open(prompts_file) as f:
            if prompts_file.endswith('.json'):
                data = json.load(f)
                if isinstance(data, list):
                    return data
                return data.get('prompts', [])
            else:
                # Assume text file with one prompt per line
                return [line.strip() for line in f if line.strip()]
    except Exception as e:
        logger.error(f"Failed to load prompts from {prompts_file}: {e}")
        return None


async def run_evaluation(args: argparse.Namespace) -> None:
    """Run comprehensive evaluation.
    
    Args:
        args: Command line arguments
    """
    # Setup configuration
    config = ResearchEvaluationConfig(
        training_corpus_path=args.corpus_path,
        dynamic_evaluation_rounds=args.dynamic_rounds,
        enable_multi_agent=args.enable_multi_agent,
        adversarial_test_count=args.adversarial_tests,
        parallel_execution=args.parallel,
        output_dir=args.output_dir,
        detailed_logging=args.verbose
    )
    
    # Load model
    model_fn = load_model_function(args.model_path)
    
    # Load test prompts
    test_prompts = load_test_prompts(args.prompts_file)
    
    # Create orchestrator
    orchestrator = ResearchEvaluationOrchestrator(config)
    
    logger.info(f"Starting evaluation of model: {args.model_identifier}")
    
    # Run evaluation
    try:
        result = await orchestrator.evaluate_model(
            model_fn=model_fn,
            model_identifier=args.model_identifier,
            test_prompts=test_prompts
        )
        
        # Print summary
        print("\n" + "="*80)
        print("RESEARCH EVALUATION SUMMARY")
        print("="*80)
        print(f"Model: {result.model_identifier}")
        print(f"Evaluation ID: {result.evaluation_id}")
        print(f"Timestamp: {result.timestamp}")
        print(f"Execution Time: {result.execution_time:.2f} seconds")
        print()
        
        print("SCORES:")
        print(f"  Overall Score:      {result.overall_score:.3f}")
        print(f"  Reliability Score:  {result.reliability_score:.3f}")
        print(f"  Safety Score:       {result.safety_score:.3f}")
        print(f"  Collaboration Score: {result.collaboration_score:.3f}")
        print()
        
        print(f"DEPLOYMENT READINESS: {result.deployment_readiness}")
        print()
        
        if result.improvement_areas:
            print("IMPROVEMENT AREAS:")
            for area in result.improvement_areas:
                print(f"  • {area}")
            print()
        
        if result.risk_factors:
            print("RISK FACTORS:")
            for risk in result.risk_factors:
                print(f"  ⚠ {risk}")
            print()
        
        print("TEST COUNTS:")
        for test_type, count in result.test_counts.items():
            print(f"  {test_type}: {count}")
        print()
        
        # Component summaries
        print("COMPONENT RESULTS:")
        
        # Contamination Detection
        contamination = result.contamination_results
        if contamination:
            print(f"  Contamination Detection:")
            print(f"    Rate: {contamination.get('contamination_rate', 0):.2%}")
            print(f"    Samples: {contamination.get('contaminated_samples', 0)}/{contamination.get('total_samples', 0)}")
        
        # Dynamic Evaluation
        dynamic = result.dynamic_evaluation_results
        if dynamic and 'summary' in dynamic:
            stats = dynamic['summary'].get('overall_statistics', {})
            print(f"  Dynamic Evaluation:")
            print(f"    Mean Score: {stats.get('mean_score', 0):.3f}")
            print(f"    Std Dev: {stats.get('std_score', 0):.3f}")
            print(f"    Response Time: {stats.get('mean_execution_time', 0):.2f}s")
        
        # Multi-Agent Evaluation
        multi_agent = result.multi_agent_results
        if multi_agent and not multi_agent.get('error'):
            print(f"  Multi-Agent Evaluation:")
            print(f"    Average Score: {multi_agent.get('average_score', 0):.3f}")
            print(f"    Collaboration Quality: {multi_agent.get('average_collaboration_quality', 0):.3f}")
        
        # Adversarial Testing
        adversarial = result.adversarial_results
        if adversarial:
            print(f"  Adversarial Testing:")
            print(f"    Vulnerability Rate: {adversarial.get('overall_vulnerability_rate', 0):.2%}")
            print(f"    Critical Vulnerabilities: {adversarial.get('critical_vulnerabilities', 0)}")
            print(f"    Safety Assessment: {adversarial.get('safety_assessment', 'UNKNOWN')}")
        
        print("="*80)
        
        # Additional detailed output if requested
        if args.detailed_output:
            print("\nDETAILED RESULTS:")
            print(json.dumps({
                'contamination': contamination,
                'dynamic': dynamic,
                'multi_agent': multi_agent,
                'adversarial': adversarial
            }, indent=2, default=str))
        
    except Exception as e:
        logger.error(f"Evaluation failed: {e}")
        print(f"ERROR: Evaluation failed - {e}")
        sys.exit(1)


def main() -> None:
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Research-driven evaluation for code explanation models",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic evaluation
  python -m code_explainer.cli_evaluation --model-path ./model --model-id "my-model-v1"
  
  # Full evaluation with custom settings
  python -m code_explainer.cli_evaluation \\
    --model-path ./model \\
    --model-id "my-model-v1" \\
    --prompts-file test_prompts.json \\
    --corpus-path training_corpus.jsonl \\
    --dynamic-rounds 10 \\
    --adversarial-tests 50 \\
    --enable-multi-agent \\
    --parallel \\
    --verbose
        """
    )
    
    # Model configuration
    parser.add_argument(
        '--model-path', 
        required=True,
        help='Path to model or model configuration'
    )
    parser.add_argument(
        '--model-id', 
        required=True,
        help='Identifier for the model being evaluated'
    )
    
    # Test configuration
    parser.add_argument(
        '--prompts-file',
        help='Path to file containing test prompts (JSON or text)'
    )
    parser.add_argument(
        '--corpus-path',
        help='Path to training corpus for contamination detection'
    )
    
    # Evaluation settings
    parser.add_argument(
        '--dynamic-rounds',
        type=int,
        default=5,
        help='Number of dynamic evaluation rounds (default: 5)'
    )
    parser.add_argument(
        '--adversarial-tests',
        type=int,
        default=25,
        help='Number of adversarial tests to run (default: 25)'
    )
    parser.add_argument(
        '--enable-multi-agent',
        action='store_true',
        help='Enable multi-agent evaluation (slower but more comprehensive)'
    )
    parser.add_argument(
        '--parallel',
        action='store_true',
        help='Run evaluation components in parallel (faster)'
    )
    
    # Output configuration
    parser.add_argument(
        '--output-dir',
        default='evaluation_results',
        help='Directory to save results (default: evaluation_results)'
    )
    parser.add_argument(
        '--detailed-output',
        action='store_true',
        help='Print detailed results to console'
    )
    parser.add_argument(
        '--verbose', '-v',
        action='store_true',
        help='Enable verbose logging'
    )
    
    # Parse arguments
    args = parser.parse_args()
    
    # Setup logging
    setup_logging(args.verbose)
    
    # Validate arguments
    if not Path(args.model_path).exists():
        print(f"ERROR: Model path does not exist: {args.model_path}")
        sys.exit(1)
    
    if args.prompts_file and not Path(args.prompts_file).exists():
        print(f"ERROR: Prompts file does not exist: {args.prompts_file}")
        sys.exit(1)
    
    if args.corpus_path and not Path(args.corpus_path).exists():
        print(f"ERROR: Corpus path does not exist: {args.corpus_path}")
        sys.exit(1)
    
    # Run evaluation
    try:
        asyncio.run(run_evaluation(args))
    except KeyboardInterrupt:
        print("\nEvaluation interrupted by user")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Unexpected error: {e}")
        print(f"ERROR: {e}")
        sys.exit(1)


if __name__ == '__main__':
    main()
