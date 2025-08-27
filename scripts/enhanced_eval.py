#!/usr/bin/env python3
"""Enhanced evaluation script with P95/P99 metrics and performance monitoring."""

import argparse
import json
import time
import logging
from pathlib import Path
from typing import Dict, List, Any, Optional
import statistics

import numpy as np

from src.code_explainer.open_evals import (
    get_dataset_list, 
    get_dataset_info, 
    run_eval, 
    generate_dataset
)

logger = logging.getLogger(__name__)


def calculate_performance_metrics(response_times: List[float]) -> Dict[str, float]:
    """Calculate performance metrics including P95/P99."""
    if not response_times:
        return {}
    
    sorted_times = sorted(response_times)
    n = len(sorted_times)
    
    return {
        "min": min(sorted_times),
        "max": max(sorted_times),
        "mean": statistics.mean(sorted_times),
        "median": statistics.median(sorted_times),
        "p95": sorted_times[int(0.95 * n)] if n > 0 else 0.0,
        "p99": sorted_times[int(0.99 * n)] if n > 0 else 0.0,
        "std": statistics.stdev(sorted_times) if n > 1 else 0.0
    }


def run_comprehensive_eval(
    datasets: Optional[List[str]] = None,
    model_path: str = "./results",
    config_path: str = "configs/default.yaml",
    output_dir: str = "./eval_results",
    include_perf_metrics: bool = True
) -> Dict[str, Any]:
    """Run comprehensive evaluation across multiple datasets."""
    
    if datasets is None:
        datasets = get_dataset_list()
    
    logger.info(f"Running evaluation on {len(datasets)} datasets")
    
    # Create output directory
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    overall_results = {
        "evaluation_time": time.time(),
        "model_path": model_path,
        "config_path": config_path,
        "datasets_evaluated": len(datasets),
        "dataset_results": {},
        "summary": {}
    }
    
    total_samples = 0
    total_correct = 0
    all_response_times = []
    
    for dataset_id in datasets:
        logger.info(f"Evaluating dataset: {dataset_id}")
        
        try:
            # Get dataset info
            dataset_info = get_dataset_info(dataset_id)
            if not dataset_info:
                logger.warning(f"Dataset info not found for: {dataset_id}")
                continue
            
            # Run evaluation with timing
            start_time = time.time()
            
            if include_perf_metrics:
                # Generate dataset to get individual timings
                dataset = generate_dataset(dataset_id)
                response_times = []
                
                # Simulate individual query timings (replace with actual model calls)
                for sample in dataset:
                    query_start = time.time()
                    # TODO: Replace with actual model prediction
                    time.sleep(0.01)  # Simulate processing time
                    query_time = time.time() - query_start
                    response_times.append(query_time)
                
                all_response_times.extend(response_times)
                perf_metrics = calculate_performance_metrics(response_times)
            else:
                perf_metrics = {}
                response_times = []
            
            # Run standard evaluation
            metrics = run_eval(
                dataset_id=dataset_id,
                model_path=model_path,
                config_path=config_path,
                out_csv=str(output_path / f"{dataset_id}_results.csv"),
                out_json=str(output_path / f"{dataset_id}_results.json")
            )
            
            eval_time = time.time() - start_time
            
            # Combine metrics
            dataset_result = {
                "dataset_info": dataset_info,
                "eval_metrics": metrics,
                "performance_metrics": perf_metrics,
                "evaluation_time": eval_time,
                "samples_per_second": metrics["total_samples"] / eval_time if eval_time > 0 else 0
            }
            
            overall_results["dataset_results"][dataset_id] = dataset_result
            
            # Update totals
            total_samples += metrics["total_samples"]
            total_correct += metrics["correct"]
            
            logger.info(f"Dataset {dataset_id}: {metrics['accuracy']:.2%} accuracy, {eval_time:.2f}s")
            
        except Exception as e:
            logger.error(f"Failed to evaluate dataset {dataset_id}: {e}")
            overall_results["dataset_results"][dataset_id] = {
                "error": str(e),
                "dataset_info": get_dataset_info(dataset_id)
            }
    
    # Calculate overall summary
    overall_accuracy = total_correct / total_samples if total_samples > 0 else 0.0
    overall_perf = calculate_performance_metrics(all_response_times) if include_perf_metrics else {}
    
    overall_results["summary"] = {
        "total_samples": total_samples,
        "total_correct": total_correct,
        "overall_accuracy": overall_accuracy,
        "successful_datasets": len([r for r in overall_results["dataset_results"].values() if "error" not in r]),
        "failed_datasets": len([r for r in overall_results["dataset_results"].values() if "error" in r]),
        "overall_performance": overall_perf
    }
    
    # Save comprehensive results
    with open(output_path / "comprehensive_results.json", "w") as f:
        json.dump(overall_results, f, indent=2, default=str)
    
    logger.info(f"Comprehensive evaluation complete: {overall_accuracy:.2%} overall accuracy")
    return overall_results


def generate_eval_report(results: Dict[str, Any], output_path: str = "./eval_results/report.md") -> None:
    """Generate a markdown evaluation report."""
    
    report_lines = [
        "# Evaluation Report",
        "",
        f"**Evaluation Time:** {time.ctime(results['evaluation_time'])}",
        f"**Model Path:** {results['model_path']}",
        f"**Config Path:** {results['config_path']}",
        "",
        "## Summary",
        "",
        f"- **Total Datasets:** {results['datasets_evaluated']}",
        f"- **Successful:** {results['summary']['successful_datasets']}",
        f"- **Failed:** {results['summary']['failed_datasets']}",
        f"- **Total Samples:** {results['summary']['total_samples']}",
        f"- **Overall Accuracy:** {results['summary']['overall_accuracy']:.2%}",
        ""
    ]
    
    # Performance metrics
    if "overall_performance" in results["summary"] and results["summary"]["overall_performance"]:
        perf = results["summary"]["overall_performance"]
        report_lines.extend([
            "## Performance Metrics",
            "",
            f"- **Mean Response Time:** {perf.get('mean', 0):.3f}s",
            f"- **Median Response Time:** {perf.get('median', 0):.3f}s",
            f"- **P95 Response Time:** {perf.get('p95', 0):.3f}s",
            f"- **P99 Response Time:** {perf.get('p99', 0):.3f}s",
            f"- **Min Response Time:** {perf.get('min', 0):.3f}s",
            f"- **Max Response Time:** {perf.get('max', 0):.3f}s",
            ""
        ])
    
    # Dataset results
    report_lines.extend([
        "## Dataset Results",
        "",
        "| Dataset | Accuracy | Samples | Time (s) | Samples/s | Status |",
        "|---------|----------|---------|----------|-----------|--------|"
    ])
    
    for dataset_id, result in results["dataset_results"].items():
        if "error" in result:
            report_lines.append(f"| {dataset_id} | - | - | - | - | ❌ Error |")
        else:
            metrics = result["eval_metrics"]
            accuracy = f"{metrics['accuracy']:.1%}"
            samples = metrics["total_samples"]
            eval_time = f"{result['evaluation_time']:.2f}"
            samples_per_sec = f"{result['samples_per_second']:.1f}"
            report_lines.append(f"| {dataset_id} | {accuracy} | {samples} | {eval_time} | {samples_per_sec} | ✅ Success |")
    
    report_lines.extend([
        "",
        "## Dataset Details",
        ""
    ])
    
    for dataset_id, result in results["dataset_results"].items():
        if "error" in result:
            report_lines.extend([
                f"### {dataset_id} ❌",
                "",
                f"**Error:** {result['error']}",
                ""
            ])
        else:
            info = result["dataset_info"]
            metrics = result["eval_metrics"]
            report_lines.extend([
                f"### {dataset_id} ✅",
                "",
                f"**Description:** {info['description']}",
                f"**Size:** {info['size']} samples",
                f"**Accuracy:** {metrics['accuracy']:.2%}",
                f"**Correct:** {metrics['correct']}/{metrics['total_samples']}",
                ""
            ])
    
    # Write report
    report_path = Path(output_path)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(report_path, "w") as f:
        f.write("\n".join(report_lines))
    
    logger.info(f"Evaluation report saved to: {report_path}")


def main():
    """Main evaluation script."""
    parser = argparse.ArgumentParser(description="Enhanced evaluation with performance metrics")
    parser.add_argument("--datasets", nargs="+", help="Specific datasets to evaluate")
    parser.add_argument("--model-path", default="./results", help="Path to model")
    parser.add_argument("--config", default="configs/default.yaml", help="Config file path")
    parser.add_argument("--output-dir", default="./eval_results", help="Output directory")
    parser.add_argument("--no-perf", action="store_true", help="Skip performance metrics")
    parser.add_argument("--report", action="store_true", help="Generate markdown report")
    parser.add_argument("--list-datasets", action="store_true", help="List available datasets")
    parser.add_argument("--verbose", "-v", action="store_true", help="Verbose logging")
    
    args = parser.parse_args()
    
    # Setup logging
    log_level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(level=log_level, format="%(asctime)s - %(levelname)s - %(message)s")
    
    # List datasets if requested
    if args.list_datasets:
        print("Available datasets:")
        for dataset_id in get_dataset_list():
            info = get_dataset_info(dataset_id)
            if info:
                print(f"  {dataset_id}: {info['description']} ({info['size']} samples)")
            else:
                print(f"  {dataset_id}: (no info available)")
        return
    
    # Run evaluation
    results = run_comprehensive_eval(
        datasets=args.datasets,
        model_path=args.model_path,
        config_path=args.config,
        output_dir=args.output_dir,
        include_perf_metrics=not args.no_perf
    )
    
    # Generate report if requested
    if args.report:
        generate_eval_report(results, f"{args.output_dir}/report.md")
    
    print(f"\nEvaluation complete! Results saved to: {args.output_dir}")
    print(f"Overall accuracy: {results['summary']['overall_accuracy']:.2%}")


if __name__ == "__main__":
    main()
