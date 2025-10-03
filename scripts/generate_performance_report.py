#!/usr/bin/env python3
"""
Performance Report Generator

Generates comprehensive performance reports comparing current benchmarks
against baseline results and detecting regressions.
"""

import json
import argparse
import sys
from pathlib import Path
from typing import Dict, Any, List, Optional
import statistics


def load_json_file(file_path: str) -> Dict[str, Any]:
    """Load JSON file with error handling."""
    try:
        with open(file_path, 'r') as f:
            return json.load(f)
    except FileNotFoundError:
        print(f"Warning: File {file_path} not found")
        return {}
    except json.JSONDecodeError as e:
        print(f"Error parsing JSON file {file_path}: {e}")
        return {}


def calculate_regression(current: float, baseline: float, threshold: float = 0.05) -> Dict[str, Any]:
    """Calculate performance regression metrics."""
    if baseline == 0:
        return {"regression": False, "change_percent": 0, "significant": False}

    change_percent = (current - baseline) / baseline
    regression = change_percent > threshold
    significant = abs(change_percent) > threshold

    return {
        "regression": regression,
        "change_percent": change_percent,
        "significant": significant,
        "current": current,
        "baseline": baseline
    }


def generate_performance_report(current_results: Dict[str, Any],
                              baseline_results: Optional[Dict[str, Any]] = None,
                              regression_threshold: float = 0.05) -> str:
    """Generate a comprehensive performance report."""

    report_lines = ["# Performance Report\n"]

    regressions = []
    improvements = []

    if baseline_results:
        report_lines.append("## Regression Analysis\n")

        # Compare key metrics
        for metric_name, current_value in current_results.get('metrics', {}).items():
            baseline_value = baseline_results.get('metrics', {}).get(metric_name)
            if baseline_value is not None:
                regression_info = calculate_regression(current_value, baseline_value, regression_threshold)

                if regression_info['regression']:
                    regressions.append(f"- 🚨 **{metric_name}**: {regression_info['change_percent']:.1%} degradation")
                elif regression_info['significant'] and regression_info['change_percent'] < 0:
                    improvements.append(f"- ✅ **{metric_name}**: {abs(regression_info['change_percent']):.1%} improvement")

        if regressions:
            report_lines.append("### Regressions Detected\n")
            report_lines.extend(regressions)
            report_lines.append("")

        if improvements:
            report_lines.append("### Improvements Detected\n")
            report_lines.extend(improvements)
            report_lines.append("")

    # Current performance metrics
    report_lines.append("## Current Performance Metrics\n")
    for metric_name, value in current_results.get('metrics', {}).items():
        if isinstance(value, float):
            report_lines.append(f"- **{metric_name}**: {value:.4f}")
        else:
            report_lines.append(f"- **{metric_name}**: {value}")

    # System information
    if 'system_info' in current_results:
        report_lines.append("\n## System Information\n")
        for key, value in current_results['system_info'].items():
            report_lines.append(f"- **{key}**: {value}")

    # Recommendations
    report_lines.append("\n## Recommendations\n")
    if baseline_results and regressions:
        report_lines.append("- Investigate and optimize components showing performance regression")
        report_lines.append("- Consider updating performance baselines if changes are expected")
        report_lines.append("- Review recent code changes for potential bottlenecks")

    report_lines.append("- Monitor memory usage and implement optimizations if needed")
    report_lines.append("- Consider caching strategies for frequently accessed data")
    report_lines.append("- Review batch processing configurations for optimal throughput")

    return "\n".join(report_lines)


def main():
    parser = argparse.ArgumentParser(description="Generate performance report")
    parser.add_argument("--current", required=True, help="Path to current benchmark results JSON")
    parser.add_argument("--baseline", help="Path to baseline benchmark results JSON")
    parser.add_argument("--output", required=True, help="Output markdown file path")
    parser.add_argument("--regression-threshold", type=float, default=0.05,
                       help="Regression threshold as decimal (default: 0.05 = 5%)")

    args = parser.parse_args()

    # Load results
    current_results = load_json_file(args.current)
    baseline_results = load_json_file(args.baseline) if args.baseline else None

    if not current_results:
        print("Error: No current results to process")
        sys.exit(1)

    # Generate report
    report = generate_performance_report(
        current_results,
        baseline_results,
        args.regression_threshold
    )

    # Write report
    with open(args.output, 'w') as f:
        f.write(report)

    print(f"Performance report generated: {args.output}")

    # Exit with error code if regressions detected
    if baseline_results:
        has_regressions = any(
            calculate_regression(
                current_results.get('metrics', {}).get(metric_name, 0),
                baseline_results.get('metrics', {}).get(metric_name, 0),
                args.regression_threshold
            )['regression']
            for metric_name in current_results.get('metrics', {})
            if metric_name in baseline_results.get('metrics', {})
        )

        if has_regressions:
            print("Performance regressions detected!")
            sys.exit(1)


if __name__ == "__main__":
    main()