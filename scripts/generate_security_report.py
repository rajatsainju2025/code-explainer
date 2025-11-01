#!/usr/bin/env python3
"""
Security Report Generator

Aggregates security scan results from multiple tools and generates
comprehensive security reports with actionable recommendations.
"""

import json
import argparse
import sys
from pathlib import Path
from typing import Dict, Any, List, Optional
from datetime import datetime


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


def parse_bandit_results(results: Dict[str, Any]) -> Dict[str, Any]:
    """Parse Bandit security scan results."""
    issues = []
    for result in results.get('results', []):
        for issue in result.get('issues', []):
            issues.append({
                'tool': 'bandit',
                'severity': issue.get('issue_severity', 'unknown'),
                'confidence': issue.get('issue_confidence', 'unknown'),
                'file': issue.get('filename', ''),
                'line': issue.get('line_number', 0),
                'code': issue.get('code', ''),
                'description': issue.get('issue_text', ''),
                'cwe': issue.get('cwe', {}).get('id', ''),
                'recommendation': issue.get('issue_cwe', {}).get('link', '')
            })

    return {
        'tool': 'Bandit',
        'description': 'Python security linter',
        'issues': issues,
        'summary': {
            'total': len(issues),
            'high': sum(1 for i in issues if i['severity'] == 'HIGH'),
            'medium': sum(1 for i in issues if i['severity'] == 'MEDIUM'),
            'low': sum(1 for i in issues if i['severity'] == 'LOW')
        }
    }


def parse_safety_results(results: Dict[str, Any]) -> Dict[str, Any]:
    """Parse Safety dependency vulnerability results."""
    issues = []
    for vuln in results.get('vulnerabilities', []):
        issues.append({
            'tool': 'safety',
            'severity': 'HIGH',  # Safety typically reports high-severity issues
            'package': vuln.get('package', ''),
            'version': vuln.get('version', ''),
            'vulnerability': vuln.get('vulnerability', ''),
            'description': vuln.get('description', ''),
            'cve': vuln.get('cve', ''),
            'recommendation': f"Update {vuln.get('package')} to a safe version"
        })

    return {
        'tool': 'Safety',
        'description': 'Python dependency vulnerability scanner',
        'issues': issues,
        'summary': {
            'total': len(issues),
            'high': len(issues),  # All safety issues are considered high
            'medium': 0,
            'low': 0
        }
    }


def parse_semgrep_results(results: Dict[str, Any]) -> Dict[str, Any]:
    """Parse Semgrep security scan results."""
    issues = []
    for result in results.get('results', []):
        for finding in result.get('extra', {}).get('findings', []):
            issues.append({
                'tool': 'semgrep',
                'severity': finding.get('severity', 'unknown'),
                'file': finding.get('file', ''),
                'line': finding.get('line', 0),
                'code': finding.get('code', ''),
                'description': finding.get('message', ''),
                'rule': finding.get('check_id', ''),
                'recommendation': finding.get('fix', '')
            })

    return {
        'tool': 'Semgrep',
        'description': 'Semantic code analysis for security',
        'issues': issues,
        'summary': {
            'total': len(issues),
            'high': sum(1 for i in issues if i['severity'] == 'ERROR'),
            'medium': sum(1 for i in issues if i['severity'] == 'WARNING'),
            'low': sum(1 for i in issues if i['severity'] == 'INFO')
        }
    }


def parse_trivy_results(results: Dict[str, Any]) -> Dict[str, Any]:
    """Parse Trivy container scan results."""
    issues = []
    for result in results.get('Results', []):
        for vuln in result.get('Vulnerabilities', []):
            issues.append({
                'tool': 'trivy',
                'severity': vuln.get('Severity', 'unknown'),
                'package': vuln.get('PkgName', ''),
                'version': vuln.get('InstalledVersion', ''),
                'vulnerability': vuln.get('VulnerabilityID', ''),
                'description': vuln.get('Description', ''),
                'cve': vuln.get('VulnerabilityID', ''),
                'recommendation': f"Update {vuln.get('PkgName')} to version {vuln.get('FixedVersion', 'latest')}"
            })

    return {
        'tool': 'Trivy',
        'description': 'Container vulnerability scanner',
        'issues': issues,
        'summary': {
            'total': len(issues),
            'high': sum(1 for i in issues if i['severity'] == 'HIGH'),
            'medium': sum(1 for i in issues if i['severity'] == 'MEDIUM'),
            'low': sum(1 for i in issues if i['severity'] == 'LOW')
        }
    }


def generate_security_report(bandit_file: Optional[str] = None,
                           safety_file: Optional[str] = None,
                           semgrep_file: Optional[str] = None,
                           trivy_file: Optional[str] = None,
                           pip_audit_file: Optional[str] = None) -> str:
    """Generate a comprehensive security report."""

    report_lines = ["# Security Scan Report\n"]
    report_lines.append(f"Generated: {datetime.now().isoformat()}\n")

    all_results = []
    total_high = 0
    total_medium = 0
    total_low = 0

    # Process each tool's results
    if bandit_file:
        bandit_results = parse_bandit_results(load_json_file(bandit_file))
        all_results.append(bandit_results)
        total_high += bandit_results['summary']['high']
        total_medium += bandit_results['summary']['medium']
        total_low += bandit_results['summary']['low']

    if safety_file:
        safety_results = parse_safety_results(load_json_file(safety_file))
        all_results.append(safety_results)
        total_high += safety_results['summary']['high']

    if semgrep_file:
        semgrep_results = parse_semgrep_results(load_json_file(semgrep_file))
        all_results.append(semgrep_results)
        total_high += semgrep_results['summary']['high']
        total_medium += semgrep_results['summary']['medium']
        total_low += semgrep_results['summary']['low']

    if trivy_file:
        trivy_results = parse_trivy_results(load_json_file(trivy_file))
        all_results.append(trivy_results)
        total_high += trivy_results['summary']['high']
        total_medium += trivy_results['summary']['medium']
        total_low += trivy_results['summary']['low']

    # Executive summary
    report_lines.append("## Executive Summary\n")
    report_lines.append(f"- **Total Issues**: {total_high + total_medium + total_low}")
    report_lines.append(f"- **High Severity**: {total_high}")
    report_lines.append(f"- **Medium Severity**: {total_medium}")
    report_lines.append(f"- **Low Severity**: {total_low}")
    report_lines.append("")

    # Risk assessment
    if total_high > 0:
        report_lines.append("🚨 **CRITICAL**: High-severity security issues detected that require immediate attention.\n")
    elif total_medium > 5:
        report_lines.append("⚠️ **WARNING**: Multiple medium-severity issues detected.\n")
    else:
        report_lines.append("✅ **GOOD**: No critical security issues detected.\n")

    # Detailed results by tool
    for result in all_results:
        report_lines.append(f"## {result['tool']} Results\n")
        report_lines.append(f"**Description**: {result['description']}\n")
        report_lines.append("**Summary**:")
        report_lines.append(f"- Total: {result['summary']['total']}")
        report_lines.append(f"- High: {result['summary']['high']}")
        report_lines.append(f"- Medium: {result['summary']['medium']}")
        report_lines.append(f"- Low: {result['summary']['low']}\n")

        if result['issues']:
            report_lines.append("**Issues**:\n")
            for issue in result['issues'][:10]:  # Limit to first 10 issues per tool
                severity_emoji = {"HIGH": "🚨", "MEDIUM": "⚠️", "LOW": "ℹ️"}.get(issue['severity'], "❓")
                report_lines.append(f"- {severity_emoji} **{issue['severity']}**: {issue['description']}")
                if issue.get('file'):
                    report_lines.append(f"  - File: `{issue['file']}`:{issue.get('line', 'N/A')}")
                if issue.get('recommendation'):
                    report_lines.append(f"  - Recommendation: {issue['recommendation']}")
                report_lines.append("")

            if len(result['issues']) > 10:
                report_lines.append(f"*... and {len(result['issues']) - 10} more issues*\n")

    # Recommendations
    report_lines.append("## Recommendations\n")

    if total_high > 0:
        report_lines.append("### Immediate Actions Required")
        report_lines.append("- Address all high-severity security issues before deployment")
        report_lines.append("- Review and fix vulnerable dependencies")
        report_lines.append("- Implement security fixes for code vulnerabilities")
        report_lines.append("")

    report_lines.append("### General Security Practices")
    report_lines.append("- Keep dependencies updated regularly")
    report_lines.append("- Run security scans in CI/CD pipeline")
    report_lines.append("- Implement proper input validation and sanitization")
    report_lines.append("- Use security headers and HTTPS")
    report_lines.append("- Regular security training for development team")
    report_lines.append("- Implement secrets management and rotation")
    report_lines.append("")

    return "\n".join(report_lines)


def main():
    parser = argparse.ArgumentParser(description="Generate security report")
    parser.add_argument("--bandit", help="Path to Bandit scan results JSON")
    parser.add_argument("--safety", help="Path to Safety scan results JSON")
    parser.add_argument("--semgrep", help="Path to Semgrep scan results JSON")
    parser.add_argument("--trivy", help="Path to Trivy scan results JSON")
    parser.add_argument("--pip-audit", help="Path to pip-audit scan results JSON")
    parser.add_argument("--output", required=True, help="Output markdown file path")

    args = parser.parse_args()

    # Generate report
    report = generate_security_report(
        args.bandit,
        args.safety,
        args.semgrep,
        args.trivy,
        args.pip_audit
    )

    # Write report
    with open(args.output, 'w') as f:
        f.write(report)

    print(f"Security report generated: {args.output}")

    # Check if there are critical issues
    has_critical_issues = False
    for arg, file_path in [('bandit', args.bandit), ('safety', args.safety),
                          ('semgrep', args.semgrep), ('trivy', args.trivy)]:
        if file_path:
            results = load_json_file(file_path)
            if arg == 'bandit' and any(result.get('issues', []) for result in results.get('results', [])):
                has_critical_issues = True
            elif arg == 'safety' and results.get('vulnerabilities'):
                has_critical_issues = True
            elif arg == 'semgrep' and results.get('results'):
                has_critical_issues = True
            elif arg == 'trivy' and any(result.get('Vulnerabilities', []) for result in results.get('Results', [])):
                has_critical_issues = True

    if has_critical_issues:
        print("Critical security issues detected!")
        sys.exit(1)


if __name__ == "__main__":
    main()