#!/usr/bin/env python3
"""
Security Scanner for Code Explainer
Performs comprehensive security analysis including:
- Dependency vulnerability scanning
- Code security analysis
- Configuration security checks
- Container security scanning
"""

import subprocess
import sys
import json
import os
from pathlib import Path
from typing import Dict, List, Any, Optional
import argparse
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class SecurityScanner:
    """Comprehensive security scanner for the codebase."""

    def __init__(self, project_root: Path):
        self.project_root = project_root
        self.results = {
            "vulnerabilities": [],
            "code_issues": [],
            "config_issues": [],
            "container_issues": [],
            "summary": {}
        }

    def run_dependency_scan(self) -> Dict[str, Any]:
        """Scan dependencies for vulnerabilities using Safety."""
        logger.info("🔍 Scanning dependencies for vulnerabilities...")

        try:
            # Check if safety is installed
            result = subprocess.run(
                [sys.executable, "-m", "safety", "check", "--json"],
                capture_output=True, text=True, cwd=self.project_root
            )

            vulnerabilities = []
            if result.returncode == 0:
                # No vulnerabilities found or successful scan
                try:
                    vuln_data = json.loads(result.stdout)
                    vulnerabilities = vuln_data.get("vulnerabilities", [])
                except json.JSONDecodeError:
                    # Try to parse as different format
                    if "No vulnerabilities found" in result.stdout:
                        vulnerabilities = []
                    else:
                        vulnerabilities = [{"error": "Failed to parse safety output", "output": result.stdout[:500]}]
            else:
                # Try to parse error output
                try:
                    vuln_data = json.loads(result.stdout)
                    vulnerabilities = vuln_data.get("vulnerabilities", [])
                except json.JSONDecodeError:
                    try:
                        vuln_data = json.loads(result.stderr)
                        vulnerabilities = vuln_data.get("vulnerabilities", [])
                    except json.JSONDecodeError:
                        vulnerabilities = [{"error": "Safety scan failed", "stderr": result.stderr[:500]}]

            return {
                "tool": "safety",
                "packages_scanned": len(vulnerabilities) if vulnerabilities else 0,
                "vulnerabilities_found": len(vulnerabilities),
                "vulnerabilities": vulnerabilities,
                "status": "completed"
            }

        except Exception as e:
            logger.error(f"Dependency scan failed: {e}")
            return {"error": str(e)}

    def run_code_security_scan(self) -> Dict[str, Any]:
        """Run code security analysis using Bandit."""
        logger.info("🔍 Running code security analysis...")

        try:
            # Use file output to avoid truncation issues
            import tempfile
            import os

            with tempfile.NamedTemporaryFile(mode='w+', suffix='.json', delete=False) as temp_file:
                temp_filename = temp_file.name

            try:
                # Run bandit with file output
                result = subprocess.run(
                    [sys.executable, "-m", "bandit", "-r", "src/", "-f", "json", "-o", temp_filename],
                    capture_output=True, text=True, cwd=self.project_root
                )

                # Read the results from file
                if os.path.exists(temp_filename):
                    with open(temp_filename, 'r') as f:
                        issues_data = json.load(f)

                    issues = issues_data.get("results", [])
                    metrics = issues_data.get("metrics", {}).get("_totals", {})

                    return {
                        "tool": "bandit",
                        "issues_found": len(issues),
                        "issues": issues,
                        "metrics": metrics,
                        "status": "completed"
                    }
                else:
                    return {"error": "Bandit output file not created"}

            finally:
                # Clean up temp file
                if os.path.exists(temp_filename):
                    os.unlink(temp_filename)

        except Exception as e:
            logger.error(f"Code security scan failed: {e}")
            return {"error": str(e)}

    def run_semgrep_scan(self) -> Dict[str, Any]:
        """Run Semgrep security analysis."""
        logger.info("🔍 Running Semgrep security analysis...")

        try:
            # Run semgrep with security rules
            result = subprocess.run(
                [sys.executable, "-m", "semgrep", "--config", "auto", "--json", "src/"],
                capture_output=True, text=True, cwd=self.project_root
            )

            findings = []
            if result.returncode == 0:
                try:
                    semgrep_data = json.loads(result.stdout)
                    findings = semgrep_data.get("results", [])
                except json.JSONDecodeError:
                    pass

            return {
                "tool": "semgrep",
                "findings": len(findings),
                "results": findings,
                "status": "completed"
            }

        except Exception as e:
            logger.error(f"Semgrep scan failed: {e}")
            return {"error": str(e)}

    def check_config_security(self) -> Dict[str, Any]:
        """Check configuration files for security issues."""
        logger.info("🔍 Checking configuration security...")

        issues = []

        # Check for hardcoded secrets in config files
        config_files = [
            "configs/*.yaml",
            "*.yaml",
            "*.yml",
            "*.json",
            "*.toml",
            "*.cfg",
            "*.ini"
        ]

        secret_patterns = [
            r"password\s*[:=]\s*['\"]?[^'\"\s]+['\"]?",
            r"secret\s*[:=]\s*['\"]?[^'\"\s]+['\"]?",
            r"token\s*[:=]\s*['\"]?[^'\"\s]+['\"]?",
            r"key\s*[:=]\s*['\"]?[^'\"\s]+['\"]?",
            r"api_key\s*[:=]\s*['\"]?[^'\"\s]+['\"]?",
        ]

        for pattern in config_files:
            for config_file in self.project_root.glob(pattern):
                if config_file.is_file():
                    try:
                        content = config_file.read_text()
                        for pattern in secret_patterns:
                            import re
                            if re.search(pattern, content, re.IGNORECASE):
                                issues.append({
                                    "file": str(config_file),
                                    "issue": "Potential hardcoded secret detected",
                                    "pattern": pattern
                                })
                    except Exception as e:
                        issues.append({
                            "file": str(config_file),
                            "issue": f"Could not read file: {e}"
                        })

        # Check for overly permissive file permissions
        sensitive_files = [
            "*.pem", "*.key", "*.p12", "*.pfx", "*.jks",
            ".env*", "secrets.*", "*secret*", "*key*"
        ]

        for pattern in sensitive_files:
            for sensitive_file in self.project_root.glob(pattern):
                if sensitive_file.is_file():
                    permissions = oct(sensitive_file.stat().st_mode)[-3:]
                    if permissions not in ["600", "400"]:
                        issues.append({
                            "file": str(sensitive_file),
                            "issue": f"File has overly permissive permissions: {permissions}",
                            "recommended": "600"
                        })

        return {
            "tool": "config_checker",
            "issues_found": len(issues),
            "issues": issues,
            "status": "completed"
        }

    def check_container_security(self) -> Dict[str, Any]:
        """Check Docker/container security."""
        logger.info("🔍 Checking container security...")

        issues = []

        # Check Dockerfile
        dockerfile = self.project_root / "Dockerfile"
        if dockerfile.exists():
            content = dockerfile.read_text()

            # Check for root user usage
            if "USER root" in content or not any("USER " in line for line in content.split('\n')):
                issues.append({
                    "file": "Dockerfile",
                    "issue": "Container may be running as root user",
                    "recommendation": "Use non-root user for security"
                })

            # Check for latest tag usage
            if ":latest" in content:
                issues.append({
                    "file": "Dockerfile",
                    "issue": "Using 'latest' tag can lead to unpredictable builds",
                    "recommendation": "Use specific version tags"
                })

        # Check docker-compose files
        compose_files = ["docker-compose.yml", "docker-compose.yaml"]
        for compose_file in compose_files:
            compose_path = self.project_root / compose_file
            if compose_path.exists():
                content = compose_path.read_text()
                if "root" in content.lower():
                    issues.append({
                        "file": compose_file,
                        "issue": "Potential root user configuration in compose file"
                    })

        return {
            "tool": "container_checker",
            "issues_found": len(issues),
            "issues": issues,
            "status": "completed"
        }

    def generate_report(self) -> Dict[str, Any]:
        """Generate comprehensive security report."""
        logger.info("📋 Generating security report...")

        # Run all scans
        self.results["vulnerabilities"] = self.run_dependency_scan()
        self.results["code_issues"] = self.run_code_security_scan()
        self.results["semgrep_findings"] = self.run_semgrep_scan()
        self.results["config_issues"] = self.check_config_security()
        self.results["container_issues"] = self.check_container_security()

        # Calculate summary
        total_issues = sum(
            result.get("issues_found", 0) + result.get("vulnerabilities_found", 0) + result.get("findings", 0)
            for result in self.results.values()
            if isinstance(result, dict)
        )

        critical_issues = sum(
            1 for result in self.results.values()
            if isinstance(result, dict) and result.get("issues_found", 0) > 0
        )

        self.results["summary"] = {
            "total_issues": total_issues,
            "critical_issues": critical_issues,
            "scan_status": "completed",
            "recommendations": self._generate_recommendations()
        }

        return self.results

    def _generate_recommendations(self) -> List[str]:
        """Generate security recommendations based on findings."""
        recommendations = []

        if self.results["vulnerabilities"].get("vulnerabilities_found", 0) > 0:
            recommendations.append("Update vulnerable dependencies to latest secure versions")

        if self.results["code_issues"].get("issues_found", 0) > 0:
            recommendations.append("Review and fix code security issues identified by Bandit")

        if self.results["config_issues"].get("issues_found", 0) > 0:
            recommendations.append("Remove hardcoded secrets and fix file permissions")

        if self.results["container_issues"].get("issues_found", 0) > 0:
            recommendations.append("Review Docker configuration for security best practices")

        if not recommendations:
            recommendations.append("✅ No critical security issues found - continue monitoring")

        return recommendations

    def save_report(self, output_file: Path) -> None:
        """Save security report to file."""
        with open(output_file, 'w') as f:
            json.dump(self.results, f, indent=2)
        logger.info(f"📄 Security report saved to {output_file}")

def main():
    parser = argparse.ArgumentParser(description="Security Scanner for Code Explainer")
    parser.add_argument("--output", "-o", type=Path, default=Path("security_report.json"),
                       help="Output file for security report")
    parser.add_argument("--project-root", type=Path, default=Path("."),
                       help="Project root directory")

    args = parser.parse_args()

    scanner = SecurityScanner(args.project_root)
    report = scanner.generate_report()
    scanner.save_report(args.output)

    # Print summary
    summary = report["summary"]
    print(f"\n🔒 Security Scan Summary:")
    print(f"Total Issues: {summary['total_issues']}")
    print(f"Critical Issues: {summary['critical_issues']}")
    print(f"\n📋 Recommendations:")
    for rec in summary["recommendations"]:
        print(f"  • {rec}")

    # Exit with error code if critical issues found
    if summary["critical_issues"] > 0:
        sys.exit(1)

if __name__ == "__main__":
    main()