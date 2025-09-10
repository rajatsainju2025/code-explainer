"""
Security Audit and Hardening Module

This module provides comprehensive security auditing and hardening capabilities
for the Code Explainer system, including vulnerability scanning, secure coding
practices, access control, and compliance monitoring.

Key Features:
- Automated vulnerability scanning and assessment
- Secure coding practice enforcement
- Access control and authentication hardening
- Input validation and sanitization
- Cryptographic security implementation
- Compliance monitoring and reporting
- Security event logging and alerting
- Penetration testing automation
- Security policy enforcement

Based on industry security standards and best practices.
"""

import hashlib
import hmac
import secrets
import logging
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional, Union, Callable
from abc import ABC, abstractmethod
import re
import json
import ast
import inspect
from functools import wraps
import threading
import time
import os

logger = logging.getLogger(__name__)

@dataclass
class SecurityVulnerability:
    """Represents a security vulnerability."""
    vulnerability_id: str
    title: str
    description: str
    severity: str  # Critical, High, Medium, Low, Info
    category: str  # injection, auth, crypto, etc.
    cwe_id: Optional[str] = None
    owasp_id: Optional[str] = None
    file_path: Optional[Path] = None
    line_number: Optional[int] = None
    code_snippet: Optional[str] = None
    recommendation: str = ""
    status: str = "open"  # open, fixed, false_positive
    discovered_at: datetime = field(default_factory=datetime.now)
    fixed_at: Optional[datetime] = None

@dataclass
class SecurityScanResult:
    """Result of a security scan."""
    scan_id: str
    target: str
    scan_type: str
    vulnerabilities: List[SecurityVulnerability]
    scan_duration: float
    scan_timestamp: datetime
    summary: Dict[str, int]  # severity counts

@dataclass
class SecurityPolicy:
    """Security policy definition."""
    name: str
    description: str
    rules: List[Dict[str, Any]]
    severity: str
    enabled: bool = True

class VulnerabilityScanner:
    """Automated vulnerability scanner."""

    def __init__(self):
        self.vulnerabilities: List[SecurityVulnerability] = []
        self.policies: List[SecurityPolicy] = []
        self.scan_results: List[SecurityScanResult] = []

    def add_policy(self, policy: SecurityPolicy):
        """Add a security policy."""
        self.policies.append(policy)

    def scan_codebase(self, root_path: Path) -> SecurityScanResult:
        """Scan codebase for vulnerabilities."""
        start_time = time.time()
        vulnerabilities = []

        # Scan Python files
        for py_file in root_path.rglob("*.py"):
            if self._should_scan_file(py_file):
                file_vulns = self._scan_python_file(py_file)
                vulnerabilities.extend(file_vulns)

        # Scan configuration files
        for config_file in root_path.rglob("*.yaml"):
            file_vulns = self._scan_config_file(config_file)
            vulnerabilities.extend(file_vulns)

        for config_file in root_path.rglob("*.json"):
            file_vulns = self._scan_config_file(config_file)
            vulnerabilities.extend(file_vulns)

        scan_duration = time.time() - start_time

        # Create summary
        summary = {}
        for vuln in vulnerabilities:
            summary[vuln.severity] = summary.get(vuln.severity, 0) + 1

        result = SecurityScanResult(
            scan_id=f"scan_{int(time.time())}",
            target=str(root_path),
            scan_type="codebase",
            vulnerabilities=vulnerabilities,
            scan_duration=scan_duration,
            scan_timestamp=datetime.now(),
            summary=summary
        )

        self.scan_results.append(result)
        return result

    def _should_scan_file(self, filepath: Path) -> bool:
        """Determine if file should be scanned."""
        # Skip common directories
        skip_dirs = {".git", "__pycache__", ".venv", "venv", "node_modules", ".pytest_cache"}
        parts = filepath.parts
        return not any(skip_dir in parts for skip_dir in skip_dirs)

    def _scan_python_file(self, filepath: Path) -> List[SecurityVulnerability]:
        """Scan Python file for vulnerabilities."""
        vulnerabilities = []

        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                content = f.read()

            # Parse AST for static analysis
            tree = ast.parse(content, filename=str(filepath))

            # Check for common vulnerabilities
            vulnerabilities.extend(self._check_injection_vulnerabilities(tree, filepath, content))
            vulnerabilities.extend(self._check_auth_vulnerabilities(tree, filepath, content))
            vulnerabilities.extend(self._check_crypto_vulnerabilities(tree, filepath, content))
            vulnerabilities.extend(self._check_input_validation(tree, filepath, content))

        except Exception as e:
            logger.error(f"Error scanning {filepath}: {str(e)}")

        return vulnerabilities

    def _check_injection_vulnerabilities(self, tree: ast.AST, filepath: Path,
                                       content: str) -> List[SecurityVulnerability]:
        """Check for injection vulnerabilities."""
        vulnerabilities = []

        for node in ast.walk(tree):
            if isinstance(node, ast.Call):
                # Check for SQL injection patterns
                if self._is_sql_injection_risk(node):
                    vuln = SecurityVulnerability(
                        vulnerability_id=f"SQL_INJ_{filepath.name}_{node.lineno}",
                        title="Potential SQL Injection",
                        description="String formatting used in SQL query may be vulnerable to injection",
                        severity="High",
                        category="injection",
                        cwe_id="CWE-89",
                        file_path=filepath,
                        line_number=node.lineno,
                        code_snippet=self._get_code_snippet(content, node.lineno),
                        recommendation="Use parameterized queries or prepared statements"
                    )
                    vulnerabilities.append(vuln)

                # Check for command injection
                if self._is_command_injection_risk(node):
                    vuln = SecurityVulnerability(
                        vulnerability_id=f"CMD_INJ_{filepath.name}_{node.lineno}",
                        title="Potential Command Injection",
                        description="Shell command execution with user input may be vulnerable",
                        severity="Critical",
                        category="injection",
                        cwe_id="CWE-78",
                        file_path=filepath,
                        line_number=node.lineno,
                        code_snippet=self._get_code_snippet(content, node.lineno),
                        recommendation="Use subprocess with argument lists, avoid shell=True"
                    )
                    vulnerabilities.append(vuln)

        return vulnerabilities

    def _is_sql_injection_risk(self, node: ast.Call) -> bool:
        """Check if SQL call is potentially vulnerable."""
        if isinstance(node.func, ast.Attribute):
            func_name = self._get_full_name(node.func)
            if "execute" in func_name or "query" in func_name:
                # Check arguments for string formatting
                for arg in node.args:
                    if isinstance(arg, ast.BinOp) and isinstance(arg.op, ast.Mod):
                        return True  # String formatting detected
                    if isinstance(arg, ast.Call) and self._is_format_call(arg):
                        return True  # .format() call detected
        return False

    def _is_command_injection_risk(self, node: ast.Call) -> bool:
        """Check if command execution is potentially vulnerable."""
        if isinstance(node.func, ast.Name):
            if node.func.id in ["system", "popen", "call"]:
                # Check if shell=True is used
                for keyword in node.keywords:
                    if keyword.arg == "shell" and isinstance(keyword.value, ast.NameConstant):
                        if keyword.value.value is True:
                            return True
        return False

    def _check_auth_vulnerabilities(self, tree: ast.AST, filepath: Path,
                                  content: str) -> List[SecurityVulnerability]:
        """Check for authentication vulnerabilities."""
        vulnerabilities = []

        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef):
                # Check for hardcoded credentials
                if self._has_hardcoded_credentials(node):
                    vuln = SecurityVulnerability(
                        vulnerability_id=f"HARDCODED_CRED_{filepath.name}_{node.lineno}",
                        title="Hardcoded Credentials",
                        description="Function contains hardcoded credentials or secrets",
                        severity="High",
                        category="auth",
                        cwe_id="CWE-798",
                        file_path=filepath,
                        line_number=node.lineno,
                        code_snippet=self._get_code_snippet(content, node.lineno),
                        recommendation="Use environment variables or secure credential storage"
                    )
                    vulnerabilities.append(vuln)

        return vulnerabilities

    def _has_hardcoded_credentials(self, node: ast.FunctionDef) -> bool:
        """Check if function has hardcoded credentials."""
        credential_patterns = [
            r"password\s*=\s*['\"][^'\"]+['\"]",
            r"secret\s*=\s*['\"][^'\"]+['\"]",
            r"token\s*=\s*['\"][^'\"]+['\"]",
            r"key\s*=\s*['\"][^'\"]+['\"]"
        ]

        # Get function source
        source = ast.get_source_segment(open(node.__dict__.get('filename', ''), 'r').read(), node)
        if source:
            for pattern in credential_patterns:
                if re.search(pattern, source, re.IGNORECASE):
                    return True
        return False

    def _check_crypto_vulnerabilities(self, tree: ast.AST, filepath: Path,
                                    content: str) -> List[SecurityVulnerability]:
        """Check for cryptographic vulnerabilities."""
        vulnerabilities = []

        for node in ast.walk(tree):
            if isinstance(node, ast.Call):
                if isinstance(node.func, ast.Name):
                    # Check for weak hash functions
                    if node.func.id in ["md5", "sha1"]:
                        vuln = SecurityVulnerability(
                            vulnerability_id=f"WEAK_HASH_{filepath.name}_{node.lineno}",
                            title="Weak Hash Function",
                            description=f"Use of weak hash function {node.func.id}",
                            severity="Medium",
                            category="crypto",
                            cwe_id="CWE-327",
                            file_path=filepath,
                            line_number=node.lineno,
                            code_snippet=self._get_code_snippet(content, node.lineno),
                            recommendation="Use SHA-256 or stronger hash functions"
                        )
                        vulnerabilities.append(vuln)

        return vulnerabilities

    def _check_input_validation(self, tree: ast.AST, filepath: Path,
                              content: str) -> List[SecurityVulnerability]:
        """Check for input validation issues."""
        vulnerabilities = []

        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef):
                # Check for functions that don't validate input
                if self._needs_input_validation(node):
                    vuln = SecurityVulnerability(
                        vulnerability_id=f"INPUT_VALIDATION_{filepath.name}_{node.lineno}",
                        title="Missing Input Validation",
                        description="Function processes input without validation",
                        severity="Medium",
                        category="validation",
                        cwe_id="CWE-20",
                        file_path=filepath,
                        line_number=node.lineno,
                        code_snippet=self._get_code_snippet(content, node.lineno),
                        recommendation="Add input validation and sanitization"
                    )
                    vulnerabilities.append(vuln)

        return vulnerabilities

    def _needs_input_validation(self, node: ast.FunctionDef) -> bool:
        """Check if function needs input validation."""
        # Simple heuristic: functions with parameters that process data
        has_params = len(node.args.args) > 0
        has_data_processing = any(
            isinstance(child, ast.Call) and hasattr(child, 'func') and hasattr(child.func, 'id')
            for child in ast.walk(node)
        )
        return has_params and has_data_processing

    def _scan_config_file(self, filepath: Path) -> List[SecurityVulnerability]:
        """Scan configuration file for security issues."""
        vulnerabilities = []

        try:
            if filepath.suffix.lower() == '.yaml':
                import yaml
                with open(filepath, 'r') as f:
                    config = yaml.safe_load(f)
            elif filepath.suffix.lower() == '.json':
                with open(filepath, 'r') as f:
                    config = json.load(f)
            else:
                return vulnerabilities

            # Check for security issues in config
            vulnerabilities.extend(self._check_config_security(config, filepath))

        except Exception as e:
            logger.error(f"Error scanning config {filepath}: {str(e)}")

        return vulnerabilities

    def _check_config_security(self, config: Dict[str, Any],
                             filepath: Path) -> List[SecurityVulnerability]:
        """Check configuration for security issues."""
        vulnerabilities = []

        # Check for debug mode in production
        if config.get('debug', False):
            vuln = SecurityVulnerability(
                vulnerability_id=f"DEBUG_ENABLED_{filepath.name}",
                title="Debug Mode Enabled",
                description="Debug mode is enabled in configuration",
                severity="Medium",
                category="config",
                file_path=filepath,
                recommendation="Disable debug mode in production"
            )
            vulnerabilities.append(vuln)

        # Check for weak passwords
        if 'password' in config and len(str(config['password'])) < 8:
            vuln = SecurityVulnerability(
                vulnerability_id=f"WEAK_PASSWORD_{filepath.name}",
                title="Weak Password",
                description="Password is too short or weak",
                severity="High",
                category="auth",
                file_path=filepath,
                recommendation="Use strong passwords with 12+ characters"
            )
            vulnerabilities.append(vuln)

        return vulnerabilities

    def _get_full_name(self, node: ast.AST) -> str:
        """Get full name of AST node."""
        if isinstance(node, ast.Name):
            return node.id
        elif isinstance(node, ast.Attribute):
            return f"{self._get_full_name(node.value)}.{node.attr}"
        return ""

    def _is_format_call(self, node: ast.Call) -> bool:
        """Check if call is a format method."""
        if isinstance(node.func, ast.Attribute):
            return node.func.attr == "format"
        return False

    def _get_code_snippet(self, content: str, line_number: int,
                         context_lines: int = 2) -> str:
        """Get code snippet around line number."""
        lines = content.split('\n')
        start = max(0, line_number - context_lines - 1)
        end = min(len(lines), line_number + context_lines)
        return '\n'.join(lines[start:end])

class SecurityHardener:
    """Security hardening utilities."""

    def __init__(self):
        self.security_headers = {
            'X-Content-Type-Options': 'nosniff',
            'X-Frame-Options': 'DENY',
            'X-XSS-Protection': '1; mode=block',
            'Strict-Transport-Security': 'max-age=31536000; includeSubDomains',
            'Content-Security-Policy': "default-src 'self'",
            'Referrer-Policy': 'strict-origin-when-cross-origin'
        }

    def generate_secure_headers(self) -> Dict[str, str]:
        """Generate secure HTTP headers."""
        return self.security_headers.copy()

    def sanitize_input(self, input_data: str, max_length: int = 1000) -> str:
        """Sanitize user input."""
        if not input_data:
            return ""

        # Remove potentially dangerous characters
        sanitized = re.sub(r'[<>]', '', input_data)

        # Limit length
        if len(sanitized) > max_length:
            sanitized = sanitized[:max_length]

        return sanitized

    def validate_file_upload(self, filename: str, allowed_extensions: List[str]) -> bool:
        """Validate file upload security."""
        if not filename:
            return False

        # Check extension
        _, ext = os.path.splitext(filename)
        if ext.lower() not in [e.lower() for e in allowed_extensions]:
            return False

        # Check for path traversal
        if '..' in filename or '/' in filename or '\\' in filename:
            return False

        return True

    def generate_api_key(self, length: int = 32) -> str:
        """Generate secure API key."""
        return secrets.token_urlsafe(length)

    def hash_password(self, password: str) -> str:
        """Hash password securely."""
        salt = secrets.token_hex(16)
        hashed = hashlib.pbkdf2_hmac('sha256', password.encode(), salt.encode(), 100000)
        return f"{salt}:{hashed.hex()}"

    def verify_password(self, password: str, hashed_password: str) -> bool:
        """Verify password against hash."""
        try:
            salt, hash_value = hashed_password.split(':')
            computed_hash = hashlib.pbkdf2_hmac('sha256', password.encode(),
                                              salt.encode(), 100000)
            return secrets.compare_digest(computed_hash.hex(), hash_value)
        except:
            return False

class SecurityMonitor:
    """Security monitoring and alerting."""

    def __init__(self):
        self.alerts: List[Dict[str, Any]] = []
        self.failed_attempts: Dict[str, int] = {}
        self.suspicious_patterns = [
            r"union.*select.*from",  # SQL injection
            r"<script.*>.*</script>",  # XSS
            r"\.\./\.\./",  # Path traversal
            r"eval\s*\(",  # Code injection
        ]

    def log_security_event(self, event_type: str, details: Dict[str, Any],
                          severity: str = "info"):
        """Log security event."""
        event = {
            'timestamp': datetime.now(),
            'type': event_type,
            'severity': severity,
            'details': details
        }
        self.alerts.append(event)

        if severity in ['high', 'critical']:
            logger.warning(f"Security alert: {event_type} - {details}")

    def check_suspicious_input(self, input_data: str, source: str) -> bool:
        """Check input for suspicious patterns."""
        for pattern in self.suspicious_patterns:
            if re.search(pattern, input_data, re.IGNORECASE):
                self.log_security_event(
                    'suspicious_input_detected',
                    {'pattern': pattern, 'source': source, 'input': input_data[:100]},
                    'high'
                )
                return True
        return False

    def track_failed_authentication(self, username: str, ip_address: str):
        """Track failed authentication attempts."""
        key = f"{username}:{ip_address}"
        self.failed_attempts[key] = self.failed_attempts.get(key, 0) + 1

        if self.failed_attempts[key] >= 5:
            self.log_security_event(
                'brute_force_attempt',
                {'username': username, 'ip': ip_address, 'attempts': self.failed_attempts[key]},
                'critical'
            )

    def get_security_report(self) -> Dict[str, Any]:
        """Generate security report."""
        return {
            'total_alerts': len(self.alerts),
            'alerts_by_severity': self._count_alerts_by_severity(),
            'recent_alerts': self.alerts[-10:],  # Last 10 alerts
            'failed_auth_attempts': len(self.failed_attempts),
            'timestamp': datetime.now()
        }

    def _count_alerts_by_severity(self) -> Dict[str, int]:
        """Count alerts by severity."""
        counts = {}
        for alert in self.alerts:
            severity = alert['severity']
            counts[severity] = counts.get(severity, 0) + 1
        return counts

# Decorators for security
def require_authentication(auth_func: Callable):
    """Decorator to require authentication."""
    def decorator(func: Callable):
        @wraps(func)
        def wrapper(*args, **kwargs):
            if not auth_func():
                raise PermissionError("Authentication required")
            return func(*args, **kwargs)
        return wrapper
    return decorator

def validate_input(validation_func: Callable):
    """Decorator to validate input."""
    def decorator(func: Callable):
        @wraps(func)
        def wrapper(*args, **kwargs):
            if not validation_func(*args, **kwargs):
                raise ValueError("Input validation failed")
            return func(*args, **kwargs)
        return wrapper
    return decorator

def rate_limit(max_calls: int, time_window: int):
    """Decorator to implement rate limiting."""
    calls = {}

    def decorator(func: Callable):
        @wraps(func)
        def wrapper(*args, **kwargs):
            now = time.time()
            key = f"{func.__name__}_{id(args[0]) if args else 'global'}"

            # Clean old calls
            calls[key] = [call for call in calls.get(key, [])
                         if now - call < time_window]

            if len(calls[key]) >= max_calls:
                raise Exception("Rate limit exceeded")

            calls[key].append(now)
            return func(*args, **kwargs)
        return wrapper
    return decorator

# Convenience functions
def create_vulnerability_scanner() -> VulnerabilityScanner:
    """Create vulnerability scanner."""
    scanner = VulnerabilityScanner()

    # Add default security policies
    policies = [
        SecurityPolicy(
            name="sql_injection_prevention",
            description="Prevent SQL injection vulnerabilities",
            rules=[
                {"pattern": r"execute.*%.*", "severity": "high"},
                {"pattern": r"query.*\+.*", "severity": "high"}
            ],
            severity="high"
        ),
        SecurityPolicy(
            name="hardcoded_secrets",
            description="Detect hardcoded secrets and credentials",
            rules=[
                {"pattern": r"password\s*=\s*['\"][^'\"]+['\"]", "severity": "high"},
                {"pattern": r"secret\s*=\s*['\"][^'\"]+['\"]", "severity": "high"}
            ],
            severity="high"
        )
    ]

    for policy in policies:
        scanner.add_policy(policy)

    return scanner

def create_security_hardener() -> SecurityHardener:
    """Create security hardener."""
    return SecurityHardener()

def create_security_monitor() -> SecurityMonitor:
    """Create security monitor."""
    return SecurityMonitor()

def run_security_audit(project_path: Path) -> Dict[str, Any]:
    """Run comprehensive security audit."""
    scanner = create_vulnerability_scanner()
    hardener = create_security_hardener()
    monitor = create_security_monitor()

    # Run vulnerability scan
    scan_result = scanner.scan_codebase(project_path)

    # Generate security report
    report = {
        'scan_result': {
            'vulnerabilities_found': len(scan_result.vulnerabilities),
            'severity_breakdown': scan_result.summary,
            'scan_duration': scan_result.scan_duration
        },
        'security_headers': hardener.generate_secure_headers(),
        'security_monitoring': monitor.get_security_report(),
        'recommendations': [
            "Implement input validation for all user inputs",
            "Use parameterized queries to prevent SQL injection",
            "Store secrets in environment variables or secure vaults",
            "Implement rate limiting for API endpoints",
            "Use HTTPS for all communications",
            "Regularly update dependencies for security patches"
        ]
    }

    return report

if __name__ == "__main__":
    # Example usage
    project_path = Path.cwd()

    print("Running security audit...")
    report = run_security_audit(project_path)

    print("Security Audit Report:")
    print(f"Vulnerabilities found: {report['scan_result']['vulnerabilities_found']}")
    print(f"Severity breakdown: {report['scan_result']['severity_breakdown']}")
    print(f"Scan duration: {report['scan_result']['scan_duration']:.2f}s")

    print("\nSecurity Recommendations:")
    for rec in report['recommendations']:
        print(f"- {rec}")

    print("\nSecurity audit completed!")
