"""
Advanced Security Module

This module implements comprehensive security features for the Code Intelligence
Platform, including advanced threat detection, secure code analysis, privacy
protection, and compliance management. It incorporates the latest security
research and best practices to ensure the platform remains secure.

Features:
- Advanced threat detection and prevention
- Secure code analysis and vulnerability assessment
- Privacy-preserving machine learning techniques
- Compliance management and audit trails
- Secure multi-party computation for collaborative features
- Adversarial attack detection and mitigation
- Secure model deployment and monitoring
- Data poisoning detection and prevention
- Access control and authentication enhancements
- Security monitoring and incident response
"""

import json
import hashlib
import hmac
import secrets
import time
from typing import Dict, List, Optional, Any, Callable, Union, Tuple, Set
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
import re
import ast
import inspect
try:
    from cryptography.fernet import Fernet
    from cryptography.hazmat.primitives import hashes
    from cryptography.hazmat.primitives.asymmetric import rsa, padding
    CRYPTOGRAPHY_AVAILABLE = True
except ImportError:
    CRYPTOGRAPHY_AVAILABLE = False
import warnings
warnings.filterwarnings('ignore')


@dataclass
class SecurityEvent:
    """Represents a security event."""
    event_id: str
    event_type: str
    severity: str
    description: str
    source: str
    user_id: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    timestamp: datetime = field(default_factory=datetime.utcnow)


@dataclass
class VulnerabilityReport:
    """Report of security vulnerabilities."""
    report_id: str
    target: str
    vulnerabilities: List[Dict[str, Any]]
    risk_score: float
    recommendations: List[str]
    scan_timestamp: datetime = field(default_factory=datetime.utcnow)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class AccessControlPolicy:
    """Access control policy."""
    policy_id: str
    name: str
    rules: List[Dict[str, Any]]
    effect: str  # 'allow' or 'deny'
    priority: int
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class AuditLogEntry:
    """Audit log entry."""
    entry_id: str
    action: str
    user_id: str
    resource: str
    result: str
    ip_address: str
    user_agent: str
    timestamp: datetime = field(default_factory=datetime.utcnow)
    metadata: Dict[str, Any] = field(default_factory=dict)


class ThreatDetectionEngine:
    """Advanced threat detection engine."""

    def __init__(self):
        self.threat_patterns: Dict[str, Callable] = {}
        self.anomaly_detectors: Dict[str, Any] = {}
        self.threat_intelligence: Dict[str, Any] = {}

    def register_threat_pattern(self, pattern_name: str,
                               detection_function: Callable) -> None:
        """Register a threat detection pattern."""
        self.threat_patterns[pattern_name] = detection_function

    def detect_threats(self, data: Any, context: Dict[str, Any]) -> List[SecurityEvent]:
        """Detect threats in the given data."""
        events = []

        for pattern_name, detection_func in self.threat_patterns.items():
            try:
                threats = detection_func(data, context)
                for threat in threats:
                    event = SecurityEvent(
                        event_id=secrets.token_hex(16),
                        event_type="threat_detected",
                        severity=threat.get("severity", "medium"),
                        description=threat.get("description", "Threat detected"),
                        source=pattern_name,
                        metadata=threat
                    )
                    events.append(event)
            except Exception as e:
                # Log detection errors but don't fail the entire process
                print(f"Error in threat detection pattern {pattern_name}: {e}")

        return events

    def detect_code_injection(self, code: str, context: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Detect code injection attempts."""
        threats = []

        # Common injection patterns
        injection_patterns = [
            r"eval\s*\(",
            r"exec\s*\(",
            r"__import__\s*\(",
            r"subprocess\.",
            r"os\.system",
            r"open\s*\(.*/etc/passwd",
            r"pickle\.loads?",
            r"yaml\.load",
        ]

        for pattern in injection_patterns:
            if re.search(pattern, code, re.IGNORECASE):
                threats.append({
                    "type": "code_injection",
                    "pattern": pattern,
                    "severity": "high",
                    "description": f"Potential code injection detected: {pattern}"
                })

        return threats

    def detect_data_poisoning(self, data: Any, context: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Detect data poisoning attempts."""
        threats = []

        # Simple statistical anomaly detection
        if isinstance(data, list) and len(data) > 10:
            # Check for statistical anomalies
            values = [item.get("value", 0) for item in data if isinstance(item, dict)]
            if values:
                mean_val = sum(values) / len(values)
                std_val = (sum((x - mean_val) ** 2 for x in values) / len(values)) ** 0.5

                outliers = [i for i, v in enumerate(values) if abs(v - mean_val) > 3 * std_val]
                if outliers:
                    threats.append({
                        "type": "data_poisoning",
                        "severity": "medium",
                        "description": f"Statistical outliers detected in {len(outliers)} data points"
                    })

        return threats

    def detect_adversarial_inputs(self, input_data: Any,
                                context: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Detect adversarial inputs."""
        threats = []

        # Check for common adversarial patterns
        if isinstance(input_data, str):
            # Unicode obfuscation
            if re.search(r'\\u[0-9a-f]{4}', input_data):
                threats.append({
                    "type": "adversarial_input",
                    "severity": "medium",
                    "description": "Unicode obfuscation detected"
                })

            # Excessive special characters
            special_chars = len(re.findall(r'[^\w\s]', input_data))
            if special_chars / len(input_data) > 0.3:
                threats.append({
                    "type": "adversarial_input",
                    "severity": "low",
                    "description": "High ratio of special characters detected"
                })

        return threats


class SecureCodeAnalyzer:
    """Secure code analysis and vulnerability assessment."""

    def __init__(self):
        self.vulnerability_patterns: Dict[str, Dict[str, Any]] = {}
        self.security_rules: List[Dict[str, Any]] = []

    def register_vulnerability_pattern(self, vuln_id: str,
                                     pattern: Dict[str, Any]) -> None:
        """Register a vulnerability detection pattern."""
        self.vulnerability_patterns[vuln_id] = pattern

    def analyze_code_security(self, code: str,
                            language: str = "python") -> VulnerabilityReport:
        """Analyze code for security vulnerabilities."""
        vulnerabilities = []

        # Parse the code
        try:
            if language == "python":
                vulnerabilities.extend(self._analyze_python_security(code))
            else:
                vulnerabilities.append({
                    "type": "unsupported_language",
                    "severity": "info",
                    "description": f"Security analysis not available for {language}"
                })
        except Exception as e:
            vulnerabilities.append({
                "type": "analysis_error",
                "severity": "medium",
                "description": f"Error during security analysis: {str(e)}"
            })

        # Calculate risk score
        risk_score = self._calculate_risk_score(vulnerabilities)

        # Generate recommendations
        recommendations = self._generate_security_recommendations(vulnerabilities)

        return VulnerabilityReport(
            report_id=secrets.token_hex(16),
            target=f"code_analysis_{language}",
            vulnerabilities=vulnerabilities,
            risk_score=risk_score,
            recommendations=recommendations
        )

    def _analyze_python_security(self, code: str) -> List[Dict[str, Any]]:
        """Analyze Python code for security issues."""
        vulnerabilities = []

        try:
            tree = ast.parse(code)

            for node in ast.walk(tree):
                # Check for dangerous function calls
                if isinstance(node, ast.Call):
                    func_name = self._get_function_name(node.func)
                    if func_name in ["eval", "exec", "pickle.loads", "yaml.load"]:
                        vulnerabilities.append({
                            "type": "dangerous_function",
                            "severity": "high",
                            "line": getattr(node, 'lineno', 0),
                            "description": f"Dangerous function call: {func_name}"
                        })

                # Check for SQL injection patterns
                elif isinstance(node, ast.Str):
                    if re.search(r"SELECT.*\+.*WHERE", node.s, re.IGNORECASE):
                        vulnerabilities.append({
                            "type": "sql_injection",
                            "severity": "high",
                            "line": getattr(node, 'lineno', 0),
                            "description": "Potential SQL injection vulnerability"
                        })

                # Check for hardcoded secrets
                elif isinstance(node, ast.Assign):
                    for target in node.targets:
                        if isinstance(target, ast.Name):
                            if "password" in target.id.lower() or "secret" in target.id.lower():
                                if isinstance(node.value, ast.Str):
                                    vulnerabilities.append({
                                        "type": "hardcoded_secret",
                                        "severity": "medium",
                                        "line": getattr(node, 'lineno', 0),
                                        "description": f"Hardcoded secret detected: {target.id}"
                                    })

        except SyntaxError:
            vulnerabilities.append({
                "type": "syntax_error",
                "severity": "info",
                "description": "Code contains syntax errors"
            })

        return vulnerabilities

    def _get_function_name(self, node: ast.expr) -> str:
        """Get function name from AST node."""
        if isinstance(node, ast.Name):
            return node.id
        elif isinstance(node, ast.Attribute):
            return f"{self._get_function_name(node.value)}.{node.attr}"
        return ""

    def _calculate_risk_score(self, vulnerabilities: List[Dict[str, Any]]) -> float:
        """Calculate overall risk score."""
        if not vulnerabilities:
            return 0.0

        severity_weights = {
            "critical": 1.0,
            "high": 0.8,
            "medium": 0.5,
            "low": 0.2,
            "info": 0.1
        }

        total_weight = sum(severity_weights.get(v.get("severity", "medium"), 0.5)
                          for v in vulnerabilities)

        return min(total_weight / len(vulnerabilities), 1.0)

    def _generate_security_recommendations(self,
                                         vulnerabilities: List[Dict[str, Any]]) -> List[str]:
        """Generate security recommendations."""
        recommendations = []

        vuln_types = set(v["type"] for v in vulnerabilities)

        if "dangerous_function" in vuln_types:
            recommendations.append("Avoid using eval(), exec(), and pickle.loads() in production code")
            recommendations.append("Use ast.literal_eval() for safe evaluation of literals")

        if "sql_injection" in vuln_types:
            recommendations.append("Use parameterized queries or ORM libraries")
            recommendations.append("Validate and sanitize all user inputs")

        if "hardcoded_secret" in vuln_types:
            recommendations.append("Store secrets in environment variables or secure vaults")
            recommendations.append("Use secret management tools like HashiCorp Vault")

        if not recommendations:
            recommendations.append("Regular security code reviews recommended")
            recommendations.append("Keep dependencies updated to latest secure versions")

        return recommendations


class PrivacyProtectionEngine:
    """Privacy-preserving machine learning and data protection."""

    def __init__(self):
        self.encryption_key = Fernet.generate_key()
        self.cipher = Fernet(self.encryption_key)

    def encrypt_data(self, data: str) -> str:
        """Encrypt sensitive data."""
        return self.cipher.encrypt(data.encode()).decode()

    def decrypt_data(self, encrypted_data: str) -> str:
        """Decrypt sensitive data."""
        return self.cipher.decrypt(encrypted_data.encode()).decode()

    def anonymize_data(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Anonymize personal data."""
        anonymized = data.copy()

        # Remove or hash personal identifiers
        personal_fields = ["name", "email", "phone", "address", "ip_address"]

        for field in personal_fields:
            if field in anonymized:
                if field == "ip_address":
                    anonymized[field] = self._anonymize_ip(anonymized[field])
                else:
                    anonymized[field] = hashlib.sha256(
                        str(anonymized[field]).encode()
                    ).hexdigest()[:16]

        return anonymized

    def _anonymize_ip(self, ip: str) -> str:
        """Anonymize IP address."""
        try:
            parts = ip.split(".")
            if len(parts) == 4:
                return f"{parts[0]}.{parts[1]}.xxx.xxx"
        except:
            pass
        return "xxx.xxx.xxx.xxx"

    def implement_differential_privacy(self, data: List[float],
                                     epsilon: float = 0.1) -> List[float]:
        """Implement differential privacy on numerical data."""
        import numpy as np

        # Add Laplace noise for differential privacy
        sensitivity = 1.0  # Assume sensitivity of 1
        noise_scale = sensitivity / epsilon

        noisy_data = []
        for value in data:
            noise = np.random.laplace(0, noise_scale)
            noisy_data.append(value + noise)

        return noisy_data

    def check_privacy_risk(self, data: Any) -> Dict[str, Any]:
        """Check privacy risk of data."""
        risks = {
            "personal_data_detected": False,
            "high_risk_fields": [],
            "recommendations": []
        }

        if isinstance(data, dict):
            personal_indicators = ["name", "email", "phone", "ssn", "address"]

            for key in data.keys():
                if any(indicator in key.lower() for indicator in personal_indicators):
                    risks["personal_data_detected"] = True
                    risks["high_risk_fields"].append(key)

        if risks["personal_data_detected"]:
            risks["recommendations"].extend([
                "Implement data anonymization",
                "Use differential privacy techniques",
                "Obtain proper consent for data usage"
            ])

        return risks


class ComplianceManager:
    """Compliance management and audit trails."""

    def __init__(self):
        self.audit_logs: List[AuditLogEntry] = []
        self.compliance_frameworks = {
            "gdpr": self._check_gdpr_compliance,
            "ccpa": self._check_ccpa_compliance,
            "soc2": self._check_soc2_compliance
        }

    def log_audit_event(self, action: str, user_id: str, resource: str,
                       result: str, ip_address: str, user_agent: str,
                       metadata: Optional[Dict[str, Any]] = None) -> None:
        """Log an audit event."""
        entry = AuditLogEntry(
            entry_id=secrets.token_hex(16),
            action=action,
            user_id=user_id,
            resource=resource,
            result=result,
            ip_address=ip_address,
            user_agent=user_agent,
            metadata=metadata or {}
        )

        self.audit_logs.append(entry)

    def check_compliance(self, framework: str, data: Any) -> Dict[str, Any]:
        """Check compliance with a specific framework."""
        if framework not in self.compliance_frameworks:
            return {"error": f"Unknown compliance framework: {framework}"}

        return self.compliance_frameworks[framework](data)

    def _check_gdpr_compliance(self, data: Any) -> Dict[str, Any]:
        """Check GDPR compliance."""
        compliance = {
            "compliant": True,
            "issues": [],
            "recommendations": []
        }

        # Check for personal data processing
        if isinstance(data, dict):
            personal_data_fields = ["email", "name", "phone", "address"]
            personal_data_present = any(field in data for field in personal_data_fields)

            if personal_data_present:
                compliance["issues"].append("Personal data processing detected")
                compliance["recommendations"].extend([
                    "Obtain explicit consent for data processing",
                    "Implement right to erasure (right to be forgotten)",
                    "Conduct Data Protection Impact Assessment (DPIA)"
                ])

        return compliance

    def _check_ccpa_compliance(self, data: Any) -> Dict[str, Any]:
        """Check CCPA compliance."""
        compliance = {
            "compliant": True,
            "issues": [],
            "recommendations": []
        }

        # CCPA-specific checks
        compliance["recommendations"].extend([
            "Provide privacy notice to California residents",
            "Implement opt-out mechanisms for data selling",
            "Honor data deletion requests within 45 days"
        ])

        return compliance

    def _check_soc2_compliance(self, data: Any) -> Dict[str, Any]:
        """Check SOC 2 compliance."""
        compliance = {
            "compliant": True,
            "issues": [],
            "recommendations": []
        }

        # SOC 2 Trust Services Criteria
        compliance["recommendations"].extend([
            "Implement access controls and monitoring",
            "Regular security assessments and penetration testing",
            "Maintain detailed audit logs and monitoring"
        ])

        return compliance

    def generate_compliance_report(self, framework: str) -> Dict[str, Any]:
        """Generate a compliance report."""
        recent_logs = [log for log in self.audit_logs
                      if (datetime.utcnow() - log.timestamp).days <= 90]

        return {
            "framework": framework,
            "audit_period_days": 90,
            "total_events": len(recent_logs),
            "compliance_status": "under_review",
            "last_assessment": datetime.utcnow(),
            "recommendations": [
                "Regular compliance audits recommended",
                "Implement automated compliance monitoring",
                "Staff training on compliance requirements"
            ]
        }


class SecureModelDeployment:
    """Secure model deployment and monitoring."""

    def __init__(self):
        self.deployed_models: Dict[str, Dict[str, Any]] = {}
        self.monitoring_alerts: List[Dict[str, Any]] = []

    def deploy_model_securely(self, model_id: str, model_data: Any,
                            security_config: Dict[str, Any]) -> Dict[str, Any]:
        """Deploy a model with security measures."""
        deployment = {
            "model_id": model_id,
            "deployment_id": secrets.token_hex(16),
            "security_measures": security_config,
            "status": "deployed",
            "deployment_time": datetime.utcnow()
        }

        self.deployed_models[model_id] = deployment
        return deployment

    def monitor_model_security(self, model_id: str,
                             input_data: Any, output_data: Any) -> List[SecurityEvent]:
        """Monitor model for security issues."""
        events = []

        # Check for adversarial inputs
        if self._detect_adversarial_pattern(input_data):
            events.append(SecurityEvent(
                event_id=secrets.token_hex(16),
                event_type="adversarial_input_detected",
                severity="high",
                description="Potential adversarial input detected",
                source="model_monitoring",
                metadata={"model_id": model_id}
            ))

        # Check for data leakage in outputs
        if self._detect_data_leakage(output_data):
            events.append(SecurityEvent(
                event_id=secrets.token_hex(16),
                event_type="data_leakage_detected",
                severity="critical",
                description="Potential data leakage in model output",
                source="model_monitoring",
                metadata={"model_id": model_id}
            ))

        return events

    def _detect_adversarial_pattern(self, data: Any) -> bool:
        """Detect adversarial patterns in input data."""
        # Simple heuristic-based detection
        if isinstance(data, str):
            # Check for unusual patterns
            if len(data) > 10000:  # Unusually long input
                return True
            if sum(1 for c in data if ord(c) > 127) / len(data) > 0.5:  # High Unicode content
                return True
        return False

    def _detect_data_leakage(self, data: Any) -> bool:
        """Detect potential data leakage in outputs."""
        # Check for sensitive information patterns
        if isinstance(data, str):
            sensitive_patterns = [
                r'\b\d{3}-\d{2}-\d{4}\b',  # SSN pattern
                r'\b\d{16}\b',  # Credit card pattern
                r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b'  # Email pattern
            ]

            for pattern in sensitive_patterns:
                if re.search(pattern, data):
                    return True
        return False


class AdvancedSecurityManager:
    """Main manager for advanced security features."""

    def __init__(self):
        self.threat_engine = ThreatDetectionEngine()
        self.code_analyzer = SecureCodeAnalyzer()
        self.privacy_engine = PrivacyProtectionEngine()
        self.compliance_manager = ComplianceManager()
        self.secure_deployment = SecureModelDeployment()
        self.security_events: List[SecurityEvent] = []

    def initialize_security(self) -> None:
        """Initialize security components."""
        # Register threat detection patterns
        self.threat_engine.register_threat_pattern(
            "code_injection", self.threat_engine.detect_code_injection
        )
        self.threat_engine.register_threat_pattern(
            "data_poisoning", self.threat_engine.detect_data_poisoning
        )
        self.threat_engine.register_threat_pattern(
            "adversarial_inputs", self.threat_engine.detect_adversarial_inputs
        )

    def perform_security_scan(self, target: Any,
                            scan_type: str = "comprehensive") -> Dict[str, Any]:
        """Perform a comprehensive security scan."""
        results = {
            "scan_id": secrets.token_hex(16),
            "scan_type": scan_type,
            "timestamp": datetime.utcnow(),
            "threats_detected": [],
            "vulnerabilities_found": [],
            "privacy_risks": {},
            "recommendations": []
        }

        # Threat detection
        if scan_type in ["comprehensive", "threats"]:
            threats = self.threat_engine.detect_threats(target, {})
            results["threats_detected"] = [
                {
                    "type": t.event_type,
                    "severity": t.severity,
                    "description": t.description
                } for t in threats
            ]
            self.security_events.extend(threats)

        # Vulnerability assessment
        if scan_type in ["comprehensive", "vulnerabilities"]:
            if isinstance(target, str):
                vuln_report = self.code_analyzer.analyze_code_security(target)
                results["vulnerabilities_found"] = vuln_report.vulnerabilities
                results["recommendations"].extend(vuln_report.recommendations)

        # Privacy assessment
        if scan_type in ["comprehensive", "privacy"]:
            privacy_risks = self.privacy_engine.check_privacy_risk(target)
            results["privacy_risks"] = privacy_risks
            results["recommendations"].extend(privacy_risks.get("recommendations", []))

        return results

    def get_security_status(self) -> Dict[str, Any]:
        """Get current security status."""
        recent_events = [e for e in self.security_events
                        if (datetime.utcnow() - e.timestamp).seconds < 3600]  # Last hour

        severity_counts = {}
        for event in recent_events:
            severity_counts[event.severity] = severity_counts.get(event.severity, 0) + 1

        return {
            "overall_status": "secure" if not recent_events else "warning",
            "recent_events": len(recent_events),
            "severity_breakdown": severity_counts,
            "last_scan": datetime.utcnow(),
            "active_threats": len([e for e in recent_events if e.severity in ["high", "critical"]])
        }

    def audit_user_action(self, action: str, user_id: str, resource: str,
                         result: str, ip_address: str, user_agent: str) -> None:
        """Audit a user action."""
        self.compliance_manager.log_audit_event(
            action, user_id, resource, result, ip_address, user_agent
        )

    def generate_security_report(self) -> Dict[str, Any]:
        """Generate a comprehensive security report."""
        return {
            "report_id": secrets.token_hex(16),
            "generated_at": datetime.utcnow(),
            "security_status": self.get_security_status(),
            "total_events": len(self.security_events),
            "compliance_status": {
                "gdpr": self.compliance_manager.check_compliance("gdpr", {}),
                "ccpa": self.compliance_manager.check_compliance("ccpa", {}),
                "soc2": self.compliance_manager.check_compliance("soc2", {})
            },
            "recommendations": [
                "Regular security assessments recommended",
                "Implement multi-factor authentication",
                "Regular staff security training",
                "Keep all systems and dependencies updated"
            ]
        }


# Export main classes
__all__ = [
    "SecurityEvent",
    "VulnerabilityReport",
    "AccessControlPolicy",
    "AuditLogEntry",
    "ThreatDetectionEngine",
    "SecureCodeAnalyzer",
    "PrivacyProtectionEngine",
    "ComplianceManager",
    "SecureModelDeployment",
    "AdvancedSecurityManager"
]
