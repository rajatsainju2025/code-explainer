"""Enhanced security validation and code redaction for sensitive content."""

import re
import ast
import logging
from typing import List, Dict, Tuple, Any, Optional
from pathlib import Path

logger = logging.getLogger(__name__)


class SensitivePatterns:
    """Registry of sensitive patterns to detect and redact."""

    # Credentials and API keys
    CREDENTIALS = [
        r'password\s*=\s*["\'][^"\']+["\']',
        r'api_key\s*=\s*["\'][^"\']+["\']',
        r'secret\s*=\s*["\'][^"\']+["\']',
        r'token\s*=\s*["\'][^"\']+["\']',
        r'auth\s*=\s*["\'][^"\']+["\']',
        r'key\s*=\s*["\'][^"\']+["\']',
    ]

    # Personal identifiable information
    PII = [
        r'\b\d{3}-\d{2}-\d{4}\b',  # SSN
        r'\b\d{4}\s?\d{4}\s?\d{4}\s?\d{4}\b',  # Credit card
        r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b',  # Email
        r'\b\d{3}-\d{3}-\d{4}\b',  # Phone number
    ]

    # File paths that might contain sensitive info
    SENSITIVE_PATHS = [
        r'["\']/?etc/passwd["\']',
        r'["\']/?etc/shadow["\']',
        r'["\'][^"\']*\.pem["\']',
        r'["\'][^"\']*\.key["\']',
        r'["\'][^"\']*id_rsa["\']',
    ]

    # Database connection strings
    DATABASE_URLS = [
        r'["\'](?:mysql|postgresql|mongodb|redis)://[^"\']+["\']',
        r'["\'](?:sqlite:///)[^"\']+["\']',
    ]

    # Cloud service URLs and keys
    CLOUD_SERVICES = [
        r'["\'](?:https?://)?[^"\']*\.amazonaws\.com[^"\']*["\']',
        r'["\'](?:https?://)?[^"\']*\.azure\.com[^"\']*["\']',
        r'["\'](?:https?://)?[^"\']*\.googleapis\.com[^"\']*["\']',
        r'AWS_[A-Z_]+\s*=\s*["\'][^"\']+["\']',
        r'AZURE_[A-Z_]+\s*=\s*["\'][^"\']+["\']',
        r'GCP_[A-Z_]+\s*=\s*["\'][^"\']+["\']',
    ]


class CodeRedactor:
    """Redacts sensitive information from code while preserving functionality."""

    def __init__(self, patterns: Optional[SensitivePatterns] = None):
        """Initialize the redactor with sensitivity patterns."""
        self.patterns = patterns or SensitivePatterns()
        self.redaction_map: Dict[str, str] = {}
        self.redaction_counter = 0

    def detect_sensitive_content(self, code: str) -> List[Dict[str, Any]]:
        """Detect sensitive content in code and return details."""
        detections = []

        pattern_categories = {
            "credentials": self.patterns.CREDENTIALS,
            "pii": self.patterns.PII,
            "sensitive_paths": self.patterns.SENSITIVE_PATHS,
            "database_urls": self.patterns.DATABASE_URLS,
            "cloud_services": self.patterns.CLOUD_SERVICES,
        }

        for category, patterns in pattern_categories.items():
            for pattern in patterns:
                matches = re.finditer(pattern, code, re.IGNORECASE)
                for match in matches:
                    detections.append({
                        "category": category,
                        "pattern": pattern,
                        "match": match.group(),
                        "start": match.start(),
                        "end": match.end(),
                        "line": code[:match.start()].count('\n') + 1
                    })

        return detections

    def generate_redaction_placeholder(self, category: str, original: str) -> str:
        """Generate a placeholder for redacted content."""
        self.redaction_counter += 1
        placeholder = f"[REDACTED_{category.upper()}_{self.redaction_counter}]"
        self.redaction_map[placeholder] = original
        return placeholder

    def redact_code(self, code: str, preserve_functionality: bool = True) -> Tuple[str, List[Dict[str, Any]]]:
        """Redact sensitive content from code.

        Args:
            code: Source code to redact
            preserve_functionality: Whether to preserve code functionality

        Returns:
            Tuple of (redacted_code, detections)
        """
        detections = self.detect_sensitive_content(code)
        redacted_code = code

        # Sort detections by position (reverse order to maintain indices)
        sorted_detections = sorted(detections, key=lambda x: x["start"], reverse=True)

        for detection in sorted_detections:
            start, end = detection["start"], detection["end"]
            original = detection["match"]
            category = detection["category"]

            if preserve_functionality:
                # Generate meaningful placeholder based on context
                placeholder = self._generate_functional_placeholder(original, category)
            else:
                placeholder = self.generate_redaction_placeholder(category, original)

            redacted_code = redacted_code[:start] + placeholder + redacted_code[end:]
            detection["redacted_with"] = placeholder

        return redacted_code, detections

    def _generate_functional_placeholder(self, original: str, category: str) -> str:
        """Generate a placeholder that maintains code functionality."""
        if category == "credentials":
            if "password" in original.lower():
                return '"[REDACTED_PASSWORD]"'
            elif "api_key" in original.lower():
                return '"[REDACTED_API_KEY]"'
            elif "token" in original.lower():
                return '"[REDACTED_TOKEN]"'
            else:
                return '"[REDACTED_CREDENTIAL]"'

        elif category == "pii":
            if "@" in original:  # Email
                return '"user@example.com"'
            elif re.match(r'\b\d{3}-\d{2}-\d{4}\b', original):  # SSN
                return '"XXX-XX-XXXX"'
            elif re.match(r'\b\d{4}\s?\d{4}\s?\d{4}\s?\d{4}\b', original):  # Credit card
                return '"XXXX-XXXX-XXXX-XXXX"'
            elif re.match(r'\b\d{3}-\d{3}-\d{4}\b', original):  # Phone
                return '"XXX-XXX-XXXX"'
            else:
                return '"[REDACTED_PII]"'

        elif category == "database_urls":
            return '"[REDACTED_DATABASE_URL]"'

        elif category == "cloud_services":
            return '"[REDACTED_CLOUD_SERVICE]"'

        elif category == "sensitive_paths":
            return '"/path/to/file"'

        else:
            return f'"[REDACTED_{category.upper()}]"'

    def get_redaction_summary(self) -> Dict[str, Any]:
        """Get summary of redactions performed."""
        # Get categories from stored detections (if we want to track them)
        # For now, return a simple summary
        return {
            "total_redactions": self.redaction_counter,
            "redaction_map_size": len(self.redaction_map)
        }


class EnhancedCodeSecurityValidator:
    """Enhanced security validator with redaction capabilities."""

    def __init__(self):
        """Initialize the enhanced security validator."""
        self.redactor = CodeRedactor()
        self.dangerous_imports = {
            'os', 'sys', 'subprocess', 'shutil', 'glob', 'pickle',
            'marshal', 'ctypes', 'importlib', '__import__', 'eval',
            'exec', 'compile', 'open'
        }
        self.dangerous_functions = {
            'eval', 'exec', 'compile', '__import__', 'getattr', 'setattr',
            'delattr', 'globals', 'locals', 'vars', 'dir', 'help'
        }
        self.dangerous_attributes = {
            '__class__', '__bases__', '__subclasses__', '__mro__',
            '__globals__', '__code__', '__dict__', '__doc__'
        }

    def validate_code_with_redaction(
        self,
        code: str,
        redact_sensitive: bool = True,
        preserve_functionality: bool = True
    ) -> Dict[str, Any]:
        """Validate code and optionally redact sensitive content.

        Args:
            code: Source code to validate
            redact_sensitive: Whether to redact sensitive content
            preserve_functionality: Whether to preserve functionality during redaction

        Returns:
            Dictionary with validation results and redacted code
        """
        result = {
            "original_code": code,
            "redacted_code": code,
            "security_issues": [],
            "sensitive_detections": [],
            "redaction_summary": {},
            "is_safe": True,
            "redaction_performed": False
        }

        # Detect and redact sensitive content
        if redact_sensitive:
            redacted_code, detections = self.redactor.redact_code(
                code, preserve_functionality=preserve_functionality
            )
            result["redacted_code"] = redacted_code
            result["sensitive_detections"] = detections
            result["redaction_summary"] = self.redactor.get_redaction_summary()
            result["redaction_performed"] = len(detections) > 0

            # Use redacted code for security validation
            code_to_validate = redacted_code
        else:
            code_to_validate = code

        # Perform security validation on (possibly redacted) code
        security_result = self._validate_security(code_to_validate)
        result["security_issues"] = security_result["issues"]
        result["is_safe"] = security_result["is_safe"]

        return result

    def _validate_security(self, code: str) -> Dict[str, Any]:
        """Perform security validation on code."""
        issues = []

        try:
            # Parse AST to check for dangerous patterns
            tree = ast.parse(code)

            for node in ast.walk(tree):
                # Check dangerous imports
                if isinstance(node, ast.Import):
                    for alias in node.names:
                        if alias.name in self.dangerous_imports:
                            issues.append(f"Dangerous import: {alias.name}")

                elif isinstance(node, ast.ImportFrom):
                    if node.module in self.dangerous_imports:
                        issues.append(f"Dangerous import from: {node.module}")

                # Check dangerous function calls
                elif isinstance(node, ast.Call):
                    if isinstance(node.func, ast.Name):
                        if node.func.id in self.dangerous_functions:
                            issues.append(f"Dangerous function call: {node.func.id}")

                # Check dangerous attribute access
                elif isinstance(node, ast.Attribute):
                    if node.attr in self.dangerous_attributes:
                        issues.append(f"Dangerous attribute access: {node.attr}")

        except SyntaxError as e:
            issues.append(f"Syntax error: {e}")
        except Exception as e:
            issues.append(f"Validation error: {e}")

        # Check for additional string-based patterns
        string_issues = self._check_string_patterns(code)
        issues.extend(string_issues)

        return {
            "is_safe": len(issues) == 0,
            "issues": issues
        }

    def _check_string_patterns(self, code: str) -> List[str]:
        """Check for dangerous patterns in code strings."""
        issues = []

        dangerous_patterns = [
            (r'rm\s+-rf', "Dangerous file deletion command"),
            (r'dd\s+if=', "Dangerous disk operation"),
            (r'chmod\s+777', "Dangerous permission change"),
            (r'sudo\s+', "Privilege escalation attempt"),
            (r'curl.*\|\s*sh', "Dangerous pipe to shell"),
            (r'wget.*\|\s*sh', "Dangerous pipe to shell"),
        ]

        for pattern, message in dangerous_patterns:
            if re.search(pattern, code, re.IGNORECASE):
                issues.append(message)

        return issues


def create_secure_code_processor(
    redact_sensitive: bool = True,
    preserve_functionality: bool = True
) -> EnhancedCodeSecurityValidator:
    """Create a configured secure code processor."""
    processor = EnhancedCodeSecurityValidator()
    return processor


# Example usage and testing
def main():
    """Example usage of the security validation and redaction system."""

    # Example code with sensitive content
    test_code = '''
import os
import requests

# Database configuration
DATABASE_URL = "postgresql://user:password123@localhost:5432/mydb"
API_KEY = "sk-1234567890abcdef"

# Personal information
email = "john.doe@company.com"
ssn = "123-45-6789"

def connect_to_api():
    headers = {
        "Authorization": f"Bearer {API_KEY}",
        "X-API-Key": "another-secret-key"
    }

    response = requests.get(
        "https://api.service.com/data",
        headers=headers
    )
    return response.json()

def dangerous_operation():
    # This is potentially dangerous
    os.system("rm -rf /tmp/*")

# More sensitive data
AWS_SECRET_KEY = "AKIAIOSFODNN7EXAMPLE"
'''

    # Create validator
    validator = EnhancedCodeSecurityValidator()

    # Validate with redaction
    result = validator.validate_code_with_redaction(
        test_code,
        redact_sensitive=True,
        preserve_functionality=True
    )

    print("=== Security Validation with Redaction ===")
    print(f"Redaction performed: {result['redaction_performed']}")
    print(f"Sensitive detections: {len(result['sensitive_detections'])}")
    print(f"Security issues: {len(result['security_issues'])}")
    print(f"Is safe: {result['is_safe']}")

    print("\n=== Redacted Code ===")
    print(result['redacted_code'])

    print("\n=== Security Issues ===")
    for issue in result['security_issues']:
        print(f"- {issue}")

    print("\n=== Sensitive Detections ===")
    for detection in result['sensitive_detections']:
        print(f"- {detection['category']}: {detection['match']} -> {detection.get('redacted_with', 'N/A')}")


if __name__ == "__main__":
    main()
