"""
Security utilities for code validation and safe execution.

Enhanced with:
- Advanced input validation
- Rate limiting with sliding window
- Content filtering for sensitive patterns
- Comprehensive audit logging
- Configurable security policies
"""

from typing import Dict, Any, List, Tuple, Optional, Set
import ast
import re
import time
import hashlib
import json
import logging
from pathlib import Path
from datetime import datetime, timedelta
from collections import defaultdict, deque

logger = logging.getLogger(__name__)

# Pre-compile dangerous patterns for efficient detection
DANGEROUS_IMPORTS = frozenset({
    'os', 'subprocess', 'sys', 'socket', '__import__',
    'exec', 'eval', 'compile', 'open', 'input'
})

DANGEROUS_FUNCTIONS = frozenset({
    'system', 'exec', 'eval', 'compile', 'getattr',
    'setattr', 'delattr', '__import__', 'open'
})

DANGEROUS_PATTERN = re.compile(
    r'\b(os|subprocess|__import__|exec|eval|compile)\s*\(',
    re.IGNORECASE | re.MULTILINE
)


# Cache time.time for micro-optimization
_time_time = time.time


class RateLimiter:
    """Rate limiter using sliding window algorithm."""

    __slots__ = ("requests_per_minute", "requests", "cleanup_interval", "last_cleanup")

    def __init__(self, requests_per_minute: int = 60):
        self.requests_per_minute = requests_per_minute
        self.requests: Dict[str, deque] = defaultdict(deque)
        self.cleanup_interval = 60  # seconds
        self.last_cleanup = _time_time()

    def is_allowed(self, client_id: str) -> Tuple[bool, Optional[str]]:
        """Check if request is allowed for client."""
        current_time = _time_time()
        
        # Periodic cleanup
        if current_time - self.last_cleanup > self.cleanup_interval:
            self._cleanup_old_requests()
        
        # Get client's request history
        client_requests = self.requests[client_id]
        
        # Remove requests older than 1 minute
        cutoff_time = current_time - 60
        while client_requests and client_requests[0] < cutoff_time:
            client_requests.popleft()
        
        # Check if limit exceeded
        if len(client_requests) >= self.requests_per_minute:
            wait_time = int(60 - (current_time - client_requests[0]))
            return False, f"Rate limit exceeded. Try again in {wait_time} seconds."
        
        # Add current request
        client_requests.append(current_time)
        return True, None

    def _cleanup_old_requests(self):
        """Remove old client entries to prevent memory leak."""
        current_time = _time_time()
        cutoff_time = current_time - 300  # Keep 5 minutes of history
        
        clients_to_remove = []
        for client_id, requests in self.requests.items():
            # Remove old requests
            while requests and requests[0] < cutoff_time:
                requests.popleft()
            
            # Mark empty clients for removal
            if not requests:
                clients_to_remove.append(client_id)
        
        # Remove empty clients
        for client_id in clients_to_remove:
            del self.requests[client_id]
        
        self.last_cleanup = current_time
        logger.debug("Cleaned up rate limiter, removed %d clients", len(clients_to_remove))


class AuditLogger:
    """Security audit logger with structured events."""

    __slots__ = ("enabled", "log_path")

    def __init__(self, log_path: Optional[str] = None, enabled: bool = True):
        self.enabled = enabled
        self.log_path = Path(log_path) if log_path else None
        
        if self.enabled and self.log_path:
            self.log_path.parent.mkdir(parents=True, exist_ok=True)

    def log_event(self, event_type: str, details: Dict[str, Any], severity: str = "INFO"):
        """Log a security event."""
        if not self.enabled:
            return
        
        event = {
            "timestamp": datetime.utcnow().isoformat(),
            "event_type": event_type,
            "severity": severity,
            "details": details
        }
        
        # Log to file
        if self.log_path:
            try:
                with open(self.log_path, "a") as f:
                    f.write(json.dumps(event) + "\n")
            except Exception as e:
                logger.error("Failed to write audit log: %s", e)
        
        # Also log to standard logger
        log_method = getattr(logger, severity.lower(), logger.info)
        log_method("[AUDIT] %s: %s", event_type, json.dumps(details))

    def log_validation_failure(self, code_hash: str, issues: List[str], client_id: Optional[str] = None):
        """Log security validation failure."""
        self.log_event(
            "validation_failure",
            {
                "code_hash": code_hash,
                "issues": issues,
                "client_id": client_id
            },
            severity="WARNING"
        )

    def log_rate_limit_exceeded(self, client_id: str):
        """Log rate limit exceeded event."""
        self.log_event(
            "rate_limit_exceeded",
            {"client_id": client_id},
            severity="WARNING"
        )

    def log_suspicious_pattern(self, code_hash: str, pattern: str, client_id: Optional[str] = None):
        """Log detection of suspicious pattern."""
        self.log_event(
            "suspicious_pattern",
            {
                "code_hash": code_hash,
                "pattern": pattern,
                "client_id": client_id
            },
            severity="WARNING"
        )


class ContentFilter:
    """Filter for detecting sensitive content in code with compiled regex patterns."""
    
    __slots__ = ('sensitive_pattern_strings', 'compiled_patterns')

    def __init__(self):
        # Define pattern strings
        self.sensitive_pattern_strings = {
            "credentials": [
                r'password\s*=\s*["\'][^"\']*["\']',
                r'api_key\s*=\s*["\'][^"\']*["\']',
                r'token\s*=\s*["\'][^"\']*["\']',
                r'secret\s*=\s*["\'][^"\']*["\']',
                r'aws_access_key',
                r'private_key',
            ],
            "suspicious_commands": [
                r'rm\s+-rf',
                r'del\s+/s',
                r'format\s+c:',
                r'dd\s+if=',
                r'mkfs\.',
            ],
            "network_operations": [
                r'socket\.connect',
                r'requests\.get',
                r'urllib\.urlopen',
                r'http\.client',
            ]
        }
        
        # Compile all patterns once for better performance
        self.compiled_patterns = {
            category: [re.compile(pattern, re.IGNORECASE) for pattern in patterns]
            for category, patterns in self.sensitive_pattern_strings.items()
        }

    def scan(self, code: str) -> Dict[str, List[str]]:
        """Scan code for sensitive content using pre-compiled patterns."""
        findings = {}
        
        for category, patterns in self.compiled_patterns.items():
            matches = [
                pattern.pattern
                for pattern in patterns
                if pattern.search(code)
            ]
            
            if matches:
                findings[category] = matches
        
        return findings


class InputValidator:
    """Advanced input validation."""
    
    __slots__ = ('max_length', 'allowed_imports')

    def __init__(self, max_length: int = 10000, allowed_imports: Optional[Set[str]] = None):
        self.max_length = max_length
        self.allowed_imports = allowed_imports or {
            'typing', 'dataclasses', 'enum', 'collections',
            'itertools', 'functools', 'math', 'random'
        }

    def validate(self, code: str) -> Tuple[bool, List[str]]:
        """Validate input code."""
        issues = []
        
        # Length check
        if len(code) > self.max_length:
            issues.append(f"Code exceeds maximum length of {self.max_length} characters")
        
        # Empty check
        if not code or not code.strip():
            issues.append("Empty code provided")
        
        # Basic syntax check
        try:
            ast.parse(code)
        except SyntaxError as e:
            issues.append(f"Syntax error: {e}")
            return False, issues
        
        # Import validation
        try:
            tree = ast.parse(code)
            for node in ast.walk(tree):
                if isinstance(node, ast.Import):
                    for alias in node.names:
                        if alias.name.split('.')[0] not in self.allowed_imports:
                            issues.append(f"Disallowed import: {alias.name}")
        except Exception as e:
            issues.append(f"Import validation error: {e}")
        
        return len(issues) == 0, issues


class CodeSecurityValidator:
    """Enhanced code security validator with multiple layers."""
    
    # Pre-compiled dangerous patterns for faster matching
    _DANGEROUS_PATTERNS_COMPILED = None
    _DANGEROUS_PATTERN_MESSAGES = {
        'import_os': ("Potentially dangerous system import detected", re.compile(r'import\s+os', re.IGNORECASE)),
        'import_subprocess': ("Potentially dangerous system import detected", re.compile(r'import\s+subprocess', re.IGNORECASE)),
        'import_sys': ("Potentially dangerous system import detected", re.compile(r'import\s+sys', re.IGNORECASE)),
        'os_system': ("Potentially dangerous system call detected", re.compile(r'os\.system', re.IGNORECASE)),
        'subprocess_call': ("Potentially dangerous subprocess call detected", re.compile(r'subprocess\.', re.IGNORECASE)),
        'eval': ("Potentially dangerous code execution detected", re.compile(r'eval\s*\(')),
        'exec': ("Potentially dangerous code execution detected", re.compile(r'exec\s*\(')),
        'dunder_import': ("Potentially dangerous dynamic import detected", re.compile(r'__import__')),
        'open_file': ("Potentially dangerous file operation detected", re.compile(r'open\s*\(')),
        'file_builtin': ("Potentially dangerous file operation detected", re.compile(r'file\s*\(')),
        'import_requests': ("Potentially dangerous network operation detected", re.compile(r'import\s+requests', re.IGNORECASE)),
        'import_urllib': ("Potentially dangerous network operation detected", re.compile(r'import\s+urllib', re.IGNORECASE)),
        'urllib_call': ("Potentially dangerous network operation detected", re.compile(r'urllib\.', re.IGNORECASE)),
        'requests_call': ("Potentially dangerous network operation detected", re.compile(r'requests\.', re.IGNORECASE)),
        'socket_call': ("Potentially dangerous network operation detected", re.compile(r'socket\.', re.IGNORECASE)),
        'import_socket': ("Potentially dangerous network operation detected", re.compile(r'import\s+socket', re.IGNORECASE)),
    }

    def __init__(self, strict_mode: bool = False):
        self.strict_mode = strict_mode
        self.content_filter = ContentFilter()
        self.input_validator = InputValidator()

    def validate_code(self, code: str) -> Tuple[bool, List[str]]:
        """Comprehensive code validation with multiple layers."""
        all_issues = []
        
        # Layer 1: Input validation
        valid_input, input_issues = self.input_validator.validate(code)
        if not valid_input:
            all_issues.extend(input_issues)
            return False, all_issues
        
        # Layer 2: Pattern matching with pre-compiled patterns (faster)
        seen_messages = set()  # Avoid duplicate messages
        for pattern_name, (message, pattern) in self._DANGEROUS_PATTERN_MESSAGES.items():
            if pattern.search(code) and message not in seen_messages:
                all_issues.append(message)
                seen_messages.add(message)
        
        # Layer 3: AST analysis
        try:
            tree = ast.parse(code)
            ast_issues = self._analyze_ast(tree)
            all_issues.extend(ast_issues)
        except SyntaxError as e:
            all_issues.append(f"Syntax error: {e}")
        
        # Layer 4: Content filtering (always scan, warn in strict mode)
        sensitive_content = self.content_filter.scan(code)
        if sensitive_content and self.strict_mode:
            for category, patterns in sensitive_content.items():
                all_issues.append(f"Sensitive content detected ({category}): {len(patterns)} patterns")
        
        return len(all_issues) == 0, all_issues


    def _analyze_ast(self, tree: ast.AST) -> List[str]:
        """Analyze AST for security issues."""
        issues = []

        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    if alias.name in ['os', 'subprocess', 'sys']:
                        issues.append(f"Dangerous import: {alias.name}")
            elif isinstance(node, ast.Call):
                if isinstance(node.func, ast.Name):
                    if node.func.id in ['eval', 'exec', 'open']:
                        issues.append(f"Dangerous function call: {node.func.id}")

        return issues


class SafeCodeExecutor:
    """Enhanced safe code execution with timeout and sandboxing."""
    
    __slots__ = ('timeout', 'audit_logger')

    def __init__(self, timeout: int = 30, audit_logger: Optional[AuditLogger] = None):
        self.timeout = timeout
        self.audit_logger = audit_logger or AuditLogger(enabled=False)

    def execute_code(self, code: str, client_id: Optional[str] = None) -> Dict[str, Any]:
        """Execute code safely with validation and auditing."""
        # Generate code hash for auditing
        code_hash = hash_code(code)
        
        # First validate the code
        validator = CodeSecurityValidator()
        is_safe, issues = validator.validate_code(code)

        if not is_safe:
            self.audit_logger.log_validation_failure(code_hash, issues, client_id)
            return {
                "success": False,
                "error": "Security validation failed",
                "issues": issues,
                "code_hash": code_hash
            }

        import subprocess
        import tempfile
        import os

        try:
            # Write code to temporary file
            with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
                f.write(code)
                temp_file = f.name

            # Execute with timeout and capture output
            start_time = time.time()
            result = subprocess.run(
                ['python', temp_file],
                capture_output=True,
                text=True,
                timeout=self.timeout
            )
            execution_time = time.time() - start_time

            # Clean up
            os.unlink(temp_file)
            
            # Audit successful execution
            self.audit_logger.log_event(
                "code_execution",
                {
                    "code_hash": code_hash,
                    "execution_time": execution_time,
                    "success": result.returncode == 0,
                    "client_id": client_id
                }
            )

            return {
                "success": result.returncode == 0,
                "output": result.stdout,
                "error": result.stderr,
                "execution_time": execution_time,
                "code_hash": code_hash
            }

        except subprocess.TimeoutExpired:
            self.audit_logger.log_event(
                "execution_timeout",
                {"code_hash": code_hash, "timeout": self.timeout, "client_id": client_id},
                severity="WARNING"
            )
            return {
                "success": False,
                "error": f"Timeout: execution timed out after {self.timeout} seconds",
                "output": "",
                "execution_time": self.timeout,
                "code_hash": code_hash
            }
        except Exception as e:
            self.audit_logger.log_event(
                "execution_error",
                {"code_hash": code_hash, "error": str(e), "client_id": client_id},
                severity="ERROR"
            )
            return {
                "success": False,
                "error": str(e),
                "output": "",
                "execution_time": 0.1,
                "code_hash": code_hash
            }


def hash_code(code: str) -> str:
    """Generate a hash for code content."""
    return hashlib.sha256(code.encode('utf-8')).hexdigest()


def sanitize_code_for_display(code: str, max_length: int = 1000) -> str:
    """Sanitize code for safe display with enhanced pattern matching."""
    # Remove potentially dangerous patterns for display
    sanitized = code.replace('import os', '# import os (removed for security)')
    sanitized = sanitized.replace('import subprocess', '# import subprocess (removed for security)')
    sanitized = sanitized.replace('eval(', '# eval( (removed for security)')
    sanitized = sanitized.replace('exec(', '# exec( (removed for security)')

    # Mask potential secrets with improved patterns
    sanitized = re.sub(r'password\s*=\s*["\'][^"\']*["\']', 'password = "***"', sanitized, flags=re.IGNORECASE)
    sanitized = re.sub(r'api[_-]?key\s*=\s*["\'][^"\']*["\']', 'api_key = "***"', sanitized, flags=re.IGNORECASE)
    sanitized = re.sub(r'token\s*=\s*["\'][^"\']*["\']', 'token = "***"', sanitized, flags=re.IGNORECASE)
    sanitized = re.sub(r'secret\s*=\s*["\'][^"\']*["\']', 'secret = "***"', sanitized, flags=re.IGNORECASE)
    sanitized = re.sub(r'auth\s*=\s*["\'][^"\']*["\']', 'auth = "***"', sanitized, flags=re.IGNORECASE)

    # Truncate if too long
    if len(sanitized) > max_length:
        truncate_at = max_length - 3
        sanitized = sanitized[:truncate_at] + "..."

    return sanitized


class SecurityManager:
    """Central security manager coordinating all security components."""

    def __init__(self, 
                 rate_limit_rpm: int = 60,
                 strict_mode: bool = False,
                 audit_log_path: Optional[str] = None,
                 max_code_length: int = 10000):
        self.rate_limiter = RateLimiter(rate_limit_rpm)
        self.validator = CodeSecurityValidator(strict_mode)
        self.executor = SafeCodeExecutor(
            timeout=30,
            audit_logger=AuditLogger(audit_log_path, enabled=audit_log_path is not None)
        )
        self.content_filter = ContentFilter()
        self.max_code_length = max_code_length

    def check_rate_limit(self, client_id: str) -> Tuple[bool, Optional[str]]:
        """Check rate limit for client."""
        allowed, message = self.rate_limiter.is_allowed(client_id)
        if not allowed:
            self.executor.audit_logger.log_rate_limit_exceeded(client_id)
        return allowed, message

    def validate_code(self, code: str, client_id: Optional[str] = None) -> Tuple[bool, List[str]]:
        """Validate code with comprehensive checks."""
        code_hash = hash_code(code)
        
        is_valid, issues = self.validator.validate_code(code)
        
        if not is_valid and client_id:
            self.executor.audit_logger.log_validation_failure(code_hash, issues, client_id)
        
        return is_valid, issues

    def execute_code_safe(self, code: str, client_id: Optional[str] = None) -> Dict[str, Any]:
        """Execute code with full security checks."""
        # Check rate limit
        allowed, message = self.check_rate_limit(client_id or "anonymous")
        if not allowed:
            return {
                "success": False,
                "error": message,
                "rate_limited": True
            }
        
        # Execute with validation
        return self.executor.execute_code(code, client_id)

    def scan_for_sensitive_content(self, code: str) -> Dict[str, List[str]]:
        """Scan code for sensitive content."""
        return self.content_filter.scan(code)
