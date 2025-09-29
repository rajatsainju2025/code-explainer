"""Container-based sandboxing for secure code execution."""

import os
import tempfile
import subprocess
import logging
import json
import time
from typing import Dict, Any, Optional, List
from pathlib import Path
import shutil

logger = logging.getLogger(__name__)


class ContainerSandbox:
    """Container-based sandbox for secure code execution."""

    def __init__(
        self,
        image: str = "python:3.11-slim",
        timeout: int = 30,
        memory_limit: str = "128m",
        cpu_limit: str = "0.5",
        network_disabled: bool = True
    ):
        """Initialize container sandbox.

        Args:
            image: Docker image to use
            timeout: Execution timeout in seconds
            memory_limit: Memory limit (e.g., "128m", "1g")
            cpu_limit: CPU limit (e.g., "0.5", "1.0")
            network_disabled: Whether to disable network access
        """
        self.image = image
        self.timeout = timeout
        self.memory_limit = memory_limit
        self.cpu_limit = cpu_limit
        self.network_disabled = network_disabled
        self._check_docker()

    def _check_docker(self) -> None:
        """Check if Docker is available."""
        try:
            subprocess.run(
                ["docker", "--version"],
                capture_output=True,
                check=True,
                timeout=5
            )
        except (subprocess.CalledProcessError, subprocess.TimeoutExpired, FileNotFoundError):
            raise RuntimeError("Docker is not available. Please install Docker to use container sandboxing.")

    def execute_code(
        self,
        code: str,
        input_data: str = "",
        allowed_imports: Optional[List[str]] = None,
        capture_output: bool = True
    ) -> Dict[str, Any]:
        """Execute code in a secure container.

        Args:
            code: Python code to execute
            input_data: Input data for the code
            allowed_imports: List of allowed import modules
            capture_output: Whether to capture stdout/stderr

        Returns:
            Execution result with output, error, and metadata
        """
        # Validate imports
        if allowed_imports is not None:
            import_violations = self._check_imports(code, allowed_imports)
            if import_violations:
                return {
                    "success": False,
                    "output": "",
                    "error": f"Forbidden imports detected: {import_violations}",
                    "execution_time": 0,
                    "memory_usage": 0
                }

        # Create temporary directory for code execution
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)

            # Write code to file
            code_file = temp_path / "code.py"
            code_file.write_text(code)

            # Write input data if provided
            input_file = temp_path / "input.txt"
            if input_data:
                input_file.write_text(input_data)

            # Build Docker command
            docker_cmd = self._build_docker_command(temp_path, capture_output)

            # Execute in container
            start_time = time.time()
            try:
                result = subprocess.run(
                    docker_cmd,
                    capture_output=capture_output,
                    text=True,
                    timeout=self.timeout
                )
                execution_time = time.time() - start_time

                return {
                    "success": result.returncode == 0,
                    "output": result.stdout if capture_output else "",
                    "error": result.stderr if capture_output else "",
                    "execution_time": execution_time,
                    "return_code": result.returncode
                }

            except subprocess.TimeoutExpired:
                return {
                    "success": False,
                    "output": "",
                    "error": f"Execution timed out after {self.timeout} seconds",
                    "execution_time": self.timeout,
                    "return_code": -1
                }
            except Exception as e:
                return {
                    "success": False,
                    "output": "",
                    "error": f"Container execution failed: {e}",
                    "execution_time": time.time() - start_time,
                    "return_code": -1
                }

    def _build_docker_command(self, temp_path: Path, capture_output: bool) -> List[str]:
        """Build Docker command for code execution."""
        cmd = [
            "docker", "run",
            "--rm",  # Remove container after execution
            "--read-only",  # Read-only filesystem
            "--tmpfs", "/tmp:rw,noexec,nosuid,size=100m",  # Writable tmp with restrictions
            "--memory", self.memory_limit,
            "--cpus", self.cpu_limit,
            "--security-opt", "no-new-privileges",  # Prevent privilege escalation
            "--cap-drop", "ALL",  # Drop all capabilities
            "--user", "1000:1000",  # Run as non-root user
            "-v", f"{temp_path}:/workspace:ro",  # Mount code directory read-only
            "-w", "/workspace",  # Set working directory
        ]

        if self.network_disabled:
            cmd.extend(["--network", "none"])

        # Add image and command
        cmd.extend([
            self.image,
            "python", "-c", """
import sys
import os
import resource

# Set resource limits
try:
    # Limit virtual memory to 256MB
    resource.setrlimit(resource.RLIMIT_AS, (256 * 1024 * 1024, 256 * 1024 * 1024))
    # Limit CPU time to 30 seconds
    resource.setrlimit(resource.RLIMIT_CPU, (30, 30))
    # Limit number of processes
    resource.setrlimit(resource.RLIMIT_NPROC, (10, 10))
except Exception:
    pass  # Some limits might not be available

# Read and execute code
try:
    with open('code.py', 'r') as f:
        code = f.read()

    # Set up input if available
    if os.path.exists('input.txt'):
        with open('input.txt', 'r') as f:
            input_data = f.read()
        sys.stdin = __import__('io').StringIO(input_data)

    exec(code)
except Exception as e:
    print(f"Execution error: {e}", file=sys.stderr)
    sys.exit(1)
"""
        ])

        return cmd

    def _check_imports(self, code: str, allowed_imports: List[str]) -> List[str]:
        """Check for forbidden imports in code."""
        import ast

        violations = []

        try:
            tree = ast.parse(code)

            for node in ast.walk(tree):
                if isinstance(node, ast.Import):
                    for alias in node.names:
                        if alias.name not in allowed_imports:
                            violations.append(alias.name)
                elif isinstance(node, ast.ImportFrom):
                    if node.module and node.module not in allowed_imports:
                        violations.append(node.module)
        except SyntaxError:
            # If code has syntax errors, let the container handle it
            pass

        return violations

    def test_security(self) -> Dict[str, Any]:
        """Test sandbox security with various attack vectors."""
        test_cases = [
            {
                "name": "File system access",
                "code": "import os; print(os.listdir('/'))",
                "should_fail": True
            },
            {
                "name": "Network access",
                "code": "import urllib.request; urllib.request.urlopen('http://example.com')",
                "should_fail": True
            },
            {
                "name": "Process spawning",
                "code": "import subprocess; subprocess.run(['ls', '/'])",
                "should_fail": True
            },
            {
                "name": "Memory bomb",
                "code": "x = 'a' * (1024 * 1024 * 1024)",  # 1GB string
                "should_fail": True
            },
            {
                "name": "CPU bomb",
                "code": "while True: pass",
                "should_fail": True
            },
            {
                "name": "Safe math",
                "code": "print(2 + 2)",
                "should_fail": False
            }
        ]

        results = {}

        for test in test_cases:
            logger.info(f"Testing: {test['name']}")
            result = self.execute_code(test["code"])

            # Check if result matches expectation
            if test["should_fail"]:
                passed = not result["success"] or "error" in result["error"].lower()
            else:
                passed = result["success"]

            results[test["name"]] = {
                "passed": passed,
                "success": result["success"],
                "output": result["output"],
                "error": result["error"],
                "execution_time": result["execution_time"]
            }

        return results


class EnhancedSecurityValidator:
    """Enhanced security validation with container support."""

    def __init__(self, use_container: bool = True):
        """Initialize security validator.

        Args:
            use_container: Whether to use container sandboxing
        """
        self.use_container = use_container
        if use_container:
            try:
                self.sandbox = ContainerSandbox()
            except RuntimeError as e:
                logger.warning(f"Container sandboxing not available: {e}")
                self.use_container = False
                self.sandbox = None
        else:
            self.sandbox = None

    def validate_and_execute(
        self,
        code: str,
        allowed_imports: Optional[List[str]] = None,
        safe_builtins_only: bool = True
    ) -> Dict[str, Any]:
        """Validate and execute code with enhanced security.

        Args:
            code: Code to validate and execute
            allowed_imports: List of allowed imports
            safe_builtins_only: Whether to restrict to safe builtins

        Returns:
            Validation and execution results
        """
        # First, run static analysis
        static_result = self._static_security_check(code)

        if not static_result["safe"]:
            return {
                "validation_passed": False,
                "execution_result": None,
                "security_issues": static_result["issues"],
                "recommendation": "Code failed static security analysis"
            }

        # If container sandboxing is available, use it
        if self.use_container and self.sandbox:
            execution_result = self.sandbox.execute_code(
                code,
                allowed_imports=allowed_imports
            )

            return {
                "validation_passed": True,
                "execution_result": execution_result,
                "security_issues": [],
                "recommendation": "Code executed in secure container"
            }

        # Fallback to restricted execution
        return self._restricted_execution(code, safe_builtins_only)

    def _static_security_check(self, code: str) -> Dict[str, Any]:
        """Perform static security analysis."""
        issues = []

        # Check for dangerous patterns
        dangerous_patterns = [
            "eval(",
            "exec(",
            "__import__",
            "compile(",
            "open(",
            "file(",
            "input(",
            "raw_input(",
            "subprocess",
            "os.system",
            "os.popen",
            "commands.",
        ]

        for pattern in dangerous_patterns:
            if pattern in code:
                issues.append(f"Potentially dangerous pattern: {pattern}")

        # AST-based analysis
        try:
            import ast
            tree = ast.parse(code)

            for node in ast.walk(tree):
                if isinstance(node, ast.Import):
                    for alias in node.names:
                        if alias.name in ["os", "sys", "subprocess", "socket"]:
                            issues.append(f"Potentially dangerous import: {alias.name}")
                elif isinstance(node, ast.ImportFrom):
                    if node.module in ["os", "sys", "subprocess", "socket"]:
                        issues.append(f"Potentially dangerous import: {node.module}")
        except SyntaxError:
            issues.append("Code contains syntax errors")

        return {
            "safe": len(issues) == 0,
            "issues": issues
        }

    def _restricted_execution(self, code: str, safe_builtins_only: bool) -> Dict[str, Any]:
        """Execute code with restricted environment."""
        # Import at function level to ensure availability
        import io
        import sys

        # This is a simplified version - for production use container sandboxing
        safe_builtins = {
            'abs', 'all', 'any', 'bin', 'bool', 'chr', 'dict', 'dir',
            'divmod', 'enumerate', 'filter', 'float', 'format', 'frozenset',
            'getattr', 'hasattr', 'hash', 'hex', 'id', 'int', 'isinstance',
            'issubclass', 'iter', 'len', 'list', 'map', 'max', 'min',
            'next', 'oct', 'ord', 'pow', 'print', 'range', 'repr',
            'reversed', 'round', 'set', 'slice', 'sorted', 'str', 'sum',
            'tuple', 'type', 'zip'
        }

        restricted_globals = {}
        if safe_builtins_only:
            restricted_globals['__builtins__'] = {
                name: getattr(__builtins__, name) for name in safe_builtins
                if hasattr(__builtins__, name)
            }

        # Save original stdout/stderr
        old_stdout = sys.stdout
        old_stderr = sys.stderr
        stdout_capture = io.StringIO()
        stderr_capture = io.StringIO()

        # Initialize timing
        start_time = time.time()

        try:
            # Capture output
            sys.stdout = stdout_capture
            sys.stderr = stderr_capture

            exec(code, restricted_globals)
            execution_time = time.time() - start_time

            return {
                "validation_passed": True,
                "execution_result": {
                    "success": True,
                    "output": stdout_capture.getvalue(),
                    "error": stderr_capture.getvalue(),
                    "execution_time": execution_time
                },
                "security_issues": [],
                "recommendation": "Code executed with restricted builtins"
            }

        except Exception as e:
            execution_time = time.time() - start_time
            return {
                "validation_passed": True,
                "execution_result": {
                    "success": False,
                    "output": stdout_capture.getvalue(),
                    "error": str(e),
                    "execution_time": execution_time
                },
                "security_issues": [],
                "recommendation": "Code execution failed"
            }
        finally:
            # Restore stdout/stderr
            sys.stdout = old_stdout
            sys.stderr = old_stderr


def main():
    """Test container sandboxing."""
    try:
        sandbox = ContainerSandbox()

        # Test basic execution
        result = sandbox.execute_code("print('Hello, secure world!')")
        print("Basic execution:", result)

        # Test security
        security_results = sandbox.test_security()
        print("\nSecurity test results:")
        for test_name, result in security_results.items():
            status = "PASS" if result["passed"] else "FAIL"
            print(f"  {test_name}: {status}")

    except RuntimeError as e:
        print(f"Container sandboxing not available: {e}")
        print("Install Docker to enable container-based sandboxing.")


if __name__ == "__main__":
    main()
