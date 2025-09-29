"""Continuous Integration and Development workflow automation."""

import os
import subprocess
import json
import time
import logging
from typing import Dict, List, Any, Optional, Union, Tuple
from pathlib import Path
from dataclasses import dataclass, field
from enum import Enum

logger = logging.getLogger(__name__)


class BuildStatus(Enum):
    """Build status enumeration."""
    PENDING = "pending"
    RUNNING = "running"
    SUCCESS = "success"
    FAILURE = "failure"
    CANCELLED = "cancelled"


class TestStatus(Enum):
    """Test status enumeration."""
    PENDING = "pending"
    RUNNING = "running"
    PASSED = "passed"
    FAILED = "failed"
    SKIPPED = "skipped"


@dataclass
class BuildStep:
    """Single build step definition."""
    name: str
    command: str
    description: str = ""
    working_dir: str = "."
    timeout_seconds: int = 300
    continue_on_error: bool = False
    environment: Dict[str, str] = field(default_factory=dict)
    artifacts: List[str] = field(default_factory=list)


@dataclass
class BuildResult:
    """Build step execution result."""
    step_name: str
    status: BuildStatus
    duration_seconds: float
    return_code: int = 0
    stdout: str = ""
    stderr: str = ""
    artifacts_collected: List[str] = field(default_factory=list)
    error_message: str = ""


@dataclass
class PipelineConfig:
    """CI/CD pipeline configuration."""
    name: str
    description: str = ""
    trigger_on_push: bool = True
    trigger_on_pr: bool = True
    python_versions: List[str] = field(default_factory=lambda: ["3.9", "3.10", "3.11"])
    os_matrix: List[str] = field(default_factory=lambda: ["ubuntu-latest", "windows-latest", "macos-latest"])
    build_steps: List[BuildStep] = field(default_factory=list)
    test_commands: List[str] = field(default_factory=list)
    artifacts_to_upload: List[str] = field(default_factory=list)
    deploy_steps: List[BuildStep] = field(default_factory=list)


class CIRunner:
    """Continuous Integration runner."""

    def __init__(self, config: PipelineConfig, workspace_dir: str = "."):
        """Initialize CI runner.

        Args:
            config: Pipeline configuration
            workspace_dir: Workspace directory
        """
        self.config = config
        self.workspace_dir = Path(workspace_dir).resolve()
        self.results: List[BuildResult] = []
        self.start_time: Optional[float] = None

    def run_command(
        self,
        command: str,
        working_dir: str = ".",
        timeout: int = 300,
        env: Optional[Dict[str, str]] = None
    ) -> Tuple[int, str, str]:
        """Run a shell command with timeout and error handling.

        Args:
            command: Command to run
            working_dir: Working directory
            timeout: Timeout in seconds
            env: Environment variables

        Returns:
            Tuple of (return_code, stdout, stderr)
        """
        full_env = os.environ.copy()
        if env:
            full_env.update(env)

        try:
            logger.info(f"Running command: {command}")

            process = subprocess.Popen(
                command,
                shell=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                cwd=working_dir,
                env=full_env
            )

            try:
                stdout, stderr = process.communicate(timeout=timeout)
                return process.returncode, stdout, stderr
            except subprocess.TimeoutExpired:
                process.kill()
                stdout, stderr = process.communicate()
                return -1, stdout, f"Command timed out after {timeout} seconds\n{stderr}"

        except Exception as e:
            logger.error(f"Command execution failed: {e}")
            return -1, "", str(e)

    def run_build_step(self, step: BuildStep) -> BuildResult:
        """Run a single build step.

        Args:
            step: Build step to execute

        Returns:
            Build result
        """
        start_time = time.time()

        logger.info(f"Executing build step: {step.name}")

        try:
            working_dir = os.path.join(self.workspace_dir, step.working_dir)

            return_code, stdout, stderr = self.run_command(
                step.command,
                working_dir=working_dir,
                timeout=step.timeout_seconds,
                env=step.environment
            )

            duration = time.time() - start_time

            # Collect artifacts
            artifacts_collected = []
            for artifact_pattern in step.artifacts:
                try:
                    import glob
                    matches = glob.glob(
                        os.path.join(working_dir, artifact_pattern),
                        recursive=True
                    )
                    artifacts_collected.extend(matches)
                except Exception as e:
                    logger.warning(f"Failed to collect artifact {artifact_pattern}: {e}")

            # Determine status
            if return_code == 0:
                status = BuildStatus.SUCCESS
                error_message = ""
            else:
                status = BuildStatus.FAILURE
                error_message = f"Command failed with code {return_code}"
                if stderr:
                    error_message += f": {stderr[:500]}"

            result = BuildResult(
                step_name=step.name,
                status=status,
                duration_seconds=duration,
                return_code=return_code,
                stdout=stdout,
                stderr=stderr,
                artifacts_collected=artifacts_collected,
                error_message=error_message
            )

            logger.info(f"Build step {step.name} completed: {status.value} ({duration:.1f}s)")
            return result

        except Exception as e:
            duration = time.time() - start_time
            error_message = f"Build step execution failed: {e}"
            logger.error(error_message)

            return BuildResult(
                step_name=step.name,
                status=BuildStatus.FAILURE,
                duration_seconds=duration,
                error_message=error_message
            )

    def run_tests(self) -> List[BuildResult]:
        """Run all configured test commands.

        Returns:
            List of test results
        """
        test_results = []

        for i, test_command in enumerate(self.config.test_commands):
            step = BuildStep(
                name=f"test_{i+1}",
                command=test_command,
                description=f"Test command: {test_command}",
                timeout_seconds=600  # Longer timeout for tests
            )

            result = self.run_build_step(step)
            test_results.append(result)

            # Stop on test failure unless configured otherwise
            if result.status == BuildStatus.FAILURE:
                logger.error(f"Test failed: {test_command}")
                break

        return test_results

    def run_full_pipeline(self) -> Dict[str, Any]:
        """Run the complete CI/CD pipeline.

        Returns:
            Pipeline execution summary
        """
        self.start_time = time.time()
        self.results = []

        logger.info(f"Starting CI pipeline: {self.config.name}")

        pipeline_status = BuildStatus.SUCCESS
        failed_step = None

        try:
            # Run build steps
            for step in self.config.build_steps:
                result = self.run_build_step(step)
                self.results.append(result)

                if result.status == BuildStatus.FAILURE:
                    pipeline_status = BuildStatus.FAILURE
                    failed_step = step.name

                    if not step.continue_on_error:
                        logger.error(f"Pipeline failed at step: {step.name}")
                        break

            # Run tests if build succeeded or continue_on_error is True
            if pipeline_status == BuildStatus.SUCCESS or any(
                step.continue_on_error for step in self.config.build_steps
            ):
                test_results = self.run_tests()
                self.results.extend(test_results)

                # Check if any tests failed
                if any(r.status == BuildStatus.FAILURE for r in test_results):
                    pipeline_status = BuildStatus.FAILURE
                    if not failed_step:
                        failed_step = "tests"

            # Run deployment steps if everything passed
            if pipeline_status == BuildStatus.SUCCESS and self.config.deploy_steps:
                for step in self.config.deploy_steps:
                    result = self.run_build_step(step)
                    self.results.append(result)

                    if result.status == BuildStatus.FAILURE:
                        pipeline_status = BuildStatus.FAILURE
                        failed_step = step.name
                        break

        except Exception as e:
            logger.error(f"Pipeline execution failed: {e}")
            pipeline_status = BuildStatus.FAILURE
            failed_step = "execution"

        total_duration = time.time() - self.start_time

        # Generate summary
        summary = {
            "pipeline_name": self.config.name,
            "status": pipeline_status.value,
            "total_duration_seconds": total_duration,
            "failed_step": failed_step,
            "total_steps": len(self.results),
            "successful_steps": sum(1 for r in self.results if r.status == BuildStatus.SUCCESS),
            "failed_steps": sum(1 for r in self.results if r.status == BuildStatus.FAILURE),
            "artifacts_collected": sum(len(r.artifacts_collected) for r in self.results),
            "step_results": [
                {
                    "name": r.step_name,
                    "status": r.status.value,
                    "duration": r.duration_seconds,
                    "return_code": r.return_code,
                    "error": r.error_message
                }
                for r in self.results
            ]
        }

        logger.info(f"Pipeline completed: {pipeline_status.value} ({total_duration:.1f}s)")
        return summary

    def generate_github_workflow(self, output_file: str = ".github/workflows/ci.yml") -> str:
        """Generate GitHub Actions workflow file.

        Args:
            output_file: Output file path

        Returns:
            Generated workflow YAML content
        """
        workflow_content = f"""name: {self.config.name}

on:
  push:
    branches: [ main, develop ]
  pull_request:
    branches: [ main ]

jobs:
  test:
    runs-on: ${{{{ matrix.os }}}}
    strategy:
      matrix:
        os: {self.config.os_matrix}
        python-version: {self.config.python_versions}

    steps:
    - uses: actions/checkout@v4

    - name: Set up Python ${{{{ matrix.python-version }}}}
      uses: actions/setup-python@v4
      with:
        python-version: ${{{{ matrix.python-version }}}}

    - name: Install dependencies
      run: |
        python -m pip install --upgrade pip
        pip install -r requirements.txt
        pip install -e .
"""

        # Add build steps
        for step in self.config.build_steps:
            workflow_content += f"""
    - name: {step.name}
      run: {step.command}
"""
            if step.working_dir != ".":
                workflow_content += f"      working-directory: {step.working_dir}\n"

            if step.environment:
                workflow_content += "      env:\n"
                for key, value in step.environment.items():
                    workflow_content += f"        {key}: {value}\n"

        # Add test commands
        for i, test_cmd in enumerate(self.config.test_commands):
            workflow_content += f"""
    - name: Run tests {i+1}
      run: {test_cmd}
"""

        # Add artifact upload
        if self.config.artifacts_to_upload:
            workflow_content += f"""
    - name: Upload artifacts
      uses: actions/upload-artifact@v3
      if: always()
      with:
        name: test-artifacts-${{{{ matrix.os }}}}-${{{{ matrix.python-version }}}}
        path: |
"""
            for artifact in self.config.artifacts_to_upload:
                workflow_content += f"          {artifact}\n"

        # Write to file
        try:
            output_path = Path(output_file)
            output_path.parent.mkdir(parents=True, exist_ok=True)

            with open(output_path, 'w') as f:
                f.write(workflow_content)

            logger.info(f"GitHub workflow written to {output_file}")
        except Exception as e:
            logger.error(f"Failed to write workflow file: {e}")

        return workflow_content


def create_default_pipeline() -> PipelineConfig:
    """Create a default CI/CD pipeline configuration.

    Returns:
        Default pipeline configuration
    """
    return PipelineConfig(
        name="Code Explainer CI/CD",
        description="Continuous integration and deployment for code explainer project",
        build_steps=[
            BuildStep(
                name="install_dependencies",
                command="pip install -r requirements.txt && pip install -e .",
                description="Install project dependencies"
            ),
            BuildStep(
                name="lint_code",
                command="python -m flake8 src tests --max-line-length=100",
                description="Lint code with flake8",
                continue_on_error=True
            ),
            BuildStep(
                name="type_check",
                command="python -m mypy src --ignore-missing-imports",
                description="Type checking with mypy",
                continue_on_error=True
            ),
            BuildStep(
                name="security_scan",
                command="python -m bandit -r src -f json -o bandit-report.json",
                description="Security scanning with bandit",
                continue_on_error=True,
                artifacts=["bandit-report.json"]
            )
        ],
        test_commands=[
            "python -m pytest tests/ -v --cov=src --cov-report=xml --cov-report=html",
            "python -m pytest tests/test_integration.py -v -m integration"
        ],
        artifacts_to_upload=[
            "coverage.xml",
            "htmlcov/",
            "bandit-report.json",
            "pytest-report.json"
        ],
        deploy_steps=[
            BuildStep(
                name="build_package",
                command="python -m build",
                description="Build distribution packages",
                artifacts=["dist/"]
            )
        ]
    )


def run_ci_locally(config_file: Optional[str] = None) -> Dict[str, Any]:
    """Run CI pipeline locally for testing.

    Args:
        config_file: Optional config file path

    Returns:
        Pipeline execution results
    """
    if config_file and os.path.exists(config_file):
        # Load config from file
        try:
            with open(config_file) as f:
                config_data = json.load(f)
            # Convert to PipelineConfig (simplified)
            config = create_default_pipeline()
            config.name = config_data.get("name", config.name)
        except Exception as e:
            logger.error(f"Failed to load config file: {e}")
            config = create_default_pipeline()
    else:
        config = create_default_pipeline()

    runner = CIRunner(config)
    return runner.run_full_pipeline()


# Quality gates and metrics
class QualityGate:
    """Quality gate checker for CI/CD."""

    def __init__(self):
        """Initialize quality gate."""
        self.thresholds = {
            "test_coverage": 80.0,
            "lint_score": 8.0,
            "security_issues": 0,
            "test_pass_rate": 95.0
        }

    def check_coverage(self, coverage_file: str = "coverage.xml") -> Tuple[bool, float]:
        """Check test coverage against threshold.

        Args:
            coverage_file: Coverage report file

        Returns:
            Tuple of (passed, coverage_percentage)
        """
        try:
            # Simple XML parsing for coverage
            if os.path.exists(coverage_file):
                with open(coverage_file) as f:
                    content = f.read()

                # Extract coverage percentage (simplified)
                import re
                match = re.search(r'line-rate="([0-9.]+)"', content)
                if match:
                    coverage = float(match.group(1)) * 100
                    passed = coverage >= self.thresholds["test_coverage"]
                    return passed, coverage

            return False, 0.0

        except Exception as e:
            logger.error(f"Coverage check failed: {e}")
            return False, 0.0

    def check_all_gates(self) -> Dict[str, Any]:
        """Check all quality gates.

        Returns:
            Quality gate results
        """
        results = {}

        # Check coverage
        coverage_passed, coverage_value = self.check_coverage()
        results["coverage"] = {
            "passed": coverage_passed,
            "value": coverage_value,
            "threshold": self.thresholds["test_coverage"]
        }

        # Overall gate status
        all_passed = all(gate["passed"] for gate in results.values())

        return {
            "overall_passed": all_passed,
            "gates": results,
            "thresholds": self.thresholds
        }
