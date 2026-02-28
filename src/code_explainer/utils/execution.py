"""Safe code execution utilities."""

import platform
import subprocess
import sys
import tempfile
from typing import Tuple

# Pre-cache for faster access
_TEMP_DIR = tempfile.gettempdir()
_PYTHON_EXE = sys.executable
_IS_UNIX = platform.system() != "Windows"

# Minimal safe environment for the sandboxed subprocess.
# Using env=None would inherit all parent vars (a security risk for untrusted code).
# Passing only the bare minimum lets the Python interpreter start reliably while
# still denying access to credentials, tokens, or other sensitive env vars.
_SANDBOX_ENV: dict = {
    "PATH": "/usr/bin:/bin",
    "HOME": _TEMP_DIR,
    "PYTHONDONTWRITEBYTECODE": "1",
}


def _safe_exec_subprocess(code: str, timeout_s: float = 1.0, mem_mb: int = 64) -> Tuple[str, str]:
    """Run code in a subprocess with basic time/memory limits and capture output.
    Returns (stdout, stderr) truncated.
    """
    # Wrapper to set resource limits (Unix only; skipped on Windows)
    if _IS_UNIX:
        cpu_limit = int(timeout_s)
        mem_limit = mem_mb * 1024 * 1024
        prelude = (
            f"import sys,resource\n"
            f"resource.setrlimit(resource.RLIMIT_CPU, ({cpu_limit}, {cpu_limit}))\n"
            f"resource.setrlimit(resource.RLIMIT_AS, ({mem_limit}, {mem_limit}))\n"
            "\n"
        )
        wrapped = prelude + code
    else:
        # On Windows, resource module doesn't exist; skip rlimit sandboxing
        wrapped = code
    actual_timeout = max(timeout_s, 0.1)
    
    try:
        proc = subprocess.run(
            [_PYTHON_EXE, "-c", wrapped],
            input=None,
            capture_output=True,
            text=True,
            timeout=actual_timeout,
            cwd=_TEMP_DIR,
            env=_SANDBOX_ENV,
        )
        out = (proc.stdout or "").strip()
        err = (proc.stderr or "").strip()
    except subprocess.TimeoutExpired:
        return "", "TimeoutExpired"
    except (subprocess.CalledProcessError, OSError, ValueError) as e:
        return "", f"ExecutionError: {e}"

    # Truncate inline
    max_len = 500
    if len(out) > max_len:
        out = out[:max_len] + "…"
    if len(err) > max_len:
        err = err[:max_len] + "…"

    return out, err