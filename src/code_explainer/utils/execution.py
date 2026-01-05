"""Safe code execution utilities."""

import subprocess
import sys
import tempfile
from typing import Tuple

# Pre-cache for faster access
_TEMP_DIR = tempfile.gettempdir()
_PYTHON_EXE = sys.executable
_EMPTY_ENV: dict = {}


def _safe_exec_subprocess(code: str, timeout_s: float = 1.0, mem_mb: int = 64) -> Tuple[str, str]:
    """Run code in a subprocess with basic time/memory limits and capture output.
    Returns (stdout, stderr) truncated.
    """
    # Wrapper to set resource limits (Unix only)
    cpu_limit = int(timeout_s)
    mem_limit = mem_mb * 1024 * 1024
    prelude = (
        f"import sys,resource,os\n"
        f"resource.setrlimit(resource.RLIMIT_CPU, ({cpu_limit}, {cpu_limit}))\n"
        f"resource.setrlimit(resource.RLIMIT_AS, ({mem_limit}, {mem_limit}))\n"
        "os.environ.clear()\n"
        "\n"
    )

    wrapped = prelude + code
    actual_timeout = max(timeout_s, 0.1)
    
    try:
        proc = subprocess.run(
            [_PYTHON_EXE, "-c", wrapped],
            input=None,
            capture_output=True,
            text=True,
            timeout=actual_timeout,
            cwd=_TEMP_DIR,
            env=_EMPTY_ENV,
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