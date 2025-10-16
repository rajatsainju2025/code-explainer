"""Safe code execution utilities."""

import os
import shlex
import subprocess
import sys
import tempfile
import textwrap
from typing import Tuple


def _safe_exec_subprocess(code: str, timeout_s: float = 1.0, mem_mb: int = 64) -> Tuple[str, str]:
    """Run code in a subprocess with basic time/memory limits and capture output.
    Returns (stdout, stderr) truncated.
    """
    # Wrapper to set resource limits (Unix only)
    prelude = (
        "import sys,resource,os\n"
        f"resource.setrlimit(resource.RLIMIT_CPU, ({int(timeout_s)}, {int(timeout_s)}))\n"
        f"resource.setrlimit(resource.RLIMIT_AS, ({mem_mb*1024*1024}, {mem_mb*1024*1024}))\n"
        "os.environ.clear()\n"
        "\n"
    )

    wrapped = prelude + code
    try:
        proc = subprocess.run(
            [sys.executable, "-c", wrapped],
            input=None,
            capture_output=True,
            text=True,
            timeout=max(timeout_s, 0.1),
            cwd=tempfile.gettempdir(),
            env={},
        )
        out = (proc.stdout or "").strip()
        err = (proc.stderr or "").strip()
    except subprocess.TimeoutExpired:
        out, err = "", "TimeoutExpired"
    except Exception as e:
        out, err = "", f"ExecutionError: {e}"

    # Truncate
    def _trunc(s: str, n: int = 500) -> str:
        return (s[:n] + "…") if len(s) > n else s

    return _trunc(out), _trunc(err)