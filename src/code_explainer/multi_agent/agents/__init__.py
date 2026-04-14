"""Multi-agent agents sub-package.

Exports a single shared SymbolicAnalyzer singleton so that all agents
(structural, verification, …) reuse the same warm AST cache rather than
each maintaining their own module-level global.
"""

from typing import Any, Optional
import logging

logger = logging.getLogger(__name__)

_symbolic_analyzer: Optional[Any] = None


def get_shared_symbolic_analyzer() -> Any:
    """Return the package-wide SymbolicAnalyzer singleton.

    Created on first call (lazy) to keep module-import fast.  All agents
    that need symbolic analysis should call this function instead of
    maintaining their own module-level global, ensuring the internal AST
    cache is shared and stays warm across all agents.
    """
    global _symbolic_analyzer
    if _symbolic_analyzer is None:
        logger.debug("Initialising shared SymbolicAnalyzer singleton")
        from ...symbolic import SymbolicAnalyzer
        _symbolic_analyzer = SymbolicAnalyzer()
    return _symbolic_analyzer


__all__ = ["get_shared_symbolic_analyzer"]
