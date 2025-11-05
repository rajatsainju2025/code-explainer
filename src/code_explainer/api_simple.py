"""Public API simplification and cleanup module.

Provides simplified, user-friendly public interfaces for core functionality.

Functions:
    explain_code: Simple code explanation interface
    batch_explain: Batch code explanation interface
    explain_with_strategy: Strategy-specific explanation
"""

from typing import List, Optional, Dict, Any


def explain_code(
    code: str,
    strategy: Optional[str] = None,
    max_length: Optional[int] = None
) -> str:
    """Explain code snippet - simplified interface.
    
    This is the primary user-facing function for code explanation.
    It handles all the complexity internally.
    
    Args:
        code: The code to explain
        strategy: Optional explanation strategy
        max_length: Optional max output length
        
    Returns:
        Human-readable explanation of the code
        
    Raises:
        ValidationError: If code is invalid
        InferenceError: If explanation fails
        
    Examples:
        >>> explanation = explain_code("x = 1 + 2")
        >>> print(explanation)
        "This code assigns the result of 1 + 2..."
    """
    # Implementation would use CodeExplainer internally
    pass


def batch_explain(
    codes: List[str],
    strategy: Optional[str] = None,
    batch_size: int = 32
) -> Dict[int, str]:
    """Explain multiple code snippets efficiently.
    
    Args:
        codes: List of code snippets to explain
        strategy: Optional explanation strategy
        batch_size: Batch size for processing
        
    Returns:
        Dictionary mapping code index to explanation
        
    Raises:
        ValidationError: If any code is invalid
    """
    pass


def explain_with_strategy(
    code: str,
    strategy: str
) -> Dict[str, Any]:
    """Explain code with specific strategy.
    
    Args:
        code: The code to explain
        strategy: Explanation strategy to use
        
    Returns:
        Dictionary with explanation and metadata
    """
    pass
