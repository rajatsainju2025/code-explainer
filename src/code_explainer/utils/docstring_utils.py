"""Docstring standardization guide and utilities."""

DOCSTRING_TEMPLATE = '''
"""Module description.

This module provides [main functionality].

Attributes:
    CONSTANT (type): Description of constant.
    
Classes:
    ClassName: Description of class.
    
Functions:
    function_name: Description of function.
"""
'''

FUNCTION_DOCSTRING_TEMPLATE = '''
def function_name(param1: str, param2: int) -> bool:
    """Brief one-line description.
    
    Longer description explaining what the function does, including
    any important notes or examples.
    
    Args:
        param1: Description of param1.
        param2: Description of param2 and valid range if applicable.
        
    Returns:
        Description of return value and its meaning.
        
    Raises:
        ValueError: When param1 is invalid.
        TypeError: When param2 is not an integer.
        
    Examples:
        >>> result = function_name("test", 42)
        >>> print(result)
        True
    """
'''

CLASS_DOCSTRING_TEMPLATE = '''
class ClassName:
    """Brief one-line description.
    
    Longer description explaining the class purpose, design decisions,
    and usage patterns.
    
    Attributes:
        attribute1 (str): Description of instance attribute.
        attribute2 (int): Description of another attribute.
        
    Example:
        >>> obj = ClassName()
        >>> obj.method()
    """
    
    def __init__(self, param: str) -> None:
        """Initialize ClassName.
        
        Args:
            param: Description of initialization parameter.
        """
'''

# Google-style docstring guidelines
DOCSTRING_GUIDELINES = """
DOCSTRING STANDARDIZATION GUIDELINES
====================================

1. MODULE DOCSTRINGS
   - One line summary (first line)
   - Blank line
   - Longer description (optional)
   - List of classes and functions provided

2. FUNCTION DOCSTRINGS
   - One line summary describing what function does
   - Blank line
   - Longer description (if needed)
   - Args section with parameter descriptions
   - Returns section with return value description
   - Raises section listing exceptions that can be raised
   - Examples section with usage examples

3. CLASS DOCSTRINGS
   - One line summary
   - Blank line
   - Longer description
   - Attributes section with instance variables
   - Methods listed (if complex)
   - Example usage

4. FORMAT RULES
   - Use triple double-quotes: \"\"\"
   - First line is summary (max 79 chars)
   - Sections separated by blank lines
   - Use backticks for code: `code_sample`
   - Use Args, Returns, Raises, Examples, Attributes

5. PARAMETER DESCRIPTIONS
   - type (required): Description of parameter
   - Include valid ranges/values if constrained
   - Note if parameter is optional

6. RETURN DESCRIPTIONS
   - Description of what is returned
   - Type should match function annotation
   - Include meaning/interpretation if complex

7. EXCEPTION DESCRIPTIONS
   - List all exceptions that can be raised
   - Describe conditions that cause the exception
   - Include how to handle if helpful

8. EXAMPLES
   - Include usage examples for complex functions
   - Use >>> for interactive examples
   - Show both normal and edge cases
"""

# Pre-compile validation markers for O(1) lookup
_ARGS_MARKER = "args:"
_RETURNS_MARKER = "returns:"
_EXAMPLE_MARKER = "example"


def validate_docstring(docstring: str) -> dict:
    """Validate docstring completeness.
    
    Args:
        docstring: The docstring to validate
        
    Returns:
        Dictionary with validation results
    """
    # Pre-compute lowercase once
    full_text = docstring.lower()
    
    # Check sections using pre-computed lowercase
    has_args = _ARGS_MARKER in full_text
    has_returns = _RETURNS_MARKER in full_text
    has_examples = _EXAMPLE_MARKER in full_text
    
    # Check for summary (first non-empty line)
    first_line = docstring.split('\n', 1)[0] if docstring else ""
    has_summary = bool(first_line.strip())
    
    # Build issues list only if needed
    issues = [] if has_summary else ["Missing summary line"]
    
    return {
        "has_summary": has_summary,
        "has_description": False,
        "has_args": has_args,
        "has_returns": has_returns,
        "has_examples": has_examples,
        "issues": issues
    }
