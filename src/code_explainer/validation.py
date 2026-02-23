"""Input validation models using Pydantic.

Optimized for performance with:
- Pre-computed validation sets for O(1) lookup
- Model configuration for performance (validate_default=False)
- Efficient batch validation with early termination
- Cached error messages to avoid repeated string formatting
- Singleton fast validator for compiled pattern reuse
"""

from typing import List, Optional
from pydantic import BaseModel, Field, field_validator, ConfigDict
from .exceptions import ValidationError

# Pre-computed sets for constant-time validation
_ALLOWED_STRATEGIES = frozenset({"vanilla", "ast_augmented", "retrieval_augmented", "execution_trace", "multi_agent", "intelligent"})
# Pre-formatted error messages (avoid repeated string formatting)
_STRATEGY_ERROR_MSG = f"Strategy must be one of: {', '.join(sorted(_ALLOWED_STRATEGIES))}"

# Constants for validation limits
_MAX_CODE_LENGTH = 10000
_MAX_BATCH_SIZE = 100
_MAX_SEQUENCE_LENGTH = 2048
_MIN_SEQUENCE_LENGTH = 50


class CodeExplanationRequest(BaseModel):
    """Request model for code explanation with optimized validation."""
    
    model_config = ConfigDict(
        validate_default=False,  # Skip validation for defaults
        str_strip_whitespace=False,  # Don't strip - we check explicitly
        extra='ignore'  # Ignore extra fields for forward compatibility
    )
    
    code: str = Field(min_length=1, max_length=_MAX_CODE_LENGTH, description="Source code to explain")
    max_length: Optional[int] = Field(default=None, ge=_MIN_SEQUENCE_LENGTH, le=_MAX_SEQUENCE_LENGTH, description="Maximum sequence length")
    strategy: Optional[str] = Field(default="vanilla", description="Prompt strategy")

    @field_validator('code')
    @classmethod
    def validate_code_not_empty(cls, v: str) -> str:
        # Ultra-fast path: single character check
        if len(v) == 0:
            raise ValidationError('Code cannot be empty', field_name='code')
        # Fast path: check first char for non-whitespace
        if not v[0].isspace():
            return v  # Likely valid, skip expensive strip()
        # Slow path: full whitespace check only when needed
        if not v.strip():
            raise ValidationError('Code cannot be whitespace only', field_name='code')
        return v

    @field_validator('strategy')
    @classmethod
    def validate_strategy(cls, v: Optional[str]) -> Optional[str]:
        if v is not None and v not in _ALLOWED_STRATEGIES:
            raise ValidationError(_STRATEGY_ERROR_MSG, field_name='strategy', field_value=v)
        return v


class BatchCodeExplanationRequest(BaseModel):
    """Request model for batch code explanation with optimized validation."""
    
    model_config = ConfigDict(
        validate_default=False,
        extra='ignore'
    )
    
    codes: List[str] = Field(description="List of code snippets")
    max_length: Optional[int] = Field(default=None, ge=_MIN_SEQUENCE_LENGTH, le=_MAX_SEQUENCE_LENGTH, description="Maximum sequence length")
    strategy: Optional[str] = Field(default="vanilla", description="Prompt strategy")

    @field_validator('codes')
    @classmethod
    def validate_codes(cls, v: List[str]) -> List[str]:
        if not v:
            raise ValidationError('Codes list cannot be empty', field_name='codes')
        
        codes_len = len(v)
        if codes_len > _MAX_BATCH_SIZE:
            raise ValidationError(
                f'Cannot process more than {_MAX_BATCH_SIZE} codes at once',
                field_name='codes',
                field_value=f'{codes_len} codes'
            )
        
        # Single-pass validation with fast paths and early termination
        max_len = _MAX_CODE_LENGTH
        for i, code in enumerate(v):
            # Fast path: length check first (cheapest operation)
            code_len = len(code)
            if code_len == 0:
                raise ValidationError(
                    f'Code at index {i} cannot be empty',
                    field_name=f'codes[{i}]'
                )
            if code_len > max_len:
                raise ValidationError(
                    f'Code at index {i} exceeds maximum length of {max_len} characters',
                    field_name=f'codes[{i}]',
                    field_value=f'{code_len} chars'
                )
            # Only check for whitespace if first char is whitespace (fast path)
            if code[0].isspace() and not code.strip():
                raise ValidationError(
                    f'Code at index {i} cannot be whitespace only',
                    field_name=f'codes[{i}]'
                )
        
        return v

    @field_validator('strategy')
    @classmethod
    def validate_strategy(cls, v: Optional[str]) -> Optional[str]:
        if v is not None and v not in _ALLOWED_STRATEGIES:
            raise ValidationError(_STRATEGY_ERROR_MSG, field_name='strategy', field_value=v)
        return v



        return v