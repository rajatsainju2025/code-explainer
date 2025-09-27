"""Input validation models using Pydantic."""

from typing import List, Optional, Literal
from pydantic import BaseModel, Field, field_validator


class CodeExplanationRequest(BaseModel):
    """Request model for code explanation."""
    code: str = Field(min_length=1, max_length=10000, description="Source code to explain")
    max_length: Optional[int] = Field(default=None, ge=50, le=2048, description="Maximum sequence length")
    strategy: Optional[str] = Field(default="vanilla", description="Prompt strategy")

    @field_validator('code')
    @classmethod
    def validate_code_not_empty(cls, v):
        if not v.strip():
            raise ValueError('Code cannot be empty or whitespace only')
        return v

    @field_validator('strategy')
    @classmethod
    def validate_strategy(cls, v):
        if v is not None and v not in ["vanilla", "ast_augmented"]:
            raise ValueError('Strategy must be either "vanilla" or "ast_augmented"')
        return v


class BatchCodeExplanationRequest(BaseModel):
    """Request model for batch code explanation."""
    codes: List[str] = Field(description="List of code snippets")
    max_length: Optional[int] = Field(default=None, ge=50, le=2048, description="Maximum sequence length")
    strategy: Optional[str] = Field(default="vanilla", description="Prompt strategy")

    @field_validator('codes')
    @classmethod
    def validate_codes(cls, v):
        if not v:
            raise ValueError('Codes list cannot be empty')
        if len(v) > 100:
            raise ValueError('Cannot process more than 100 codes at once')
        for i, code in enumerate(v):
            if not code.strip():
                raise ValueError(f'Code at index {i} cannot be empty or whitespace only')
            if len(code) > 10000:
                raise ValueError(f'Code at index {i} exceeds maximum length of 10000 characters')
        return v

    @field_validator('strategy')
    @classmethod
    def validate_strategy(cls, v):
        if v is not None and v not in ["vanilla", "ast_augmented"]:
            raise ValueError('Strategy must be either "vanilla" or "ast_augmented"')
        return v


class ModelConfigValidation(BaseModel):
    """Validation model for model configuration."""
    name: str = Field(min_length=1, description="Model name or path")
    arch: Literal["causal", "seq2seq"] = Field(default="causal", description="Model architecture")
    torch_dtype: str = Field(default="auto", description="PyTorch data type")
    load_in_8bit: bool = Field(default=False, description="Whether to load in 8-bit precision")
    max_length: int = Field(default=512, ge=1, le=4096, description="Maximum sequence length")
    device_map: Optional[str] = Field(default=None, description="Device mapping for model")


class LoggingConfigValidation(BaseModel):
    """Validation model for logging configuration."""
    level: Literal["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"] = Field(default="INFO", description="Logging level")
    log_file: Optional[str] = Field(default=None, description="Log file path")
    format: str = Field(default="%(asctime)s - %(name)s - %(levelname)s - %(message)s", description="Log format")
    date_format: str = Field(default="%Y-%m-%d %H:%M:%S", description="Date format")


class CacheConfigValidation(BaseModel):
    """Validation model for cache configuration."""
    enabled: bool = Field(default=True, description="Whether caching is enabled")
    directory: str = Field(default=".cache/explanations", description="Cache directory path")
    max_size: int = Field(default=1000, ge=1, le=10000, description="Maximum cache size")