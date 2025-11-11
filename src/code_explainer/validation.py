"""Input validation models using Pydantic."""

from typing import List, Optional, Literal
from pydantic import BaseModel, Field, field_validator
from .exceptions import ValidationError

# Pre-computed sets for constant-time validation
_ALLOWED_STRATEGIES = frozenset({"vanilla", "ast_augmented", "retrieval_augmented", "execution_trace"})
_ALLOWED_CACHE_STRATEGIES = frozenset({"lru", "lfu", "fifo", "size_based", "adaptive"})


class CodeExplanationRequest(BaseModel):
    """Request model for code explanation."""
    code: str = Field(min_length=1, max_length=10000, description="Source code to explain")
    max_length: Optional[int] = Field(default=None, ge=50, le=2048, description="Maximum sequence length")
    strategy: Optional[str] = Field(default="vanilla", description="Prompt strategy")

    @field_validator('code')
    @classmethod
    def validate_code_not_empty(cls, v):
        if not v.strip():
            raise ValidationError('Code cannot be empty or whitespace only', field_name='code')
        return v

    @field_validator('strategy')
    @classmethod
    def validate_strategy(cls, v):
        if v is not None and v not in _ALLOWED_STRATEGIES:
            allowed_str = ", ".join(sorted(_ALLOWED_STRATEGIES))
            raise ValidationError(f'Strategy must be one of: {allowed_str}', field_name='strategy', field_value=v)
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
            raise ValidationError('Codes list cannot be empty', field_name='codes')
        if len(v) > 100:
            raise ValidationError('Cannot process more than 100 codes at once', field_name='codes', field_value=f'{len(v)} codes')
        
        # Single-pass validation for maximum efficiency
        for i, code in enumerate(v):
            if not code.strip():
                raise ValidationError(f'Code at index {i} cannot be empty or whitespace only', field_name=f'codes[{i}]')
            if len(code) > 10000:
                raise ValidationError(f'Code at index {i} exceeds maximum length of 10000 characters', field_name=f'codes[{i}]', field_value=f'{len(code)} chars')
        
        return v

    @field_validator('strategy')
    @classmethod
    def validate_strategy(cls, v):
        if v is not None and v not in _ALLOWED_STRATEGIES:
            allowed_str = ", ".join(sorted(_ALLOWED_STRATEGIES))
            raise ValidationError(f'Strategy must be one of: {allowed_str}', field_name='strategy', field_value=v)
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

    # Advanced cache settings
    advanced_cache_enabled: bool = Field(default=False, description="Whether advanced caching is enabled")
    advanced_cache_dir: str = Field(default=".cache/advanced", description="Advanced cache directory")
    max_memory_entries: int = Field(default=1000, ge=100, le=10000, description="Maximum memory cache entries")
    max_disk_entries: int = Field(default=10000, ge=1000, le=100000, description="Maximum disk cache entries")
    default_ttl: int = Field(default=86400, ge=3600, le=604800, description="Default cache TTL in seconds")
    cache_strategy: str = Field(default="lru", description="Cache eviction strategy")
    compression_threshold: int = Field(default=1000, ge=100, le=10000, description="Compression threshold in bytes")
    enable_monitoring: bool = Field(default=True, description="Enable cache performance monitoring")

    @field_validator('cache_strategy')
    @classmethod
    def validate_cache_strategy(cls, v):
        if v not in _ALLOWED_CACHE_STRATEGIES:
            allowed_str = ", ".join(sorted(_ALLOWED_CACHE_STRATEGIES))
            raise ValueError(f'Cache strategy must be one of: {allowed_str}')
        return v