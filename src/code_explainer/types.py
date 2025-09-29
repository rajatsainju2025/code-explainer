"""Type definitions and annotations for code explainer components."""

from typing import (
    Any, Dict, List, Optional, Union, Tuple, Callable, Protocol,
    TypeVar, Generic, Literal, TypedDict, NamedTuple, runtime_checkable
)
from typing_extensions import NotRequired
from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum
import sys

if sys.version_info >= (3, 10):
    from typing import ParamSpec
else:
    from typing_extensions import ParamSpec

# Type variables
T = TypeVar('T')
P = ParamSpec('P')
ExplanationT = TypeVar('ExplanationT', bound='ExplanationResult')


# Enums for better type safety
class ExplanationStrategy(str, Enum):
    """Available explanation strategies."""
    VANILLA = "vanilla"
    AST_AUGMENTED = "ast_augmented"
    ENHANCED_RAG = "enhanced_rag"
    MULTI_AGENT = "multi_agent"
    EXECUTION_TRACE = "execution_trace"


class SecurityRiskLevel(str, Enum):
    """Security risk levels."""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"
    UNKNOWN = "unknown"


class ModelArchitecture(str, Enum):
    """Model architectures."""
    CAUSAL = "causal"
    SEQ2SEQ = "seq2seq"
    ENCODER_DECODER = "encoder_decoder"


class CacheType(str, Enum):
    """Cache types."""
    MEMORY = "memory"
    DISK = "disk"
    REDIS = "redis"
    HYBRID = "hybrid"


# TypedDict for structured data
class SecurityValidationResult(TypedDict):
    """Result of security validation."""
    is_safe: bool
    issues: List[str]
    recommendations: List[str]
    risk_level: SecurityRiskLevel
    scan_time_ms: float


class CodeAnalysisResult(TypedDict):
    """Result of code analysis."""
    complexity_score: int
    function_count: int
    class_count: int
    line_count: int
    has_imports: bool
    has_loops: bool
    ast_valid: bool
    quality_metrics: Dict[str, Union[int, float, str]]
    suggestions: List[str]


class ExplanationMetadata(TypedDict, total=False):
    """Metadata for explanations."""
    timestamp: str
    code_length: int
    strategy: ExplanationStrategy
    confidence_score: NotRequired[float]
    model_version: NotRequired[str]
    processing_time_ms: NotRequired[float]
    cached: NotRequired[bool]
    cache_key: NotRequired[str]


class MetricEvent(TypedDict):
    """Metric event data."""
    name: str
    value: float
    timestamp: float
    tags: Dict[str, str]
    metadata: Dict[str, Any]


class PerformanceMetrics(TypedDict):
    """Performance metrics data."""
    count: int
    sum: float
    avg: float
    min: float
    max: float
    median: float
    p95: float
    p99: float
    std: NotRequired[float]


# NamedTuple for immutable data structures
class CodeLocation(NamedTuple):
    """Location in code."""
    line: int
    column: int
    filename: Optional[str] = None


class ExplanationSpan(NamedTuple):
    """Span of text in explanation."""
    start: int
    end: int
    label: str
    confidence: float


class ModelConfig(NamedTuple):
    """Model configuration."""
    name: str
    architecture: ModelArchitecture
    max_length: int
    temperature: float
    top_p: float
    top_k: int


# Dataclasses for complex objects
@dataclass(frozen=True)
class ExplanationResult:
    """Result of code explanation."""
    explanation: str
    strategy: ExplanationStrategy
    execution_time_ms: float
    confidence_score: Optional[float] = None
    cached: bool = False
    metadata: Optional[ExplanationMetadata] = None
    security_validation: Optional[SecurityValidationResult] = None
    code_analysis: Optional[CodeAnalysisResult] = None

    def __post_init__(self):
        """Validate result after initialization."""
        if self.confidence_score is not None:
            if not 0.0 <= self.confidence_score <= 1.0:
                raise ValueError(f"Confidence score must be between 0 and 1, got {self.confidence_score}")

        if self.execution_time_ms < 0:
            raise ValueError(f"Execution time cannot be negative, got {self.execution_time_ms}")


@dataclass
class BatchExplanationRequest:
    """Request for batch explanation."""
    codes: List[str]
    strategy: ExplanationStrategy = ExplanationStrategy.ENHANCED_RAG
    batch_size: int = 10
    include_security_check: bool = True
    parallel_processing: bool = True

    def __post_init__(self):
        """Validate request after initialization."""
        if not self.codes:
            raise ValueError("Codes list cannot be empty")

        if self.batch_size < 1:
            raise ValueError(f"Batch size must be positive, got {self.batch_size}")

        if len(self.codes) > 1000:
            raise ValueError(f"Too many codes, maximum 1000 allowed, got {len(self.codes)}")


@dataclass
class CacheConfig:
    """Cache configuration."""
    type: CacheType
    max_size: int
    ttl_seconds: int
    eviction_policy: Literal["lru", "fifo", "random"] = "lru"
    compression_enabled: bool = False

    def __post_init__(self):
        """Validate configuration."""
        if self.max_size <= 0:
            raise ValueError(f"Cache size must be positive, got {self.max_size}")

        if self.ttl_seconds <= 0:
            raise ValueError(f"TTL must be positive, got {self.ttl_seconds}")


# Protocol definitions for interface contracts
@runtime_checkable
class ExplainerProtocol(Protocol):
    """Protocol for code explainers."""

    def explain(self, code: str, strategy: ExplanationStrategy) -> str:
        """Explain the given code."""
        ...

    def batch_explain(self, codes: List[str], strategy: ExplanationStrategy) -> List[str]:
        """Explain multiple code snippets."""
        ...


@runtime_checkable
class SecurityValidatorProtocol(Protocol):
    """Protocol for security validators."""

    def validate_code(self, code: str) -> SecurityValidationResult:
        """Validate code security."""
        ...

    def is_safe(self, code: str) -> bool:
        """Quick safety check."""
        ...


@runtime_checkable
class CacheProtocol(Protocol[T]):
    """Protocol for cache implementations."""

    def get(self, key: str) -> Optional[T]:
        """Get item from cache."""
        ...

    def put(self, key: str, value: T, ttl: Optional[int] = None) -> None:
        """Put item in cache."""
        ...

    def delete(self, key: str) -> bool:
        """Delete item from cache."""
        ...

    def clear(self) -> None:
        """Clear all items from cache."""
        ...


@runtime_checkable
class MetricsCollectorProtocol(Protocol):
    """Protocol for metrics collection."""

    def record_event(self, name: str, value: float, tags: Optional[Dict[str, str]] = None) -> None:
        """Record a metric event."""
        ...

    def increment_counter(self, name: str, value: float = 1.0) -> None:
        """Increment a counter."""
        ...

    def set_gauge(self, name: str, value: float) -> None:
        """Set a gauge value."""
        ...


# Abstract base classes
class BaseExplainer(ABC):
    """Abstract base class for code explainers."""

    @abstractmethod
    def explain(self, code: str, strategy: ExplanationStrategy) -> ExplanationResult:
        """Explain the given code."""
        pass

    @abstractmethod
    def get_supported_strategies(self) -> List[ExplanationStrategy]:
        """Get list of supported strategies."""
        pass


class BaseCache(ABC, Generic[T]):
    """Abstract base class for cache implementations."""

    @abstractmethod
    def get(self, key: str) -> Optional[T]:
        """Get item from cache."""
        pass

    @abstractmethod
    def put(self, key: str, value: T, ttl: Optional[int] = None) -> None:
        """Put item in cache."""
        pass

    @abstractmethod
    def delete(self, key: str) -> bool:
        """Delete item from cache."""
        pass

    @abstractmethod
    def clear(self) -> None:
        """Clear all items."""
        pass

    @abstractmethod
    def size(self) -> int:
        """Get cache size."""
        pass


class BaseSecurityValidator(ABC):
    """Abstract base class for security validators."""

    @abstractmethod
    def validate_code(self, code: str) -> SecurityValidationResult:
        """Validate code security."""
        pass

    @abstractmethod
    def get_risk_patterns(self) -> Dict[str, List[str]]:
        """Get risk patterns by category."""
        pass


# Union types for common combinations
CodeInput = Union[str, List[str]]
ExplanationOutput = Union[ExplanationResult, List[ExplanationResult]]
CacheKey = Union[str, Tuple[str, ...]]
MetricValue = Union[int, float]
ConfigValue = Union[str, int, float, bool, List[Any], Dict[str, Any]]


# Callback types
ExplanationCallback = Callable[[str, ExplanationResult], None]
SecurityCallback = Callable[[str, SecurityValidationResult], None]
MetricCallback = Callable[[MetricEvent], None]
ErrorCallback = Callable[[Exception], None]


# Type aliases for complex types
ModelParameters = Dict[str, Union[str, int, float, bool]]
TrainingConfig = Dict[str, Union[str, int, float, bool, List[str]]]
EvaluationMetrics = Dict[str, Union[float, int, List[float]]]
StrategyConfig = Dict[ExplanationStrategy, Dict[str, Any]]


# Generic types for extensibility
class Configurable(Generic[T]):
    """Base class for configurable components."""

    def __init__(self, config: T):
        self.config = config

    def get_config(self) -> T:
        """Get current configuration."""
        return self.config

    def update_config(self, config: T) -> None:
        """Update configuration."""
        self.config = config


class Observable(Generic[T]):
    """Base class for observable components."""

    def __init__(self):
        self._observers: List[Callable[[T], None]] = []

    def add_observer(self, observer: Callable[[T], None]) -> None:
        """Add an observer."""
        self._observers.append(observer)

    def remove_observer(self, observer: Callable[[T], None]) -> None:
        """Remove an observer."""
        if observer in self._observers:
            self._observers.remove(observer)

    def notify_observers(self, event: T) -> None:
        """Notify all observers."""
        for observer in self._observers:
            try:
                observer(event)
            except Exception as e:
                # Log error but don't fail
                import logging
                logging.error(f"Observer error: {e}")


# Type guards for runtime type checking
def is_explanation_result(obj: Any) -> bool:
    """Check if object is an ExplanationResult."""
    return isinstance(obj, ExplanationResult)


def is_security_result(obj: Any) -> bool:
    """Check if object is a SecurityValidationResult."""
    return (isinstance(obj, dict) and
            'is_safe' in obj and
            'issues' in obj and
            'recommendations' in obj)


def is_valid_strategy(strategy: str) -> bool:
    """Check if strategy is valid."""
    try:
        ExplanationStrategy(strategy)
        return True
    except ValueError:
        return False


# Type converter utilities
def ensure_strategy(strategy: Union[str, ExplanationStrategy]) -> ExplanationStrategy:
    """Ensure strategy is ExplanationStrategy enum."""
    if isinstance(strategy, str):
        return ExplanationStrategy(strategy)
    return strategy


def ensure_risk_level(level: Union[str, SecurityRiskLevel]) -> SecurityRiskLevel:
    """Ensure risk level is SecurityRiskLevel enum."""
    if isinstance(level, str):
        try:
            return SecurityRiskLevel(level)
        except ValueError:
            return SecurityRiskLevel.UNKNOWN
    return level


# Factory function types
ExplainerFactory = Callable[[ModelConfig], BaseExplainer]
CacheFactory = Callable[[CacheConfig], BaseCache[Any]]
ValidatorFactory = Callable[[], BaseSecurityValidator]


# Error types
class CodeExplainerError(Exception):
    """Base exception for code explainer."""
    pass


class ValidationError(CodeExplainerError):
    """Validation error."""
    pass


class SecurityError(CodeExplainerError):
    """Security error."""
    pass


class CacheError(CodeExplainerError):
    """Cache error."""
    pass


class ModelError(CodeExplainerError):
    """Model error."""
    pass


# Configuration schema types
class LoggingConfig(TypedDict):
    """Logging configuration."""
    level: str
    log_file: Optional[str]
    format: str
    rotation: bool


class APIConfig(TypedDict):
    """API configuration."""
    host: str
    port: int
    workers: int
    timeout: int
    cors_enabled: bool
    rate_limiting: bool


class SystemConfig(TypedDict):
    """System configuration."""
    model: ModelParameters
    cache: CacheConfig
    logging: LoggingConfig
    api: APIConfig
    security: Dict[str, Any]
    monitoring: Dict[str, Any]


# Export all types for external use
__all__ = [
    # Enums
    'ExplanationStrategy', 'SecurityRiskLevel', 'ModelArchitecture', 'CacheType',

    # TypedDict
    'SecurityValidationResult', 'CodeAnalysisResult', 'ExplanationMetadata',
    'MetricEvent', 'PerformanceMetrics', 'LoggingConfig', 'APIConfig', 'SystemConfig',

    # NamedTuple
    'CodeLocation', 'ExplanationSpan', 'ModelConfig',

    # Dataclasses
    'ExplanationResult', 'BatchExplanationRequest', 'CacheConfig',

    # Protocols
    'ExplainerProtocol', 'SecurityValidatorProtocol', 'CacheProtocol', 'MetricsCollectorProtocol',

    # Abstract classes
    'BaseExplainer', 'BaseCache', 'BaseSecurityValidator',

    # Generic classes
    'Configurable', 'Observable',

    # Union types
    'CodeInput', 'ExplanationOutput', 'CacheKey', 'MetricValue', 'ConfigValue',

    # Callback types
    'ExplanationCallback', 'SecurityCallback', 'MetricCallback', 'ErrorCallback',

    # Type aliases
    'ModelParameters', 'TrainingConfig', 'EvaluationMetrics', 'StrategyConfig',

    # Factory types
    'ExplainerFactory', 'CacheFactory', 'ValidatorFactory',

    # Error types
    'CodeExplainerError', 'ValidationError', 'SecurityError', 'CacheError', 'ModelError',

    # Type guards
    'is_explanation_result', 'is_security_result', 'is_valid_strategy',

    # Converters
    'ensure_strategy', 'ensure_risk_level',
]
