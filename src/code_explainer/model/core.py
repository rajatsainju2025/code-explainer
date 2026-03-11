"""Core model class for code explanation."""

import gc
import logging
from pathlib import Path
from typing import Any, Optional, Union

import torch
from transformers import (
    PreTrainedModel,
    PreTrainedTokenizerBase,
)

from ..model_loader import ModelLoader, ModelResources
from ..multi_agent import MultiAgentOrchestrator
from ..symbolic import SymbolicAnalyzer
from ..cache import ExplanationCache

# Import mixins
from .properties import CodeExplainerPropertiesMixin
from .initialization import CodeExplainerInitializationMixin
from .config_validation import CodeExplainerConfigValidationMixin
from .utilities import CodeExplainerUtilitiesMixin
from .explanation import CodeExplainerExplanationMixin
from .monitoring import CodeExplainerMonitoringMixin
try:
    from .intelligent_explainer import (
        IntelligentExplanationGenerator,
        ExplanationAudience,
        ExplanationStyle,
        EnhancedExplanation
    )
    INTELLIGENT_EXPLAINER_AVAILABLE = True
except ImportError as e:
    logging.getLogger(__name__).warning(f"Intelligent explainer not available: {e}")
    INTELLIGENT_EXPLAINER_AVAILABLE = False
    IntelligentExplanationGenerator = None
    ExplanationAudience = None
    ExplanationStyle = None
    EnhancedExplanation = None

logger = logging.getLogger(__name__)


class CodeExplainer(
    CodeExplainerPropertiesMixin,
    CodeExplainerInitializationMixin,
    CodeExplainerConfigValidationMixin,
    CodeExplainerUtilitiesMixin,
    CodeExplainerExplanationMixin,
    CodeExplainerMonitoringMixin
):
    """Main class for code explanation inference.

    Attributes:
        config (Config): Hydra configuration object
        explanation_cache (Optional[ExplanationCache]): Cache for generated explanations
        symbolic_analyzer (SymbolicAnalyzer): Analyzer for symbolic code analysis
        multi_agent_orchestrator (MultiAgentOrchestrator): Orchestrator for multi-agent interactions
    """
    # Class-level attributes for better type checking
    config: Any
    logger: Any
    model_loader: Optional["ModelLoader"]
    _injected_model: Optional[PreTrainedModel]
    _injected_tokenizer: Optional[PreTrainedTokenizerBase]
    explanation_cache: Optional["ExplanationCache"]
    cache_manager: Optional[Any]

    def __init__(
        self,
        model_path: Optional[Union[str, Path, Any]] = "./results",
        config_path: Optional[str] = "configs/default.yaml"
    ) -> None:
        """Initialize the code explainer.

        Args:
            model_path: Path to trained model directory. If None, uses default from config.
            config_path: Path to configuration file. If None, uses default config.
        """
        # Early logger so mixins can safely use `self.logger` during init
        self.logger = logging.getLogger(__name__)

        # Initialize configuration and logging
        self.config, resolved_model_path = self._initialize_config(
            model_path, config_path
        )
        # Allow _setup_logging to configure handlers/levels on top of the
        # instance logger created above.
        self.logger = self._setup_logging()

        # Store model path for lazy loading
        self._model_path = resolved_model_path
        
        # Defer model loading until first access (lazy loading)
        # This significantly improves startup time
        self._resources: Optional["ModelResources"] = None
        self.model_loader: Optional[ModelLoader] = None

        # Initialize additional components
        self._initialize_components()

        # Lazy initialization for expensive components - only create when needed
        self._symbolic_analyzer: Optional["SymbolicAnalyzer"] = None
        self._multi_agent_orchestrator: Optional["MultiAgentOrchestrator"] = None
    
    def __enter__(self):
        """Context manager entry - no special setup needed."""
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit - cleanup resources."""
        self.cleanup_memory()
        return False  # Don't suppress exceptions
    
    def cleanup_memory(self) -> None:
        """Explicitly cleanup memory and release resources."""
        # Flush cache if present
        if hasattr(self, 'explanation_cache') and self.explanation_cache is not None:
            self.explanation_cache.flush()

        # Inspect _resources directly to avoid triggering the lazy-load
        # property, which would load the model just to immediately discard it.
        if getattr(self, '_resources', None) is not None:
            if hasattr(torch.cuda, 'empty_cache'):
                torch.cuda.empty_cache()

        # Trigger garbage collection
        gc.collect()
    
    @property
    def symbolic_analyzer(self) -> "SymbolicAnalyzer":
        """Lazy-loaded symbolic analyzer."""
        if self._symbolic_analyzer is None:
            self._symbolic_analyzer = SymbolicAnalyzer()
        return self._symbolic_analyzer
    
    @property
    def multi_agent_orchestrator(self) -> "MultiAgentOrchestrator":
        """Lazy-loaded multi-agent orchestrator."""
        if self._multi_agent_orchestrator is None:
            self._multi_agent_orchestrator = MultiAgentOrchestrator(self)
        return self._multi_agent_orchestrator

    def explain_code_with_symbolic(
        self,
        code: str,
        include_symbolic: bool = True,
        max_length: Optional[int] = None,
        strategy: Optional[str] = None,
    ) -> str:
        """Generate an explanation combined with optional symbolic analysis.

        This convenience method is a backward-compatible shim used by tests
        and higher-level code that expects a single string containing both
        a symbolic analysis section and the model-generated explanation.
        """
        # Generate the model explanation (textual)
        textual = self.explain_code(code, max_length=max_length, strategy=strategy)

        if not include_symbolic:
            return textual

        # Generate symbolic analysis and format it, fall back gracefully
        try:
            from ..symbolic import format_symbolic_explanation

            symbolic = self.symbolic_analyzer.analyze_code(code)
            formatted = format_symbolic_explanation(symbolic)
        except Exception:
            formatted = "No symbolic conditions detected."

        # Combine with clear headings for tests that look for section markers
        parts = ["## Symbolic Analysis", formatted, "", "## Code Explanation", textual]
        return "\n".join(parts)