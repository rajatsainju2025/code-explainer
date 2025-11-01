"""Core model class for code explanation."""

import gc
import logging
from pathlib import Path
from typing import Any, Optional, Union
from dataclasses import dataclass

import torch
from transformers import (
    PreTrainedModel,
    PreTrainedTokenizerBase,
)

from ..model_loader import ModelLoader
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

# Import OmegaConf for config conversion

# Import new intelligent explanation components
try:
    from .intelligent_explainer import (
        IntelligentExplanationGenerator,
        ExplanationAudience,
        ExplanationStyle,
        EnhancedExplanation
    )
    INTELLIGENT_EXPLAINER_AVAILABLE = True
except ImportError as e:
    import logging
    logging.getLogger(__name__).warning(f"Intelligent explainer not available: {e}")
    INTELLIGENT_EXPLAINER_AVAILABLE = False
    IntelligentExplanationGenerator = None
    ExplanationAudience = None
    ExplanationStyle = None
    EnhancedExplanation = None

logger = logging.getLogger(__name__)


@dataclass
class ModelConfig:
    """Configuration for model loading and inference."""
    name: str
    arch: str = "causal"
    torch_dtype: Union[str, torch.dtype] = "auto"
    load_in_8bit: bool = False
    max_length: int = 512
    device_map: Optional[str] = None


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
    _resources: Optional[Any]
    _injected_model: Optional[PreTrainedModel]
    _injected_tokenizer: Optional[PreTrainedTokenizerBase]
    explanation_cache: Optional["ExplanationCache"]
    symbolic_analyzer: "SymbolicAnalyzer"
    multi_agent_orchestrator: "MultiAgentOrchestrator"
    cache_manager: Optional[Any]
    advanced_cache: Optional[Any]

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
        # Initialize configuration and logging
        self.config, resolved_model_path = self._initialize_config(
            model_path, config_path
        )
        self.logger = self._setup_logging()

        # Initialize model resources
        self._resources = self._initialize_model_resources(resolved_model_path)

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
        
        # Clear model cache
        if hasattr(self, 'model') and self.model is not None:
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