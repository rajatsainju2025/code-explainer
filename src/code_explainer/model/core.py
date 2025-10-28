"""Core model class for code explanation."""

import logging
from pathlib import Path
from typing import Any, Optional, Union, Tuple, TYPE_CHECKING
from dataclasses import dataclass

import torch
from transformers import (
    AutoModelForCausalLM,
    AutoModelForSeq2SeqLM,
    AutoTokenizer,
    PreTrainedModel,
    PreTrainedTokenizer,
    PreTrainedTokenizerBase,
)

from ..config import Config, init_config
from ..enhanced_error_handling import get_logger, setup_logging
from ..model_loader import ModelLoader, ModelResources, ModelError
from ..exceptions import ValidationError, ConfigurationError
from ..multi_agent import MultiAgentOrchestrator
from ..symbolic import SymbolicAnalyzer, format_symbolic_explanation
from ..utils import get_device, load_config, prompt_for_language
from ..validation import CodeExplanationRequest, BatchCodeExplanationRequest
from ..cache import ExplanationCache

# Import mixins
from .properties import CodeExplainerPropertiesMixin
from .initialization import CodeExplainerInitializationMixin
from .config_validation import CodeExplainerConfigValidationMixin
from .utilities import CodeExplainerUtilitiesMixin
from .explanation import CodeExplainerExplanationMixin
from .monitoring import CodeExplainerMonitoringMixin

# Import OmegaConf for config conversion
from omegaconf import OmegaConf
from omegaconf import DictConfig, ListConfig  # type: ignore

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

        # Initialize additional components
        self.symbolic_analyzer = SymbolicAnalyzer()
        self.multi_agent_orchestrator = MultiAgentOrchestrator(self)