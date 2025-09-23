"""Main model class for code explanation."""

import logging
import concurrent.futures
from pathlib import Path
from typing import Any, Dict, List, Optional, Union, Tuple, cast
from dataclasses import dataclass

import torch
from torch.nn import Module
from transformers import (
    AutoModelForCausalLM,
    AutoModelForSeq2SeqLM,
    AutoTokenizer,
    PreTrainedModel,
    PreTrainedTokenizer,
    PreTrainedTokenizerBase,
)

from .cache import ExplanationCache
from .config import Config, init_config
from .model_loader import ModelLoader, ModelResources, ModelError
from .multi_agent import MultiAgentOrchestrator
from .symbolic import SymbolicAnalyzer, format_symbolic_explanation
from .utils import get_device, load_config, prompt_for_language

# Import OmegaConf for config conversion
from omegaconf import OmegaConf

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


class CodeExplainer:
    """Main class for code explanation inference.
    
    Attributes:
        config (Config): Hydra configuration object
        explanation_cache (Optional[ExplanationCache]): Cache for generated explanations
        symbolic_analyzer (SymbolicAnalyzer): Analyzer for symbolic code analysis
        multi_agent_orchestrator (MultiAgentOrchestrator): Orchestrator for multi-agent interactions
    """
    
    @property
    def model(self) -> PreTrainedModel:
        """Get the loaded model."""
        if self._resources is None:
            raise RuntimeError("Model resources not initialized")
        return self._resources.model

    @property
    def tokenizer(self) -> PreTrainedTokenizerBase:
        """Get the loaded tokenizer."""
        if self._resources is None:
            raise RuntimeError("Model resources not initialized")
        return self._resources.tokenizer

    @property
    def device(self) -> torch.device:
        """Get the compute device."""
        if self._resources is None:
            raise RuntimeError("Model resources not initialized")
        return self._resources.device

    @property
    def arch(self) -> str:
        """Get the model architecture type."""
        if self._resources is None:
            raise RuntimeError("Model resources not initialized")
        return self._resources.model_type

    def __init__(
        self,
        model_path: Optional[str] = "./results",
        config_path: Optional[str] = "configs/default.yaml"
    ) -> None:
        """Initialize the code explainer.

        Args:
            model_path: Path to trained model directory. If None, uses default from config.
            config_path: Path to configuration file. If None, uses default config.
        """
        # Initialize configuration
        self.config = init_config(config_path)
        
        # Set up model loader and load model resources
        self.model_loader = ModelLoader(self.config.model)
        self._resources: Optional[ModelResources] = None
        
        try:
            self._resources = self.model_loader.load(model_path)
        except ModelError as e:
            logger.error(f"Failed to load model from {model_path}: {e}")
            logger.info("Attempting to load base model...")
            self._resources = self.model_loader.load()  # Load from config name
        
        # Initialize caching if enabled
        if self.config.cache.enabled:
            self.explanation_cache = ExplanationCache(
                self.config.cache.directory,
                self.config.cache.max_size
            )
        else:
            self.explanation_cache = None

        # Initialize additional components
        self.symbolic_analyzer = SymbolicAnalyzer()
        self.multi_agent_orchestrator = MultiAgentOrchestrator(self)

    def explain_code(
        self,
        code: str,
        max_length: Optional[int] = None,
        strategy: Optional[str] = None
    ) -> str:
        """Generate explanation for the given code.

        Args:
            code: Source code to explain
            max_length: Optional max sequence length
            strategy: Optional prompt strategy override (e.g., "vanilla", "ast_augmented")

        Returns:
            str: Generated explanation for the code

        Raises:
            RuntimeError: If model or tokenizer is not initialized
            ValueError: If invalid prompt strategy is provided
        """
        if max_length is None:
            max_length = self.config.get("model", {}).get("max_length", 512)

        # Get strategy for caching
        used_strategy: str = strategy or self.config.get("prompt", {}).get("strategy", "vanilla")
        model_name: str = getattr(self, 'model_name', 'unknown')

        # Check cache first
        if self.explanation_cache is not None:
            cached_explanation = self.explanation_cache.get(code, used_strategy, model_name)
            if cached_explanation is not None:
                logger.debug("Using cached explanation")
                return cached_explanation

        # Ensure model and tokenizer are loaded
        if self.tokenizer is None or self.model is None:
            raise RuntimeError("Model and tokenizer must be initialized before generating explanations")
            
        tok: PreTrainedTokenizerBase = self.tokenizer
        mdl: PreTrainedModel = self.model

        # Language-aware prompt with optional strategy override
        if strategy is not None:
            import copy

            cfg = copy.deepcopy(self.config)
            cfg.setdefault("prompt", {})["strategy"] = strategy
            prompt = prompt_for_language(cast(Dict[str, Any], OmegaConf.to_container(cfg, resolve=True)), code)
        else:
            prompt = prompt_for_language(cast(Dict[str, Any], OmegaConf.to_container(self.config, resolve=True)), code)

        # Prepare inputs
        inputs = tok(prompt, return_tensors="pt")
        device = torch.device(self.device)
        inputs = {k: v.to(device) for k, v in inputs.items()}

        # Generate explanation
        with torch.no_grad():
            gen_max = int(max_length) if max_length is not None else 512
            outputs = mdl.generate(
                input_ids=inputs["input_ids"],
                attention_mask=inputs.get("attention_mask"),
                max_length=min(gen_max, inputs["input_ids"].shape[1] + 150),
                temperature=self.config["model"]["temperature"],
                top_p=self.config["model"]["top_p"],
                top_k=self.config["model"]["top_k"],
                do_sample=True,
                pad_token_id=tok.eos_token_id,
                no_repeat_ngram_size=2,
                early_stopping=True,
            )

        generated_text = tok.decode(outputs[0], skip_special_tokens=True)
        explanation = generated_text[len(prompt) :].strip()

        # Cache the explanation
        if self.explanation_cache is not None:
            self.explanation_cache.put(code, used_strategy, model_name, explanation)

        return explanation

    def explain_code_batch(
        self,
        codes: List[str],
        max_length: Optional[int] = None,
        strategy: Optional[str] = None
    ) -> List[str]:
        """Generate explanations for multiple code snippets efficiently using batch processing.

        Args:
            codes: List of Python code snippets
            max_length: Optional max sequence length
            strategy: Optional prompt strategy override

        Returns:
            List of generated explanations
        """
        if not codes:
            return []

        if max_length is None:
            max_length = self.config["model"]["max_length"]

        # Get strategy for caching
        used_strategy = strategy or self.config.get("prompt", {}).get("strategy", "vanilla")
        model_name = getattr(self, 'model_name', 'unknown')

        # Check cache for all codes first
        explanations = []
        uncached_codes = []
        uncached_indices = []

        for i, code in enumerate(codes):
            if self.explanation_cache is not None:
                cached_explanation = self.explanation_cache.get(code, used_strategy, model_name)
                if cached_explanation is not None:
                    explanations.append(cached_explanation)
                    continue
            uncached_codes.append(code)
            uncached_indices.append(i)
            explanations.append(None)  # Placeholder

        # If all were cached, return early
        if not uncached_codes:
            return explanations

        # Process uncached codes in batch
        batch_explanations = self._explain_code_batch_internal(
            uncached_codes,
            max_length or self.config["model"]["max_length"],
            strategy
        )

        # Fill in the results
        for idx, explanation in zip(uncached_indices, batch_explanations):
            explanations[idx] = explanation

        return explanations

    def _explain_code_batch_internal(
        self,
        codes: List[str],
        max_length: int,
        strategy: Optional[str] = None
    ) -> List[str]:
        """Internal method for batch processing of explanations."""
        assert self.tokenizer is not None and self.model is not None
        tok = self.tokenizer
        mdl = self.model

        # Prepare prompts for all codes
        prompts = []
        for code in codes:
            if strategy is not None:
                import copy
                cfg = copy.deepcopy(self.config)
                cfg.setdefault("prompt", {})["strategy"] = strategy
                prompt = prompt_for_language(cast(Dict[str, Any], OmegaConf.to_container(cfg, resolve=True)), code)
            else:
                prompt = prompt_for_language(cast(Dict[str, Any], OmegaConf.to_container(self.config, resolve=True)), code)
            prompts.append(prompt)

        # Batch tokenize
        inputs = tok(prompts, return_tensors="pt", padding=True, truncation=True)
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        # Get strategy for caching
        used_strategy = strategy or self.config.get("prompt", {}).get("strategy", "vanilla")
        model_name = getattr(self, 'model_name', 'unknown')

        with torch.no_grad():
            gen_max = int(max_length) if max_length is not None else 512
            # Adjust max_length based on input length
            max_input_length = inputs["input_ids"].shape[1]
            adjusted_max_length = min(gen_max, max_input_length + 150)

            outputs = mdl.generate(
                inputs["input_ids"],
                attention_mask=inputs.get("attention_mask"),
                max_length=adjusted_max_length,
                temperature=self.config["model"]["temperature"],
                top_p=self.config["model"]["top_p"],
                top_k=self.config["model"]["top_k"],
                do_sample=True,
                pad_token_id=tok.eos_token_id,
                no_repeat_ngram_size=2,
                early_stopping=True,
                num_return_sequences=1,  # One explanation per input
            )

        # Decode all outputs
        batch_explanations = []
        for i, output in enumerate(outputs):
            generated_text = tok.decode(output, skip_special_tokens=True)
            explanation = generated_text[len(prompts[i]):].strip()
            batch_explanations.append(explanation)

            # Cache the explanation
            if self.explanation_cache is not None:
                self.explanation_cache.put(codes[i], used_strategy, model_name, explanation)

        return batch_explanations

    def analyze_code(self, code: str, strategy: Optional[str] = None) -> Dict[str, Any]:
        """Perform comprehensive code analysis.

        Args:
            code: Python code to analyze
            strategy: Optional prompt strategy override

        Returns:
            Dictionary with analysis results
        """
        explanation = self.explain_code(code, strategy=strategy)

        # Basic analysis
        lines = code.strip().split("\n")
        analysis = {
            "explanation": explanation,
            "line_count": len(lines),
            "character_count": len(code),
            "contains_functions": "def " in code,
            "contains_classes": "class " in code,
            "contains_loops": any(keyword in code for keyword in ["for ", "while "]),
            "contains_conditionals": any(keyword in code for keyword in ["if ", "elif ", "else:"]),
            "contains_imports": any(
                line.strip().startswith(("import ", "from ")) for line in lines
            ),
        }

        return analysis

    def explain_code_with_symbolic(
        self,
        code: str,
        include_symbolic: bool = True,
        max_length: Optional[int] = None,
        strategy: Optional[str] = None,
    ) -> str:
        """Generate explanation with optional symbolic analysis.

        Args:
            code: Source code to explain
            include_symbolic: Whether to include symbolic analysis
            max_length: Optional max sequence length
            strategy: Optional prompt strategy override

        Returns:
            Enhanced explanation with symbolic analysis
        """
        # Get standard explanation
        standard_explanation = self.explain_code(code, max_length, strategy)

        if not include_symbolic:
            return standard_explanation

        # Add symbolic analysis
        symbolic_explanation = self.symbolic_analyzer.analyze_code(code)
        symbolic_text = format_symbolic_explanation(symbolic_explanation)

        if symbolic_text and symbolic_text != "No symbolic conditions detected.":
            enhanced_explanation = f"""## Code Explanation

{standard_explanation}

## Symbolic Analysis

{symbolic_text}

## Summary

The code has been analyzed both semantically and symbolically. The symbolic analysis
provides formal conditions and properties that can be verified through testing."""
            return enhanced_explanation

        return standard_explanation

    def explain_code_multi_agent(
        self, code: str, max_length: Optional[int] = None, strategy: Optional[str] = None
    ) -> str:
        """Generate explanation using multi-agent collaboration.

        Args:
            code: Source code to explain
            max_length: Optional max sequence length (unused for multi-agent)
            strategy: Optional prompt strategy (passed to semantic agent)

        Returns:
            Collaborative explanation from multiple agents
        """
        if self.multi_agent_orchestrator is None:
            # Fallback to regular explanation if multi-agent not available
            return self.explain_code(code, max_length, strategy)

        return self.multi_agent_orchestrator.explain_code_collaborative(code)

    def explain_code_parallel(
        self,
        codes: List[str],
        max_length: Optional[int] = None,
        strategy: Optional[str] = None,
        max_workers: Optional[int] = None
    ) -> List[str]:
        """Generate explanations for multiple code snippets using parallel processing.

        Args:
            codes: List of Python code snippets
            max_length: Optional max sequence length
            strategy: Optional prompt strategy override
            max_workers: Maximum number of worker threads (default: CPU count)

        Returns:
            List of generated explanations
        """
        if not codes:
            return []

        if max_workers is None:
            max_workers = min(len(codes), max(1, concurrent.futures.ThreadPoolExecutor()._max_workers))

        # For small batches, use sequential processing to avoid overhead
        if len(codes) <= 4:
            return self.explain_code_batch(codes, max_length, strategy)

        # Split codes into chunks for parallel processing
        chunk_size = max(1, len(codes) // max_workers)
        code_chunks = [codes[i:i + chunk_size] for i in range(0, len(codes), chunk_size)]

        results = []
        with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
            # Submit tasks for each chunk
            future_to_chunk = {
                executor.submit(self._process_code_chunk, chunk, max_length, strategy): chunk
                for chunk in code_chunks
            }

            # Collect results in order
            for future in concurrent.futures.as_completed(future_to_chunk):
                try:
                    chunk_results = future.result()
                    results.extend(chunk_results)
                except Exception as e:
                    logger.error(f"Error processing code chunk: {str(e)}")
                    # Add empty strings for failed chunks
                    chunk = future_to_chunk[future]
                    results.extend([""] * len(chunk))

        return results

    def _process_code_chunk(
        self,
        codes: List[str],
        max_length: Optional[int] = None,
        strategy: Optional[str] = None
    ) -> List[str]:
        """Process a chunk of codes (used by parallel processing)."""
        try:
            return self.explain_code_batch(codes, max_length, strategy)
        except Exception as e:
            logger.error(f"Error in code chunk processing: {str(e)}")
            return [""] * len(codes)

    def explain_code_with_threading(
        self,
        codes: List[str],
        max_length: Optional[int] = None,
        strategy: Optional[str] = None,
        max_workers: Optional[int] = None
    ) -> List[str]:
        """Generate explanations using threading with model sharing.

        This method is more memory-efficient than process-based parallelism
        but requires careful handling of PyTorch model sharing.

        Args:
            codes: List of Python code snippets
            max_length: Optional max sequence length
            strategy: Optional prompt strategy override
            max_workers: Maximum number of worker threads

        Returns:
            List of generated explanations
        """
        if not codes:
            return []

        if max_workers is None:
            max_workers = min(len(codes), 4)  # Conservative default

        # Use threading with shared model (PyTorch handles thread safety)
        with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = [
                executor.submit(self.explain_code, code, max_length, strategy)
                for code in codes
            ]

            results = []
            for future in concurrent.futures.as_completed(futures):
                try:
                    result = future.result()
                    results.append(result)
                except Exception as e:
                    logger.error(f"Error in threaded explanation: {str(e)}")
                    results.append("")

            # Sort results back to original order
            return results
