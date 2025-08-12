"""Main model class for code explanation."""

import logging
from typing import List, Dict, Any, Optional
from pathlib import Path

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoModelForSeq2SeqLM

from .utils import load_config, get_device, prompt_for_language
from .symbolic import SymbolicAnalyzer, format_symbolic_explanation
from .multi_agent import MultiAgentOrchestrator

logger = logging.getLogger(__name__)


class CodeExplainer:
    """Main class for code explanation inference."""
    
    def __init__(self, model_path: str = "./results", config_path: str = "configs/default.yaml"):
        """Initialize the code explainer.
        
        Args:
            model_path: Path to trained model directory
            config_path: Path to configuration file
        """
        self.config = load_config(config_path)
        self.device = get_device()
        self.model_path = Path(model_path)
        self.arch = self.config.get("model", {}).get("arch", "causal")
        
        # Initialize symbolic analyzer
        self.symbolic_analyzer = SymbolicAnalyzer()
        
        # Initialize multi-agent orchestrator (will be set up after model loading)
        self.multi_agent_orchestrator = None
        
        # Initialize model and tokenizer
        self.tokenizer = None
        self.model = None
        self._load_model()
        
        # Set up multi-agent system after model is loaded
        self.multi_agent_orchestrator = MultiAgentOrchestrator(self)
        
    def _load_model(self) -> None:
        """Load the trained model and tokenizer."""
        dtype_str = self.config.get("model", {}).get("torch_dtype", "auto")
        dtype_map = {
            "float16": torch.float16,
            "fp16": torch.float16,
            "bfloat16": torch.bfloat16,
            "bf16": torch.bfloat16,
            "auto": "auto",
        }
        torch_dtype = dtype_map.get(str(dtype_str).lower(), "auto")
        load_in_8bit = bool(self.config.get("model", {}).get("load_in_8bit", False))
        if load_in_8bit and self.device in ("cpu", "mps"):
            load_in_8bit = False
        
        from typing import Union
        def _load_from(src: Union[Path, str]):
            tok = AutoTokenizer.from_pretrained(src)
            if getattr(tok, "pad_token", None) is None:
                tok.pad_token = tok.eos_token  # type: ignore[assignment]
            model_kwargs: Dict[str, Any] = {"torch_dtype": torch_dtype}
            if load_in_8bit:
                model_kwargs.update({"load_in_8bit": True, "device_map": "auto"})
            if self.arch == "seq2seq":
                mdl = AutoModelForSeq2SeqLM.from_pretrained(src, **model_kwargs)
            else:
                mdl = AutoModelForCausalLM.from_pretrained(src, **model_kwargs)
                if getattr(mdl.config, "pad_token_id", None) is None:
                    mdl.config.pad_token_id = tok.pad_token_id  # type: ignore[assignment]
            if not load_in_8bit:
                mdl.to(self.device)
            mdl.eval()
            return tok, mdl
        
        try:
            logger.info(f"Loading model from {self.model_path}")
            self.tokenizer, self.model = _load_from(self.model_path)
            logger.info("Model loaded successfully")
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            # Fallback to base model
            logger.info("Falling back to base model")
            model_name = self.config["model"]["name"]
            self.tokenizer, self.model = _load_from(model_name)
            
    def explain_code(self, code: str, max_length: Optional[int] = None, strategy: Optional[str] = None) -> str:
        """Generate explanation for the given code.
        
        Args:
            code: Source code to explain
            max_length: Optional max sequence length
            strategy: Optional prompt strategy override (e.g., "vanilla", "ast_augmented")
        """
        if max_length is None:
            max_length = self.config["model"]["max_length"]
        
        # Bind tokenizer/model to satisfy type checkers
        assert self.tokenizer is not None and self.model is not None
        tok = self.tokenizer
        mdl = self.model
        
        # Language-aware prompt with optional strategy override
        if strategy is not None:
            import copy
            cfg = copy.deepcopy(self.config)
            cfg.setdefault("prompt", {})["strategy"] = strategy
            prompt = prompt_for_language(cfg, code)
        else:
            prompt = prompt_for_language(self.config, code)
        
        inputs = tok(prompt, return_tensors="pt")
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        with torch.no_grad():
            gen_max = int(max_length) if max_length is not None else 512
            outputs = mdl.generate(
                inputs["input_ids"],
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
        explanation = generated_text[len(prompt):].strip()
        return explanation
    
    def explain_code_batch(self, codes: List[str]) -> List[str]:
        """Generate explanations for multiple code snippets.
        
        Args:
            codes: List of Python code snippets
            
        Returns:
            List of generated explanations
        """
        explanations = []
        for code in codes:
            explanation = self.explain_code(code)
            explanations.append(explanation)
            
        return explanations
    
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
        lines = code.strip().split('\n')
        analysis = {
            "explanation": explanation,
            "line_count": len(lines),
            "character_count": len(code),
            "contains_functions": "def " in code,
            "contains_classes": "class " in code,
            "contains_loops": any(keyword in code for keyword in ["for ", "while "]),
            "contains_conditionals": any(keyword in code for keyword in ["if ", "elif ", "else:"]),
            "contains_imports": any(line.strip().startswith(("import ", "from ")) for line in lines),
        }
        
        return analysis
    
    def explain_code_with_symbolic(self, code: str, include_symbolic: bool = True, 
                                   max_length: Optional[int] = None, 
                                   strategy: Optional[str] = None) -> str:
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

The code has been analyzed both semantically and symbolically. The symbolic analysis provides formal conditions and properties that can be verified through testing."""
            return enhanced_explanation
        
        return standard_explanation
    
    def explain_code_multi_agent(self, code: str, max_length: Optional[int] = None, 
                                 strategy: Optional[str] = None) -> str:
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
