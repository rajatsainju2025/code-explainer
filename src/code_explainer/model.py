"""Main model class for code explanation."""

import logging
from typing import List, Dict, Any, Optional
from pathlib import Path

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoModelForSeq2SeqLM

from .utils import load_config, get_device, prompt_for_language

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
        
        # Initialize model and tokenizer
        self.tokenizer = None
        self.model = None
        self._load_model()
        
    def _load_model(self) -> None:
        """Load the trained model and tokenizer."""
        try:
            logger.info(f"Loading model from {self.model_path}")
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_path)
            if self.arch == "seq2seq":
                self.model = AutoModelForSeq2SeqLM.from_pretrained(self.model_path)
            else:
                self.model = AutoModelForCausalLM.from_pretrained(self.model_path)
            self.model.to(self.device)
            self.model.eval()
            logger.info("Model loaded successfully")
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            # Fallback to base model
            logger.info("Falling back to base model")
            model_name = self.config["model"]["name"]
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
            if self.arch == "seq2seq":
                self.model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
            else:
                self.model = AutoModelForCausalLM.from_pretrained(model_name)
            self.model.to(self.device)
            self.model.eval()
            
    def explain_code(self, code: str, max_length: Optional[int] = None) -> str:
        """Generate explanation for the given code."""
        if max_length is None:
            max_length = self.config["model"]["max_length"]
        
        # Bind tokenizer/model to satisfy type checkers
        assert self.tokenizer is not None and self.model is not None
        tok = self.tokenizer
        mdl = self.model
        
        # Language-aware prompt
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
    
    def analyze_code(self, code: str) -> Dict[str, Any]:
        """Perform comprehensive code analysis.
        
        Args:
            code: Python code to analyze
            
        Returns:
            Dictionary with analysis results
        """
        explanation = self.explain_code(code)
        
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
