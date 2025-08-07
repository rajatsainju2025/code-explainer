"""Main model class for code explanation."""

import logging
from typing import List, Dict, Any, Optional
from pathlib import Path

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

from .utils import load_config, get_device

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
        
        # Initialize model and tokenizer
        self.tokenizer = None
        self.model = None
        self._load_model()
        
    def _load_model(self) -> None:
        """Load the trained model and tokenizer."""
        try:
            logger.info(f"Loading model from {self.model_path}")
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_path)
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
            self.model = AutoModelForCausalLM.from_pretrained(model_name)
            self.model.to(self.device)
            self.model.eval()
            
    def explain_code(self, code: str, max_length: Optional[int] = None) -> str:
        """Generate explanation for the given code.
        
        Args:
            code: Python code to explain
            max_length: Maximum length of generated explanation
            
        Returns:
            Generated explanation
        """
        if max_length is None:
            max_length = self.config["model"]["max_length"]
            
        # Format prompt
        prompt_template = self.config["prompt"]["template"]
        prompt = prompt_template.format(code=code.strip())
        
        # Tokenize input
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
        
        # Generate explanation
        with torch.no_grad():
            outputs = self.model.generate(
                inputs.input_ids,
                max_length=len(inputs.input_ids[0]) + 150,  # Add space for explanation
                temperature=self.config["model"]["temperature"],
                top_p=self.config["model"]["top_p"],
                top_k=self.config["model"]["top_k"],
                do_sample=True,
                pad_token_id=self.tokenizer.eos_token_id,
                no_repeat_ngram_size=2,
                early_stopping=True
            )
        
        # Decode and extract explanation
        generated_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
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
