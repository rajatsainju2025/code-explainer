"""AI-powered code generation engine with templates and patterns."""

import asyncio
import json
import logging
from typing import Dict, List, Any, Optional, Callable
from dataclasses import dataclass, field
from pathlib import Path

logger = logging.getLogger(__name__)

@dataclass
class CodeTemplate:
    """Template for code generation."""
    name: str
    language: str
    category: str
    template: str
    variables: List[str]
    examples: List[str] = field(default_factory=list)

@dataclass
class GenerationRequest:
    """Request for code generation."""
    prompt: str
    language: str
    context: Dict[str, Any] = field(default_factory=dict)
    constraints: Dict[str, Any] = field(default_factory=dict)

class CodeGenerationEngine:
    """AI-powered code generation with templates."""
    
    def __init__(self, model_fn: Callable[[str], str], templates_path: Optional[str] = None):
        self.model_fn = model_fn
        self.templates: Dict[str, CodeTemplate] = {}
        self.templates_path = templates_path or "templates"
        self._load_templates()
    
    def _load_templates(self):
        """Load code templates."""
        # Default templates
        self.templates = {
            "function": CodeTemplate(
                name="function",
                language="python",
                category="basic",
                template="def {name}({params}):\n    \"\"\"{docstring}\"\"\"\n    {body}\n    return {return_value}",
                variables=["name", "params", "docstring", "body", "return_value"],
                examples=["def add(a, b): return a + b"]
            ),
            "class": CodeTemplate(
                name="class",
                language="python",
                category="oop",
                template="class {name}:\n    \"\"\"{docstring}\"\"\"\n    \n    def __init__(self, {params}):\n        {init_body}\n    \n    {methods}",
                variables=["name", "docstring", "params", "init_body", "methods"],
                examples=["class Calculator:\n    def add(self, a, b): return a + b"]
            )
        }
    
    async def generate_code(self, request: GenerationRequest) -> str:
        """Generate code based on request."""
        # Use template if available
        template = self._find_template(request.language, request.prompt)
        if template:
            return self._fill_template(template, request)
        
        # AI generation
        prompt = f"Generate {request.language} code for: {request.prompt}"
        if request.constraints:
            prompt += f"\nConstraints: {json.dumps(request.constraints)}"
        
        return await asyncio.to_thread(self.model_fn, prompt)
    
    def _find_template(self, language: str, prompt: str) -> Optional[CodeTemplate]:
        """Find matching template."""
        for template in self.templates.values():
            if template.language == language and any(keyword in prompt.lower() for keyword in template.name.split()):
                return template
        return None
    
    def _fill_template(self, template: CodeTemplate, request: GenerationRequest) -> str:
        """Fill template with request data."""
        # Simple template filling - in practice would use more sophisticated logic
        filled = template.template
        for var in template.variables:
            if var in request.context:
                filled = filled.replace(f"{{{var}}}", str(request.context[var]))
        return filled
    
    def add_template(self, template: CodeTemplate):
        """Add new template."""
        self.templates[template.name] = template
    
    def get_templates(self, language: str) -> List[CodeTemplate]:
        """Get templates for language."""
        return [t for t in self.templates.values() if t.language == language]

# Example usage
async def demo_code_generation():
    """Demo code generation."""
    def mock_model(prompt: str) -> str:
        return f"Generated code for: {prompt[:50]}..."
    
    engine = CodeGenerationEngine(mock_model)
    
    request = GenerationRequest(
        prompt="Create a function to calculate fibonacci",
        language="python",
        context={"name": "fibonacci", "params": "n"}
    )
    
    code = await engine.generate_code(request)
    print(f"Generated: {code}")

if __name__ == "__main__":
    asyncio.run(demo_code_generation())
