"""
Intelligent Explanation Generator

This module provides adaptive, context-aware code explanation generation
using enhanced language processing and pattern recognition.
"""

import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum
from typing import Dict, List, Optional, Any, Union

from .enhanced_language_processor import EnhancedLanguageDetector, LanguageAnalysis, CodePattern, FrameworkInfo

logger = logging.getLogger(__name__)


class ExplanationAudience(Enum):
    """Target audience for explanations."""
    BEGINNER = "beginner"
    INTERMEDIATE = "intermediate"
    EXPERT = "expert"
    AUTOMATIC = "automatic"  # Auto-detect based on code complexity


class ExplanationStyle(Enum):
    """Style of explanation generation."""
    CONCISE = "concise"
    DETAILED = "detailed"
    TUTORIAL = "tutorial"
    REFERENCE = "reference"


@dataclass
class ExplanationContext:
    """Context for generating explanations."""
    audience: ExplanationAudience
    style: ExplanationStyle
    include_examples: bool
    include_best_practices: bool
    include_security_notes: bool
    max_length: Optional[int]
    focus_areas: List[str]  # e.g., ["performance", "security", "maintainability"]


@dataclass
class EnhancedExplanation:
    """Enhanced explanation with structured information."""
    primary_explanation: str
    language_info: str
    structure_analysis: str
    pattern_analysis: str
    framework_info: str
    best_practices: List[str]
    security_notes: List[str]
    examples: List[str]
    related_concepts: List[str]
    complexity_assessment: str
    metadata: Dict[str, Any]


class ExplanationStrategy(ABC):
    """Abstract base class for explanation strategies."""
    
    @abstractmethod
    def generate_explanation(
        self,
        code: str,
        analysis: LanguageAnalysis,
        context: ExplanationContext
    ) -> EnhancedExplanation:
        """Generate explanation based on code analysis and context."""
        pass
    
    @abstractmethod
    def get_strategy_name(self) -> str:
        """Get the name of this strategy."""
        pass


class PatternAwareStrategy(ExplanationStrategy):
    """Strategy that focuses on design patterns and code structure."""
    
    def get_strategy_name(self) -> str:
        return "pattern_aware"
    
    def generate_explanation(
        self,
        code: str,
        analysis: LanguageAnalysis,
        context: ExplanationContext
    ) -> EnhancedExplanation:
        """Generate pattern-focused explanation."""
        
        # Build primary explanation
        explanation_parts = [
            f"This is a {analysis.language.value} code snippet"
        ]
        
        # Add complexity assessment
        if analysis.complexity_score < 0.3:
            complexity_desc = "with low complexity"
        elif analysis.complexity_score < 0.7:
            complexity_desc = "with moderate complexity"
        else:
            complexity_desc = "with high complexity"
        
        explanation_parts[0] += f" {complexity_desc}."
        
        # Structure analysis
        structure_parts = []
        if analysis.functions:
            func_names = [f["name"] for f in analysis.functions[:5]]
            structure_parts.append(f"Contains {len(analysis.functions)} function(s): {', '.join(func_names)}")
        
        if analysis.classes:
            class_names = [c["name"] for c in analysis.classes[:3]]
            structure_parts.append(f"Defines {len(analysis.classes)} class(es): {', '.join(class_names)}")
        
        if analysis.imports:
            structure_parts.append(f"Imports {len(analysis.imports)} module(s)")
        
        structure_analysis = ". ".join(structure_parts) + "." if structure_parts else "No major structural elements detected."
        
        # Pattern analysis
        pattern_parts = []
        design_patterns = [p for p in analysis.patterns if p.type == "design_pattern"]
        if design_patterns:
            pattern_names = [p.name for p in design_patterns]
            pattern_parts.append(f"Implements design pattern(s): {', '.join(pattern_names)}")
        
        algorithms = analysis.algorithms
        if algorithms:
            pattern_parts.append(f"Uses algorithm(s): {', '.join(algorithms)}")
        
        code_smells = [p for p in analysis.patterns if p.type == "anti_pattern"]
        if code_smells and context.include_best_practices:
            smell_names = [p.name for p in code_smells]
            pattern_parts.append(f"⚠️ Code smells detected: {', '.join(smell_names)}")
        
        pattern_analysis = ". ".join(pattern_parts) + "." if pattern_parts else "No specific patterns detected."
        
        # Framework analysis
        framework_parts = []
        if analysis.frameworks:
            for fw in analysis.frameworks:
                confidence_desc = "likely" if fw.confidence > 0.7 else "possibly"
                framework_parts.append(f"{confidence_desc} uses {fw.name}")
        
        framework_info = ". ".join(framework_parts) + "." if framework_parts else "No frameworks detected."
        
        # Best practices
        best_practices = analysis.best_practices.copy()
        if context.include_best_practices:
            if design_patterns:
                best_practices.append("Good use of design patterns for maintainable code")
            if code_smells:
                best_practices.extend([
                    f"Address {smell.name}: {smell.description}"
                    for smell in code_smells
                ])
        
        # Security notes
        security_notes = []
        if context.include_security_notes:
            # Check for potential security issues
            if "exec(" in code or "eval(" in code:
                security_notes.append("⚠️ Potential code injection risk: avoid exec() and eval()")
            if "shell=True" in code:
                security_notes.append("⚠️ Shell injection risk: avoid shell=True in subprocess calls")
            if "pickle.load" in code:
                security_notes.append("⚠️ Pickle deserialization can be unsafe with untrusted data")
        
        # Examples (would be enhanced with actual example generation)
        examples = []
        if context.include_examples and context.audience == ExplanationAudience.BEGINNER:
            if analysis.functions:
                examples.append("Example usage patterns would be shown here")
        
        # Related concepts
        related_concepts = []
        for pattern in design_patterns:
            if pattern.name == "Singleton":
                related_concepts.extend(["Object-oriented programming", "Creational patterns", "Global state"])
            elif pattern.name == "Observer":
                related_concepts.extend(["Event-driven programming", "Pub/sub pattern", "Behavioral patterns"])
        
        if algorithms:
            if "sorting" in algorithms:
                related_concepts.extend(["Algorithm complexity", "Big O notation"])
            if "recursion" in algorithms:
                related_concepts.extend(["Base cases", "Stack overflow", "Dynamic programming"])
        
        # Combine primary explanation
        primary_parts = [explanation_parts[0]]
        
        if context.style == ExplanationStyle.DETAILED:
            primary_parts.extend([
                f"\n\n**Purpose**: {self._infer_code_purpose(analysis)}",
                f"\n**Key Components**: {structure_analysis}",
            ])
            
            if pattern_analysis != "No specific patterns detected.":
                primary_parts.append(f"\n**Patterns**: {pattern_analysis}")
        
        primary_explanation = "".join(primary_parts)
        
        return EnhancedExplanation(
            primary_explanation=primary_explanation,
            language_info=f"Language: {analysis.language.value.title()} (confidence: {analysis.confidence:.1%})",
            structure_analysis=structure_analysis,
            pattern_analysis=pattern_analysis,
            framework_info=framework_info,
            best_practices=best_practices,
            security_notes=security_notes,
            examples=examples,
            related_concepts=related_concepts,
            complexity_assessment=f"Complexity: {complexity_desc}",
            metadata={
                "analysis": analysis,
                "context": context,
                "strategy": self.get_strategy_name()
            }
        )
    
    def _infer_code_purpose(self, analysis: LanguageAnalysis) -> str:
        """Infer the purpose of the code from analysis."""
        purposes = []
        
        # Function-based purpose
        if analysis.functions:
            func_names = [f["name"].lower() for f in analysis.functions]
            if any("test" in name for name in func_names):
                purposes.append("testing")
            if any("main" in name for name in func_names):
                purposes.append("application entry point")
            if any(name.startswith("get_") for name in func_names):
                purposes.append("data retrieval")
            if any(name.startswith("set_") or name.startswith("update_") for name in func_names):
                purposes.append("data modification")
        
        # Pattern-based purpose
        design_patterns = [p.name for p in analysis.patterns if p.type == "design_pattern"]
        if "Singleton" in design_patterns:
            purposes.append("single instance management")
        if "Factory" in design_patterns:
            purposes.append("object creation")
        if "Observer" in design_patterns:
            purposes.append("event handling")
        
        # Framework-based purpose
        if analysis.frameworks:
            fw_names = [fw.name.lower() for fw in analysis.frameworks]
            if any("web" in fw or fw in ["django", "flask", "fastapi"] for fw in fw_names):
                purposes.append("web application")
            if any(fw in ["numpy", "pandas", "sklearn"] for fw in fw_names):
                purposes.append("data analysis")
            if any(fw in ["torch", "tensorflow"] for fw in fw_names):
                purposes.append("machine learning")
        
        if purposes:
            return f"This code appears to be for {', '.join(purposes[:3])}"
        else:
            return "General-purpose code implementation"


class AdaptiveExplanationStrategy(ExplanationStrategy):
    """Strategy that adapts explanation based on code complexity and audience."""
    
    def get_strategy_name(self) -> str:
        return "adaptive"
    
    def generate_explanation(
        self,
        code: str,
        analysis: LanguageAnalysis,
        context: ExplanationContext
    ) -> EnhancedExplanation:
        """Generate adaptive explanation based on complexity and audience."""
        
        # Auto-detect audience if needed
        if context.audience == ExplanationAudience.AUTOMATIC:
            context.audience = self._detect_audience(analysis)
        
        # Choose sub-strategy based on code characteristics
        if analysis.patterns and len(analysis.patterns) > 2:
            # Use pattern-aware strategy for pattern-rich code
            pattern_strategy = PatternAwareStrategy()
            return pattern_strategy.generate_explanation(code, analysis, context)
        elif analysis.complexity_score > 0.7:
            # Use detailed explanation for complex code, but prefer caller's choice if provided
            if context.style == ExplanationStyle.CONCISE:
                # caller explicitly asked concise, keep it
                pass
            else:
                context.style = ExplanationStyle.DETAILED
            pattern_strategy = PatternAwareStrategy()
            return pattern_strategy.generate_explanation(code, analysis, context)
        else:
            # Use concise explanation for simple code if caller didn't request detailed/tutorial/reference
            if context.style not in (ExplanationStyle.DETAILED, ExplanationStyle.TUTORIAL, ExplanationStyle.REFERENCE):
                context.style = ExplanationStyle.CONCISE
            return self._generate_simple_explanation(code, analysis, context)
    
    def _detect_audience(self, analysis: LanguageAnalysis) -> ExplanationAudience:
        """Detect appropriate audience based on code complexity."""
        if analysis.complexity_score < 0.3 and not analysis.patterns:
            return ExplanationAudience.BEGINNER
        elif analysis.complexity_score > 0.7 or len(analysis.patterns) > 3:
            return ExplanationAudience.EXPERT
        else:
            return ExplanationAudience.INTERMEDIATE
    
    def _generate_simple_explanation(
        self,
        code: str,
        analysis: LanguageAnalysis,
        context: ExplanationContext
    ) -> EnhancedExplanation:
        """Generate simple explanation for straightforward code."""
        
        # Simple primary explanation, audience-aware wording
        parts = [f"This {analysis.language.value} code"]
        
        if analysis.functions:
            if len(analysis.functions) == 1:
                parts.append(f"defines a function called '{analysis.functions[0]['name']}'")
            else:
                parts.append(f"defines {len(analysis.functions)} functions")
        elif analysis.classes:
            if len(analysis.classes) == 1:
                parts.append(f"defines a class called '{analysis.classes[0]['name']}'")
            else:
                parts.append(f"defines {len(analysis.classes)} classes")
        else:
            parts.append("contains basic programming logic")
        
        primary_explanation = " ".join(parts) + "."
        # Add audience-tailored suffix to differentiate outputs
        if context.audience == ExplanationAudience.BEGINNER:
            primary_explanation += " It uses straightforward constructs suitable for learning."
        elif context.audience == ExplanationAudience.EXPERT:
            primary_explanation += " Focus is on core semantics without step-by-step guidance."
        
        # Minimal structure analysis
        structure_analysis = f"Contains {analysis.loc} lines of code."
        
        return EnhancedExplanation(
            primary_explanation=primary_explanation,
            language_info=f"Language: {analysis.language.value.title()}",
            structure_analysis=structure_analysis,
            pattern_analysis="Simple, straightforward implementation.",
            framework_info="No frameworks detected." if not analysis.frameworks else f"Uses {analysis.frameworks[0].name}.",
            best_practices=analysis.best_practices[:2],  # Limit to top suggestions
            security_notes=[],
            examples=[],
            related_concepts=[],
            complexity_assessment="Low complexity",
            metadata={
                "analysis": analysis,
                "context": context,
                "strategy": self.get_strategy_name()
            }
        )


class IntelligentExplanationGenerator:
    """Main class for generating intelligent, context-aware explanations."""
    
    def __init__(self):
        self.language_detector = EnhancedLanguageDetector()
        self.strategies = {
            "pattern_aware": PatternAwareStrategy(),
            "adaptive": AdaptiveExplanationStrategy(),
        }
        self.default_strategy = "adaptive"
    
    def explain_code(
        self,
        code: str,
        strategy: Optional[str] = None,
        audience: ExplanationAudience = ExplanationAudience.AUTOMATIC,
        style: ExplanationStyle = ExplanationStyle.DETAILED,
        include_examples: bool = False,
        include_best_practices: bool = True,
        include_security_notes: bool = True,
        max_length: Optional[int] = None,
        focus_areas: Optional[List[str]] = None,
        filename: Optional[str] = None
    ) -> EnhancedExplanation:
        """Generate intelligent explanation for code."""
        
        # Analyze code
        analysis = self.language_detector.analyze_code(code, filename)
        
        # Create context
        context = ExplanationContext(
            audience=audience,
            style=style,
            include_examples=include_examples,
            include_best_practices=include_best_practices,
            include_security_notes=include_security_notes,
            max_length=max_length,
            focus_areas=focus_areas or []
        )
        
        # Choose strategy
        strategy_name = strategy or self.default_strategy
        if strategy_name not in self.strategies:
            logger.warning(f"Unknown strategy '{strategy_name}', using default")
            strategy_name = self.default_strategy
        
        explanation_strategy = self.strategies[strategy_name]
        
        # Generate explanation
        return explanation_strategy.generate_explanation(code, analysis, context)
    
    def explain_code_batch(
        self,
        codes: List[str],
        **kwargs
    ) -> List[EnhancedExplanation]:
        """Generate explanations for multiple code snippets."""
        return [self.explain_code(code, **kwargs) for code in codes]
    
    def get_available_strategies(self) -> List[str]:
        """Get list of available explanation strategies."""
        return list(self.strategies.keys())
    
    def add_strategy(self, name: str, strategy: ExplanationStrategy):
        """Add a new explanation strategy."""
        self.strategies[name] = strategy
    
    def format_explanation(
        self,
        explanation: EnhancedExplanation,
        format_type: str = "markdown"
    ) -> str:
        """Format explanation for display."""
        if format_type == "markdown":
            return self._format_markdown(explanation)
        elif format_type == "plain":
            return self._format_plain(explanation)
        else:
            return explanation.primary_explanation
    
    def _format_markdown(self, explanation: EnhancedExplanation) -> str:
        """Format explanation as Markdown."""
        parts = [
            "# Code Explanation\n",
            explanation.primary_explanation,
            f"\n\n## {explanation.language_info}",
            f"\n**{explanation.complexity_assessment}**",
            f"\n\n## Structure Analysis\n{explanation.structure_analysis}",
        ]
        
        if explanation.pattern_analysis != "No specific patterns detected.":
            parts.append(f"\n\n## Pattern Analysis\n{explanation.pattern_analysis}")
        
        if explanation.framework_info != "No frameworks detected.":
            parts.append(f"\n\n## Frameworks\n{explanation.framework_info}")
        
        if explanation.best_practices:
            parts.append("\n\n## Best Practices")
            for practice in explanation.best_practices:
                parts.append(f"\n- {practice}")
        
        if explanation.security_notes:
            parts.append("\n\n## Security Notes")
            for note in explanation.security_notes:
                parts.append(f"\n- {note}")
        
        if explanation.related_concepts:
            parts.append(f"\n\n## Related Concepts\n{', '.join(explanation.related_concepts)}")
        
        return "".join(parts)
    
    def _format_plain(self, explanation: EnhancedExplanation) -> str:
        """Format explanation as plain text."""
        parts = [
            "CODE EXPLANATION\n",
            "=" * 50,
            f"\n{explanation.primary_explanation}",
            f"\n\n{explanation.language_info}",
            f"{explanation.complexity_assessment}",
            f"\n\nStructure: {explanation.structure_analysis}",
        ]
        
        if explanation.best_practices:
            parts.append(f"\n\nBest Practices:\n- " + "\n- ".join(explanation.best_practices))
        
        return "".join(parts)


# Global instance for easy access
intelligent_explainer = IntelligentExplanationGenerator()