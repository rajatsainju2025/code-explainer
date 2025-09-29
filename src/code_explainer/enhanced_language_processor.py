"""
Enhanced Language Processing with Tree-sitter Integration

This module provides robust multi-language parsing, analysis, and pattern recognition
using Tree-sitter for accurate syntax tree generation and language-specific intelligence.
"""

import ast
import json
import re
import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple, Any, Union
from collections import defaultdict

logger = logging.getLogger(__name__)

# Tree-sitter imports (gracefully handle if not installed)
try:
    import tree_sitter
    from tree_sitter import Language, Parser, Tree, Node
    TREE_SITTER_AVAILABLE = True
except ImportError:
    logger.warning("Tree-sitter not available. Install with: pip install tree-sitter")
    TREE_SITTER_AVAILABLE = False
    # Create dummy classes for type hints
    class Language: pass
    class Parser: pass
    class Tree: pass
    class Node: pass


class SupportedLanguage(Enum):
    """Supported programming languages with enhanced analysis."""
    PYTHON = "python"
    JAVASCRIPT = "javascript"
    TYPESCRIPT = "typescript"
    JAVA = "java"
    CPP = "cpp"
    CSHARP = "csharp"
    GO = "go"
    RUST = "rust"
    PHP = "php"
    RUBY = "ruby"


@dataclass
class CodePattern:
    """Represents a detected code pattern."""
    name: str
    type: str  # design_pattern, algorithm, idiom, anti_pattern
    confidence: float
    description: str
    location: Tuple[int, int]  # line_start, line_end
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class FrameworkInfo:
    """Information about detected frameworks."""
    name: str
    version: Optional[str]
    confidence: float
    imports: List[str]
    patterns: List[str]


@dataclass
class LanguageAnalysis:
    """Comprehensive analysis result for a code snippet."""
    language: SupportedLanguage
    confidence: float
    syntax_valid: bool

    # Structure information
    functions: List[Dict[str, Any]]
    classes: List[Dict[str, Any]]
    imports: List[str]
    variables: List[str]

    # Pattern recognition
    patterns: List[CodePattern]
    frameworks: List[FrameworkInfo]
    algorithms: List[str]

    # Quality metrics
    complexity_score: float
    code_smells: List[str]
    best_practices: List[str]

    # Additional metadata
    loc: int  # lines of code
    cyclomatic_complexity: Optional[int]
    maintainability_index: Optional[float]


class LanguageProcessor(ABC):
    """Abstract base class for language-specific processors."""

    @abstractmethod
    def get_language(self) -> SupportedLanguage:
        """Get the language this processor handles."""
        pass

    @abstractmethod
    def parse_code(self, code: str) -> Any:
        """Parse code and return language-specific AST/parse tree."""
        pass

    @abstractmethod
    def extract_structure(self, code: str) -> Dict[str, Any]:
        """Extract structural information (functions, classes, etc.)."""
        pass

    @abstractmethod
    def detect_patterns(self, code: str) -> List[CodePattern]:
        """Detect design patterns and code smells."""
        pass

    @abstractmethod
    def detect_frameworks(self, code: str) -> List[FrameworkInfo]:
        """Detect frameworks and libraries being used."""
        pass


class PythonProcessor(LanguageProcessor):
    """Enhanced Python code processor."""

    def __init__(self):
        self.language = SupportedLanguage.PYTHON
        self.parser = None

        if TREE_SITTER_AVAILABLE:
            try:
                # Try to load Tree-sitter Python grammar
                # Note: This requires tree-sitter-python to be built/installed
                import tree_sitter
                python_lang = tree_sitter.Language("tree_sitter/languages.so", "python")
                self.parser = tree_sitter.Parser()
                self.parser.set_language(python_lang)
            except Exception as e:
                logger.warning(f"Could not load Tree-sitter Python parser: {e}")
                self.parser = None

    def get_language(self) -> SupportedLanguage:
        return self.language

    def parse_code(self, code: str) -> Union[ast.AST, Tree, None]:
        """Parse Python code using Tree-sitter or fallback to ast."""
        if self.parser and TREE_SITTER_AVAILABLE:
            try:
                tree = self.parser.parse(bytes(code, "utf8"))
                return tree
            except Exception as e:
                logger.warning(f"Tree-sitter parsing failed: {e}, falling back to ast")

        # Fallback to standard ast
        try:
            return ast.parse(code)
        except SyntaxError:
            return None

    def extract_structure(self, code: str) -> Dict[str, Any]:
        """Extract Python code structure."""
        try:
            tree = ast.parse(code)
        except SyntaxError:
            return {"functions": [], "classes": [], "imports": [], "variables": []}

        functions = []
        classes = []
        imports = []
        variables = []

        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef):
                func_info = {
                    "name": node.name,
                    "line": node.lineno,
                    "args": [arg.arg for arg in node.args.args],
                    "decorators": [ast.unparse(d) for d in node.decorator_list],
                    "is_async": isinstance(node, ast.AsyncFunctionDef),
                    "has_docstring": (
                        len(node.body) > 0
                        and isinstance(node.body[0], ast.Expr)
                        and isinstance(node.body[0].value, ast.Constant)
                        and isinstance(node.body[0].value.value, str)
                    )
                }
                functions.append(func_info)

            elif isinstance(node, ast.ClassDef):
                class_info = {
                    "name": node.name,
                    "line": node.lineno,
                    "bases": [ast.unparse(base) for base in node.bases],
                    "decorators": [ast.unparse(d) for d in node.decorator_list],
                    "methods": [n.name for n in node.body if isinstance(n, ast.FunctionDef)],
                    "has_docstring": (
                        len(node.body) > 0
                        and isinstance(node.body[0], ast.Expr)
                        and isinstance(node.body[0].value, ast.Constant)
                        and isinstance(node.body[0].value.value, str)
                    )
                }
                classes.append(class_info)

            elif isinstance(node, (ast.Import, ast.ImportFrom)):
                if isinstance(node, ast.Import):
                    for alias in node.names:
                        imports.append(alias.name)
                else:
                    module = node.module or ""
                    for alias in node.names:
                        imports.append(f"{module}.{alias.name}" if module else alias.name)

            elif isinstance(node, ast.Assign):
                for target in node.targets:
                    if isinstance(target, ast.Name):
                        variables.append(target.id)

        return {
            "functions": functions,
            "classes": classes,
            "imports": imports,
            "variables": variables
        }

    def detect_patterns(self, code: str) -> List[CodePattern]:
        """Detect Python design patterns and idioms."""
        patterns = []

        try:
            tree = ast.parse(code)
        except SyntaxError:
            return patterns

        # Singleton pattern detection
        if self._detect_singleton(tree):
            patterns.append(CodePattern(
                name="Singleton",
                type="design_pattern",
                confidence=0.8,
                description="Singleton pattern implementation detected",
                location=(1, len(code.split('\n'))),
                metadata={"pattern_type": "creational"}
            ))

        # Factory pattern detection
        if self._detect_factory(tree):
            patterns.append(CodePattern(
                name="Factory",
                type="design_pattern",
                confidence=0.7,
                description="Factory pattern implementation detected",
                location=(1, len(code.split('\n'))),
                metadata={"pattern_type": "creational"}
            ))

        # Decorator pattern detection
        if self._detect_decorator_pattern(tree):
            patterns.append(CodePattern(
                name="Decorator",
                type="design_pattern",
                confidence=0.8,
                description="Decorator pattern implementation detected",
                location=(1, len(code.split('\n'))),
                metadata={"pattern_type": "structural"}
            ))

        # Observer pattern detection
        if self._detect_observer(tree):
            patterns.append(CodePattern(
                name="Observer",
                type="design_pattern",
                confidence=0.7,
                description="Observer pattern implementation detected",
                location=(1, len(code.split('\n'))),
                metadata={"pattern_type": "behavioral"}
            ))

        # Code smell detection
        smells = self._detect_code_smells(tree)
        patterns.extend(smells)

        return patterns

    def detect_frameworks(self, code: str) -> List[FrameworkInfo]:
        """Detect Python frameworks and libraries."""
        frameworks = []
        imports = self.extract_structure(code)["imports"]

        # Framework detection based on imports
        framework_patterns = {
            "django": {"imports": ["django", "django."], "patterns": ["models.Model", "HttpResponse"]},
            "flask": {"imports": ["flask"], "patterns": ["Flask", "@app.route"]},
            "fastapi": {"imports": ["fastapi"], "patterns": ["FastAPI", "@app.get", "@app.post"]},
            "numpy": {"imports": ["numpy", "np"], "patterns": ["np.array", "numpy."]},
            "pandas": {"imports": ["pandas", "pd"], "patterns": ["pd.DataFrame", "pandas."]},
            "sklearn": {"imports": ["sklearn", "scikit-learn"], "patterns": ["fit", "predict", "transform"]},
            "torch": {"imports": ["torch", "pytorch"], "patterns": ["torch.nn", "torch.optim"]},
            "tensorflow": {"imports": ["tensorflow", "tf"], "patterns": ["tf.keras", "tf.nn"]}
        }

        for framework, info in framework_patterns.items():
            framework_imports = [imp for imp in imports if any(imp.startswith(pattern) for pattern in info["imports"])]

            if framework_imports:
                confidence = min(1.0, len(framework_imports) / len(info["imports"]))

                # Boost confidence if we find framework-specific patterns in code
                pattern_matches = sum(1 for pattern in info["patterns"] if pattern in code)
                if pattern_matches > 0:
                    confidence = min(1.0, confidence + 0.2 * pattern_matches)

                frameworks.append(FrameworkInfo(
                    name=framework,
                    version=None,  # Could extract from imports or requirements
                    confidence=confidence,
                    imports=framework_imports,
                    patterns=info["patterns"]
                ))

        return frameworks

    def _detect_singleton(self, tree: ast.AST) -> bool:
        """Detect Singleton pattern."""
        for node in ast.walk(tree):
            if isinstance(node, ast.ClassDef):
                # Look for __new__ method override
                has_new = any(isinstance(n, ast.FunctionDef) and n.name == "__new__" for n in node.body)
                # Look for class variables like _instance
                has_instance_var = any(
                    isinstance(n, ast.Assign)
                    and any(isinstance(t, ast.Name) and t.id.startswith("_instance") for t in n.targets)
                    for n in node.body
                )
                if has_new or has_instance_var:
                    return True
        return False

    def _detect_factory(self, tree: ast.AST) -> bool:
        """Detect Factory pattern."""
        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef):
                # Factory method naming patterns
                if any(keyword in node.name.lower() for keyword in ["create", "factory", "make", "build"]):
                    # Check if it returns different types based on parameters
                    has_conditional_return = any(
                        isinstance(n, ast.If) and any(isinstance(b, ast.Return) for b in n.body + n.orelse)
                        for n in ast.walk(node)
                    )
                    if has_conditional_return:
                        return True
        return False

    def _detect_decorator_pattern(self, tree: ast.AST) -> bool:
        """Detect Decorator pattern (not Python decorators, but the design pattern)."""
        classes = [node for node in ast.walk(tree) if isinstance(node, ast.ClassDef)]

        for cls in classes:
            # Look for composition (has another object as attribute)
            has_wrapped_object = any(
                isinstance(n, ast.Assign)
                for n in cls.body
                if isinstance(n, ast.Assign)
            )

            # Look for delegation methods
            has_delegation = any(
                isinstance(n, ast.FunctionDef) and any(
                    isinstance(stmt, ast.Return) and isinstance(stmt.value, ast.Call)
                    for stmt in n.body if isinstance(stmt, ast.Return)
                )
                for n in cls.body if isinstance(n, ast.FunctionDef)
            )

            if has_wrapped_object and has_delegation:
                return True
        return False

    def _detect_observer(self, tree: ast.AST) -> bool:
        """Detect Observer pattern."""
        for node in ast.walk(tree):
            if isinstance(node, ast.ClassDef):
                methods = [n.name for n in node.body if isinstance(n, ast.FunctionDef)]
                # Observer pattern methods
                observer_methods = {"attach", "detach", "notify", "update", "subscribe", "unsubscribe"}
                if len(set(methods) & observer_methods) >= 2:
                    return True
        return False

    def _detect_code_smells(self, tree: ast.AST) -> List[CodePattern]:
        """Detect code smells and anti-patterns."""
        smells = []

        for node in ast.walk(tree):
            # Long method detection
            if isinstance(node, ast.FunctionDef) and len(node.body) > 20:
                smells.append(CodePattern(
                    name="Long Method",
                    type="anti_pattern",
                    confidence=0.8,
                    description=f"Method '{node.name}' is too long ({len(node.body)} statements)",
                    location=(node.lineno, node.lineno + len(node.body)),
                    metadata={"lines": len(node.body)}
                ))

            # God class detection
            if isinstance(node, ast.ClassDef) and len(node.body) > 30:
                methods = [n for n in node.body if isinstance(n, ast.FunctionDef)]
                if len(methods) > 15:
                    smells.append(CodePattern(
                        name="God Class",
                        type="anti_pattern",
                        confidence=0.9,
                        description=f"Class '{node.name}' has too many responsibilities ({len(methods)} methods)",
                        location=(node.lineno, node.lineno + len(node.body)),
                        metadata={"methods": len(methods)}
                    ))

            # Deep nesting detection
            if isinstance(node, (ast.If, ast.For, ast.While)):
                nesting_depth = self._calculate_nesting_depth(node)
                if nesting_depth > 4:
                    smells.append(CodePattern(
                        name="Deep Nesting",
                        type="anti_pattern",
                        confidence=0.7,
                        description=f"Code has deep nesting (depth: {nesting_depth})",
                        location=(node.lineno, getattr(node, 'end_lineno', node.lineno + 1)),
                        metadata={"depth": nesting_depth}
                    ))

        return smells

    def _calculate_nesting_depth(self, node: ast.AST, current_depth: int = 0) -> int:
        """Calculate maximum nesting depth starting from a node."""
        max_depth = current_depth

        for child in ast.iter_child_nodes(node):
            if isinstance(child, (ast.If, ast.For, ast.While, ast.With, ast.Try)):
                child_depth = self._calculate_nesting_depth(child, current_depth + 1)
                max_depth = max(max_depth, child_depth)

        return max_depth


class JavaScriptProcessor(LanguageProcessor):
    """JavaScript/TypeScript code processor."""

    def __init__(self):
        self.language = SupportedLanguage.JAVASCRIPT
        self.parser = None

    def get_language(self) -> SupportedLanguage:
        return self.language

    def parse_code(self, code: str) -> Any:
        """Parse JavaScript code (placeholder - would need JS parser)."""
        # For now, return None - would implement with tree-sitter-javascript
        logger.warning("JavaScript parsing not fully implemented yet")
        return None

    def extract_structure(self, code: str) -> Dict[str, Any]:
        """Extract JavaScript code structure using regex patterns."""
        # Basic regex-based extraction (would be better with proper parser)
        functions = []
        classes = []
        imports = []

        # Function detection
        func_pattern = r'(?:function\s+(\w+)|const\s+(\w+)\s*=\s*(?:async\s+)?\([^)]*\)\s*=>|(\w+)\s*:\s*(?:async\s+)?function)'
        for match in re.finditer(func_pattern, code):
            func_name = match.group(1) or match.group(2) or match.group(3)
            functions.append({
                "name": func_name,
                "line": code[:match.start()].count('\n') + 1,
                "type": "function"
            })

        # Class detection
        class_pattern = r'class\s+(\w+)'
        for match in re.finditer(class_pattern, code):
            classes.append({
                "name": match.group(1),
                "line": code[:match.start()].count('\n') + 1
            })

        # Import detection
        import_pattern = r'(?:import\s+[^;]+from\s+["\']([^"\']+)["\']|require\(["\']([^"\']+)["\']\))'
        for match in re.finditer(import_pattern, code):
            module = match.group(1) or match.group(2)
            imports.append(module)

        return {
            "functions": functions,
            "classes": classes,
            "imports": imports,
            "variables": []
        }

    def detect_patterns(self, code: str) -> List[CodePattern]:
        """Detect JavaScript patterns."""
        patterns = []

        # Module pattern detection
        if re.search(r'\(function\s*\([^)]*\)\s*\{.*\}\)\s*\([^)]*\)', code, re.DOTALL):
            patterns.append(CodePattern(
                name="Module Pattern",
                type="design_pattern",
                confidence=0.8,
                description="IIFE module pattern detected",
                location=(1, len(code.split('\n')))
            ))

        # Promise pattern
        if "Promise" in code or ".then(" in code or "async" in code or "await" in code:
            patterns.append(CodePattern(
                name="Promise/Async Pattern",
                type="idiom",
                confidence=0.9,
                description="Asynchronous code pattern detected",
                location=(1, len(code.split('\n')))
            ))

        return patterns

    def detect_frameworks(self, code: str) -> List[FrameworkInfo]:
        """Detect JavaScript frameworks."""
        frameworks = []
        imports = self.extract_structure(code)["imports"]

        # React detection
        if any("react" in imp.lower() for imp in imports) or "jsx" in code.lower():
            frameworks.append(FrameworkInfo(
                name="React",
                version=None,
                confidence=0.9,
                imports=[imp for imp in imports if "react" in imp.lower()],
                patterns=["jsx", "useState", "useEffect"]
            ))

        # Vue detection
        if any("vue" in imp.lower() for imp in imports):
            frameworks.append(FrameworkInfo(
                name="Vue",
                version=None,
                confidence=0.8,
                imports=[imp for imp in imports if "vue" in imp.lower()],
                patterns=["Vue", "v-"]
            ))

        return frameworks


class EnhancedLanguageDetector:
    """Enhanced language detection with confidence scoring."""

    def __init__(self):
        self.processors = {
            SupportedLanguage.PYTHON: PythonProcessor(),
            SupportedLanguage.JAVASCRIPT: JavaScriptProcessor(),
            # Would add more processors here
        }

        # Language detection patterns with confidence weights
        self.patterns = {
            SupportedLanguage.PYTHON: {
                "extensions": [".py", ".pyw"],
                "keywords": ["def ", "import ", "from ", "class ", "if __name__"],
                "patterns": [r"def\s+\w+\s*\(", r"import\s+\w+", r"from\s+\w+\s+import"],
                "shebangs": ["#!/usr/bin/env python", "#!/usr/bin/python"]
            },
            SupportedLanguage.JAVASCRIPT: {
                "extensions": [".js", ".jsx", ".ts", ".tsx"],
                "keywords": ["function ", "const ", "let ", "var ", "=>"],
                "patterns": [r"function\s+\w+\s*\(", r"const\s+\w+\s*=", r"=>\s*\{"],
                "shebangs": ["#!/usr/bin/env node"]
            },
            SupportedLanguage.JAVA: {
                "extensions": [".java"],
                "keywords": ["public class", "private ", "public ", "static"],
                "patterns": [r"public\s+class\s+\w+", r"public\s+static\s+void\s+main"],
                "shebangs": []
            }
        }

    def detect_language(self, code: str, filename: Optional[str] = None) -> Tuple[SupportedLanguage, float]:
        """Detect language with confidence score."""
        scores = defaultdict(float)

        # File extension scoring
        if filename:
            for lang, patterns in self.patterns.items():
                for ext in patterns["extensions"]:
                    if filename.endswith(ext):
                        scores[lang] += 0.4

        # Shebang scoring
        first_line = code.split('\n')[0] if code else ""
        for lang, patterns in self.patterns.items():
            for shebang in patterns["shebangs"]:
                if first_line.startswith(shebang):
                    scores[lang] += 0.3

        # Keyword and pattern scoring
        for lang, patterns in self.patterns.items():
            # Keyword scoring
            keyword_score = sum(1 for keyword in patterns["keywords"] if keyword in code)
            scores[lang] += min(0.3, keyword_score * 0.05)

            # Regex pattern scoring
            pattern_score = sum(1 for pattern in patterns["patterns"] if re.search(pattern, code))
            scores[lang] += min(0.3, pattern_score * 0.1)

        # Return best match
        if scores:
            best_lang = max(scores.keys(), key=lambda k: scores[k])
            confidence = min(1.0, scores[best_lang])
            return best_lang, confidence

        # Fallback to Python (most common in our use case)
        return SupportedLanguage.PYTHON, 0.3

    def analyze_code(self, code: str, filename: Optional[str] = None) -> LanguageAnalysis:
        """Perform comprehensive code analysis."""
        # Detect language
        language, confidence = self.detect_language(code, filename)

        # Get appropriate processor
        processor = self.processors.get(language)
        if not processor:
            # Use Python processor as fallback
            processor = self.processors[SupportedLanguage.PYTHON]

        # Extract structure
        structure = processor.extract_structure(code)

        # Detect patterns
        patterns = processor.detect_patterns(code)

        # Detect frameworks
        frameworks = processor.detect_frameworks(code)

        # Calculate metrics
        loc = len([line for line in code.split('\n') if line.strip()])
        complexity_score = self._calculate_complexity_score(patterns)

        # Extract specific pattern types
        code_smells = [p.name for p in patterns if p.type == "anti_pattern"]
        algorithms = self._detect_algorithms(code, patterns)
        best_practices = self._suggest_best_practices(code, patterns)

        # If Python and AST parses successfully, boost confidence
        if language == SupportedLanguage.PYTHON:
            try:
                if PythonProcessor().parse_code(code) is not None:
                    confidence = max(confidence, 0.6)
            except Exception:
                pass

        return LanguageAnalysis(
            language=language,
            confidence=confidence,
            syntax_valid=processor.parse_code(code) is not None,
            functions=structure["functions"],
            classes=structure["classes"],
            imports=structure["imports"],
            variables=structure.get("variables", []),
            patterns=patterns,
            frameworks=frameworks,
            algorithms=algorithms,
            complexity_score=complexity_score,
            code_smells=code_smells,
            best_practices=best_practices,
            loc=loc,
            cyclomatic_complexity=None,  # Would implement with proper complexity analysis
            maintainability_index=None
        )

    def _calculate_complexity_score(self, patterns: List[CodePattern]) -> float:
        """Calculate overall code complexity score (0-1, higher = more complex)."""
        base_score = 0.1

        # Add complexity for anti-patterns
        anti_patterns = [p for p in patterns if p.type == "anti_pattern"]
        anti_pattern_penalty = len(anti_patterns) * 0.1

        # Add complexity for nesting
        nesting_patterns = [p for p in patterns if p.name == "Deep Nesting"]
        nesting_penalty = sum(p.metadata.get("depth", 0) * 0.02 for p in nesting_patterns)

        return min(1.0, base_score + anti_pattern_penalty + nesting_penalty)

    def _detect_algorithms(self, code: str, patterns: List[CodePattern]) -> List[str]:
        """Detect specific algorithms in the code."""
        algorithms = []

        # Algorithm detection patterns
        if "sort" in code.lower():
            algorithms.append("sorting")
        if "search" in code.lower() or "find" in code.lower():
            algorithms.append("searching")
        if "recursive" in code.lower() or re.search(r'def\s+(\w+).*\1\(', code):
            algorithms.append("recursion")
        if "fibonacci" in code.lower():
            algorithms.append("fibonacci")
        if "factorial" in code.lower():
            algorithms.append("factorial")
        if re.search(r'for.*for.*for', code, re.DOTALL):
            algorithms.append("nested_loops")

        return algorithms

    def _suggest_best_practices(self, code: str, patterns: List[CodePattern]) -> List[str]:
        """Suggest best practices based on code analysis."""
        suggestions = []

        # Check for common issues
        if not any("docstring" in str(p.metadata) for p in patterns):
            suggestions.append("Add docstrings to functions and classes")

        if any(p.type == "anti_pattern" for p in patterns):
            suggestions.append("Refactor identified code smells")

        if "TODO" in code or "FIXME" in code:
            suggestions.append("Address TODO and FIXME comments")

        if len(code.split('\n')) > 100:
            suggestions.append("Consider breaking large files into smaller modules")

        return suggestions


# Create aliases for backward compatibility and easier imports
EnhancedLanguageProcessor = EnhancedLanguageDetector
CodeLanguage = SupportedLanguage
AnalysisResult = LanguageAnalysis

# Global instance for easy access
enhanced_detector = EnhancedLanguageDetector()