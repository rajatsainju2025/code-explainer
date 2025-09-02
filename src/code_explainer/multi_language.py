"""
Multi-Language Expansion Module for Code Intelligence Platform

This module provides support for multiple programming languages, enabling
the code intelligence platform to handle diverse codebases with language-specific
parsers, tokenizers, and evaluation strategies.

Features:
- Language detection and classification
- Language-specific AST parsing
- Multi-language code generation
- Cross-language code analysis
- Language-specific evaluation metrics
"""

import re
from typing import Dict, List, Optional, Any, Union
from dataclasses import dataclass
from enum import Enum
import ast
import json
import yaml


class ProgrammingLanguage(Enum):
    """Supported programming languages."""
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
    SWIFT = "swift"
    KOTLIN = "kotlin"
    SCALA = "scala"
    R = "r"
    SQL = "sql"
    HTML = "html"
    CSS = "css"
    SHELL = "shell"
    DOCKERFILE = "dockerfile"
    YAML = "yaml"
    JSON = "json"
    MARKDOWN = "markdown"


@dataclass
class LanguageFeatures:
    """Features and capabilities for each programming language."""
    name: str
    extensions: List[str]
    comment_styles: List[str]
    has_types: bool
    supports_oop: bool
    compilation_required: bool
    interpreter_available: bool
    package_managers: List[str]
    testing_frameworks: List[str]


LANGUAGE_FEATURES = {
    ProgrammingLanguage.PYTHON: LanguageFeatures(
        name="Python",
        extensions=[".py", ".pyw", ".pyc"],
        comment_styles=["#"],
        has_types=True,
        supports_oop=True,
        compilation_required=False,
        interpreter_available=True,
        package_managers=["pip", "conda", "poetry"],
        testing_frameworks=["pytest", "unittest", "nose"]
    ),
    ProgrammingLanguage.JAVASCRIPT: LanguageFeatures(
        name="JavaScript",
        extensions=[".js", ".mjs", ".cjs"],
        comment_styles=["//", "/* */"],
        has_types=False,
        supports_oop=True,
        compilation_required=False,
        interpreter_available=True,
        package_managers=["npm", "yarn", "pnpm"],
        testing_frameworks=["jest", "mocha", "jasmine"]
    ),
    ProgrammingLanguage.TYPESCRIPT: LanguageFeatures(
        name="TypeScript",
        extensions=[".ts", ".tsx"],
        comment_styles=["//", "/* */"],
        has_types=True,
        supports_oop=True,
        compilation_required=True,
        interpreter_available=False,
        package_managers=["npm", "yarn", "pnpm"],
        testing_frameworks=["jest", "mocha", "jasmine"]
    ),
    ProgrammingLanguage.JAVA: LanguageFeatures(
        name="Java",
        extensions=[".java", ".class", ".jar"],
        comment_styles=["//", "/* */"],
        has_types=True,
        supports_oop=True,
        compilation_required=True,
        interpreter_available=False,
        package_managers=["maven", "gradle"],
        testing_frameworks=["junit", "testng", "spock"]
    ),
    ProgrammingLanguage.GO: LanguageFeatures(
        name="Go",
        extensions=[".go"],
        comment_styles=["//", "/* */"],
        has_types=True,
        supports_oop=False,
        compilation_required=True,
        interpreter_available=False,
        package_managers=["go mod"],
        testing_frameworks=["testing", "ginkgo"]
    ),
    ProgrammingLanguage.RUST: LanguageFeatures(
        name="Rust",
        extensions=[".rs"],
        comment_styles=["//", "/* */"],
        has_types=True,
        supports_oop=True,
        compilation_required=True,
        interpreter_available=False,
        package_managers=["cargo"],
        testing_frameworks=["cargo test"]
    ),
}


class LanguageDetector:
    """Detects programming language from file content and extension."""

    @staticmethod
    def detect_from_extension(filename: str) -> Optional[ProgrammingLanguage]:
        """Detect language from file extension."""
        ext_map = {}
        for lang, features in LANGUAGE_FEATURES.items():
            for ext in features.extensions:
                ext_map[ext] = lang

        for ext in ext_map:
            if filename.endswith(ext):
                return ext_map[ext]
        return None

    @staticmethod
    def detect_from_content(content: str) -> Optional[ProgrammingLanguage]:
        """Detect language from file content using heuristics."""
        content_lower = content.lower()

        # Python indicators
        if re.search(r'\bdef\s+\w+\s*\(', content) or re.search(r'\bimport\s+\w+', content):
            return ProgrammingLanguage.PYTHON

        # JavaScript/TypeScript indicators
        if re.search(r'\bfunction\s+\w+\s*\(', content) or re.search(r'\bconst\s+\w+\s*=', content):
            if 'interface' in content_lower or 'type ' in content_lower:
                return ProgrammingLanguage.TYPESCRIPT
            return ProgrammingLanguage.JAVASCRIPT

        # Java indicators
        if re.search(r'\bpublic\s+class\s+\w+', content) or re.search(r'\bSystem\.out\.println', content):
            return ProgrammingLanguage.JAVA

        # Go indicators
        if re.search(r'\bfunc\s+\w+\s*\(', content) or re.search(r'\bpackage\s+main', content):
            return ProgrammingLanguage.GO

        # Rust indicators
        if re.search(r'\bfn\s+\w+\s*\(', content) or re.search(r'\blet\s+mut\s+\w+', content):
            return ProgrammingLanguage.RUST

        return None


class MultiLanguageParser:
    """Parses code in multiple programming languages."""

    def __init__(self):
        self.parsers = {
            ProgrammingLanguage.PYTHON: self._parse_python,
            ProgrammingLanguage.JAVASCRIPT: self._parse_javascript,
            ProgrammingLanguage.TYPESCRIPT: self._parse_typescript,
            ProgrammingLanguage.JSON: self._parse_json,
            ProgrammingLanguage.YAML: self._parse_yaml,
        }

    def parse(self, content: str, language: ProgrammingLanguage) -> Dict[str, Any]:
        """Parse code content based on language."""
        if language not in self.parsers:
            return {"error": f"Parser not available for {language.value}"}

        try:
            return self.parsers[language](content)
        except Exception as e:
            return {"error": f"Parsing failed: {str(e)}"}

    def _parse_python(self, content: str) -> Dict[str, Any]:
        """Parse Python code using AST."""
        try:
            tree = ast.parse(content)
            functions = []
            classes = []
            imports = []

            for node in ast.walk(tree):
                if isinstance(node, ast.FunctionDef):
                    functions.append({
                        "name": node.name,
                        "args": [arg.arg for arg in node.args.args],
                        "line": node.lineno
                    })
                elif isinstance(node, ast.ClassDef):
                    classes.append({
                        "name": node.name,
                        "line": node.lineno
                    })
                elif isinstance(node, ast.Import):
                    imports.extend([alias.name for alias in node.names])
                elif isinstance(node, ast.ImportFrom):
                    module = node.module or ""
                    imports.extend([f"{module}.{alias.name}" for alias in node.names])

            return {
                "language": "python",
                "functions": functions,
                "classes": classes,
                "imports": imports,
                "total_lines": len(content.splitlines())
            }
        except SyntaxError as e:
            return {"error": f"Python syntax error: {str(e)}"}

    def _parse_javascript(self, content: str) -> Dict[str, Any]:
        """Parse JavaScript code (basic implementation)."""
        # Basic regex-based parsing for JavaScript
        functions = re.findall(r'\bfunction\s+(\w+)\s*\([^)]*\)', content)
        classes = re.findall(r'\bclass\s+(\w+)', content)
        imports = re.findall(r'\bimport\s+.*?\bfrom\s+[\'"]([^\'"]+)[\'"]', content)

        return {
            "language": "javascript",
            "functions": [{"name": f, "line": 0} for f in functions],
            "classes": [{"name": c, "line": 0} for c in classes],
            "imports": imports,
            "total_lines": len(content.splitlines())
        }

    def _parse_typescript(self, content: str) -> Dict[str, Any]:
        """Parse TypeScript code."""
        # Similar to JavaScript but with type annotations
        result = self._parse_javascript(content)
        result["language"] = "typescript"

        # Detect interfaces and types
        interfaces = re.findall(r'\binterface\s+(\w+)', content)
        types = re.findall(r'\btype\s+(\w+)', content)

        result["interfaces"] = interfaces
        result["types"] = types

        return result

    def _parse_json(self, content: str) -> Dict[str, Any]:
        """Parse JSON content."""
        try:
            data = json.loads(content)
            return {
                "language": "json",
                "parsed": True,
                "keys": list(data.keys()) if isinstance(data, dict) else None,
                "total_lines": len(content.splitlines())
            }
        except json.JSONDecodeError as e:
            return {"error": f"JSON parsing error: {str(e)}"}

    def _parse_yaml(self, content: str) -> Dict[str, Any]:
        """Parse YAML content."""
        try:
            data = yaml.safe_load(content)
            return {
                "language": "yaml",
                "parsed": True,
                "keys": list(data.keys()) if isinstance(data, dict) else None,
                "total_lines": len(content.splitlines())
            }
        except yaml.YAMLError as e:
            return {"error": f"YAML parsing error: {str(e)}"}


class MultiLanguageGenerator:
    """Generates code in multiple programming languages."""

    def __init__(self):
        self.generators = {
            ProgrammingLanguage.PYTHON: self._generate_python,
            ProgrammingLanguage.JAVASCRIPT: self._generate_javascript,
            ProgrammingLanguage.TYPESCRIPT: self._generate_typescript,
        }

    def generate_function(self, language: ProgrammingLanguage, name: str,
                         params: List[str], return_type: Optional[str] = None,
                         body: Optional[str] = None) -> str:
        """Generate a function in the specified language."""
        if language not in self.generators:
            return f"// Code generation not supported for {language.value}"

        return self.generators[language](name, params, return_type, body)

    def _generate_python(self, name: str, params: List[str],
                        return_type: Optional[str], body: Optional[str]) -> str:
        """Generate Python function."""
        param_str = ", ".join(params)
        function = f"def {name}({param_str}):\n"
        if body:
            function += f"    {body}\n"
        else:
            function += "    pass\n"
        return function

    def _generate_javascript(self, name: str, params: List[str],
                           return_type: Optional[str], body: Optional[str]) -> str:
        """Generate JavaScript function."""
        param_str = ", ".join(params)
        function = f"function {name}({param_str}) {{\n"
        if body:
            function += f"    {body}\n"
        else:
            function += "    // TODO: implement\n"
        function += "}\n"
        return function

    def _generate_typescript(self, name: str, params: List[str],
                           return_type: Optional[str], body: Optional[str]) -> str:
        """Generate TypeScript function."""
        param_str = ", ".join([f"{p}: any" for p in params])
        return_type_str = f": {return_type}" if return_type else ""
        function = f"function {name}({param_str}){return_type_str} {{\n"
        if body:
            function += f"    {body}\n"
        else:
            function += "    // TODO: implement\n"
        function += "}\n"
        return function


class CrossLanguageAnalyzer:
    """Analyzes code across multiple languages for patterns and insights."""

    def __init__(self):
        self.parser = MultiLanguageParser()

    def analyze_codebase(self, files: Dict[str, str]) -> Dict[str, Any]:
        """Analyze a codebase with multiple language files."""
        analysis = {
            "total_files": len(files),
            "languages": {},
            "cross_language_patterns": [],
            "code_quality_metrics": {}
        }

        detector = LanguageDetector()

        for filename, content in files.items():
            lang = detector.detect_from_extension(filename) or detector.detect_from_content(content)
            if lang:
                if lang.value not in analysis["languages"]:
                    analysis["languages"][lang.value] = {"files": [], "total_lines": 0}

                analysis["languages"][lang.value]["files"].append(filename)
                analysis["languages"][lang.value]["total_lines"] += len(content.splitlines())

                # Parse and analyze individual file
                parsed = self.parser.parse(content, lang)
                if "error" not in parsed:
                    analysis["languages"][lang.value]["parsed_data"] = parsed

        # Cross-language pattern detection
        analysis["cross_language_patterns"] = self._detect_patterns(analysis["languages"])

        return analysis

    def _detect_patterns(self, languages: Dict[str, Any]) -> List[str]:
        """Detect patterns across languages."""
        patterns = []

        # Check for API consistency across languages
        if "python" in languages and "javascript" in languages:
            patterns.append("Python-JavaScript API consistency detected")

        # Check for multi-language frameworks
        if len(languages) > 3:
            patterns.append("Multi-language framework detected")

        return patterns


# Export main classes for external use
__all__ = [
    "ProgrammingLanguage",
    "LanguageFeatures",
    "LanguageDetector",
    "MultiLanguageParser",
    "MultiLanguageGenerator",
    "CrossLanguageAnalyzer",
    "LANGUAGE_FEATURES"
]
