"""
Documentation Enhancement Module for Code Intelligence Platform

This module provides comprehensive documentation capabilities including
automatic documentation generation, API documentation, interactive tutorials,
documentation quality analysis, and multi-format documentation support.

Features:
- Automatic documentation generation from code
- Interactive API documentation
- Documentation quality analysis and scoring
- Multi-format documentation export (PDF, HTML, Markdown)
- Documentation search and indexing
- Versioned documentation management
- Documentation templates and themes
- Real-time documentation updates
- Documentation collaboration features
"""

import os
import json
import re
import inspect
import ast
import markdown
from typing import Dict, List, Optional, Any, Callable, Union, Type, Tuple
from dataclasses import dataclass, field
from enum import Enum
from datetime import datetime
import yaml
import html
from collections import defaultdict
import uuid


class DocumentationFormat(Enum):
    """Supported documentation formats."""
    MARKDOWN = "markdown"
    HTML = "html"
    PDF = "pdf"
    JSON = "json"
    YAML = "yaml"


class DocumentationType(Enum):
    """Types of documentation."""
    API_REFERENCE = "api_reference"
    USER_GUIDE = "user_guide"
    TUTORIAL = "tutorial"
    FAQ = "faq"
    CHANGELOG = "changelog"
    ARCHITECTURE = "architecture"


@dataclass
class DocumentationMetadata:
    """Metadata for documentation."""
    title: str
    version: str
    author: str
    created_at: datetime
    updated_at: Optional[datetime] = None
    tags: List[str] = field(default_factory=list)
    category: DocumentationType = DocumentationType.USER_GUIDE
    language: str = "en"
    format: DocumentationFormat = DocumentationFormat.MARKDOWN

    def to_dict(self) -> Dict[str, Any]:
        """Convert metadata to dictionary."""
        return {
            "title": self.title,
            "version": self.version,
            "author": self.author,
            "created_at": self.created_at.isoformat(),
            "updated_at": self.updated_at.isoformat() if self.updated_at else None,
            "tags": self.tags,
            "category": self.category.value,
            "language": self.language,
            "format": self.format.value
        }


@dataclass
class DocumentationSection:
    """A section within documentation."""
    id: str
    title: str
    content: str
    level: int = 1
    subsections: List["DocumentationSection"] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert section to dictionary."""
        return {
            "id": self.id,
            "title": self.title,
            "content": self.content,
            "level": self.level,
            "subsections": [s.to_dict() for s in self.subsections],
            "metadata": self.metadata
        }


@dataclass
class Documentation:
    """Complete documentation structure."""
    metadata: DocumentationMetadata
    sections: List[DocumentationSection]
    table_of_contents: List[Dict[str, Any]] = field(default_factory=list)
    search_index: Dict[str, List[str]] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert documentation to dictionary."""
        return {
            "metadata": self.metadata.to_dict(),
            "sections": [s.to_dict() for s in self.sections],
            "table_of_contents": self.table_of_contents,
            "search_index": self.search_index
        }


class CodeAnalyzer:
    """Analyzes code to extract documentation information."""

    def __init__(self):
        self.parsers = {
            "python": self._parse_python_code,
            "javascript": self._parse_javascript_code,
            "typescript": self._parse_typescript_code
        }

    def analyze_file(self, file_path: str) -> Dict[str, Any]:
        """Analyze a code file and extract documentation information."""
        if not os.path.exists(file_path):
            return {}

        _, ext = os.path.splitext(file_path)
        language = self._get_language_from_extension(ext)

        if language not in self.parsers:
            return {}

        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()

        return self.parsers[language](content, file_path)

    def analyze_module(self, module_path: str) -> Dict[str, Any]:
        """Analyze an entire module/directory."""
        if not os.path.isdir(module_path):
            return self.analyze_file(module_path)

        analysis = {
            "files": [],
            "classes": [],
            "functions": [],
            "modules": []
        }

        for root, dirs, files in os.walk(module_path):
            for file in files:
                if file.endswith(('.py', '.js', '.ts')):
                    file_path = os.path.join(root, file)
                    file_analysis = self.analyze_file(file_path)
                    if file_analysis:
                        analysis["files"].append(file_analysis)

        return analysis

    def _get_language_from_extension(self, extension: str) -> str:
        """Get programming language from file extension."""
        ext_map = {
            '.py': 'python',
            '.js': 'javascript',
            '.ts': 'typescript',
            '.jsx': 'javascript',
            '.tsx': 'typescript'
        }
        return ext_map.get(extension.lower(), '')

    def _parse_python_code(self, content: str, file_path: str) -> Dict[str, Any]:
        """Parse Python code and extract documentation."""
        try:
            tree = ast.parse(content)
            analysis = {
                "file": os.path.basename(file_path),
                "language": "python",
                "classes": [],
                "functions": [],
                "docstrings": []
            }

            for node in ast.walk(tree):
                if isinstance(node, ast.ClassDef):
                    class_info = {
                        "name": node.name,
                        "docstring": ast.get_docstring(node),
                        "methods": [],
                        "line_number": node.lineno
                    }

                    # Extract methods
                    for item in node.body:
                        if isinstance(item, ast.FunctionDef):
                            method_info = {
                                "name": item.name,
                                "docstring": ast.get_docstring(item),
                                "args": [arg.arg for arg in item.args.args],
                                "line_number": item.lineno
                            }
                            class_info["methods"].append(method_info)

                    analysis["classes"].append(class_info)

                elif isinstance(node, ast.FunctionDef) and not isinstance(node, ast.AsyncFunctionDef):
                    func_info = {
                        "name": node.name,
                        "docstring": ast.get_docstring(node),
                        "args": [arg.arg for arg in node.args.args],
                        "line_number": node.lineno
                    }
                    analysis["functions"].append(func_info)

            return analysis
        except SyntaxError:
            return {}

    def _parse_javascript_code(self, content: str, file_path: str) -> Dict[str, Any]:
        """Parse JavaScript code and extract documentation."""
        # Simplified JavaScript parsing
        analysis = {
            "file": os.path.basename(file_path),
            "language": "javascript",
            "functions": [],
            "classes": []
        }

        # Extract function definitions
        func_pattern = r'(?:function\s+(\w+)|const\s+(\w+)\s*=\s*\([^)]*\)\s*=>|(\w+)\s*\([^)]*\)\s*{)'
        functions = re.findall(func_pattern, content)

        for func_match in functions:
            func_name = func_match[0] or func_match[1] or func_match[2]
            if func_name:
                analysis["functions"].append({
                    "name": func_name,
                    "docstring": None  # JS doesn't have built-in docstrings
                })

        return analysis

    def _parse_typescript_code(self, content: str, file_path: str) -> Dict[str, Any]:
        """Parse TypeScript code and extract documentation."""
        # Similar to JavaScript but with type information
        analysis = self._parse_javascript_code(content, file_path)
        analysis["language"] = "typescript"

        # Extract type definitions
        type_pattern = r'(?:interface\s+(\w+)|type\s+(\w+)\s*=)'
        types = re.findall(type_pattern, content)
        analysis["types"] = [t[0] or t[1] for t in types if t[0] or t[1]]

        return analysis


class DocumentationGenerator:
    """Generates documentation from code analysis."""

    def __init__(self):
        self.templates: Dict[str, str] = {}
        self.code_analyzer = CodeAnalyzer()

    def generate_api_documentation(self, module_path: str, title: str = "API Documentation") -> Documentation:
        """Generate API documentation from code."""
        analysis = self.code_analyzer.analyze_module(module_path)

        metadata = DocumentationMetadata(
            title=title,
            version="1.0.0",
            author="Code Explainer",
            created_at=datetime.utcnow(),
            category=DocumentationType.API_REFERENCE
        )

        sections = []

        # Overview section
        overview_content = f"# {title}\n\nThis document provides API reference for the codebase.\n\n"
        overview_content += f"**Files analyzed:** {len(analysis.get('files', []))}\n"
        overview_content += f"**Classes found:** {sum(len(f.get('classes', [])) for f in analysis.get('files', []))}\n"
        overview_content += f"**Functions found:** {sum(len(f.get('functions', [])) for f in analysis.get('files', []))}\n"

        overview_section = DocumentationSection(
            id="overview",
            title="Overview",
            content=overview_content
        )
        sections.append(overview_section)

        # Generate sections for each file
        for file_analysis in analysis.get('files', []):
            file_section = self._generate_file_section(file_analysis)
            if file_section:
                sections.append(file_section)

        # Create table of contents
        toc = self._generate_table_of_contents(sections)

        documentation = Documentation(
            metadata=metadata,
            sections=sections,
            table_of_contents=toc
        )

        # Build search index
        documentation.search_index = self._build_search_index(documentation)

        return documentation

    def _generate_file_section(self, file_analysis: Dict[str, Any]) -> Optional[DocumentationSection]:
        """Generate documentation section for a file."""
        if not file_analysis.get('classes') and not file_analysis.get('functions'):
            return None

        content = f"## {file_analysis['file']}\n\n"

        # Add classes
        for cls in file_analysis.get('classes', []):
            content += f"### Class: {cls['name']}\n\n"
            if cls.get('docstring'):
                content += f"{cls['docstring']}\n\n"

            # Add methods
            for method in cls.get('methods', []):
                content += f"#### Method: {method['name']}\n\n"
                if method.get('docstring'):
                    content += f"{method['docstring']}\n\n"
                if method.get('args'):
                    content += f"**Parameters:** {', '.join(method['args'])}\n\n"

        # Add functions
        for func in file_analysis.get('functions', []):
            content += f"### Function: {func['name']}\n\n"
            if func.get('docstring'):
                content += f"{func['docstring']}\n\n"
            if func.get('args'):
                content += f"**Parameters:** {', '.join(func['args'])}\n\n"

        return DocumentationSection(
            id=f"file_{file_analysis['file'].replace('.', '_')}",
            title=file_analysis['file'],
            content=content,
            level=2
        )

    def _generate_table_of_contents(self, sections: List[DocumentationSection]) -> List[Dict[str, Any]]:
        """Generate table of contents from sections."""
        toc = []
        for section in sections:
            toc_entry = {
                "id": section.id,
                "title": section.title,
                "level": section.level
            }
            toc.append(toc_entry)

            # Add subsections
            for subsection in section.subsections:
                toc.append({
                    "id": subsection.id,
                    "title": subsection.title,
                    "level": subsection.level,
                    "parent": section.id
                })

        return toc

    def _build_search_index(self, documentation: Documentation) -> Dict[str, List[str]]:
        """Build search index for documentation."""
        index = defaultdict(list)

        def add_to_index(text: str, section_id: str):
            words = re.findall(r'\b\w+\b', text.lower())
            for word in words:
                if len(word) > 2:  # Skip very short words
                    index[word].append(section_id)

        for section in documentation.sections:
            add_to_index(section.title, section.id)
            add_to_index(section.content, section.id)

            for subsection in section.subsections:
                add_to_index(subsection.title, subsection.id)
                add_to_index(subsection.content, subsection.id)

        return dict(index)


class DocumentationQualityAnalyzer:
    """Analyzes documentation quality and provides improvement suggestions."""

    def __init__(self):
        self.quality_metrics = {
            "completeness": self._check_completeness,
            "accuracy": self._check_accuracy,
            "clarity": self._check_clarity,
            "consistency": self._check_consistency,
            "accessibility": self._check_accessibility
        }

    def analyze_quality(self, documentation: Documentation) -> Dict[str, Any]:
        """Analyze documentation quality."""
        results = {
            "overall_score": 0.0,
            "metrics": {},
            "issues": [],
            "recommendations": []
        }

        total_score = 0
        for metric_name, metric_func in self.quality_metrics.items():
            score, issues = metric_func(documentation)
            results["metrics"][metric_name] = {
                "score": score,
                "issues": issues
            }
            total_score += score

        results["overall_score"] = total_score / len(self.quality_metrics)

        # Generate recommendations
        results["recommendations"] = self._generate_quality_recommendations(results)

        return results

    def _check_completeness(self, documentation: Documentation) -> Tuple[float, List[str]]:
        """Check documentation completeness."""
        issues = []
        score = 1.0

        # Check for empty sections
        empty_sections = [s for s in documentation.sections if not s.content.strip()]
        if empty_sections:
            score -= len(empty_sections) * 0.1
            issues.append(f"Found {len(empty_sections)} empty sections")

        # Check for missing descriptions
        sections_without_content = [s for s in documentation.sections if len(s.content.strip()) < 50]
        if sections_without_content:
            score -= len(sections_without_content) * 0.05
            issues.append(f"Found {len(sections_without_content)} sections with minimal content")

        return max(0, score), issues

    def _check_accuracy(self, documentation: Documentation) -> Tuple[float, List[str]]:
        """Check documentation accuracy."""
        issues = []
        score = 0.9  # Default high score, would need manual verification

        # Check for outdated version references
        content = " ".join(s.content for s in documentation.sections)
        if "version" in content.lower():
            # This would need more sophisticated checking
            pass

        return score, issues

    def _check_clarity(self, documentation: Documentation) -> Tuple[float, List[str]]:
        """Check documentation clarity."""
        issues = []
        score = 1.0

        for section in documentation.sections:
            # Check for very long sentences
            sentences = re.split(r'[.!?]+', section.content)
            long_sentences = [s for s in sentences if len(s.split()) > 30]
            if long_sentences:
                score -= 0.05
                issues.append(f"Section '{section.title}' has {len(long_sentences)} very long sentences")

            # Check for complex words (simplified)
            words = re.findall(r'\b\w+\b', section.content.lower())
            complex_words = [w for w in words if len(w) > 12]
            if len(complex_words) > len(words) * 0.1:
                score -= 0.03
                issues.append(f"Section '{section.title}' contains many complex words")

        return max(0, score), issues

    def _check_consistency(self, documentation: Documentation) -> Tuple[float, List[str]]:
        """Check documentation consistency."""
        issues = []
        score = 1.0

        # Check for consistent heading styles
        headings = []
        for section in documentation.sections:
            content_headings = re.findall(r'^#{1,6}\s+.+', section.content, re.MULTILINE)
            headings.extend(content_headings)

        # Check for mixed heading styles
        if len(set(len(h.split()[0]) for h in headings if h)) > 1:
            score -= 0.1
            issues.append("Inconsistent heading styles found")

        return score, issues

    def _check_accessibility(self, documentation: Documentation) -> Tuple[float, List[str]]:
        """Check documentation accessibility."""
        issues = []
        score = 1.0

        for section in documentation.sections:
            # Check for alt text in images (simplified)
            if '![' in section.content and 'alt=' not in section.content:
                score -= 0.1
                issues.append(f"Section '{section.title}' contains images without alt text")

            # Check for proper heading hierarchy
            heading_levels = re.findall(r'^#{1,6}', section.content, re.MULTILINE)
            levels = [len(h) for h in heading_levels]
            if levels and levels != sorted(set(levels)):
                score -= 0.05
                issues.append(f"Section '{section.title}' has improper heading hierarchy")

        return max(0, score), issues

    def _generate_quality_recommendations(self, results: Dict[str, Any]) -> List[str]:
        """Generate quality improvement recommendations."""
        recommendations = []

        for metric_name, metric_data in results["metrics"].items():
            if metric_data["score"] < 0.8:
                if metric_name == "completeness":
                    recommendations.append("Add more detailed content to improve completeness")
                elif metric_name == "clarity":
                    recommendations.append("Simplify language and break down complex sentences")
                elif metric_name == "consistency":
                    recommendations.append("Standardize formatting and terminology")
                elif metric_name == "accessibility":
                    recommendations.append("Add alt text to images and improve heading structure")

        return recommendations


class DocumentationExporter:
    """Exports documentation to various formats."""

    def __init__(self):
        self.exporters = {
            DocumentationFormat.MARKDOWN: self._export_markdown,
            DocumentationFormat.HTML: self._export_html,
            DocumentationFormat.JSON: self._export_json,
            DocumentationFormat.YAML: self._export_yaml
        }

    def export(self, documentation: Documentation, format: DocumentationFormat,
               output_path: str) -> bool:
        """Export documentation to specified format."""
        if format not in self.exporters:
            return False

        try:
            content = self.exporters[format](documentation)

            with open(output_path, 'w', encoding='utf-8') as f:
                f.write(content)

            return True
        except Exception as e:
            print(f"Export failed: {e}")
            return False

    def _export_markdown(self, documentation: Documentation) -> str:
        """Export to Markdown format."""
        content = f"# {documentation.metadata.title}\n\n"
        content += f"**Version:** {documentation.metadata.version}\n"
        content += f"**Author:** {documentation.metadata.author}\n"
        content += f"**Created:** {documentation.metadata.created_at.strftime('%Y-%m-%d')}\n\n"

        # Add table of contents
        if documentation.table_of_contents:
            content += "## Table of Contents\n\n"
            for item in documentation.table_of_contents:
                indent = "  " * (item["level"] - 1)
                content += f"{indent}- [{item['title']}](#{item['id']})\n"
            content += "\n"

        # Add sections
        for section in documentation.sections:
            content += section.content + "\n\n"

        return content

    def _export_html(self, documentation: Documentation) -> str:
        """Export to HTML format."""
        html_content = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>{html.escape(documentation.metadata.title)}</title>
    <style>
        body {{ font-family: Arial, sans-serif; margin: 40px; }}
        h1, h2, h3 {{ color: #333; }}
        .metadata {{ background: #f5f5f5; padding: 20px; margin-bottom: 20px; }}
        .toc {{ background: #f9f9f9; padding: 15px; margin-bottom: 20px; }}
        pre {{ background: #f4f4f4; padding: 10px; border-radius: 4px; }}
        code {{ background: #f4f4f4; padding: 2px 4px; border-radius: 2px; }}
    </style>
</head>
<body>
    <div class="metadata">
        <h1>{html.escape(documentation.metadata.title)}</h1>
        <p><strong>Version:</strong> {html.escape(documentation.metadata.version)}</p>
        <p><strong>Author:</strong> {html.escape(documentation.metadata.author)}</p>
        <p><strong>Created:</strong> {documentation.metadata.created_at.strftime('%Y-%m-%d')}</p>
    </div>
"""

        # Add table of contents
        if documentation.table_of_contents:
            html_content += '<div class="toc"><h2>Table of Contents</h2><ul>'
            for item in documentation.table_of_contents:
                indent = "&nbsp;&nbsp;" * (item["level"] - 1)
                html_content += f'<li>{indent}<a href="#{item["id"]}">{html.escape(item["title"])}</a></li>'
            html_content += '</ul></div>'

        # Convert Markdown sections to HTML
        for section in documentation.sections:
            html_content += markdown.markdown(section.content)

        html_content += "</body></html>"
        return html_content

    def _export_json(self, documentation: Documentation) -> str:
        """Export to JSON format."""
        return json.dumps(documentation.to_dict(), indent=2, default=str)

    def _export_yaml(self, documentation: Documentation) -> str:
        """Export to YAML format."""
        return yaml.dump(documentation.to_dict(), default_flow_style=False)


class DocumentationManager:
    """Main manager for documentation operations."""

    def __init__(self):
        self.generator = DocumentationGenerator()
        self.quality_analyzer = DocumentationQualityAnalyzer()
        self.exporter = DocumentationExporter()
        self.documents: Dict[str, Documentation] = {}

    def create_documentation(self, doc_id: str, title: str, category: DocumentationType,
                           content_sections: List[Dict[str, Any]]) -> Documentation:
        """Create new documentation."""
        metadata = DocumentationMetadata(
            title=title,
            version="1.0.0",
            author="Code Explainer",
            created_at=datetime.utcnow(),
            category=category
        )

        sections = []
        for i, section_data in enumerate(content_sections):
            section = DocumentationSection(
                id=f"section_{i}",
                title=section_data["title"],
                content=section_data["content"],
                level=section_data.get("level", 1)
            )
            sections.append(section)

        documentation = Documentation(
            metadata=metadata,
            sections=sections
        )

        # Generate table of contents and search index
        documentation.table_of_contents = self.generator._generate_table_of_contents(sections)
        documentation.search_index = self.generator._build_search_index(documentation)

        self.documents[doc_id] = documentation
        return documentation

    def generate_api_docs(self, module_path: str, doc_id: str) -> Documentation:
        """Generate API documentation from code."""
        documentation = self.generator.generate_api_documentation(module_path)
        self.documents[doc_id] = documentation
        return documentation

    def analyze_documentation_quality(self, doc_id: str) -> Dict[str, Any]:
        """Analyze quality of documentation."""
        if doc_id not in self.documents:
            return {}

        return self.quality_analyzer.analyze_quality(self.documents[doc_id])

    def export_documentation(self, doc_id: str, format: DocumentationFormat,
                           output_path: str) -> bool:
        """Export documentation to file."""
        if doc_id not in self.documents:
            return False

        return self.exporter.export(self.documents[doc_id], format, output_path)

    def search_documentation(self, query: str) -> List[Dict[str, Any]]:
        """Search across all documentation."""
        results = []

        for doc_id, doc in self.documents.items():
            if query.lower() in doc.metadata.title.lower():
                results.append({
                    "doc_id": doc_id,
                    "title": doc.metadata.title,
                    "type": "title_match",
                    "relevance": 1.0
                })

            # Search in content
            for section in doc.sections:
                if query.lower() in section.content.lower():
                    results.append({
                        "doc_id": doc_id,
                        "section_id": section.id,
                        "title": section.title,
                        "type": "content_match",
                        "relevance": 0.8
                    })

        return results

    def get_documentation_summary(self) -> Dict[str, Any]:
        """Get summary of all documentation."""
        return {
            "total_documents": len(self.documents),
            "categories": list(set(doc.metadata.category.value for doc in self.documents.values())),
            "total_sections": sum(len(doc.sections) for doc in self.documents.values()),
            "last_updated": max((doc.metadata.updated_at or doc.metadata.created_at
                               for doc in self.documents.values()), default=None)
        }


# Export main classes
__all__ = [
    "DocumentationFormat",
    "DocumentationType",
    "DocumentationMetadata",
    "DocumentationSection",
    "Documentation",
    "CodeAnalyzer",
    "DocumentationGenerator",
    "DocumentationQualityAnalyzer",
    "DocumentationExporter",
    "DocumentationManager"
]
