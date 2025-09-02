"""
Comprehensive Documentation Module for Code Intelligence Platform

This module provides comprehensive documentation generation, management, and
serving capabilities for the code intelligence platform, including API docs,
user guides, developer documentation, and interactive help systems.

Features:
- Auto-generated API documentation from code
- Interactive documentation browser
- Multi-format documentation export (HTML, PDF, Markdown)
- Documentation search and indexing
- Versioned documentation management
- User guide generation and tutorials
- Developer documentation with examples
- Documentation quality metrics and validation
"""

import os
import json
import re
import time
from typing import Dict, List, Optional, Any, Callable, Union
from dataclasses import dataclass, field
from enum import Enum
import inspect
import ast
import markdown
from abc import ABC, abstractmethod
import logging

logger = logging.getLogger(__name__)


class DocFormat(Enum):
    """Supported documentation formats."""
    HTML = "html"
    MARKDOWN = "markdown"
    PDF = "pdf"
    JSON = "json"
    PLAIN_TEXT = "txt"


class DocType(Enum):
    """Types of documentation."""
    API_REFERENCE = "api_reference"
    USER_GUIDE = "user_guide"
    DEVELOPER_GUIDE = "developer_guide"
    TUTORIAL = "tutorial"
    FAQ = "faq"
    CHANGELOG = "changelog"


@dataclass
class DocumentationItem:
    """Represents a documentation item."""
    id: str
    title: str
    content: str
    doc_type: DocType
    format: DocFormat
    version: str
    tags: List[str] = field(default_factory=list)
    created_at: float = field(default_factory=time.time)
    updated_at: float = field(default_factory=time.time)
    author: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class APIDocumentation:
    """API documentation for a function/class/module."""
    name: str
    type: str  # "function", "class", "module"
    signature: str
    docstring: Optional[str]
    parameters: List[Dict[str, Any]] = field(default_factory=list)
    return_type: Optional[str] = None
    examples: List[str] = field(default_factory=list)
    module: str = ""
    line_number: Optional[int] = None


class DocumentationGenerator(ABC):
    """Abstract base class for documentation generators."""

    @abstractmethod
    def generate(self, source: Any) -> str:
        """Generate documentation from source."""
        pass

    @abstractmethod
    def get_format(self) -> DocFormat:
        """Get the output format."""
        pass


class PythonAPIGenerator(DocumentationGenerator):
    """Generate API documentation from Python code."""

    def __init__(self):
        self.format_type = DocFormat.MARKDOWN

    def generate(self, source: Any) -> str:
        """Generate API documentation from Python source."""
        if inspect.ismodule(source):
            return self._generate_module_docs(source)
        elif inspect.isclass(source):
            return self._generate_class_docs(source)
        elif inspect.isfunction(source) or inspect.ismethod(source):
            return self._generate_function_docs(source)
        else:
            return f"Unsupported source type: {type(source)}"

    def get_format(self) -> DocFormat:
        return self.format_type

    def _generate_module_docs(self, module) -> str:
        """Generate documentation for a Python module."""
        docs = f"# {module.__name__}\n\n"

        if module.__doc__:
            docs += f"{module.__doc__}\n\n"

        # Get all classes, functions, and submodules
        members = inspect.getmembers(module)

        classes = [m for m in members if inspect.isclass(m[1]) and m[1].__module__ == module.__name__]
        functions = [m for m in members if inspect.isfunction(m[1]) and m[1].__module__ == module.__name__]

        if classes:
            docs += "## Classes\n\n"
            for name, cls in classes:
                docs += f"- [{name}](#{name.lower()})\n"
            docs += "\n"

        if functions:
            docs += "## Functions\n\n"
            for name, func in functions:
                docs += f"- [{name}](#{name.lower()})\n"
            docs += "\n"

        # Generate detailed docs for each
        for name, cls in classes:
            docs += self._generate_class_docs(cls) + "\n\n"

        for name, func in functions:
            docs += self._generate_function_docs(func) + "\n\n"

        return docs

    def _generate_class_docs(self, cls) -> str:
        """Generate documentation for a Python class."""
        docs = f"## {cls.__name__}\n\n"

        if cls.__doc__:
            docs += f"{cls.__doc__}\n\n"

        # Get methods
        methods = [m for m in inspect.getmembers(cls, predicate=inspect.isfunction)
                  if not m[0].startswith('_') or m[0] == '__init__']

        if methods:
            docs += "### Methods\n\n"
            for name, method in methods:
                docs += f"#### {name}\n\n"
                docs += self._generate_function_docs(method, indent="") + "\n"

        return docs

    def _generate_function_docs(self, func, indent: str = "") -> str:
        """Generate documentation for a Python function."""
        try:
            sig = inspect.signature(func)
            docs = f"{indent}**{func.__name__}**({', '.join(str(p) for p in sig.parameters.values())})\n\n"

            if func.__doc__:
                docs += f"{indent}{func.__doc__}\n\n"

            # Parse parameters from docstring
            if func.__doc__:
                param_matches = re.findall(r':param (\w+): (.+)', func.__doc__)
                if param_matches:
                    docs += f"{indent}**Parameters:**\n\n"
                    for param, desc in param_matches:
                        docs += f"{indent}- `{param}`: {desc}\n"
                    docs += "\n"

            # Parse return from docstring
            return_match = re.search(r':returns?: (.+)', func.__doc__)
            if return_match:
                docs += f"{indent}**Returns:** {return_match.group(1)}\n\n"

        except Exception as e:
            docs = f"{indent}**{func.__name__}**\n\n"
            docs += f"{indent}*Error generating signature: {e}*\n\n"

        return docs


class MarkdownGenerator(DocumentationGenerator):
    """Generate documentation from Markdown sources."""

    def __init__(self):
        self.format_type = DocFormat.HTML

    def generate(self, source: str) -> str:
        """Convert Markdown to HTML."""
        return markdown.markdown(source, extensions=['extra', 'codehilite', 'toc'])

    def get_format(self) -> DocFormat:
        return self.format_type


class DocumentationManager:
    """Manage documentation items and generation."""

    def __init__(self, docs_path: str = "docs"):
        self.docs_path = docs_path
        os.makedirs(docs_path, exist_ok=True)
        self.generators: Dict[DocFormat, DocumentationGenerator] = {
            DocFormat.MARKDOWN: PythonAPIGenerator(),
            DocFormat.HTML: MarkdownGenerator()
        }
        self.documentation_items: Dict[str, DocumentationItem] = {}
        self.search_index: Dict[str, List[str]] = {}
        self._load_documentation()

    def add_documentation(self, item: DocumentationItem) -> None:
        """Add a documentation item."""
        self.documentation_items[item.id] = item
        self._update_search_index(item)
        self._save_documentation()

    def get_documentation(self, doc_id: str) -> Optional[DocumentationItem]:
        """Get a documentation item by ID."""
        return self.documentation_items.get(doc_id)

    def search_documentation(self, query: str, doc_type: Optional[DocType] = None) -> List[DocumentationItem]:
        """Search documentation by query."""
        query_lower = query.lower()
        results = []

        for item in self.documentation_items.values():
            if doc_type and item.doc_type != doc_type:
                continue

            # Search in title, content, and tags
            searchable_text = f"{item.title} {item.content} {' '.join(item.tags)}".lower()

            if query_lower in searchable_text:
                results.append(item)

        return sorted(results, key=lambda x: x.updated_at, reverse=True)

    def generate_api_docs(self, module_path: str, output_format: DocFormat = DocFormat.MARKDOWN) -> str:
        """Generate API documentation from Python module."""
        try:
            import importlib.util
            spec = importlib.util.spec_from_file_location("temp_module", module_path)
            if spec is None or spec.loader is None:
                return f"Error: Could not load module from {module_path}"

            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)

            generator = self.generators.get(output_format, PythonAPIGenerator())
            return generator.generate(module)

        except Exception as e:
            return f"Error generating API docs: {e}"

    def export_documentation(self, doc_ids: List[str], output_format: DocFormat,
                           output_path: str) -> str:
        """Export documentation to specified format."""
        combined_content = ""

        for doc_id in doc_ids:
            item = self.get_documentation(doc_id)
            if item:
                if output_format == DocFormat.HTML:
                    combined_content += f"<h1>{item.title}</h1>\n{item.content}\n\n"
                elif output_format == DocFormat.MARKDOWN:
                    combined_content += f"# {item.title}\n\n{item.content}\n\n"
                elif output_format == DocFormat.JSON:
                    combined_content += json.dumps({
                        "id": item.id,
                        "title": item.title,
                        "content": item.content,
                        "type": item.doc_type.value
                    }, indent=2) + "\n"

        # Write to file
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(combined_content)

        return output_path

    def validate_documentation(self, doc_id: str) -> Dict[str, Any]:
        """Validate documentation quality."""
        item = self.get_documentation(doc_id)
        if not item:
            return {"valid": False, "errors": ["Documentation not found"]}

        errors = []
        warnings = []

        # Check title
        if len(item.title.strip()) < 5:
            errors.append("Title too short")

        # Check content
        if len(item.content.strip()) < 50:
            errors.append("Content too short")

        # Check for broken links (basic check)
        links = re.findall(r'\[([^\]]+)\]\(([^)]+)\)', item.content)
        for link_text, link_url in links:
            if not link_url.startswith(('http', '/', '#')):
                warnings.append(f"Potentially broken relative link: {link_url}")

        # Check code blocks
        code_blocks = re.findall(r'```(\w+)?\n(.*?)\n```', item.content, re.DOTALL)
        if not code_blocks:
            warnings.append("No code examples found")

        return {
            "valid": len(errors) == 0,
            "errors": errors,
            "warnings": warnings,
            "score": max(0, 100 - len(errors) * 20 - len(warnings) * 5)
        }

    def get_documentation_stats(self) -> Dict[str, Any]:
        """Get documentation statistics."""
        total_docs = len(self.documentation_items)
        if total_docs == 0:
            return {"total_documents": 0}

        doc_types = {}
        formats = {}
        versions = {}

        for item in self.documentation_items.values():
            doc_types[item.doc_type.value] = doc_types.get(item.doc_type.value, 0) + 1
            formats[item.format.value] = formats.get(item.format.value, 0) + 1
            versions[item.version] = versions.get(item.version, 0) + 1

        # Calculate average quality score
        quality_scores = []
        for item in self.documentation_items.values():
            validation = self.validate_documentation(item.id)
            quality_scores.append(validation["score"])

        avg_quality = sum(quality_scores) / len(quality_scores) if quality_scores else 0

        return {
            "total_documents": total_docs,
            "document_types": doc_types,
            "formats": formats,
            "versions": versions,
            "average_quality_score": avg_quality,
            "last_updated": max(item.updated_at for item in self.documentation_items.values())
        }

    def _update_search_index(self, item: DocumentationItem) -> None:
        """Update search index with new item."""
        words = re.findall(r'\b\w+\b', f"{item.title} {item.content}")
        for word in words:
            word_lower = word.lower()
            if word_lower not in self.search_index:
                self.search_index[word_lower] = []
            if item.id not in self.search_index[word_lower]:
                self.search_index[word_lower].append(item.id)

    def _save_documentation(self) -> None:
        """Save documentation to storage."""
        data = {}
        for item in self.documentation_items.values():
            data[item.id] = {
                "id": item.id,
                "title": item.title,
                "content": item.content,
                "doc_type": item.doc_type.value,
                "format": item.format.value,
                "version": item.version,
                "tags": item.tags,
                "created_at": item.created_at,
                "updated_at": item.updated_at,
                "author": item.author,
                "metadata": item.metadata
            }

        with open(os.path.join(self.docs_path, "documentation.json"), 'w') as f:
            json.dump(data, f, indent=2)

    def _load_documentation(self) -> None:
        """Load documentation from storage."""
        docs_file = os.path.join(self.docs_path, "documentation.json")
        if os.path.exists(docs_file):
            try:
                with open(docs_file, 'r') as f:
                    data = json.load(f)

                for item_data in data.values():
                    item = DocumentationItem(
                        id=item_data["id"],
                        title=item_data["title"],
                        content=item_data["content"],
                        doc_type=DocType(item_data["doc_type"]),
                        format=DocFormat(item_data["format"]),
                        version=item_data["version"],
                        tags=item_data.get("tags", []),
                        created_at=item_data["created_at"],
                        updated_at=item_data["updated_at"],
                        author=item_data.get("author"),
                        metadata=item_data.get("metadata", {})
                    )
                    self.documentation_items[item.id] = item
                    self._update_search_index(item)

            except Exception as e:
                logger.error(f"Failed to load documentation: {e}")


class InteractiveHelpSystem:
    """Interactive help and documentation browser."""

    def __init__(self, docs_manager: DocumentationManager):
        self.docs_manager = docs_manager
        self.current_context: Optional[str] = None
        self.browse_history: List[str] = []

    def get_help_for_topic(self, topic: str) -> Dict[str, Any]:
        """Get help documentation for a specific topic."""
        results = self.docs_manager.search_documentation(topic)

        if not results:
            return {
                "found": False,
                "suggestions": self._get_similar_topics(topic)
            }

        # Return the most relevant result
        best_match = results[0]

        return {
            "found": True,
            "topic": best_match.title,
            "content": best_match.content,
            "type": best_match.doc_type.value,
            "related_topics": [r.title for r in results[1:5]]  # Top 4 related
        }

    def get_contextual_help(self, context: str) -> Dict[str, Any]:
        """Get contextual help based on current user context."""
        self.current_context = context

        # Search for context-specific documentation
        context_results = self.docs_manager.search_documentation(context)

        if context_results:
            return {
                "context": context,
                "help_items": [
                    {
                        "title": item.title,
                        "summary": item.content[:200] + "..." if len(item.content) > 200 else item.content,
                        "type": item.doc_type.value
                    }
                    for item in context_results[:3]
                ]
            }

        return {"context": context, "help_items": []}

    def browse_documentation(self, category: Optional[DocType] = None) -> List[Dict[str, Any]]:
        """Browse documentation by category."""
        items = list(self.docs_manager.documentation_items.values())

        if category:
            items = [item for item in items if item.doc_type == category]

        return [
            {
                "id": item.id,
                "title": item.title,
                "type": item.doc_type.value,
                "updated": item.updated_at,
                "tags": item.tags
            }
            for item in sorted(items, key=lambda x: x.updated_at, reverse=True)
        ]

    def _get_similar_topics(self, topic: str) -> List[str]:
        """Get similar topics for suggestions."""
        # Simple similarity based on word overlap
        topic_words = set(topic.lower().split())
        suggestions = []

        for item in self.docs_manager.documentation_items.values():
            item_words = set(f"{item.title} {item.content}".lower().split())
            overlap = len(topic_words.intersection(item_words))
            if overlap > 0:
                suggestions.append((item.title, overlap))

        # Sort by overlap and return top suggestions
        suggestions.sort(key=lambda x: x[1], reverse=True)
        return [title for title, _ in suggestions[:5]]


class DocumentationOrchestrator:
    """Main orchestrator for comprehensive documentation features."""

    def __init__(self):
        self.docs_manager = DocumentationManager()
        self.help_system = InteractiveHelpSystem(self.docs_manager)
        self.api_generator = PythonAPIGenerator()

    def create_user_guide(self, title: str, content: str, version: str = "1.0") -> str:
        """Create a user guide documentation item."""
        doc_id = f"user_guide_{title.lower().replace(' ', '_')}"

        item = DocumentationItem(
            id=doc_id,
            title=title,
            content=content,
            doc_type=DocType.USER_GUIDE,
            format=DocFormat.MARKDOWN,
            version=version,
            tags=["user", "guide", "tutorial"]
        )

        self.docs_manager.add_documentation(item)
        return doc_id

    def create_api_reference(self, module_path: str, title: str, version: str = "1.0") -> str:
        """Create API reference documentation."""
        api_content = self.docs_manager.generate_api_docs(module_path)

        doc_id = f"api_{title.lower().replace(' ', '_')}"

        item = DocumentationItem(
            id=doc_id,
            title=f"API Reference: {title}",
            content=api_content,
            doc_type=DocType.API_REFERENCE,
            format=DocFormat.MARKDOWN,
            version=version,
            tags=["api", "reference", "developer"]
        )

        self.docs_manager.add_documentation(item)
        return doc_id

    def generate_complete_documentation(self, output_dir: str = "generated_docs") -> Dict[str, Any]:
        """Generate complete documentation package."""
        os.makedirs(output_dir, exist_ok=True)

        # Generate different types of documentation
        results = {
            "user_guides": [],
            "api_references": [],
            "developer_guides": [],
            "exports": [],
            "statistics": {}
        }

        # Export all documentation
        all_doc_ids = list(self.docs_manager.documentation_items.keys())

        if all_doc_ids:
            # Export as HTML
            html_path = os.path.join(output_dir, "documentation.html")
            self.docs_manager.export_documentation(all_doc_ids, DocFormat.HTML, html_path)
            results["exports"].append({"format": "HTML", "path": html_path})

            # Export as Markdown
            md_path = os.path.join(output_dir, "documentation.md")
            self.docs_manager.export_documentation(all_doc_ids, DocFormat.MARKDOWN, md_path)
            results["exports"].append({"format": "Markdown", "path": md_path})

        # Generate statistics
        results["statistics"] = self.docs_manager.get_documentation_stats()

        return results

    def get_documentation_health_report(self) -> Dict[str, Any]:
        """Generate documentation health and quality report."""
        items = list(self.docs_manager.documentation_items.values())

        if not items:
            return {"status": "No documentation found"}

        validations = []
        for item in items:
            validation = self.docs_manager.validate_documentation(item.id)
            validations.append({
                "id": item.id,
                "title": item.title,
                "valid": validation["valid"],
                "score": validation["score"],
                "errors": validation["errors"],
                "warnings": validation["warnings"]
            })

        # Calculate overall health
        total_score = sum(v["score"] for v in validations)
        avg_score = total_score / len(validations)

        health_status = "excellent" if avg_score >= 90 else \
                       "good" if avg_score >= 75 else \
                       "needs_improvement" if avg_score >= 60 else \
                       "critical"

        return {
            "overall_health": health_status,
            "average_score": avg_score,
            "total_documents": len(items),
            "valid_documents": len([v for v in validations if v["valid"]]),
            "document_validations": validations,
            "recommendations": self._generate_recommendations(validations)
        }

    def _generate_recommendations(self, validations: List[Dict[str, Any]]) -> List[str]:
        """Generate improvement recommendations."""
        recommendations = []

        error_counts = {}
        for validation in validations:
            for error in validation["errors"]:
                error_counts[error] = error_counts.get(error, 0) + 1

        # Sort by frequency
        sorted_errors = sorted(error_counts.items(), key=lambda x: x[1], reverse=True)

        for error, count in sorted_errors[:3]:  # Top 3 issues
            if "too short" in error.lower():
                recommendations.append(f"Expand content for {count} documents with insufficient detail")
            elif "broken" in error.lower():
                recommendations.append(f"Fix {count} broken links across documentation")
            else:
                recommendations.append(f"Address '{error}' in {count} documents")

        if not recommendations:
            recommendations.append("Documentation quality is excellent!")

        return recommendations


# Export main classes
__all__ = [
    "DocFormat",
    "DocType",
    "DocumentationItem",
    "APIDocumentation",
    "DocumentationGenerator",
    "PythonAPIGenerator",
    "MarkdownGenerator",
    "DocumentationManager",
    "InteractiveHelpSystem",
    "DocumentationOrchestrator"
]
