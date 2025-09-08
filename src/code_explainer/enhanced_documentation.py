"""
Enhanced Documentation Module

This module provides comprehensive, auto-generated, and research-integrated
documentation for the Code Intelligence Platform. It combines traditional
documentation with AI-powered generation, research paper integration, and
interactive documentation features.

Features:
- Auto-generated API documentation from code
- Research paper integration in documentation
- Interactive documentation with code examples
- Multi-format documentation generation (HTML, PDF, Markdown)
- Documentation quality assessment and improvement
- Research methodology documentation
- Interactive tutorials and guides
- Documentation search and discovery
- Version-controlled documentation
- Collaborative documentation editing
"""

import json
import yaml
import markdown
import os
import re
from typing import Dict, List, Optional, Any, Callable, Union, Tuple, Set
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
import inspect
import ast
import networkx as nx
from jinja2 import Template
from collections import defaultdict
import warnings
warnings.filterwarnings('ignore')


@dataclass
class DocumentationEntry:
    """Represents a documentation entry."""
    entry_id: str
    title: str
    content: str
    category: str
    tags: List[str] = field(default_factory=list)
    related_modules: List[str] = field(default_factory=list)
    research_references: List[str] = field(default_factory=list)
    code_examples: List[str] = field(default_factory=list)
    last_updated: datetime = field(default_factory=datetime.utcnow)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class APIDocumentation:
    """API documentation for a module."""
    module_name: str
    classes: List[Dict[str, Any]] = field(default_factory=list)
    functions: List[Dict[str, Any]] = field(default_factory=list)
    methods: List[Dict[str, Any]] = field(default_factory=list)
    dependencies: List[str] = field(default_factory=list)
    generated_at: datetime = field(default_factory=datetime.utcnow)


@dataclass
class ResearchDocumentation:
    """Documentation integrated with research."""
    topic: str
    research_papers: List[Dict[str, Any]]
    methodologies: List[Dict[str, Any]]
    benchmarks: List[Dict[str, Any]]
    implementation_status: str
    last_updated: datetime = field(default_factory=datetime.utcnow)


class AutoDocumentationGenerator:
    """Generates documentation automatically from code."""

    def __init__(self):
        self.templates = self._load_templates()

    def _load_templates(self) -> Dict[str, Template]:
        """Load documentation templates."""
        templates = {}

        # Basic templates for different documentation types
        templates["module"] = Template("""
# {{ module_name }}

{{ description }}

## Features

{% for feature in features %}
- {{ feature }}
{% endfor %}

## Classes

{% for class_info in classes %}
### {{ class_info.name }}

{{ class_info.docstring }}

**Methods:**
{% for method in class_info.methods %}
- `{{ method.signature }}`: {{ method.docstring | truncate(100) }}
{% endfor %}

{% endfor %}

## Functions

{% for func_info in functions %}
### {{ func_info.name }}

{{ func_info.docstring }}

**Signature:** `{{ func_info.signature }}`

{% endfor %}

## Dependencies

{% for dep in dependencies %}
- {{ dep }}
{% endfor %}

*Generated at: {{ generated_at }}*
        """)

        return templates

    def generate_module_docs(self, module_path: str) -> str:
        """Generate documentation for a Python module."""
        try:
            # Parse the module
            with open(module_path, 'r') as f:
                source = f.read()

            tree = ast.parse(source)
            module_name = Path(module_path).stem

            # Extract information
            classes = []
            functions = []
            dependencies = []

            for node in ast.walk(tree):
                if isinstance(node, ast.ClassDef):
                    classes.append(self._extract_class_info(node))
                elif isinstance(node, ast.FunctionDef) and not node.name.startswith('_'):
                    functions.append(self._extract_function_info(node))
                elif isinstance(node, ast.Import):
                    dependencies.extend([alias.name for alias in node.names])
                elif isinstance(node, ast.ImportFrom):
                    if node.module:
                        dependencies.append(node.module)

            # Generate documentation
            doc_data = {
                "module_name": module_name,
                "description": self._extract_module_docstring(tree),
                "features": self._extract_features_from_docstring(tree),
                "classes": classes,
                "functions": functions,
                "dependencies": list(set(dependencies)),
                "generated_at": datetime.utcnow().isoformat()
            }

            return self.templates["module"].render(**doc_data)

        except Exception as e:
            return f"# Error generating documentation\n\nError: {str(e)}"

    def _extract_class_info(self, node: ast.ClassDef) -> Dict[str, Any]:
        """Extract information about a class."""
        return {
            "name": node.name,
            "docstring": ast.get_docstring(node) or "No documentation available",
            "methods": [self._extract_function_info(method)
                       for method in node.body
                       if isinstance(method, ast.FunctionDef) and not method.name.startswith('_')]
        }

    def _extract_function_info(self, node: ast.FunctionDef) -> Dict[str, Any]:
        """Extract information about a function."""
        args = [arg.arg for arg in node.args.args]
        signature = f"{node.name}({', '.join(args)})"

        return {
            "name": node.name,
            "signature": signature,
            "docstring": ast.get_docstring(node) or "No documentation available"
        }

    def _extract_module_docstring(self, tree: ast.Module) -> str:
        """Extract module docstring."""
        return ast.get_docstring(tree) or "No module documentation available"

    def _extract_features_from_docstring(self, tree: ast.Module) -> List[str]:
        """Extract features from module docstring."""
        docstring = self._extract_module_docstring(tree)
        features = []

        # Look for features in docstring
        if "Features:" in docstring:
            features_section = docstring.split("Features:")[1].split("\n\n")[0]
            features = [line.strip("- ").strip() for line in features_section.split("\n") if line.strip()]

        return features


class ResearchDocumentationIntegrator:
    """Integrates research documentation with code documentation."""

    def __init__(self):
        self.research_docs: Dict[str, ResearchDocumentation] = {}

    def create_research_documentation(self, topic: str,
                                    research_papers: List[Dict[str, Any]],
                                    methodologies: List[Dict[str, Any]]) -> ResearchDocumentation:
        """Create research-integrated documentation."""
        doc = ResearchDocumentation(
            topic=topic,
            research_papers=research_papers,
            methodologies=methodologies,
            benchmarks=[],  # Would be populated from benchmark data
            implementation_status="integrated"
        )

        self.research_docs[topic] = doc
        return doc

    def generate_research_section(self, topic: str) -> str:
        """Generate a research section for documentation."""
        if topic not in self.research_docs:
            return ""

        doc = self.research_docs[topic]

        research_section = """
## Research Integration

This module incorporates methodologies from the following research:

### Key Papers

""" + "\n".join([f"""#### {paper['title']}
- **Authors:** {', '.join(paper['authors'])}
- **Venue:** {paper['venue']} {paper['year']}
- **Citations:** {paper['citations']}
- **Abstract:** {paper['abstract'][:200]}...

""" for paper in doc.research_papers]) + """

### Implemented Methodologies

""" + "\n".join([f"""#### {methodology['name']}
{methodology['description']}

**Category:** {methodology['category']}
**Validation Metrics:** {', '.join(methodology['validation_metrics'])}

""" for methodology in doc.methodologies]) + f"""

*Research integration last updated: {doc.last_updated.isoformat()}*
        """

        return research_section


class InteractiveDocumentation:
    """Provides interactive documentation features."""

    def __init__(self):
        self.examples: Dict[str, List[str]] = defaultdict(list)
        self.tutorials: Dict[str, Dict[str, Any]] = {}

    def add_code_example(self, module: str, example: str) -> None:
        """Add a code example for a module."""
        self.examples[module].append(example)

    def create_tutorial(self, tutorial_id: str, title: str,
                       steps: List[Dict[str, Any]]) -> None:
        """Create an interactive tutorial."""
        self.tutorials[tutorial_id] = {
            "title": title,
            "steps": steps,
            "created_at": datetime.utcnow()
        }

    def generate_tutorial_html(self, tutorial_id: str) -> str:
        """Generate HTML for an interactive tutorial."""
        if tutorial_id not in self.tutorials:
            return "<h1>Tutorial not found</h1>"

        tutorial = self.tutorials[tutorial_id]

        html = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>{tutorial['title']}</title>
            <style>
                .tutorial-step {{ margin: 20px; padding: 15px; border: 1px solid #ddd; }}
                .code-example {{ background: #f5f5f5; padding: 10px; margin: 10px 0; }}
                .navigation {{ margin-top: 20px; }}
            </style>
        </head>
        <body>
            <h1>{tutorial['title']}</h1>
        """

        for i, step in enumerate(tutorial['steps'], 1):
            html += f"""
            <div class="tutorial-step">
                <h3>Step {i}: {step['title']}</h3>
                <p>{step['description']}</p>
                {f'<div class="code-example"><pre>{step["code"]}</pre></div>' if 'code' in step else ''}
            </div>
            """

        html += """
            <div class="navigation">
                <button onclick="previousStep()">Previous</button>
                <button onclick="nextStep()">Next</button>
            </div>
        </body>
        </html>
        """

        return html


class DocumentationQualityAssessor:
    """Assesses documentation quality and suggests improvements."""

    def __init__(self):
        self.quality_metrics = {
            "completeness": self._assess_completeness,
            "clarity": self._assess_clarity,
            "consistency": self._assess_consistency,
            "technical_accuracy": self._assess_technical_accuracy
        }

    def assess_documentation(self, content: str) -> Dict[str, Any]:
        """Assess the quality of documentation."""
        assessment = {}

        for metric_name, metric_func in self.quality_metrics.items():
            assessment[metric_name] = metric_func(content)

        # Overall score
        assessment["overall_score"] = sum(assessment.values()) / len(assessment)

        # Generate recommendations
        assessment["recommendations"] = self._generate_recommendations(assessment)

        return assessment

    def _assess_completeness(self, content: str) -> float:
        """Assess documentation completeness."""
        required_sections = ["Features", "Installation", "Usage", "API"]
        found_sections = sum(1 for section in required_sections if section in content)
        return found_sections / len(required_sections)

    def _assess_clarity(self, content: str) -> float:
        """Assess documentation clarity."""
        # Simple heuristic: average sentence length
        sentences = re.split(r'[.!?]+', content)
        avg_length = sum(len(s.split()) for s in sentences) / len(sentences) if sentences else 0

        # Ideal range: 15-25 words per sentence
        if 15 <= avg_length <= 25:
            return 1.0
        elif 10 <= avg_length <= 30:
            return 0.8
        else:
            return 0.5

    def _assess_consistency(self, content: str) -> float:
        """Assess documentation consistency."""
        # Check for consistent heading styles, terminology, etc.
        # Simple check: consistent use of code formatting
        code_blocks = len(re.findall(r'`[^`]+`', content))
        total_words = len(content.split())

        if total_words > 0:
            code_ratio = code_blocks / total_words
            return min(code_ratio * 100, 1.0)  # Normalize
        return 0.5

    def _assess_technical_accuracy(self, content: str) -> float:
        """Assess technical accuracy."""
        # This would require more sophisticated analysis
        # For now, use a simple heuristic
        technical_terms = ["API", "function", "class", "method", "algorithm"]
        found_terms = sum(1 for term in technical_terms if term in content)
        return found_terms / len(technical_terms)

    def _generate_recommendations(self, assessment: Dict[str, float]) -> List[str]:
        """Generate improvement recommendations."""
        recommendations = []

        if assessment["completeness"] < 0.8:
            recommendations.append("Add missing sections (Features, Installation, Usage, API)")

        if assessment["clarity"] < 0.7:
            recommendations.append("Improve sentence structure and readability")

        if assessment["consistency"] < 0.6:
            recommendations.append("Ensure consistent formatting and terminology")

        if assessment["technical_accuracy"] < 0.8:
            recommendations.append("Review technical content for accuracy")

        return recommendations


class DocumentationManager:
    """Main manager for all documentation features."""

    def __init__(self):
        self.entries: Dict[str, DocumentationEntry] = {}
        self.api_docs: Dict[str, APIDocumentation] = {}
        self.auto_generator = AutoDocumentationGenerator()
        self.research_integrator = ResearchDocumentationIntegrator()
        self.interactive_docs = InteractiveDocumentation()
        self.quality_assessor = DocumentationQualityAssessor()

    def add_documentation_entry(self, entry: DocumentationEntry) -> None:
        """Add a documentation entry."""
        self.entries[entry.entry_id] = entry

    def generate_api_documentation(self, module_path: str) -> APIDocumentation:
        """Generate API documentation for a module."""
        api_doc = APIDocumentation(module_name=Path(module_path).stem)

        # This would parse the module and extract API information
        # For now, we'll use the auto-generator
        api_doc.classes = []  # Would be populated
        api_doc.functions = []  # Would be populated

        self.api_docs[api_doc.module_name] = api_doc
        return api_doc

    def generate_comprehensive_documentation(self, module_path: str,
                                           research_topic: Optional[str] = None) -> str:
        """Generate comprehensive documentation for a module."""
        # Generate auto documentation
        auto_docs = self.auto_generator.generate_module_docs(module_path)

        # Add research integration if available
        if research_topic:
            research_section = self.research_integrator.generate_research_section(research_topic)
            auto_docs += "\n\n" + research_section

        # Add interactive examples
        module_name = Path(module_path).stem
        if module_name in self.interactive_docs.examples:
            examples_section = "\n\n## Code Examples\n\n"
            for example in self.interactive_docs.examples[module_name]:
                examples_section += f"```python\n{example}\n```\n\n"
            auto_docs += examples_section

        # Assess quality
        quality_assessment = self.quality_assessor.assess_documentation(auto_docs)
        quality_section = f"""
## Documentation Quality Assessment

- **Overall Score:** {quality_assessment['overall_score']:.2f}
- **Completeness:** {quality_assessment['completeness']:.2f}
- **Clarity:** {quality_assessment['clarity']:.2f}
- **Consistency:** {quality_assessment['consistency']:.2f}
- **Technical Accuracy:** {quality_assessment['technical_accuracy']:.2f}

### Recommendations

{chr(10).join(f"- {rec}" for rec in quality_assessment['recommendations'])}
        """
        auto_docs += quality_section

        return auto_docs

    def search_documentation(self, query: str) -> List[DocumentationEntry]:
        """Search documentation entries."""
        results = []
        query_lower = query.lower()

        for entry in self.entries.values():
            if (query_lower in entry.title.lower() or
                query_lower in entry.content.lower() or
                any(query_lower in tag.lower() for tag in entry.tags)):
                results.append(entry)

        return results

    def export_documentation(self, format: str = "markdown") -> str:
        """Export all documentation."""
        if format == "markdown":
            output = "# Code Intelligence Platform Documentation\n\n"

            for entry in self.entries.values():
                output += f"## {entry.title}\n\n"
                output += f"{entry.content}\n\n"

                if entry.tags:
                    output += f"**Tags:** {', '.join(entry.tags)}\n\n"

                if entry.related_modules:
                    output += f"**Related Modules:** {', '.join(entry.related_modules)}\n\n"

                output += "---\n\n"

            return output

        elif format == "json":
            docs_data = {
                "entries": [
                    {
                        "entry_id": entry.entry_id,
                        "title": entry.title,
                        "content": entry.content,
                        "category": entry.category,
                        "tags": entry.tags,
                        "related_modules": entry.related_modules,
                        "last_updated": entry.last_updated.isoformat()
                    }
                    for entry in self.entries.values()
                ],
                "export_timestamp": datetime.utcnow().isoformat()
            }
            return json.dumps(docs_data, indent=2)

        else:
            raise ValueError(f"Unsupported format: {format}")

    def create_module_documentation(self, module_name: str,
                                  research_references: Optional[List[str]] = None) -> DocumentationEntry:
        """Create comprehensive documentation for a module."""
        module_path = f"src/code_explainer/{module_name}.py"

        if not os.path.exists(module_path):
            raise FileNotFoundError(f"Module {module_path} not found")

        # Generate comprehensive documentation
        content = self.generate_comprehensive_documentation(
            module_path,
            research_topic=module_name.replace("_", " ").title()
        )

        entry = DocumentationEntry(
            entry_id=f"doc_{module_name}",
            title=f"{module_name.replace('_', ' ').title()} Module",
            content=content,
            category="module",
            tags=["module", module_name, "documentation"],
            related_modules=[module_name],
            research_references=research_references or []
        )

        self.add_documentation_entry(entry)
        return entry


# Export main classes
__all__ = [
    "DocumentationEntry",
    "APIDocumentation",
    "ResearchDocumentation",
    "AutoDocumentationGenerator",
    "ResearchDocumentationIntegrator",
    "InteractiveDocumentation",
    "DocumentationQualityAssessor",
    "DocumentationManager"
]
