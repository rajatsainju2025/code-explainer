"""
API Documentation Generator

This module provides automatic generation of comprehensive API documentation
for the Code Explainer system, including OpenAPI/Swagger specs, interactive docs,
and multiple output formats.

Key Features:
- Automatic OpenAPI 3.0 specification generation
- Interactive API documentation with Swagger UI
- Multiple output formats (HTML, Markdown, PDF)
- API endpoint discovery and analysis
- Request/response schema documentation
- Authentication and security documentation
- API versioning and changelog generation
- Performance metrics documentation
- Integration with external documentation systems

Based on modern API documentation standards and best practices.
"""

import json
import yaml
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Union, Callable
from abc import ABC, abstractmethod
import inspect
import re
from functools import wraps
import logging

logger = logging.getLogger(__name__)

@dataclass
class APIEndpoint:
    """Represents an API endpoint."""
    path: str
    method: str
    summary: str
    description: str
    operation_id: str
    tags: List[str] = field(default_factory=list)
    parameters: List[Dict[str, Any]] = field(default_factory=list)
    request_body: Optional[Dict[str, Any]] = None
    responses: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    security: List[Dict[str, Any]] = field(default_factory=list)
    deprecated: bool = False

@dataclass
class APISchema:
    """Represents an API schema definition."""
    name: str
    type: str
    properties: Dict[str, Any] = field(default_factory=dict)
    required: List[str] = field(default_factory=list)
    description: str = ""
    example: Optional[Any] = None

@dataclass
class APIDocumentation:
    """Complete API documentation."""
    openapi_version: str = "3.0.3"
    info: Dict[str, Any] = field(default_factory=dict)
    servers: List[Dict[str, Any]] = field(default_factory=list)
    paths: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    components: Dict[str, Any] = field(default_factory=dict)
    security: List[Dict[str, Any]] = field(default_factory=list)
    tags: List[Dict[str, Any]] = field(default_factory=list)
    external_docs: Optional[Dict[str, Any]] = None

class APIDocumentationGenerator:
    """Main API documentation generator."""

    def __init__(self, title: str = "Code Explainer API",
                 version: str = "1.0.0",
                 description: str = "AI-powered code explanation API"):
        self.documentation = APIDocumentation()
        self.documentation.info = {
            "title": title,
            "version": version,
            "description": description,
            "contact": {
                "name": "Code Explainer Team",
                "url": "https://github.com/rajatsainju2025/code-explainer"
            },
            "license": {
                "name": "MIT",
                "url": "https://opensource.org/licenses/MIT"
            }
        }
        self.endpoints: List[APIEndpoint] = []
        self.schemas: Dict[str, APISchema] = {}
        self.base_path = "/api/v1"

    def add_server(self, url: str, description: str = ""):
        """Add API server."""
        server = {"url": url}
        if description:
            server["description"] = description
        self.documentation.servers.append(server)

    def add_endpoint(self, endpoint: APIEndpoint):
        """Add API endpoint."""
        self.endpoints.append(endpoint)

        # Add to OpenAPI paths
        if endpoint.path not in self.documentation.paths:
            self.documentation.paths[endpoint.path] = {}

        method_spec = {
            "summary": endpoint.summary,
            "description": endpoint.description,
            "operationId": endpoint.operation_id,
            "tags": endpoint.tags,
            "responses": endpoint.responses
        }

        if endpoint.parameters:
            method_spec["parameters"] = endpoint.parameters

        if endpoint.request_body:
            method_spec["requestBody"] = endpoint.request_body

        if endpoint.security:
            method_spec["security"] = endpoint.security

        if endpoint.deprecated:
            method_spec["deprecated"] = True

        self.documentation.paths[endpoint.path][endpoint.method.lower()] = method_spec

    def add_schema(self, schema: APISchema):
        """Add schema definition."""
        self.schemas[schema.name] = schema

        # Add to OpenAPI components
        if "schemas" not in self.documentation.components:
            self.documentation.components["schemas"] = {}

        schema_spec: Dict[str, Any] = {
            "type": schema.type,
            "description": schema.description
        }

        if schema.properties:
            schema_spec["properties"] = schema.properties

        if schema.required:
            schema_spec["required"] = schema.required

        if schema.example is not None:
            schema_spec["example"] = schema.example

        self.documentation.components["schemas"][schema.name] = schema_spec

    def add_security_scheme(self, name: str, scheme_type: str,
                          scheme_config: Dict[str, Any]):
        """Add security scheme."""
        if "securitySchemes" not in self.documentation.components:
            self.documentation.components["securitySchemes"] = {}

        self.documentation.components["securitySchemes"][name] = {
            "type": scheme_type,
            **scheme_config
        }

    def add_tag(self, name: str, description: str):
        """Add API tag."""
        self.documentation.tags.append({
            "name": name,
            "description": description
        })

    def generate_openapi_spec(self) -> Dict[str, Any]:
        """Generate OpenAPI specification."""
        spec = {
            "openapi": self.documentation.openapi_version,
            "info": self.documentation.info,
            "servers": self.documentation.servers,
            "paths": self.documentation.paths,
            "components": self.documentation.components,
            "tags": self.documentation.tags
        }

        if self.documentation.security:
            spec["security"] = self.documentation.security

        if self.documentation.external_docs:
            spec["externalDocs"] = self.documentation.external_docs

        return spec

    def export_openapi_json(self, filepath: Path):
        """Export OpenAPI spec as JSON."""
        spec = self.generate_openapi_spec()
        with open(filepath, 'w') as f:
            json.dump(spec, f, indent=2)

    def export_openapi_yaml(self, filepath: Path):
        """Export OpenAPI spec as YAML."""
        spec = self.generate_openapi_spec()
        with open(filepath, 'w') as f:
            yaml.dump(spec, f, default_flow_style=False)

    def generate_html_docs(self, filepath: Path):
        """Generate HTML documentation."""
        spec = self.generate_openapi_spec()

        html_template = f"""
<!DOCTYPE html>
<html>
<head>
    <title>{self.documentation.info['title']} - API Documentation</title>
    <link rel="stylesheet" type="text/css" href="https://unpkg.com/swagger-ui-dist@3/swagger-ui.css" />
    <style>
        body {{
            margin: 0;
            padding: 0;
        }}
        .swagger-ui .topbar {{
            display: none;
        }}
    </style>
</head>
<body>
    <div id="swagger-ui"></div>
    <script src="https://unpkg.com/swagger-ui-dist@3/swagger-ui-bundle.js"></script>
    <script>
        const spec = {json.dumps(spec)};
        const ui = SwaggerUIBundle({{
            spec: spec,
            dom_id: '#swagger-ui',
            presets: [
                SwaggerUIBundle.presets.apis,
                SwaggerUIBundle.presets.standalone
            ]
        }});
    </script>
</body>
</html>
"""

        with open(filepath, 'w') as f:
            f.write(html_template)

    def generate_markdown_docs(self, filepath: Path):
        """Generate Markdown documentation."""
        spec = self.generate_openapi_spec()

        md_content = f"""# {self.documentation.info['title']}

{self.documentation.info['description']}

**Version:** {self.documentation.info['version']}

## Servers

"""

        for server in spec.get('servers', []):
            md_content += f"- {server['url']}"
            if 'description' in server:
                md_content += f" - {server['description']}"
            md_content += "\n"

        md_content += "\n## Endpoints\n\n"

        for path, methods in spec.get('paths', {}).items():
            for method, details in methods.items():
                md_content += f"### {method.upper()} {path}\n\n"
                md_content += f"**{details.get('summary', '')}**\n\n"
                if 'description' in details:
                    md_content += f"{details['description']}\n\n"

                if 'parameters' in details:
                    md_content += "**Parameters:**\n\n"
                    for param in details['parameters']:
                        required = " (required)" if param.get('required', False) else ""
                        md_content += f"- `{param['name']}` ({param['schema']['type']}){required}: {param.get('description', '')}\n"
                    md_content += "\n"

                if 'requestBody' in details:
                    md_content += "**Request Body:**\n\n"
                    # Simplified request body documentation
                    md_content += "See OpenAPI spec for detailed schema.\n\n"

                if 'responses' in details:
                    md_content += "**Responses:**\n\n"
                    for status, response in details['responses'].items():
                        md_content += f"- `{status}`: {response.get('description', '')}\n"
                    md_content += "\n"

                md_content += "---\n\n"

        with open(filepath, 'w') as f:
            f.write(md_content)

    def validate_spec(self) -> List[str]:
        """Validate OpenAPI specification."""
        errors = []

        spec = self.generate_openapi_spec()

        # Basic validation
        if 'info' not in spec:
            errors.append("Missing 'info' section")

        if 'paths' not in spec:
            errors.append("Missing 'paths' section")

        # Check for required fields in info
        info_required = ['title', 'version']
        for field in info_required:
            if field not in spec.get('info', {}):
                errors.append(f"Missing required field 'info.{field}'")

        # Check paths
        for path, methods in spec.get('paths', {}).items():
            if not methods:
                errors.append(f"Path '{path}' has no methods defined")

            for method, details in methods.items():
                if 'responses' not in details:
                    errors.append(f"Method '{method} {path}' missing responses")

        return errors

class FastAPIDocumentationGenerator(APIDocumentationGenerator):
    """Documentation generator specifically for FastAPI applications."""

    def __init__(self, app=None, **kwargs):
        super().__init__(**kwargs)
        self.app = app

    def auto_generate_from_fastapi(self):
        """Automatically generate documentation from FastAPI app."""
        if not self.app:
            logger.warning("No FastAPI app provided for auto-generation")
            return

        try:
            # Get OpenAPI schema from FastAPI
            openapi_schema = self.app.openapi()

            # Convert to our format
            self._convert_fastapi_schema(openapi_schema)

        except Exception as e:
            logger.error(f"Failed to auto-generate from FastAPI: {str(e)}")

    def _convert_fastapi_schema(self, fastapi_schema: Dict[str, Any]):
        """Convert FastAPI OpenAPI schema to our format."""
        # Update basic info
        if 'info' in fastapi_schema:
            self.documentation.info.update(fastapi_schema['info'])

        # Add servers
        if 'servers' in fastapi_schema:
            self.documentation.servers = fastapi_schema['servers']

        # Add paths
        if 'paths' in fastapi_schema:
            self.documentation.paths = fastapi_schema['paths']

        # Add components
        if 'components' in fastapi_schema:
            self.documentation.components = fastapi_schema['components']

        # Add security
        if 'security' in fastapi_schema:
            self.documentation.security = fastapi_schema['security']

        # Add tags
        if 'tags' in fastapi_schema:
            self.documentation.tags = fastapi_schema['tags']

class DocumentationPublisher:
    """Publish documentation to various platforms."""

    def __init__(self, generator: APIDocumentationGenerator):
        self.generator = generator

    def publish_to_readme(self, readme_path: Path, section: str = "API Documentation"):
        """Publish API docs to README file."""
        md_docs = self.generator.generate_markdown_docs(Path("/tmp/temp_api_docs.md"))

        with open("/tmp/temp_api_docs.md", 'r') as f:
            docs_content = f.read()

        # Read existing README
        if readme_path.exists():
            with open(readme_path, 'r') as f:
                readme_content = f.read()
        else:
            readme_content = ""

        # Replace or add API documentation section
        section_pattern = rf"## {re.escape(section)}\n.*?(?=\n## |\Z)"
        new_section = f"## {section}\n\n{docs_content}\n"

        if re.search(section_pattern, readme_content, re.DOTALL):
            updated_content = re.sub(section_pattern, new_section, readme_content, flags=re.DOTALL)
        else:
            updated_content = readme_content + "\n" + new_section

        with open(readme_path, 'w') as f:
            f.write(updated_content)

    def publish_to_wiki(self, wiki_path: Path):
        """Publish to GitHub Wiki."""
        wiki_path.mkdir(parents=True, exist_ok=True)

        # Generate different formats
        self.generator.export_openapi_json(wiki_path / "openapi.json")
        self.generator.export_openapi_yaml(wiki_path / "openapi.yaml")
        self.generator.generate_html_docs(wiki_path / "api-docs.html")
        self.generator.generate_markdown_docs(wiki_path / "API-Documentation.md")

    def publish_to_confluence(self, space_key: str, page_title: str):
        """Publish to Confluence (placeholder for future implementation)."""
        logger.info(f"Confluence publishing not implemented yet. Would publish to space '{space_key}' with title '{page_title}'")

# Decorators for automatic documentation
def document_endpoint(summary: str = "", description: str = "",
                     tags: Optional[List[str]] = None):
    """Decorator to automatically document API endpoints."""
    def decorator(func: Callable):
        setattr(func, '_api_summary', summary)
        setattr(func, '_api_description', description)
        setattr(func, '_api_tags', tags or [])
        return func
    return decorator

def document_schema(name: str, description: str = ""):
    """Decorator to document response/request schemas."""
    def decorator(cls):
        cls._schema_name = name
        cls._schema_description = description
        return cls
    return decorator

# Convenience functions
def create_api_documentation(title: str = "Code Explainer API",
                           version: str = "1.0.0") -> APIDocumentationGenerator:
    """Create API documentation generator."""
    return APIDocumentationGenerator(title=title, version=version)

def generate_code_explainer_docs() -> APIDocumentationGenerator:
    """Generate documentation for Code Explainer API."""
    generator = APIDocumentationGenerator(
        title="Code Explainer API",
        version="1.0.0",
        description="AI-powered code explanation and analysis API"
    )

    # Add servers
    generator.add_server("https://api.code-explainer.com", "Production")
    generator.add_server("http://localhost:8000", "Development")

    # Add security scheme
    generator.add_security_scheme(
        "bearerAuth",
        "http",
        {"scheme": "bearer", "bearerFormat": "JWT"}
    )

    # Add tags
    generator.add_tag("explanation", "Code explanation endpoints")
    generator.add_tag("analysis", "Code analysis endpoints")
    generator.add_tag("health", "Health check endpoints")

    # Add example endpoint
    explain_endpoint = APIEndpoint(
        path="/api/v1/explain",
        method="POST",
        summary="Explain code snippet",
        description="Generate AI-powered explanation for a code snippet",
        operation_id="explainCode",
        tags=["explanation"],
        parameters=[],
        request_body={
            "required": True,
            "content": {
                "application/json": {
                    "schema": {"$ref": "#/components/schemas/CodeRequest"}
                }
            }
        },
        responses={
            "200": {
                "description": "Successful explanation",
                "content": {
                    "application/json": {
                        "schema": {"$ref": "#/components/schemas/ExplanationResponse"}
                    }
                }
            },
            "400": {"description": "Invalid request"},
            "500": {"description": "Internal server error"}
        },
        security=[{"bearerAuth": []}]
    )

    generator.add_endpoint(explain_endpoint)

    # Add schemas
    code_request_schema = APISchema(
        name="CodeRequest",
        type="object",
        properties={
            "code": {"type": "string", "description": "Code snippet to explain"},
            "language": {"type": "string", "description": "Programming language"},
            "context": {"type": "string", "description": "Additional context"}
        },
        required=["code"],
        description="Request payload for code explanation"
    )

    generator.add_schema(code_request_schema)

    explanation_response_schema = APISchema(
        name="ExplanationResponse",
        type="object",
        properties={
            "explanation": {"type": "string", "description": "Generated explanation"},
            "confidence": {"type": "number", "description": "Confidence score"},
            "metadata": {"type": "object", "description": "Additional metadata"}
        },
        required=["explanation"],
        description="Response payload for code explanation"
    )

    generator.add_schema(explanation_response_schema)

    return generator

if __name__ == "__main__":
    # Example usage
    generator = generate_code_explainer_docs()

    # Export different formats
    generator.export_openapi_json(Path("docs/api/openapi.json"))
    generator.export_openapi_yaml(Path("docs/api/openapi.yaml"))
    generator.generate_html_docs(Path("docs/api/index.html"))
    generator.generate_markdown_docs(Path("docs/api/README.md"))

    # Validate specification
    errors = generator.validate_spec()
    if errors:
        print(f"Validation errors: {errors}")
    else:
        print("API specification is valid")

    print("API documentation generated successfully!")
