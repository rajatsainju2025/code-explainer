"""
CI/CD Pipeline Setup Module for Code Intelligence Platform

This module provides comprehensive CI/CD capabilities including automated pipelines,
deployment strategies, environment management, and DevOps best practices to ensure
reliable and efficient software delivery.

Features:
- GitHub Actions workflow automation
- Multi-environment deployment (dev, staging, production)
- Docker containerization and orchestration
- Automated testing and quality gates
- Deployment strategies (blue-green, canary, rolling)
- Infrastructure as Code (IaC) management
- Monitoring and alerting integration
- Security scanning and compliance checks
- Release management and versioning
"""

import os
import yaml
import json
from typing import Dict, List, Optional, Any, Callable, Union
from dataclasses import dataclass, field
from enum import Enum
import subprocess
import time
import logging

logger = logging.getLogger(__name__)


class Environment(Enum):
    """Deployment environments."""
    DEVELOPMENT = "development"
    STAGING = "staging"
    PRODUCTION = "production"
    TESTING = "testing"


class DeploymentStrategy(Enum):
    """Deployment strategies."""
    BLUE_GREEN = "blue_green"
    CANARY = "canary"
    ROLLING = "rolling"
    RECREATE = "recreate"


class PipelineStage(Enum):
    """CI/CD pipeline stages."""
    BUILD = "build"
    TEST = "test"
    SECURITY_SCAN = "security_scan"
    DEPLOY = "deploy"
    MONITOR = "monitor"
    ROLLBACK = "rollback"


@dataclass
class PipelineConfig:
    """Configuration for CI/CD pipeline."""
    name: str
    environments: List[Environment]
    stages: List[PipelineStage]
    triggers: List[str] = field(default_factory=list)
    secrets: Dict[str, str] = field(default_factory=dict)
    variables: Dict[str, str] = field(default_factory=dict)
    timeout: int = 3600  # 1 hour
    retries: int = 3


@dataclass
class DeploymentConfig:
    """Configuration for deployment."""
    environment: Environment
    strategy: DeploymentStrategy
    replicas: int = 1
    resources: Dict[str, Any] = field(default_factory=dict)
    health_checks: List[Dict[str, Any]] = field(default_factory=list)
    rollback_enabled: bool = True


class GitHubActionsGenerator:
    """Generate GitHub Actions workflows."""

    def __init__(self):
        self.templates = {
            "python_ci": self._generate_python_ci_workflow,
            "docker_build": self._generate_docker_build_workflow,
            "security_scan": self._generate_security_scan_workflow,
            "deploy": self._generate_deploy_workflow,
        }

    def generate_workflow(self, workflow_type: str, config: Dict[str, Any]) -> str:
        """Generate GitHub Actions workflow YAML."""
        if workflow_type in self.templates:
            return self.templates[workflow_type](config)
        else:
            return self._generate_basic_workflow(config)

    def _generate_python_ci_workflow(self, config: Dict[str, Any]) -> str:
        """Generate Python CI workflow."""
        workflow = {
            "name": "Python CI",
            "on": {
                "push": {"branches": ["main", "develop"]},
                "pull_request": {"branches": ["main"]}
            },
            "jobs": {
                "test": {
                    "runs-on": "ubuntu-latest",
                    "strategy": {
                        "matrix": {
                            "python-version": ["3.8", "3.9", "3.10", "3.11"]
                        }
                    },
                    "steps": [
                        {
                            "name": "Checkout code",
                            "uses": "actions/checkout@v4"
                        },
                        {
                            "name": "Set up Python",
                            "uses": "actions/setup-python@v4",
                            "with": {
                                "python-version": "${{ matrix.python-version }}"
                            }
                        },
                        {
                            "name": "Install dependencies",
                            "run": "pip install -r requirements.txt"
                        },
                        {
                            "name": "Run tests",
                            "run": "python -m pytest tests/ -v --cov=src/"
                        },
                        {
                            "name": "Upload coverage",
                            "uses": "codecov/codecov-action@v3"
                        }
                    ]
                },
                "lint": {
                    "runs-on": "ubuntu-latest",
                    "steps": [
                        {
                            "name": "Checkout code",
                            "uses": "actions/checkout@v4"
                        },
                        {
                            "name": "Set up Python",
                            "uses": "actions/setup-python@v4",
                            "with": {"python-version": "3.9"}
                        },
                        {
                            "name": "Install linting tools",
                            "run": "pip install flake8 black isort mypy"
                        },
                        {
                            "name": "Run linters",
                            "run": "flake8 src/ --max-line-length=88"
                        },
                        {
                            "name": "Check formatting",
                            "run": "black --check src/"
                        }
                    ]
                }
            }
        }

        return yaml.dump(workflow, default_flow_style=False, sort_keys=False)

    def _generate_docker_build_workflow(self, config: Dict[str, Any]) -> str:
        """Generate Docker build workflow."""
        workflow = {
            "name": "Docker Build and Push",
            "on": {
                "push": {"branches": ["main"]},
                "release": {"types": ["published"]}
            },
            "jobs": {
                "build": {
                    "runs-on": "ubuntu-latest",
                    "steps": [
                        {
                            "name": "Checkout code",
                            "uses": "actions/checkout@v4"
                        },
                        {
                            "name": "Set up Docker Buildx",
                            "uses": "docker/setup-buildx-action@v3"
                        },
                        {
                            "name": "Log in to Docker Hub",
                            "uses": "docker/login-action@v3",
                            "with": {
                                "username": "${{ secrets.DOCKER_USERNAME }}",
                                "password": "${{ secrets.DOCKER_PASSWORD }}"
                            }
                        },
                        {
                            "name": "Build and push Docker image",
                            "uses": "docker/build-push-action@v5",
                            "with": {
                                "context": ".",
                                "push": True,
                                "tags": "myapp:latest,myapp:${{ github.sha }}",
                                "cache-from": "type=gha",
                                "cache-to": "type=gha,mode=max"
                            }
                        }
                    ]
                }
            }
        }

        return yaml.dump(workflow, default_flow_style=False, sort_keys=False)

    def _generate_security_scan_workflow(self, config: Dict[str, Any]) -> str:
        """Generate security scanning workflow."""
        workflow = {
            "name": "Security Scan",
            "on": {
                "push": {"branches": ["main", "develop"]},
                "pull_request": {"branches": ["main"]},
                "schedule": [{"cron": "0 0 * * 1"}]  # Weekly on Monday
            },
            "jobs": {
                "security": {
                    "runs-on": "ubuntu-latest",
                    "steps": [
                        {
                            "name": "Checkout code",
                            "uses": "actions/checkout@v4"
                        },
                        {
                            "name": "Run Trivy vulnerability scanner",
                            "uses": "aquasecurity/trivy-action@master",
                            "with": {
                                "scan-type": "fs",
                                "scan-ref": ".",
                                "format": "sarif",
                                "output": "trivy-results.sarif"
                            }
                        },
                        {
                            "name": "Upload Trivy scan results",
                            "uses": "github/codeql-action/upload-sarif@v2",
                            "with": {
                                "sarif_file": "trivy-results.sarif"
                            }
                        },
                        {
                            "name": "Run Bandit security linter",
                            "run": "pip install bandit && bandit -r src/ -f json -o bandit-results.json"
                        }
                    ]
                }
            }
        }

        return yaml.dump(workflow, default_flow_style=False, sort_keys=False)

    def _generate_deploy_workflow(self, config: Dict[str, Any]) -> str:
        """Generate deployment workflow."""
        workflow = {
            "name": "Deploy to Production",
            "on": {
                "release": {"types": ["published"]},
                "workflow_dispatch": {}
            },
            "jobs": {
                "deploy": {
                    "runs-on": "ubuntu-latest",
                    "environment": "production",
                    "steps": [
                        {
                            "name": "Checkout code",
                            "uses": "actions/checkout@v4"
                        },
                        {
                            "name": "Configure AWS credentials",
                            "uses": "aws-actions/configure-aws-credentials@v4",
                            "with": {
                                "aws-access-key-id": "${{ secrets.AWS_ACCESS_KEY_ID }}",
                                "aws-secret-access-key": "${{ secrets.AWS_SECRET_ACCESS_KEY }}",
                                "aws-region": "us-east-1"
                            }
                        },
                        {
                            "name": "Deploy to ECS",
                            "run": "aws ecs update-service --cluster my-cluster --service my-service --force-new-deployment"
                        },
                        {
                            "name": "Wait for deployment",
                            "run": "aws ecs wait services-stable --cluster my-cluster --services my-service"
                        },
                        {
                            "name": "Run health checks",
                            "run": "curl -f https://myapp.com/health || exit 1"
                        }
                    ]
                }
            }
        }

        return yaml.dump(workflow, default_flow_style=False, sort_keys=False)

    def _generate_basic_workflow(self, config: Dict[str, Any]) -> str:
        """Generate basic workflow template."""
        workflow = {
            "name": config.get("name", "CI/CD Pipeline"),
            "on": config.get("triggers", ["push"]),
            "jobs": {
                "build": {
                    "runs-on": "ubuntu-latest",
                    "steps": [
                        {"name": "Checkout", "uses": "actions/checkout@v4"},
                        {"name": "Run", "run": "echo 'Hello, World!'"}
                    ]
                }
            }
        }

        return yaml.dump(workflow, default_flow_style=False, sort_keys=False)


class DockerManager:
    """Docker containerization management."""

    def __init__(self):
        self.dockerfiles: Dict[str, str] = {}

    def generate_dockerfile(self, app_type: str = "python", config: Optional[Dict[str, Any]] = None) -> str:
        """Generate Dockerfile based on application type."""
        if config is None:
            config = {}

        if app_type == "python":
            return self._generate_python_dockerfile(config)
        elif app_type == "node":
            return self._generate_node_dockerfile(config)
        else:
            return self._generate_basic_dockerfile(config)

    def _generate_python_dockerfile(self, config: Dict[str, Any]) -> str:
        """Generate Python Dockerfile."""
        python_version = config.get("python_version", "3.9")
        port = config.get("port", 8000)

        dockerfile = f'''FROM python:{python_version}-slim

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# Set work directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \\
    gcc \\
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy project
COPY . .

# Create non-root user
RUN useradd --create-home --shell /bin/bash app \\
    && chown -R app:app /app
USER app

# Expose port
EXPOSE {port}

# Run the application
CMD ["python", "app.py"]
'''
        return dockerfile

    def _generate_node_dockerfile(self, config: Dict[str, Any]) -> str:
        """Generate Node.js Dockerfile."""
        node_version = config.get("node_version", "18")
        port = config.get("port", 3000)

        dockerfile = f'''FROM node:{node_version}-slim

# Set work directory
WORKDIR /app

# Copy package files
COPY package*.json ./

# Install dependencies
RUN npm ci --only=production

# Copy application code
COPY . .

# Create non-root user
RUN useradd --create-home --shell /bin/bash app \\
    && chown -R app:app /app
USER app

# Expose port
EXPOSE {port}

# Run the application
CMD ["npm", "start"]
'''
        return dockerfile

    def _generate_basic_dockerfile(self, config: Dict[str, Any]) -> str:
        """Generate basic Dockerfile."""
        dockerfile = '''FROM ubuntu:20.04

# Install basic tools
RUN apt-get update && apt-get install -y \\
    curl \\
    wget \\
    && rm -rf /var/lib/apt/lists/*

# Set work directory
WORKDIR /app

# Copy application
COPY . .

# Run the application
CMD ["echo", "Hello from Docker!"]
'''
        return dockerfile

    def generate_docker_compose(self, services: Dict[str, Any]) -> str:
        """Generate docker-compose.yml file."""
        compose_config = {
            "version": "3.8",
            "services": services,
            "networks": {
                "app-network": {
                    "driver": "bridge"
                }
            }
        }

        return yaml.dump(compose_config, default_flow_style=False, sort_keys=False)


class KubernetesManager:
    """Kubernetes deployment management."""

    def __init__(self):
        self.manifests: Dict[str, str] = {}

    def generate_deployment_manifest(self, app_name: str, config: Dict[str, Any]) -> str:
        """Generate Kubernetes deployment manifest."""
        replicas = config.get("replicas", 3)
        image = config.get("image", f"{app_name}:latest")
        port = config.get("port", 8000)

        manifest = {
            "apiVersion": "apps/v1",
            "kind": "Deployment",
            "metadata": {
                "name": app_name,
                "labels": {"app": app_name}
            },
            "spec": {
                "replicas": replicas,
                "selector": {
                    "matchLabels": {"app": app_name}
                },
                "template": {
                    "metadata": {
                        "labels": {"app": app_name}
                    },
                    "spec": {
                        "containers": [{
                            "name": app_name,
                            "image": image,
                            "ports": [{"containerPort": port}],
                            "env": [
                                {"name": "ENV", "value": "production"}
                            ],
                            "resources": {
                                "requests": {
                                    "memory": "256Mi",
                                    "cpu": "250m"
                                },
                                "limits": {
                                    "memory": "512Mi",
                                    "cpu": "500m"
                                }
                            },
                            "livenessProbe": {
                                "httpGet": {
                                    "path": "/health",
                                    "port": port
                                },
                                "initialDelaySeconds": 30,
                                "periodSeconds": 10
                            },
                            "readinessProbe": {
                                "httpGet": {
                                    "path": "/ready",
                                    "port": port
                                },
                                "initialDelaySeconds": 5,
                                "periodSeconds": 5
                            }
                        }]
                    }
                }
            }
        }

        return yaml.dump(manifest, default_flow_style=False, sort_keys=False)

    def generate_service_manifest(self, app_name: str, config: Dict[str, Any]) -> str:
        """Generate Kubernetes service manifest."""
        port = config.get("port", 8000)
        service_type = config.get("service_type", "ClusterIP")

        manifest = {
            "apiVersion": "v1",
            "kind": "Service",
            "metadata": {
                "name": f"{app_name}-service",
                "labels": {"app": app_name}
            },
            "spec": {
                "selector": {"app": app_name},
                "ports": [{
                    "name": "http",
                    "port": 80,
                    "targetPort": port,
                    "protocol": "TCP"
                }],
                "type": service_type
            }
        }

        return yaml.dump(manifest, default_flow_style=False, sort_keys=False)

    def generate_ingress_manifest(self, app_name: str, config: Dict[str, Any]) -> str:
        """Generate Kubernetes ingress manifest."""
        domain = config.get("domain", f"{app_name}.example.com")

        manifest = {
            "apiVersion": "networking.k8s.io/v1",
            "kind": "Ingress",
            "metadata": {
                "name": f"{app_name}-ingress",
                "annotations": {
                    "nginx.ingress.kubernetes.io/rewrite-target": "/",
                    "cert-manager.io/cluster-issuer": "letsencrypt-prod"
                }
            },
            "spec": {
                "tls": [{
                    "hosts": [domain],
                    "secretName": f"{app_name}-tls"
                }],
                "rules": [{
                    "host": domain,
                    "http": {
                        "paths": [{
                            "path": "/",
                            "pathType": "Prefix",
                            "backend": {
                                "service": {
                                    "name": f"{app_name}-service",
                                    "port": {"number": 80}
                                }
                            }
                        }]
                    }
                }]
            }
        }

        return yaml.dump(manifest, default_flow_style=False, sort_keys=False)


class TerraformManager:
    """Infrastructure as Code with Terraform."""

    def __init__(self):
        self.modules: Dict[str, str] = {}

    def generate_aws_infrastructure(self, config: Dict[str, Any]) -> str:
        """Generate AWS infrastructure Terraform configuration."""
        region = config.get("region", "us-east-1")
        environment = config.get("environment", "production")

        terraform_config = f'''terraform {{
  required_providers {{
    aws = {{
      source  = "hashicorp/aws"
      version = "~> 5.0"
    }}
  }}
}}

provider "aws" {{
  region = "{region}"
}}

# VPC
resource "aws_vpc" "main" {{
  cidr_block = "10.0.0.0/16"

  tags = {{
    Name        = "code-explainer-{environment}"
    Environment = "{environment}"
  }}
}}

# Subnets
resource "aws_subnet" "public" {{
  count             = 2
  vpc_id            = aws_vpc.main.id
  cidr_block        = "10.0.${{count.index + 1}}.0/24"
  availability_zone = data.aws_availability_zones.available.names[count.index]

  tags = {{
    Name = "code-explainer-public-${{count.index + 1}}"
  }}
}}

# ECS Cluster
resource "aws_ecs_cluster" "main" {{
  name = "code-explainer-{environment}"

  tags = {{
    Environment = "{environment}"
  }}
}}

# ECR Repository
resource "aws_ecr_repository" "app" {{
  name = "code-explainer-app"

  tags = {{
    Environment = "{environment}"
  }}
}}

# Application Load Balancer
resource "aws_lb" "app" {{
  name               = "code-explainer-alb-{environment}"
  internal           = false
  load_balancer_type = "application"
  security_groups    = [aws_security_group.alb.id]
  subnets            = aws_subnet.public[*].id

  tags = {{
    Environment = "{environment}"
  }}
}}

# Security Groups
resource "aws_security_group" "alb" {{
  name   = "code-explainer-alb-sg"
  vpc_id = aws_vpc.main.id

  ingress {{
    from_port   = 80
    to_port     = 80
    protocol    = "tcp"
    cidr_blocks = ["0.0.0.0/0"]
  }}

  ingress {{
    from_port   = 443
    to_port     = 443
    protocol    = "tcp"
    cidr_blocks = ["0.0.0.0/0"]
  }}

  egress {{
    from_port   = 0
    to_port     = 0
    protocol    = "-1"
    cidr_blocks = ["0.0.0.0/0"]
  }}
}}

# Data source for availability zones
data "aws_availability_zones" "available" {{
  state = "available"
}}
'''
        return terraform_config

    def generate_gcp_infrastructure(self, config: Dict[str, Any]) -> str:
        """Generate GCP infrastructure Terraform configuration."""
        project = config.get("project", "my-project")
        region = config.get("region", "us-central1")

        terraform_config = f'''terraform {{
  required_providers {{
    google = {{
      source  = "hashicorp/google"
      version = "~> 4.0"
    }}
  }}
}}

provider "google" {{
  project = "{project}"
  region  = "{region}"
}}

# VPC Network
resource "google_compute_network" "vpc" {{
  name                    = "code-explainer-vpc"
  auto_create_subnetworks = false
}}

# Subnet
resource "google_compute_subnetwork" "subnet" {{
  name          = "code-explainer-subnet"
  ip_cidr_range = "10.0.1.0/24"
  region        = "{region}"
  network       = google_compute_network.vpc.id
}}

# GKE Cluster
resource "google_container_cluster" "primary" {{
  name     = "code-explainer-cluster"
  location = "{region}"

  # We can't create a cluster with no node pool defined, but we want to only use
  # separately managed node pools. So we create the smallest possible default
  # node pool and immediately delete it.
  remove_default_node_pool = true
  initial_node_count       = 1

  network    = google_compute_network.vpc.name
  subnetwork = google_compute_subnetwork.subnet.name
}}

# Cloud Storage Bucket
resource "google_storage_bucket" "app_storage" {{
  name          = "code-explainer-{project}-storage"
  location      = "{region}"
  force_destroy = true

  uniform_bucket_level_access = true
}}
'''
        return terraform_config


class PipelineOrchestrator:
    """Main orchestrator for CI/CD pipeline management."""

    def __init__(self):
        self.github_actions = GitHubActionsGenerator()
        self.docker = DockerManager()
        self.kubernetes = KubernetesManager()
        self.terraform = TerraformManager()
        self.pipelines: Dict[str, PipelineConfig] = {}

    def create_complete_pipeline(self, project_name: str, config: Dict[str, Any]) -> Dict[str, str]:
        """Create complete CI/CD pipeline setup."""
        pipeline_files = {}

        # Generate GitHub Actions workflows
        workflows = [
            "python_ci",
            "docker_build",
            "security_scan",
            "deploy"
        ]

        for workflow in workflows:
            workflow_yaml = self.github_actions.generate_workflow(workflow, config)
            pipeline_files[f".github/workflows/{workflow}.yml"] = workflow_yaml

        # Generate Docker configuration
        dockerfile = self.docker.generate_dockerfile("python", config)
        pipeline_files["Dockerfile"] = dockerfile

        docker_compose = self.docker.generate_docker_compose({
            "app": {
                "build": ".",
                "ports": ["8000:8000"],
                "environment": ["ENV=development"]
            },
            "db": {
                "image": "postgres:13",
                "environment": ["POSTGRES_DB=app", "POSTGRES_USER=user", "POSTGRES_PASSWORD=password"]
            }
        })
        pipeline_files["docker-compose.yml"] = docker_compose

        # Generate Kubernetes manifests
        deployment = self.kubernetes.generate_deployment_manifest(project_name, config)
        service = self.kubernetes.generate_service_manifest(project_name, config)
        ingress = self.kubernetes.generate_ingress_manifest(project_name, config)

        pipeline_files["k8s/deployment.yml"] = deployment
        pipeline_files["k8s/service.yml"] = service
        pipeline_files["k8s/ingress.yml"] = ingress

        # Generate Terraform configuration
        terraform_config = self.terraform.generate_aws_infrastructure(config)
        pipeline_files["terraform/main.tf"] = terraform_config

        return pipeline_files

    def save_pipeline_files(self, files: Dict[str, str], base_path: str = ".") -> None:
        """Save pipeline files to disk."""
        for file_path, content in files.items():
            full_path = os.path.join(base_path, file_path)
            os.makedirs(os.path.dirname(full_path), exist_ok=True)

            with open(full_path, 'w') as f:
                f.write(content)

            logger.info(f"Created pipeline file: {full_path}")

    def validate_pipeline(self, pipeline_config: PipelineConfig) -> Dict[str, Any]:
        """Validate pipeline configuration."""
        issues = []

        # Check required fields
        if not pipeline_config.name:
            issues.append("Pipeline name is required")

        if not pipeline_config.environments:
            issues.append("At least one environment must be specified")

        if not pipeline_config.stages:
            issues.append("At least one pipeline stage must be specified")

        # Check stage dependencies
        required_stages = [PipelineStage.BUILD, PipelineStage.TEST]
        for stage in required_stages:
            if stage not in pipeline_config.stages:
                issues.append(f"Required stage '{stage.value}' is missing")

        # Check timeout
        if pipeline_config.timeout < 300:
            issues.append("Pipeline timeout should be at least 5 minutes")

        return {
            "valid": len(issues) == 0,
            "issues": issues,
            "recommendations": self._generate_pipeline_recommendations(pipeline_config)
        }

    def _generate_pipeline_recommendations(self, config: PipelineConfig) -> List[str]:
        """Generate pipeline improvement recommendations."""
        recommendations = []

        if PipelineStage.SECURITY_SCAN not in config.stages:
            recommendations.append("Consider adding security scanning stage")

        if len(config.environments) < 2:
            recommendations.append("Consider adding staging environment for better testing")

        if config.retries < 1:
            recommendations.append("Consider enabling retries for failed jobs")

        return recommendations

    def get_pipeline_status(self) -> Dict[str, Any]:
        """Get overall pipeline status and metrics."""
        return {
            "active_pipelines": len(self.pipelines),
            "total_workflows": 4,  # CI, Docker, Security, Deploy
            "infrastructure_components": {
                "docker": True,
                "kubernetes": True,
                "terraform": True
            },
            "supported_providers": ["GitHub Actions", "AWS", "GCP", "Docker", "Kubernetes"]
        }


# Export main classes
__all__ = [
    "Environment",
    "DeploymentStrategy",
    "PipelineStage",
    "PipelineConfig",
    "DeploymentConfig",
    "GitHubActionsGenerator",
    "DockerManager",
    "KubernetesManager",
    "TerraformManager",
    "PipelineOrchestrator"
]
