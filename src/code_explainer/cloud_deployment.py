"""Cloud-native deployment configurations."""

import os
import yaml
from typing import Dict, List, Any, Optional
from pathlib import Path

class CloudDeployment:
    """Cloud-native deployment configurations."""
    
    def __init__(self, config_path: Optional[str] = None):
        self.config_path = config_path or "deployments"
        self.templates = self._load_templates()
    
    def _load_templates(self) -> Dict[str, Dict[str, Any]]:
        """Load deployment templates."""
        return {
            "kubernetes": {
                "apiVersion": "apps/v1",
                "kind": "Deployment",
                "metadata": {"name": "code-explainer"},
                "spec": {
                    "replicas": 3,
                    "selector": {"matchLabels": {"app": "code-explainer"}},
                    "template": {
                        "metadata": {"labels": {"app": "code-explainer"}},
                        "spec": {
                            "containers": [{
                                "name": "code-explainer",
                                "image": "code-explainer:latest",
                                "ports": [{"containerPort": 8000}],
                                "env": [
                                    {"name": "REDIS_URL", "value": "redis-service:6379"},
                                    {"name": "DB_URL", "value": "postgres://..."}
                                ],
                                "resources": {
                                    "requests": {"memory": "512Mi", "cpu": "500m"},
                                    "limits": {"memory": "1Gi", "cpu": "1000m"}
                                }
                            }]
                        }
                    }
                }
            },
            "docker_compose": {
                "version": "3.8",
                "services": {
                    "api": {
                        "build": ".",
                        "ports": ["8000:8000"],
                        "environment": ["REDIS_URL=redis:6379"],
                        "depends_on": ["redis", "db"]
                    },
                    "redis": {
                        "image": "redis:alpine",
                        "ports": ["6379:6379"]
                    },
                    "db": {
                        "image": "postgres:13",
                        "environment": {
                            "POSTGRES_DB": "code_explainer",
                            "POSTGRES_USER": "user",
                            "POSTGRES_PASSWORD": "password"
                        }
                    }
                }
            },
            "serverless": {
                "functions": {
                    "explain_code": {
                        "runtime": "python3.9",
                        "handler": "lambda_function.lambda_handler",
                        "memory": 1024,
                        "timeout": 30,
                        "environment": {
                            "MODEL_PATH": "s3://models/code-explainer-model"
                        }
                    }
                },
                "api_gateway": {
                    "name": "code-explainer-api",
                    "endpoints": [
                        {"path": "/explain", "method": "POST", "function": "explain_code"}
                    ]
                }
            }
        }
    
    def generate_kubernetes_manifests(self, output_dir: str = "k8s"):
        """Generate Kubernetes manifests."""
        output_path = Path(output_dir)
        output_path.mkdir(exist_ok=True)
        
        # Deployment
        deployment = self.templates["kubernetes"]
        with open(output_path / "deployment.yaml", 'w') as f:
            yaml.dump(deployment, f)
        
        # Service
        service = {
            "apiVersion": "v1",
            "kind": "Service",
            "metadata": {"name": "code-explainer-service"},
            "spec": {
                "selector": {"app": "code-explainer"},
                "ports": [{"port": 8000, "targetPort": 8000}],
                "type": "LoadBalancer"
            }
        }
        with open(output_path / "service.yaml", 'w') as f:
            yaml.dump(service, f)
        
        # ConfigMap
        configmap = {
            "apiVersion": "v1",
            "kind": "ConfigMap",
            "metadata": {"name": "code-explainer-config"},
            "data": {
                "config.yaml": yaml.dump({
                    "model": {"path": "/models"},
                    "cache": {"type": "redis", "ttl": 3600}
                })
            }
        }
        with open(output_path / "configmap.yaml", 'w') as f:
            yaml.dump(configmap, f)
    
    def generate_docker_compose(self, output_file: str = "docker-compose.yml"):
        """Generate Docker Compose file."""
        compose = self.templates["docker_compose"]
        with open(output_file, 'w') as f:
            yaml.dump(compose, f)
    
    def generate_serverless_config(self, output_file: str = "serverless.yml"):
        """Generate Serverless Framework config."""
        serverless = self.templates["serverless"]
        with open(output_file, 'w') as f:
            yaml.dump(serverless, f)
    
    def generate_terraform_config(self, output_dir: str = "terraform"):
        """Generate Terraform configurations."""
        output_path = Path(output_dir)
        output_path.mkdir(exist_ok=True)
        
        # Main config
        main_tf = '''
terraform {
  required_providers {
    aws = {
      source  = "hashicorp/aws"
      version = "~> 4.0"
    }
  }
}

provider "aws" {
  region = var.region
}

resource "aws_ecs_cluster" "code_explainer" {
  name = "code-explainer-cluster"
}

resource "aws_ecs_task_definition" "code_explainer" {
  family                   = "code-explainer"
  network_mode             = "awsvpc"
  requires_compatibilities = ["FARGATE"]
  cpu                      = "512"
  memory                   = "1024"
  
  container_definitions = jsonencode([{
    name  = "code-explainer"
    image = var.container_image
    portMappings = [{
      containerPort = 8000
      hostPort      = 8000
    }]
    environment = [
      { name = "MODEL_PATH", value = var.model_path }
    ]
  }])
}

resource "aws_ecs_service" "code_explainer" {
  name            = "code-explainer-service"
  cluster         = aws_ecs_cluster.code_explainer.id
  task_definition = aws_ecs_task_definition.code_explainer.arn
  desired_count   = var.desired_count
  
  network_configuration {
    subnets         = var.subnet_ids
    security_groups = [aws_security_group.code_explainer.id]
  }
  
  load_balancer {
    target_group_arn = aws_lb_target_group.code_explainer.arn
    container_name   = "code-explainer"
    container_port   = 8000
  }
}

resource "aws_security_group" "code_explainer" {
  name_prefix = "code-explainer-"
  
  ingress {
    from_port   = 8000
    to_port     = 8000
    protocol    = "tcp"
    cidr_blocks = ["0.0.0.0/0"]
  }
}

resource "aws_lb" "code_explainer" {
  name               = "code-explainer-lb"
  internal           = false
  load_balancer_type = "application"
  subnets            = var.subnet_ids
}

resource "aws_lb_target_group" "code_explainer" {
  name_prefix = "ce-"
  port        = 8000
  protocol    = "HTTP"
  vpc_id      = var.vpc_id
  
  health_check {
    path = "/health"
  }
}

resource "aws_lb_listener" "code_explainer" {
  load_balancer_arn = aws_lb.code_explainer.arn
  port              = "80"
  protocol          = "HTTP"
  
  default_action {
    type             = "forward"
    target_group_arn = aws_lb_target_group.code_explainer.arn
  }
}
'''
        
        with open(output_path / "main.tf", 'w') as f:
            f.write(main_tf)
        
        # Variables
        variables_tf = '''
variable "region" {
  description = "AWS region"
  type        = string
  default     = "us-east-1"
}

variable "container_image" {
  description = "Container image URI"
  type        = string
}

variable "model_path" {
  description = "Path to model artifacts"
  type        = string
}

variable "desired_count" {
  description = "Desired number of tasks"
  type        = number
  default     = 2
}

variable "vpc_id" {
  description = "VPC ID"
  type        = string
}

variable "subnet_ids" {
  description = "Subnet IDs"
  type        = list(string)
}
'''
        
        with open(output_path / "variables.tf", 'w') as f:
            f.write(variables_tf)

# Example usage
def generate_all_deployments():
    """Generate all deployment configurations."""
    deployment = CloudDeployment()
    
    # Kubernetes
    deployment.generate_kubernetes_manifests()
    
    # Docker Compose
    deployment.generate_docker_compose()
    
    # Serverless
    deployment.generate_serverless_config()
    
    # Terraform
    deployment.generate_terraform_config()
    
    print("All deployment configurations generated!")

if __name__ == "__main__":
    generate_all_deployments()
