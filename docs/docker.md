# Docker Development Environment

This guide covers using Docker for development, testing, and deployment of the Code Explainer application.

## Quick Start

### Development Setup

```bash
# Clone the repository
git clone https://github.com/rsainju/code-explainer.git
cd code-explainer

# Start development environment
make docker-compose-dev

# Or manually:
docker-compose up -d api web docs
```

### Access Services

- **API**: http://localhost:8000
- **API Docs**: http://localhost:8000/docs
- **Web UI**: http://localhost:7860
- **Documentation**: http://localhost:8002

## Docker Architecture

The Dockerfile uses multi-stage builds for optimal image sizes and development workflows:

### Stages

1. **base**: Common dependencies and system setup
2. **development**: Full development environment with hot reload
3. **production**: Optimized production image
4. **testing**: CI/testing environment with test dependencies

### Images

- `code-explainer:dev`: Development with all tools and hot reload
- `code-explainer:prod`: Production-optimized image
- `code-explainer:latest`: Default development image

## Development Workflow

### Using Docker Compose

```bash
# Start all development services
docker-compose --profile docs up -d api web docs

# View logs
docker-compose logs -f api

# Run tests
docker-compose --profile testing run --rm test

# Stop services
docker-compose down
```

### Using Makefile Targets

```bash
# Build development image
make docker-build-dev

# Run development container
make docker-dev

# Run tests in container
make docker-compose-test

# Clean up
make docker-clean
```

### Hot Reload

The development container automatically reloads on code changes:

```bash
# Mount your source code
docker run -v $(pwd):/app -p 8000:8000 code-explainer:dev
```

## Configuration

### Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `DEV` | `false` | Enable development mode |
| `CODE_EXPLAINER_MODEL_PATH` | `./results` | Path to model directory |
| `CODE_EXPLAINER_CONFIG_PATH` | `configs/default.yaml` | Configuration file path |
| `CODE_EXPLAINER_RETRIEVAL_WARMUP` | `false` | Warm up retrieval on startup |
| `CODE_EXPLAINER_RATE_LIMIT` | `60/minute` | API rate limiting |

### Volumes

Mount these directories for development:

```yaml
volumes:
  - .:/app                    # Source code
  - ./models:/app/models      # Model files
  - ./data:/app/data          # Data files
  - ./results:/app/results    # Results directory
```

## Production Deployment

### Building Production Image

```bash
# Build optimized production image
make docker-build-prod

# Or manually
docker build --target production -t code-explainer:prod .
```

### Running in Production

```bash
# Run with proper configuration
docker run -d \
  --name code-explainer-prod \
  -p 8000:8000 \
  -e CODE_EXPLAINER_MODEL_PATH=/app/models \
  -v /path/to/models:/app/models:ro \
  code-explainer:prod
```

### Docker Compose Production

```yaml
version: '3.8'
services:
  api:
    image: code-explainer:prod
    ports:
      - "8000:8000"
    environment:
      - CODE_EXPLAINER_MODEL_PATH=/app/models
    volumes:
      - ./models:/app/models:ro
    restart: unless-stopped
```

## Testing

### Running Tests in Docker

```bash
# Using docker-compose
docker-compose --profile testing run --rm test

# Using Makefile
make docker-compose-test

# Manual test run
docker run --rm -v $(pwd):/app code-explainer:testing pytest tests/
```

### CI/CD Integration

The testing stage is optimized for CI pipelines:

```yaml
- name: Test
  run: |
    docker build --target testing -t test-image .
    docker run --rm test-image pytest --cov-report=xml
```

## Debugging

### Accessing Container Shell

```bash
# Development container
docker run -it --rm -v $(pwd):/app code-explainer:dev bash

# Running container
docker exec -it <container_id> bash
```

### Viewing Logs

```bash
# Docker Compose
docker-compose logs -f api

# Individual container
docker logs -f <container_id>
```

### Health Checks

```bash
# Check API health
curl http://localhost:8000/health

# Check container health
docker ps
docker inspect <container_id> | grep -A 10 "Health"
```

## Performance Optimization

### Image Size

- **Development**: ~2GB (includes all tools)
- **Production**: ~800MB (minimal runtime dependencies)
- **Base**: ~500MB (Python + system deps)

### Build Caching

Use `.dockerignore` to exclude unnecessary files:

```
.git
__pycache__
*.pyc
tests/
docs/
```

### Multi-Stage Benefits

- **Faster builds**: Dependencies cached in base stage
- **Smaller images**: Development tools not in production
- **Security**: Non-root user in production
- **Flexibility**: Different images for different use cases

## Troubleshooting

### Common Issues

#### Port Conflicts
```bash
# Check what's using ports
lsof -i :8000

# Use different ports
docker run -p 8001:8000 code-explainer:dev
```

#### Permission Issues
```bash
# Fix file permissions
sudo chown -R $USER:$USER .

# Or run as current user
docker run --user $(id -u):$(id -g) -v $(pwd):/app code-explainer:dev
```

#### Memory Issues
```bash
# Increase Docker memory limit
# Docker Desktop: Preferences > Resources > Memory

# Or limit container memory
docker run --memory=4g --memory-swap=4g code-explainer:dev
```

#### Model Loading Issues
```bash
# Ensure model volume is mounted correctly
docker run -v /path/to/models:/app/models code-explainer:prod

# Check model path environment variable
docker run -e CODE_EXPLAINER_MODEL_PATH=/app/models code-explainer:prod
```

### Logs and Debugging

```bash
# Enable debug logging
docker run -e PYTHONPATH=/app/src -e LOG_LEVEL=DEBUG code-explainer:dev

# View detailed logs
docker-compose logs --tail=100 -f api
```

## Advanced Usage

### Custom Docker Images

```dockerfile
FROM code-explainer:prod

# Add custom dependencies
RUN pip install custom-package

# Add custom configuration
COPY custom-config.yaml /app/configs/
ENV CODE_EXPLAINER_CONFIG_PATH=/app/configs/custom-config.yaml
```

### GPU Support

```bash
# Run with GPU support (requires NVIDIA Docker)
docker run --gpus all -p 8000:8000 code-explainer:dev
```

### Kubernetes Deployment

```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: code-explainer
spec:
  replicas: 3
  selector:
    matchLabels:
      app: code-explainer
  template:
    metadata:
      labels:
        app: code-explainer
    spec:
      containers:
      - name: api
        image: code-explainer:prod
        ports:
        - containerPort: 8000
        env:
        - name: CODE_EXPLAINER_MODEL_PATH
          value: "/app/models"
        volumeMounts:
        - name: model-storage
          mountPath: /app/models
      volumes:
      - name: model-storage
        persistentVolumeClaim:
          claimName: model-pvc
```

## Contributing

When contributing Docker changes:

1. Test all stages: `docker build --target development`
2. Update documentation if adding new features
3. Ensure `.dockerignore` excludes unnecessary files
4. Test with `make docker-compose-dev`
5. Update Makefile targets if needed