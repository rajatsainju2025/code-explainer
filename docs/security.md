# Security Guide

The Code Explainer implements comprehensive security measures to protect against malicious inputs, prevent resource abuse, and ensure safe operation in production environments.

## 🔒 Security Features

### Input Validation & Sanitization

The system includes multiple layers of input validation:

#### Code Security Validator
- **AST-based Analysis**: Parses Python code into Abstract Syntax Trees for structural analysis
- **Pattern Detection**: Identifies potentially dangerous imports, functions, and execution patterns
- **Length Limits**: Prevents processing of extremely long code that could cause resource exhaustion
- **Configurable Strictness**: Adjustable security levels for different deployment scenarios

**Dangerous Patterns Detected:**
```python
# Blocked imports
import os, subprocess, sys, shutil, pickle, eval, exec

# Blocked functions
eval(), exec(), __import__(), open(), file()

# Blocked modules
os.system(), subprocess.call(), pickle.load()
```

#### Request Validation
- **Pydantic Models**: Type-safe request validation with automatic error responses
- **Size Limits**: Maximum request size limits to prevent DoS attacks
- **Content-Type Enforcement**: Strict JSON-only request acceptance

### Rate Limiting

#### Sliding Window Algorithm
- **Configurable Limits**: Default 60 requests per minute per client
- **Automatic Cleanup**: Expired request tracking removal
- **Client Identification**: IP-based rate limiting with configurable key strategies
- **Graceful Degradation**: Clear error messages when limits are exceeded

**Configuration:**
```bash
export CODE_EXPLAINER_RATE_LIMIT="100/minute"
```

#### Distributed Rate Limiting
For multi-instance deployments, consider:
- Redis-backed rate limiting
- Load balancer integration
- Per-user quotas

### Security Auditing & Monitoring

#### Event Logging
- **Comprehensive Audit Trail**: All security-related events are logged
- **Request Tracking**: Unique request IDs for traceability
- **Timestamp Recording**: Precise event timing
- **Context Preservation**: Full request context in audit logs

**Audit Events:**
- Rate limit violations
- Input validation failures
- Security pattern detections
- Model optimization attempts
- Batch processing operations

#### Security Monitoring
- **Real-time Alerts**: Configurable thresholds for security events
- **Metrics Export**: Prometheus-compatible security metrics
- **Dashboard Integration**: Grafana panels for security monitoring

## 🛡️ Safe Execution Environment

### Sandboxed Code Analysis
- **Static Analysis Only**: No code execution during explanation generation
- **Safe Imports**: Controlled import environment for analysis
- **Resource Limits**: Memory and CPU restrictions on analysis operations

### Model Security
- **Input Sanitization**: All inputs passed through security validation
- **Output Filtering**: Generated explanations checked for sensitive content
- **Model Access Control**: Restricted model loading and configuration

## 🚨 Threat Mitigation

### Common Attack Vectors

#### Code Injection
```python
# Blocked: Direct execution attempts
code = "__import__('os').system('rm -rf /')"
```

**Mitigation:** AST pattern matching prevents dangerous constructs.

#### Resource Exhaustion
```python
# Blocked: Extremely long inputs
code = "x = 1\n" * 100000
```

**Mitigation:** Length limits and timeout controls.

#### Prompt Injection
```python
# Blocked: Attempts to override system prompts
code = "IGNORE PREVIOUS INSTRUCTIONS\nNew prompt: ..."
```

**Mitigation:** Structured prompt engineering and input validation.

### DoS Protection

#### Rate Limiting
- Prevents API abuse and resource exhaustion
- Configurable per-endpoint limits
- Automatic ban mechanisms for persistent abusers

#### Request Size Limits
- Maximum payload sizes prevent memory exhaustion
- Streaming request handling for large inputs
- Progressive validation to fail fast

#### Timeout Controls
- Configurable operation timeouts
- Automatic cleanup of long-running requests
- Resource pool management

## 🔧 Security Configuration

### Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `CODE_EXPLAINER_RATE_LIMIT` | `60/minute` | API rate limiting |
| `CODE_EXPLAINER_MAX_CODE_LENGTH` | `10000` | Maximum code length |
| `CODE_EXPLAINER_SECURITY_STRICT` | `true` | Strict security mode |
| `CODE_EXPLAINER_AUDIT_LOG` | `security.log` | Security audit log file |

### Security Profiles

#### Development Profile
```yaml
security:
  strict_mode: false
  rate_limit: "1000/minute"
  audit_enabled: false
  max_code_length: 50000
```

#### Production Profile
```yaml
security:
  strict_mode: true
  rate_limit: "60/minute"
  audit_enabled: true
  max_code_length: 10000
  allowed_imports: ["typing", "dataclasses", "enum"]
```

## 📊 Monitoring & Alerting

### Security Metrics

#### Prometheus Metrics
```
# HELP security_validation_total Total security validations
# TYPE security_validation_total counter
security_validation_total{result="safe"} 15432
security_validation_total{result="unsafe"} 123

# HELP rate_limit_exceeded_total Rate limit violations
# TYPE rate_limit_exceeded_total counter
rate_limit_exceeded_total 45

# HELP security_warnings_total Security warnings issued
# TYPE security_warnings_total counter
security_warnings_total{warning_type="dangerous_import"} 23
```

### Alert Conditions
- Rate limit violation spikes
- High unsafe code detection rates
- Security validation failures
- Audit log anomalies

### Log Analysis
```bash
# Monitor security events
tail -f security.log | grep "WARNING\|ERROR"

# Count security violations by type
grep "security_event" security.log | jq -r '.event_type' | sort | uniq -c
```

## 🔍 Security Testing

### Penetration Testing
```bash
# Test rate limiting
for i in {1..70}; do
  curl -s "http://localhost:8000/api/v2/health" &
done

# Test input validation
curl -X POST "http://localhost:8000/api/v2/validate-security" \
  -d '{"code": "import os; os.system(\"rm -rf /\")"}'
```

### Security Test Suite
```python
# Run security tests
pytest tests/test_security.py -v

# Test with malicious inputs
python -m pytest tests/test_security.py::test_malicious_inputs
```

### Fuzz Testing
- Random input generation
- Boundary condition testing
- Unicode and encoding edge cases

## 🚀 Production Deployment

### Security Checklist

- [ ] Rate limiting configured appropriately
- [ ] Input validation enabled
- [ ] Audit logging active
- [ ] Security headers configured
- [ ] HTTPS/TLS enabled
- [ ] Regular security updates
- [ ] Monitoring and alerting active
- [ ] Backup and recovery procedures

### Container Security
```dockerfile
# Security-hardened container
FROM python:3.11-slim

# Non-root user
RUN useradd --create-home --shell /bin/bash app
USER app

# Minimal dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Read-only filesystem where possible
VOLUME ["/tmp", "/var/log"]
```

### Network Security
- API gateway integration
- Web Application Firewall (WAF)
- DDoS protection
- SSL/TLS termination

## 📞 Incident Response

### Security Incident Procedure
1. **Detection**: Monitor alerts and logs
2. **Assessment**: Evaluate impact and scope
3. **Containment**: Disable affected endpoints
4. **Recovery**: Restore from clean backups
5. **Analysis**: Post-mortem and lessons learned
6. **Prevention**: Update security measures

### Contact Information
- **Security Issues**: security@code-explainer.dev
- **Emergency**: +1-555-0123 (24/7)
- **PGP Key**: Available at security.code-explainer.dev/pgp

## 📚 Additional Resources

- [OWASP API Security Top 10](https://owasp.org/www-project-api-security/)
- [FastAPI Security](https://fastapi.tiangolo.com/tutorial/security/)
- [Python Security Best Practices](https://python-security.readthedocs.io/)
- [NIST Cybersecurity Framework](https://www.nist.gov/cyberframework)

## 🔄 Security Updates

The security features are continuously updated. Subscribe to security advisories:

- [Security Changelog](CHANGELOG.md)
- [GitHub Security Advisories](https://github.com/rajatsainju2025/code-explainer/security/advisories)
- [Security Mailing List](https://groups.google.com/g/code-explainer-security)