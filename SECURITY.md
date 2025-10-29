# Security Policy

## Supported Versions

We release patches for security vulnerabilities in the following versions:

| Version | Supported          |
| ------- | ------------------ |
| 0.2.x   | :white_check_mark: |
| 0.1.x   | :white_check_mark: |
| < 0.1   | :x:                |

Python Support: 3.9, 3.10, 3.11, 3.12

## Reporting a Vulnerability

We take the security of Code Explainer seriously. If you believe you have found a security vulnerability, please report it to us as described below.

### Where to Report

**Please do NOT report security vulnerabilities through public GitHub issues.**

Instead, please report them via:
- Email: security@code-explainer.dev (or your.email@example.com)
- GitHub Security Advisory: https://github.com/rajatsainju2025/code-explainer/security/advisories/new

### What to Include

Please include the following information in your report:
- Type of issue (e.g., injection, XSS, authentication bypass, etc.)
- Full paths of source file(s) related to the manifestation of the issue
- The location of the affected source code (tag/branch/commit or direct URL)
- Any special configuration required to reproduce the issue
- Step-by-step instructions to reproduce the issue
- Proof-of-concept or exploit code (if possible)
- Impact of the issue, including how an attacker might exploit it

### What to Expect

After you submit a report, we will:
1. **Acknowledge receipt** within 48 hours
2. **Provide an assessment** of the vulnerability within 5 business days
3. **Work on a fix** and keep you updated on progress
4. **Coordinate disclosure** with you once a fix is ready
5. **Credit you** in the security advisory (if desired)

## Security Best Practices

### For Users

1. **API Keys**: Always use strong, randomly generated API keys
   ```bash
   # Generate secure API key
   python -c "import secrets; print(secrets.token_urlsafe(32))"
   ```

2. **Environment Variables**: Never commit `.env` files with real credentials
   ```bash
   # Add to .gitignore
   echo ".env" >> .gitignore
   ```

3. **HTTPS**: Use HTTPS in production environments

4. **Input Validation**: The API validates input, but sanitize on your side too

5. **Rate Limiting**: Implement rate limiting in production deployments

### For Developers

1. **Dependencies**: Keep dependencies updated
   ```bash
   pip install --upgrade -r requirements.txt
   ```

2. **Code Review**: All security-related changes require review

3. **Testing**: Run security tests before commits
   ```bash
   make test
   bandit -r src/ || true
   ```

4. **Secrets**: Use environment variables, never hardcode secrets
   ```python
   # Good ✅
   api_key = os.environ.get("API_KEY")
   
   # Bad ❌
   api_key = "sk-1234567890abcdef"
   ```

## Known Security Features

### Implemented

- ✅ API key authentication with SHA-256 hashing
- ✅ Constant-time comparison to prevent timing attacks
- ✅ Request rate limiting support
- ✅ Input validation and sanitization
- ✅ Error handling that doesn't leak sensitive information
- ✅ CORS configuration
- ✅ Environment variable configuration
- ✅ Secure middleware stack

### Roadmap

- 🔄 OAuth2 authentication support
- 🔄 JWT token-based authentication
- 🔄 IP whitelisting
- 🔄 Audit logging
- 🔄 Automated security scanning in CI/CD

## Security Updates

Security updates are released as soon as possible after a vulnerability is confirmed. We recommend:

1. **Subscribe** to GitHub Security Advisories for this repository
2. **Enable** Dependabot alerts
3. **Monitor** the CHANGELOG.md for security-related updates
4. **Update** promptly when security patches are released

## Responsible Disclosure

We practice responsible disclosure:
- We will work with security researchers to verify and address vulnerabilities
- We will credit researchers who report vulnerabilities (if desired)
- We will coordinate public disclosure timing with reporters
- We aim to disclose within 90 days of receiving a report

## Security Hall of Fame

We recognize security researchers who help keep Code Explainer safe:

<!-- Add contributors here -->
- *Be the first to report a security issue!*

## Contact

For security-related questions or concerns:
- Email: security@code-explainer.dev
- GitHub: https://github.com/rajatsainju2025/code-explainer/security

---

*Last updated: 2024*
