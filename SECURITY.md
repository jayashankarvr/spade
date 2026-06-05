# Security Policy

## Supported Versions

| Version | Supported          |
| ------- | ------------------ |
| 0.2.x   | :white_check_mark: |
| < 0.2   | :x:                |

## Reporting a Vulnerability

If you discover a security vulnerability, please report it privately:

1. **Do NOT** open a public GitHub issue
2. Email the maintainers with details of the vulnerability
3. Include steps to reproduce if possible

We aim to respond within 48 hours and will work with you to understand and address the issue.

## Security Considerations

SPADE is designed for image forensics research. When deploying:

- **API Server**: The REST API binds to `0.0.0.0` by default. In production, use a reverse proxy with authentication.
- **File Uploads**: The API accepts image uploads. Validate file types and sizes in production deployments.
- **Index Files**: Index files (`.spade`) contain metadata but no executable code. They use JSON format (not pickle) for security.

## Dependencies

We regularly update dependencies to address known vulnerabilities. Run `pip install --upgrade` periodically.
