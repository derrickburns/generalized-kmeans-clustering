# Security Policy

## Supported Versions

We release patches for security vulnerabilities. Currently supported versions:

| Version | Supported          |
| ------- | ------------------ |
| 0.6.x   | :white_check_mark: |
| < 0.6   | :x:                |

## Reporting a Vulnerability

We take the security of generalized-kmeans-clustering seriously. If you believe you have found a security vulnerability, please report it to us as described below.

**Please do not report security vulnerabilities through public GitHub issues.**

Instead, please report them via email to the maintainer at the email address listed in the project's Maven POM file or GitHub profile.

You should receive a response within 48 hours. If for some reason you do not, please follow up via email to ensure we received your original message.

Please include the following information in your report:

* Type of issue (e.g. buffer overflow, SQL injection, cross-site scripting, etc.)
* Full paths of source file(s) related to the manifestation of the issue
* The location of the affected source code (tag/branch/commit or direct URL)
* Any special configuration required to reproduce the issue
* Step-by-step instructions to reproduce the issue
* Proof-of-concept or exploit code (if possible)
* Impact of the issue, including how an attacker might exploit it

This information will help us triage your report more quickly.

## Preferred Languages

We prefer all communications to be in English.

## Policy

We follow the principle of [Coordinated Vulnerability Disclosure](https://en.wikipedia.org/wiki/Coordinated_vulnerability_disclosure).

## Security Best Practices

When using this library:

1. **Keep Dependencies Updated**: Regularly update to the latest version to receive security patches
2. **Validate Input Data**: Always validate and sanitize input data before clustering
3. **Resource Limits**: Set appropriate limits for cluster count (k) and iterations to prevent resource exhaustion
4. **Access Control**: Ensure appropriate access controls are in place for model files and training data
5. **Dependency Scanning**: Use tools like OWASP Dependency-Check or Snyk to scan for vulnerable dependencies

## Security Updates

Security updates will be released as patch versions and announced via:
- GitHub Security Advisories
- Release notes in CHANGELOG.md
- GitHub Releases page

## Attribution

We will acknowledge security researchers who responsibly disclose vulnerabilities in our release notes, unless they prefer to remain anonymous.
