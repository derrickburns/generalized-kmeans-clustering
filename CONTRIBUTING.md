# Contributing to Generalized K-Means Clustering

Thank you for your interest in contributing! This document provides guidelines for contributing to this project.

## Table of Contents

- [Code of Conduct](#code-of-conduct)
- [Getting Started](#getting-started)
- [Development Environment](#development-environment)
- [Making Changes](#making-changes)
- [Testing](#testing)
- [Code Style](#code-style)
- [Submitting Changes](#submitting-changes)

## Code of Conduct

This project adheres to a code of conduct adapted from the [Contributor Covenant](https://www.contributor-covenant.org/).

### Our Standards

- **Be respectful**: Value each other's ideas, styles, and viewpoints
- **Be constructive**: Provide helpful feedback and be open to receiving it
- **Be collaborative**: Work together toward the best outcome for the project
- **Be inclusive**: Welcome newcomers and help them get oriented

## Getting Started

1. **Fork the repository** on GitHub
2. **Clone your fork**:
   \`\`\`bash
   git clone https://github.com/YOUR-USERNAME/generalized-kmeans-clustering.git
   cd generalized-kmeans-clustering
   \`\`\`
3. **Add upstream remote**:
   \`\`\`bash
   git remote add upstream https://github.com/derrickburns/generalized-kmeans-clustering.git
   \`\`\`
4. **Create a branch**:
   \`\`\`bash
   git checkout -b feature/your-feature-name
   \`\`\`

## Development Environment

### Requirements

- **Java**: JDK 17 (required; Spark/Hadoop will fail on newer JDKs)
- **Scala**: 2.12.18 or 2.13.14 (managed by sbt)
- **SBT**: 1.9.x or later
- **Spark**: 3.4.0+ or 3.5.1+ (managed by sbt)

### Setup

\`\`\`bash
# Compile
sbt compile

# Run tests
sbt test

# Run style checks
sbt scalastyle
\`\`\`

## Making Changes

### Branch Naming

- `feature/` - New features
- `fix/` - Bug fixes
- `docs/` - Documentation
- `refactor/` - Refactoring
- `test/` - Tests

### Commit Messages

Follow conventional commits:

\`\`\`
<type>(<scope>): <subject>
\`\`\`

Types: `feat`, `fix`, `docs`, `style`, `refactor`, `perf`, `test`, `build`, `ci`, `chore`

## Testing

\`\`\`bash
# Run all tests
sbt test

# Run specific suite
sbt "testOnly *KMeansSuite"

# Run for specific Scala version
sbt ++2.13.14 test

# With coverage
sbt clean coverage test coverageReport
\`\`\`

### Test Requirements

- Aim for >95% coverage
- Test happy paths and edge cases
- Include performance regression tests for critical paths

## Code Style

### Scalastyle

\`\`\`bash
sbt scalastyle
sbt test:scalastyle
\`\`\`

### Guidelines

- **Indentation**: 2 spaces
- **Line length**: Max 120 characters
- **Naming**: PascalCase for classes, camelCase for methods/vars
- **Documentation**: All public APIs must have Scaladoc

### Before Submitting

- [ ] Code compiles without warnings
- [ ] All tests pass
- [ ] Scalastyle passes
- [ ] Public APIs have Scaladoc
- [ ] Commit messages follow conventions

## Submitting Changes

1. **Update branch**:
   \`\`\`bash
   git fetch upstream
   git rebase upstream/master
   \`\`\`

2. **Push**:
   \`\`\`bash
   git push origin feature/your-feature-name
   \`\`\`

3. **Create Pull Request**:
   - Descriptive title
   - Reference issues (`Closes #123`)
   - Describe changes
   - Fill out PR template

4. **Respond to feedback** and iterate

## Getting Help

- **Questions**: [Discussions](https://github.com/derrickburns/generalized-kmeans-clustering/discussions)
- **Bugs**: [Issues](https://github.com/derrickburns/generalized-kmeans-clustering/issues)
- **Docs**: Check [README](README.md)

Thank you for contributing! 🎉
