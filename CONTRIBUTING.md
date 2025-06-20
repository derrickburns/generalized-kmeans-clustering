# Contributing to Generalized K-Means Clustering

Thank you for your interest in contributing to this project! This document provides guidelines for contributing to the generalized K-means clustering library.

## Development Environment Setup

### Prerequisites

- **Java 17** or higher
- **SBT 1.x** (Scala Build Tool)
- **Scala 2.12.18** (managed by SBT)
- **Apache Spark 3.4.0** (managed by SBT)

### Getting Started

1. **Clone the repository:**
   ```bash
   git clone https://github.com/derrickburns/generalized-kmeans-clustering.git
   cd generalized-kmeans-clustering
   ```

2. **Compile the project:**
   ```bash
   sbt compile
   ```

3. **Run tests:**
   ```bash
   sbt test
   ```

4. **Check code style:**
   ```bash
   sbt scalastyle
   ```

## Code Style Guidelines

### Scala Style

- Follow standard Scala naming conventions
- Use 2-space indentation
- Line length should not exceed 120 characters
- Use meaningful variable and function names
- Add scaladoc documentation for all public APIs

### Code Quality

- **Linting:** Run `sbt scalastyle` before submitting
- **Testing:** Ensure all tests pass with `sbt test`
- **Coverage:** Maintain or improve test coverage
- **Dependencies:** Check for dependency updates with `sbt dependencyUpdates`

### Error Handling

- Use `ValidationUtils` for common validation patterns
- Provide meaningful error messages with context
- Handle edge cases gracefully
- Use SLF4J logging instead of print statements

## Project Structure

```
src/
├── main/scala/com/massivedatascience/
│   ├── clusterer/          # Core clustering algorithms
│   ├── divergence/         # Bregman divergence implementations  
│   ├── linalg/            # Linear algebra utilities
│   ├── transforms/        # Data transformation utilities
│   └── util/              # Common utilities and validation
└── test/scala/com/massivedatascience/
    └── clusterer/         # Test suites
```

## Architecture Overview

### Core Components

- **BregmanDivergence**: Defines distance functions for clustering
- **BregmanPointOps**: Point operations and factory methods
- **KMeansModel**: Trained model with prediction capabilities
- **MultiKMeansClusterer**: Interface for different clustering implementations

### Key Design Patterns

- **Weighted Vectors**: All operations use `WeightedVector` for weighted clustering
- **Pluggable Distance Functions**: Easy addition of new Bregman divergences
- **Iterative Training**: Multi-stage training support

## Testing

### Test Requirements

- All new features must include comprehensive tests
- Tests should cover edge cases and error conditions
- Use ScalaTest framework with the existing `LocalClusterSparkContext` trait
- Test files should be in `src/test/scala/com/massivedatascience/clusterer/`

### Running Tests

```bash
# Run all tests
sbt test

# Run specific test suite
sbt "testOnly *KMeansSuite"

# Run with coverage
sbt coverage test coverageReport
```

## Pull Request Process

### Before Submitting

1. **Ensure all tests pass:**
   ```bash
   sbt test
   ```

2. **Check code style:**
   ```bash
   sbt scalastyle
   ```

3. **Update documentation** if you've made API changes

4. **Add tests** for new functionality

### Pull Request Guidelines

- **Title**: Use descriptive titles (e.g., "Add validation for negative weights in BregmanPointOps")
- **Description**: Clearly explain what changes you made and why
- **Testing**: Describe how you tested your changes
- **Breaking Changes**: Clearly mark any breaking changes

### Commit Message Format

Use conventional commit messages:

```
type(scope): brief description

Longer description if needed

- List specific changes
- Include reasoning for complex changes
```

**Types:**
- `feat`: New feature
- `fix`: Bug fix
- `docs`: Documentation changes
- `style`: Code style changes
- `refactor`: Code refactoring
- `test`: Adding or updating tests
- `perf`: Performance improvements

## Common Development Tasks

### Adding a New Bregman Divergence

1. Create a new trait or object extending `BregmanDivergence`
2. Implement required methods: `convex`, `convexHomogeneous`, `gradientOfConvex`, `gradientOfConvexHomogeneous`
3. Use `ValidationUtils` for input validation
4. Add comprehensive tests in the test suite
5. Update documentation

### Improving Performance

1. Profile your changes using appropriate tools
2. Add benchmarks if introducing performance-critical code
3. Consider memory usage and garbage collection impact
4. Test with realistic data sizes

### Adding Configuration Options

1. Add new options to `KMeansConfig` if applicable
2. Ensure backward compatibility
3. Add validation for new configuration values
4. Document the new options

## Code Review Criteria

### Code Quality
- [ ] Code follows project style guidelines
- [ ] Error handling is appropriate and consistent
- [ ] No code duplication
- [ ] Performance considerations addressed

### Testing
- [ ] Adequate test coverage
- [ ] Tests cover edge cases
- [ ] Tests are maintainable and readable

### Documentation
- [ ] Public APIs are documented
- [ ] Complex algorithms are explained
- [ ] Breaking changes are clearly marked

## Getting Help

- **Issues**: Check existing [GitHub issues](https://github.com/derrickburns/generalized-kmeans-clustering/issues)
- **Discussions**: Start a discussion for questions about implementation
- **Code Review**: Request review from maintainers

## License

By contributing to this project, you agree that your contributions will be licensed under the Apache License 2.0.

## Recognition

Contributors will be acknowledged in release notes and the project README.

Thank you for contributing to the generalized K-means clustering library!