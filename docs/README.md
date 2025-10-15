# API Documentation

This directory contains the generated Scaladoc API reference for the generalized-kmeans-clustering library.

## Viewing the Documentation

The API documentation is automatically generated and published to GitHub Pages:

**https://derrickburns.github.io/generalized-kmeans-clustering/**

## Generating Locally

To generate the Scaladoc locally:

```bash
sbt unidoc
```

This will create the API documentation in `docs/api/`.

To view it locally, open `docs/api/index.html` in your browser.

## What's Documented

The API reference includes complete documentation for all public classes, traits, and methods in the library:

### Core Packages

- **`com.massivedatascience.clusterer`** - RDD-based clustering algorithms
  - K-Means variants (standard, mini-batch, streaming, etc.)
  - Advanced algorithms (Bisecting, X-Means, Soft K-Means)
  - Constrained and Annealed K-Means
  - Coreset approximation

- **`com.massivedatascience.clusterer.ml`** - DataFrame API
  - `GeneralizedKMeans` - Main clustering estimator
  - `GeneralizedKMeansModel` - Trained model
  - ML Pipeline integration

- **`com.massivedatascience.clusterer.ml.df`** - DataFrame implementation details
  - `BregmanKernel` - Divergence functions (Euclidean, KL, Itakura-Saito, L1, etc.)
  - `LloydsIterator` - Core Lloyd's algorithm
  - Strategy classes (Assignment, Update, EmptyCluster, Convergence)

- **`com.massivedatascience.divergence`** - Bregman divergences
  - Distance functions and their mathematical properties
  - Gradient computations

- **`com.massivedatascience.linalg`** - Linear algebra utilities
  - Vector operations
  - Weighted vectors

- **`com.massivedatascience.transforms`** - Data transformations
  - Embeddings (Haar, Random Indexing)
  - Dimensionality reduction

## Documentation Features

The Scaladoc includes:

- **Type signatures** for all methods
- **Parameter descriptions** with validation constraints
- **Return value documentation**
- **Usage examples** in code comments
- **Cross-references** between related classes
- **Inheritance diagrams** (when Graphviz is available)
- **Grouping** of related functionality

## Contributing to Documentation

To improve the documentation:

1. Add or enhance Scaladoc comments in source files
2. Use standard Scaladoc tags:
   - `@param` - Parameter descriptions
   - `@return` - Return value description
   - `@throws` - Exceptions that may be thrown
   - `@example` - Usage examples
   - `@see` - References to related classes/methods
   - `@note` - Important notes
   - `@since` - Version when feature was added

3. Regenerate docs: `sbt unidoc`
4. Verify locally before committing

## Build Configuration

The documentation is configured in `build.sbt`:

```scala
enablePlugins(ScalaUnidocPlugin)

Compile / doc / scalacOptions ++= Seq(
  "-doc-title", "Generalized K-Means Clustering API",
  "-doc-version", version.value,
  "-groups",
  "-implicits",
  "-diagrams"
)
```

## CI/CD

Documentation is automatically generated and deployed on every push to `master` via GitHub Actions (`.github/workflows/docs.yml`).

The workflow:
1. Checks out the code
2. Sets up Java 17
3. Installs Graphviz for diagrams
4. Runs `sbt unidoc`
5. Deploys to GitHub Pages

## Troubleshooting

### Diagrams not showing

Diagrams require Graphviz to be installed:

```bash
# macOS
brew install graphviz

# Ubuntu/Debian
sudo apt-get install graphviz

# Windows
choco install graphviz
```

### Build fails with "unable to expand" errors

This typically indicates Scaladoc syntax errors in comments. Check:
- Unmatched braces in code examples
- Invalid `@` tag references
- Malformed HTML in Scaladoc

### Outdated documentation

Clear the docs and regenerate:

```bash
rm -rf docs/api/
sbt clean unidoc
```

## Related Documentation

- **[Architecture Guide](../ARCHITECTURE.md)** - Technical deep-dive
- **[Usage Examples](../DATAFRAME_API_EXAMPLES.md)** - Code examples
- **[Performance Tuning](../PERFORMANCE_TUNING.md)** - Optimization guide
- **[Migration Guide](../MIGRATION_GUIDE.md)** - RDD → DataFrame migration

---

*Generated documentation for generalized-kmeans-clustering*
*Copyright © 2025 Massive Data Science, LLC*
*Licensed under Apache License 2.0*
