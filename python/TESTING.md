# Testing the PySpark Wrapper

This document describes how to test the PySpark wrapper for the generalized-kmeans-clustering library.

## Prerequisites

The PySpark wrapper requires:
- Python 3.7 or later
- PySpark 3.4.0 or later
- NumPy 1.20.0 or later

## Installation for Testing

### Option 1: Install from Source

```bash
# From the repository root
cd python

# Install in development mode
pip install -e .

# Or install with test dependencies
pip install -e .[dev]
```

### Option 2: Install Dependencies Manually

```bash
# Install core dependencies
pip install pyspark>=3.4.0 numpy>=1.20.0

# Install test dependencies
pip install pytest>=7.0.0 pytest-cov>=3.0.0
```

## Running Tests

### Run All Tests

```bash
cd python
pytest tests/
```

### Run with Coverage

```bash
cd python
pytest --cov=massivedatascience tests/
```

### Run Specific Test

```bash
cd python
pytest tests/test_generalized_kmeans.py::GeneralizedKMeansTest::test_basic_clustering
```

### Run with Verbose Output

```bash
cd python
pytest -v tests/
```

## Test Suite Overview

The test suite (`tests/test_generalized_kmeans.py`) contains 13 comprehensive tests:

### Basic Functionality Tests
- `test_basic_clustering` - Basic clustering with Squared Euclidean distance
- `test_kl_divergence` - Clustering with KL divergence
- `test_weighted_clustering` - Weighted clustering functionality
- `test_distance_column` - Distance column output
- `test_predict_single_point` - Single point prediction
- `test_compute_cost` - Cost computation (WCSS)

### Quality and Reliability Tests
- `test_reproducibility` - Deterministic results with fixed seed
- `test_different_k_values` - Testing various k values
- `test_initialization_modes` - Different initialization strategies
- `test_assignment_strategies` - Different assignment strategies

### API Tests
- `test_parameter_getters` - Parameter getter methods
- `test_parameter_setters` - Parameter setter methods
- `test_model_persistence` - Model save/load functionality

## Verification Without PySpark Installation

If PySpark is not installed, you can still verify the package structure:

### Check Python Syntax

```bash
# Verify all Python files compile
python3 -m py_compile python/massivedatascience/__init__.py
python3 -m py_compile python/massivedatascience/clusterer/__init__.py
python3 -m py_compile python/massivedatascience/clusterer/kmeans.py
python3 -m py_compile python/tests/__init__.py
python3 -m py_compile python/tests/test_generalized_kmeans.py
python3 -m py_compile python/examples/*.py
```

### Validate setup.py

```bash
cd python
python3 setup.py check
```

### Build Source Distribution

```bash
cd python
python3 setup.py sdist
```

This creates `dist/massivedatascience-clusterer-0.6.0.tar.gz`

## Running Examples

After installation, run the example scripts:

```bash
cd python/examples

# Basic clustering
python3 basic_clustering.py

# KL divergence
python3 kl_divergence_clustering.py

# Weighted clustering
python3 weighted_clustering.py

# Finding optimal k
python3 finding_optimal_k.py

# Model persistence
python3 model_persistence.py
```

## Continuous Integration

The PySpark wrapper can be tested in CI/CD pipelines:

### GitHub Actions Example

```yaml
name: Python Tests

on: [push, pull_request]

jobs:
  test:
    runs-on: ubuntu-latest
    strategy:
      matrix:
        python-version: ['3.8', '3.9', '3.10', '3.11']
        spark-version: ['3.4.0', '3.5.1']

    steps:
    - uses: actions/checkout@v3

    - name: Set up Python
      uses: actions/setup-python@v4
      with:
        python-version: ${{ matrix.python-version }}

    - name: Install dependencies
      run: |
        pip install pyspark==${{ matrix.spark-version }}
        pip install numpy pytest pytest-cov

    - name: Install package
      run: |
        cd python
        pip install -e .

    - name: Run tests
      run: |
        cd python
        pytest --cov=massivedatascience tests/
```

## Troubleshooting

### Issue: `ModuleNotFoundError: No module named 'pyspark'`

**Solution:**
```bash
pip install pyspark>=3.4.0
```

### Issue: `ModuleNotFoundError: No module named 'numpy'`

**Solution:**
```bash
pip install numpy>=1.20.0
```

### Issue: Tests hang or take too long

**Cause:** Spark session initialization can be slow the first time.

**Solution:**
- Wait for first test to complete (may take 30-60 seconds)
- Subsequent tests will be faster
- Use `pytest -v` to see progress

### Issue: Java version errors

**Cause:** PySpark requires Java 8, 11, or 17.

**Solution:**
```bash
# Check Java version
java -version

# Install Java 11 if needed (macOS)
brew install openjdk@11

# Set JAVA_HOME
export JAVA_HOME=/usr/local/opt/openjdk@11
```

### Issue: Permission errors during model save/load tests

**Cause:** Insufficient permissions in temp directory.

**Solution:**
- Tests use Python's `tempfile.mkdtemp()` which should work on all systems
- If issues persist, check `/tmp` permissions

## Expected Test Output

All tests should pass:

```
=============================== test session starts ================================
platform darwin -- Python 3.11.0, pytest-7.4.0, pluggy-1.0.0
rootdir: /path/to/generalized-kmeans-clustering/python
plugins: cov-4.1.0
collected 13 items

tests/test_generalized_kmeans.py .............                              [100%]

================================ 13 passed in 45.23s ================================
```

## Performance Benchmarks

Expected test execution times (on a modern laptop):

- Individual test: 2-5 seconds
- Full test suite: 30-60 seconds (first run), 20-30 seconds (subsequent runs)
- With coverage: +5-10 seconds

## Validation Checklist

Before submitting changes, verify:

- [ ] All Python files compile without syntax errors
- [ ] `setup.py check` passes
- [ ] All 13 tests pass
- [ ] Test coverage > 80%
- [ ] All examples run without errors
- [ ] Documentation is up-to-date
- [ ] No PySpark deprecation warnings

## Additional Resources

- **PySpark Documentation**: https://spark.apache.org/docs/latest/api/python/
- **pytest Documentation**: https://docs.pytest.org/
- **Package Documentation**: See `python/examples/README.md`
