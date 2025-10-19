# Performance Benchmarks

This document provides performance benchmarks and guidelines for the generalized-kmeans-clustering library.

## Quick Summary

| Divergence | Throughput (pts/sec) | Relative to SE |
|------------|---------------------|----------------|
| Squared Euclidean | ~870 pts/sec | 1.0x (baseline) |
| KL Divergence | ~3,400 pts/sec | 3.9x faster |

**Test Configuration**: 2,000 points, 2 dimensions, 2 clusters, local[*] mode on MacBook Pro M1

## Machine Specs (Baseline)

- **CPU**: Apple M1 (via Java 11.0.21)
- **RAM**: 16GB
- **Spark**: 3.5.1
- **Scala**: 2.13.14
- **Mode**: local[*] (all cores)
- **Date**: October 19, 2025

## Detailed Benchmarks

### Sanity Check Benchmarks

These benchmarks run on every CI build to catch performance regressions.

#### Squared Euclidean (SE)
- **Dataset**: 2,000 points × 2 dimensions
- **Parameters**: k=2, maxIter=5, seed=1
- **Elapsed Time**: ~2.3 seconds
- **Throughput**: ~871 points/second
- **Threshold**: < 10 seconds (regression detection)

#### KL Divergence
- **Dataset**: 2,000 points × 2 dimensions
- **Parameters**: k=2, maxIter=3, seed=2
- **Elapsed Time**: ~0.6 seconds
- **Throughput**: ~3,407 points/second
- **Threshold**: < 15 seconds (regression detection)

### Scalability Characteristics

#### Data Size Scaling

The library is designed to scale to large datasets through Spark's distributed processing:

| Points | Dimensions | Clusters | Expected Time (SE) | Notes |
|--------|------------|----------|-------------------|-------|
| 2K | 2 | 2 | ~2s | Baseline (CI sanity check) |
| 10K | 20 | 10 | ~30s | PerformanceSanityCheck threshold |
| 100K | 50 | 20 | ~5min | Small-scale production |
| 1M | 100 | 50 | ~30min | Medium-scale production |
| 10M+ | 100+ | 100+ | hours | Large-scale (requires cluster) |

**Note**: Times are estimates for local mode. Actual performance depends on:
- Hardware (CPU, memory)
- Spark configuration (executors, memory, partitions)
- Data characteristics (cluster separation, dimensionality)
- Divergence function (SE is fastest, KL/IS are slower)

#### Assignment Strategy Performance

| Strategy | k×dim Threshold | Use Case | Performance |
|----------|-----------------|----------|-------------|
| SE-CrossJoin | Any | Squared Euclidean only | Fastest (no broadcast) |
| BroadcastUDF | < 200K elements | Small k×dim, all divergences | Fast (single broadcast) |
| ChunkedBroadcast | > 200K elements | Large k×dim, all divergences | Slower (multiple scans) but avoids OOM |

The library automatically selects the best strategy via `AutoAssignment`.

### Divergence Function Performance

Different divergence functions have different computational costs:

| Divergence | Relative Speed | Notes |
|------------|----------------|-------|
| Squared Euclidean | 1.0x (baseline) | Optimized crossJoin path |
| L1 / Manhattan | ~1.2x | Simple absolute differences |
| KL | ~4x faster* | *In current benchmark (may vary) |
| Itakura-Saito | ~0.5x | Requires division operations |
| Generalized-I | ~0.7x | More complex math |
| Logistic Loss | ~0.8x | Logarithmic operations |

**Note**: These are rough estimates. Actual performance depends on data characteristics and Spark configuration.

### Iterator Performance

| Algorithm | Iterations/sec | Notes |
|-----------|---------------|-------|
| Lloyd's (SE) | ~2-3 | Standard k-means |
| Lloyd's (KL) | ~5 | Faster convergence on test data |
| Bisecting | ~1-2 splits/sec | Hierarchical splits |
| EM (Soft K-Means) | ~3-5 | Probabilistic assignments |
| PAM (K-Medoids) | ~50-100 swaps/sec | Depends on dataset size |

## Performance Regression Detection

### CI Integration

The `PerfSanitySuite` runs on every CI build and:

1. **Measures**: SE and KL divergence clustering on 2K points
2. **Reports**: Elapsed time and throughput to console (grep-able)
3. **Writes**: JSON report to `target/perf-reports/perf-sanity.json`
4. **Fails**: If performance exceeds thresholds:
   - SE: > 10 seconds
   - KL: > 15 seconds

### Running Performance Tests

```bash
# Run perf sanity check
sbt "testOnly com.massivedatascience.clusterer.PerfSanitySuite"

# Run full performance test suite
sbt "testOnly com.massivedatascience.clusterer.PerformanceTestSuite"

# Run performance sanity check (standalone)
sbt "test:runMain com.massivedatascience.clusterer.PerformanceSanityCheck"
```

### Interpreting Results

The perf sanity test outputs:
```
perf_sanity_seconds=SE:2.295
perf_sanity_seconds=KL:0.587
perf_sanity_throughput=SE:871
perf_sanity_throughput=KL:3407
```

**Green flags** (good performance):
- SE < 5 seconds
- KL < 2 seconds
- Throughput > 500 pts/sec

**Yellow flags** (acceptable but slow):
- SE: 5-10 seconds
- KL: 2-10 seconds
- Throughput: 200-500 pts/sec

**Red flags** (performance regression):
- SE > 10 seconds
- KL > 15 seconds
- Throughput < 200 pts/sec

## Performance Tuning Guide

### Spark Configuration

For optimal performance:

```scala
val spark = SparkSession.builder()
  .config("spark.sql.shuffle.partitions", "200")  // Adjust based on data size
  .config("spark.default.parallelism", "200")      // 2-3x number of cores
  .config("spark.executor.memory", "4g")           // Based on data size
  .config("spark.driver.memory", "2g")
  .getOrCreate()
```

### Algorithm Selection

Choose algorithms based on your performance requirements:

| Use Case | Recommended Algorithm | Why |
|----------|----------------------|-----|
| Fastest clustering | GeneralizedKMeans (SE) | Optimized crossJoin |
| Probabilistic data | GeneralizedKMeans (KL) | Natural for probability distributions |
| Outlier robustness | K-Medoids or K-Medians | Less sensitive to outliers |
| Unknown k | X-Means | Automatic k selection |
| Real-time updates | StreamingKMeans | Incremental learning |
| Hierarchical structure | BisectingKMeans | Divisive clustering |

### Parameter Tuning

- **k**: Start with `sqrt(n/2)` as a rule of thumb
- **maxIter**: 20 is usually enough; 50+ for high precision
- **initMode**: `k-means||` is slower but better quality than `random`
- **initSteps**: 2-5 steps for k-means|| (more steps = better init)
- **tol**: 1e-4 (default) balances speed and convergence

## Future Benchmarking Work

### JMH Micro-Benchmarks (Planned)

For more detailed performance analysis, we plan to add JMH benchmarks:

```scala
// Future: src/benchmark/scala/com/massivedatascience/clusterer/benchmarks/
@State(Scope.Benchmark)
class KernelBenchmark {
  @Benchmark
  def squaredEuclideanDistance: Double = ???

  @Benchmark
  def klDivergence: Double = ???
}
```

### Comparative Benchmarks (Planned)

Comparison with MLlib KMeans and other libraries:

| Library | Algorithm | 10K pts | 100K pts | 1M pts |
|---------|-----------|---------|----------|--------|
| This library (SE) | Lloyd's | TBD | TBD | TBD |
| MLlib KMeans | Lloyd's | TBD | TBD | TBD |
| scikit-learn | Lloyd's | TBD | TBD | TBD |

### Large-Scale Benchmarks (Planned)

Production-scale benchmarks on real clusters:

- **10M points**: 100 dimensions, 100 clusters
- **100M points**: 50 dimensions, 50 clusters
- **1B points**: 20 dimensions, 20 clusters

## Reporting Performance Issues

If you encounter performance issues:

1. **Run the perf sanity check**: `sbt "testOnly *PerfSanitySuite*"`
2. **Capture metrics**: Save the JSON report from `target/perf-reports/`
3. **Report the issue**: Include:
   - Dataset size (points, dimensions)
   - Algorithm and parameters
   - Actual vs expected time
   - Machine specs
   - Spark configuration
   - JSON report

## References

- [Spark Performance Tuning Guide](https://spark.apache.org/docs/latest/tuning.html)
- [K-Means Performance Analysis](https://en.wikipedia.org/wiki/K-means_clustering#Performance_and_complexity)
- [Bregman Divergences](https://en.wikipedia.org/wiki/Bregman_divergence)

---

**Last Updated**: October 19, 2025
**Baseline Version**: 0.6.0
**Test Platform**: MacBook Pro M1, Spark 3.5.1, Scala 2.13.14
