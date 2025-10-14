# Release Notes - Version 0.6.0

## DataFrame API - Major Release

Version 0.6.0 introduces a complete DataFrame-native clustering API with full Spark ML Pipeline integration, representing a major architectural improvement over the legacy RDD-based implementation.

## 🎉 What's New

### DataFrame API (Recommended)

A modern, type-safe API that integrates seamlessly with Spark ML:

```scala
import com.massivedatascience.clusterer.ml.GeneralizedKMeans

val kmeans = new GeneralizedKMeans()
  .setK(5)
  .setDivergence("kl")
  .setMaxIter(20)

val model = kmeans.fit(dataset)
val predictions = model.transform(dataset)
```

### Key Features

- **5 Bregman Divergences**: Squared Euclidean, KL, Itakura-Saito, Generalized I, Logistic Loss
- **Spark ML Integration**: Native Estimator/Model pattern, pipeline compatibility, model persistence
- **Pluggable Strategies**: Assignment (BroadcastUDF, SECrossJoin, Auto), Update, Empty Cluster Handling
- **Advanced Options**: Weighted clustering, multiple initialization modes, checkpointing
- **LloydsIterator Pattern**: Single unified implementation eliminates 1000+ lines of duplicated code

### Architecture Improvements

**Before:** Multiple implementations of Lloyd's algorithm scattered across different clusterers, with significant code duplication.

**After:** Single `LloydsIterator` implementation with pluggable strategies for all variations. Clean separation of concerns.

## 📊 Statistics

- **2,329 lines** of new production code
- **272 lines** of integration tests
- **231 lines** of documentation/examples
- **193 tests passing** (182 existing RDD + 11 new DataFrame)
- **Zero regressions**

## 🔄 Migration Guide

### From RDD API to DataFrame API

**Old (RDD):**
```scala
import com.massivedatascience.clusterer.KMeans
val model = KMeans.train(rdd, runs, k, maxIterations)
```

**New (DataFrame):**
```scala
import com.massivedatascience.clusterer.ml.GeneralizedKMeans
val model = new GeneralizedKMeans()
  .setK(k)
  .setMaxIter(maxIterations)
  .fit(dataframe)
```

### Backward Compatibility

The RDD-based API remains fully supported for existing projects. No breaking changes.

## 📖 Documentation

- **Quick Start**: See [README.md](README.md)
- **Complete Examples**: [DATAFRAME_API_EXAMPLES.md](DATAFRAME_API_EXAMPLES.md)
- **Architecture Details**: [DF_ML_REFACTORING_PLAN.md](DF_ML_REFACTORING_PLAN.md)
- **Phase 1 & 2 History**: [REFACTORING_SUMMARY.md](REFACTORING_SUMMARY.md)

## 🏗️ Build Changes

- **Spark**: 3.5.1 (overridable via `-Dspark.version`)
- **Scala**: 2.12.18 / 2.13.14 (cross-compiled)
- **Java**: 17 with proper module opens
- **Test Dependencies**: ScalaTest 3.2.19, ScalaCheck 1.17.0

## 🔮 Future Roadmap

The DataFrame API provides a solid foundation for future enhancements:

- **Phase 4**: Enhanced metrics (silhouette, Davies-Bouldin index)
- **Phase 5**: Expanded test coverage (persistence, property-based)
- **Phase 6**: PySpark wrapper
- **Phase 7**: Additional documentation (architecture guide, API docs)
- **Phase 8**: Official v0.6.0 release to Maven Central

## 💡 Why Upgrade?

1. **Modern API**: Type-safe, composable, integrates with Spark ML pipelines
2. **Better Performance**: Leverages Catalyst optimizer for expression-based operations
3. **Cleaner Code**: Pluggable strategies eliminate duplication
4. **Future-Proof**: Spark's strategic direction is DataFrame/Dataset API
5. **Easier Debugging**: Better error messages, logging, and parameter validation

## 🙏 Acknowledgments

This release represents a significant architectural improvement, implementing the LloydsIterator pattern to unify clustering implementations while maintaining full backward compatibility.

Special thanks to the Spark community for the excellent ML Pipeline framework that made this integration seamless.

---

**For questions or issues**, please visit: https://github.com/derrickburns/generalized-kmeans-clustering/issues
