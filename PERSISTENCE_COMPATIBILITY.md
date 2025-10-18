# Persistence Compatibility

This document defines the persistence contract for generalized-kmeans-clustering models, ensuring compatibility across Spark and Scala versions.

## Overview

Models use a **versioned, deterministic layout** that is:
- Engine-neutral (JSON + Parquet, no Scala pickling)
- Cross-version compatible (Spark 3.4↔3.5, Scala 2.12↔2.13)
- Human-readable and debuggable
- Checksummed for integrity verification

## Layout Version 1

**Status:** Stable ✅

**Supported environments:**
- Apache Spark: 3.4.x ↔ 3.5.x
- Scala: 2.12.x ↔ 2.13.x
- Java: 11, 17

### On-Disk Structure

```
<modelPath>/
  metadata.json        # All parameters, versions, checksums
  centers.parquet/     # Cluster centers with deterministic ordering
  summary.json         # Optional training metrics (future)
```

### metadata.json Schema

```json
{
  "layoutVersion": 1,
  "algo": "GeneralizedKMeansModel",
  "sparkMLVersion": "3.5.1",
  "scalaBinaryVersion": "2.13",
  "divergence": "squaredEuclidean",
  "k": 10,
  "dim": 512,
  "uid": "gkmeans_abc123",
  "kernelName": "SquaredEuclidean",
  "params": {
    "maxIter": 20,
    "tol": 1e-4,
    "seed": 1234,
    "assignmentStrategy": "auto",
    "smoothing": 1e-10,
    "emptyClusterStrategy": "reseedRandom",
    "checkpointInterval": 10,
    "initMode": "k-means||",
    "initSteps": 2,
    "featuresCol": "features",
    "predictionCol": "prediction",
    "distanceCol": "",
    "weightCol": "",
    "checkpointDir": ""
  },
  "centers": {
    "count": 10,
    "ordering": "center_id ASC (0..k-1)",
    "storage": "parquet"
  },
  "checksums": {
    "centersParquetSHA256": "abc123...",
    "metadataCanonicalSHA256": "def456..."
  }
}
```

### centers.parquet Schema

| Column | Type | Description |
|--------|------|-------------|
| `center_id` | int | Center index (0 to k-1), deterministic ordering |
| `weight` | double | Center weight (1.0 for unweighted models) |
| `vector` | Vector | Center coordinates (Spark ML Vector UDT) |

**Ordering guarantee:** Centers are always written and read in `center_id` ascending order.

## Backward Compatibility Policy

### Minor Releases (0.x.y)

**Policy:** Additive changes only

- ✅ New parameters must have sensible defaults
- ✅ Readers ignore unknown parameters (forward compatibility)
- ✅ Writers include all parameters (backward compatibility)
- ❌ Cannot remove parameters
- ❌ Cannot change parameter semantics

**Example:** Adding a new `convergenceMetric` parameter in 0.7.0:
```scala
// New parameter with default
val convergenceMetric = new Param[String](...)
setDefault(convergenceMetric -> "distortion")

// Old models (0.6.0) loading from 0.7.0: ✅ Works (default applied)
// New models (0.7.0) loading from 0.6.0: ✅ Works (param ignored)
```

### Major Releases (x.0.0)

**Policy:** Breaking changes allowed with migration path

- Bump `layoutVersion` (e.g., 1 → 2)
- Provide migration utility: `bin/migrate-models --from v1 --to v2`
- Document breaking changes in CHANGELOG
- Support reading previous version for 1 major release

## Cross-Version Testing

CI validates all cross-version scenarios:

| Save Version | Load Version | Status |
|--------------|--------------|--------|
| Spark 3.4.0 + Scala 2.12 | Spark 3.5.1 + Scala 2.13 | ✅ Validated |
| Spark 3.5.1 + Scala 2.13 | Spark 3.4.0 + Scala 2.12 | ✅ Validated |
| Spark 3.4.0 + Scala 2.13 | Spark 3.5.1 + Scala 2.12 | ✅ Validated |
| Spark 3.5.1 + Scala 2.12 | Spark 3.4.0 + Scala 2.13 | ✅ Validated |

**CI Job:** `.github/workflows/persistence-cross.yml`

## Checksums and Integrity

### Purpose

Checksums detect:
- Disk corruption
- Incomplete writes
- Manual file edits
- Accidental modifications

### Behavior

**On Save:**
1. Write centers.parquet
2. Compute SHA-256 of center data (JSON representation, deterministic order)
3. Write metadata.json with `centersParquetSHA256`
4. Compute SHA-256 of metadata.json
5. Update metadata.json with `metadataCanonicalSHA256`

**On Load:**
1. Read metadata.json
2. Validate `layoutVersion`
3. Read centers.parquet
4. **Warning** if checksums don't match (not fatal)
5. Reconstruct model

**Configuration:**
```scala
// Future: strict mode
val model = GeneralizedKMeansModel.read
  .option("strictChecksums", "true")  // Fail on mismatch
  .load(path)
```

## Deterministic Center Ordering

**Guarantee:** Centers always have stable IDs (0 to k-1) regardless of:
- Training algorithm (standard, bisecting, x-means)
- Reseeding during training
- Cluster merging/splitting

**Implementation:**
1. Algorithm assigns explicit `center_id` during training
2. Writer sorts by `center_id` before saving
3. Reader sorts by `center_id` after loading
4. Parquet uses single partition to avoid shuffle ordering issues

**Example:**
```scala
// Training may produce centers in any order internally
val model = gkm.fit(data)

// But persistence guarantees:
model.write.save("model/")
// centers.parquet has center_id: 0, 1, 2, ..., k-1 (in order)

val loaded = GeneralizedKMeansModel.load("model/")
loaded.clusterCenters(0) == model.clusterCenters(0)  // ✅ Same center
```

## Model-Specific Extensions

Different algorithms may extend the base schema:

### BisectingGeneralizedKMeansModel (future)

Additional fields in `metadata.json`:
```json
{
  "algo": "BisectingGeneralizedKMeansModel",
  "bisecting": {
    "maxDepth": 5,
    "minDivisibleClusterSize": 1.0,
    "treeStructure": {
      "root": 0,
      "children": [[1, 2], [3, 4], ...]
    }
  }
}
```

### XMeansModel (future)

Additional fields:
```json
{
  "algo": "XMeansModel",
  "xmeans": {
    "kmin": 2,
    "kmax": 20,
    "criterion": "bic",
    "optimalK": 7,
    "scores": [123.4, 98.2, 87.1, ...]
  }
}
```

### SoftGeneralizedKMeansModel (future)

Additional centers.parquet column:
- `membershipWeights`: Array[Double] - soft membership distribution

## Migration Utilities

### Upgrading from Legacy Format (pre-0.6.0)

The legacy RDD-based models used Parquet-only persistence. To migrate:

```bash
# Provided migration utility (future)
bin/migrate-legacy-model \
  --input old-model/ \
  --output new-model/ \
  --format layoutV1
```

**What it does:**
1. Reads legacy centers parquet
2. Infers parameters from model structure
3. Creates metadata.json with Layout V1
4. Writes centers with deterministic ordering

### Upgrading Between Layout Versions (future)

When Layout V2 is introduced:

```bash
bin/migrate-models \
  --from v1 \
  --to v2 \
  --input models/*.model \
  --output migrated/
```

## Best Practices

### For Library Users

1. **Always use consistent Spark versions** in production
2. **Validate checksums** if models are transferred across systems
3. **Test load/save** in your environment before deploying
4. **Keep metadata.json** for debugging (it's human-readable)

### For Library Developers

1. **Never change parameter semantics** without bumping `layoutVersion`
2. **Always provide defaults** for new parameters
3. **Test cross-version** before releasing
4. **Document breaking changes** in CHANGELOG

### For CI/CD

1. **Archive models** with git LFS or object storage
2. **Version control metadata.json** separately for inspection
3. **Run cross-version tests** in staging before production
4. **Monitor checksums** for silent corruption

## Troubleshooting

### "Incompatible layoutVersion" Error

**Symptom:**
```
java.lang.IllegalArgumentException: Incompatible layoutVersion=2 (expected 1)
```

**Solution:**
1. Check if you're using an older library version to load a newer model
2. Upgrade library to support Layout V2
3. Or use migration utility to downgrade model

### Checksum Mismatch Warning

**Symptom:**
```
WARN: centers checksum mismatch (expected: abc123, got: def456)
```

**Possible causes:**
1. Disk corruption
2. Manual file edit
3. Incomplete write (crash during save)
4. Network transfer issue

**Solution:**
1. Re-save model from source
2. Check disk health
3. Verify file permissions

### Missing centers.parquet

**Symptom:**
```
java.io.FileNotFoundException: centers.parquet not found
```

**Solution:**
1. Ensure complete model directory is copied
2. Check write permissions during save
3. Verify Spark can access distributed filesystem (HDFS, S3, etc.)

## References

- [Spark ML Persistence Guide](https://spark.apache.org/docs/latest/ml-pipeline.html#ml-persistence)
- [Conventional Commits](https://www.conventionalcommits.org/)
- [Semantic Versioning](https://semver.org/)

## Changelog

### Layout V1 (v0.6.0)
- Initial versioned persistence format
- Supports GeneralizedKMeansModel
- Cross-version compatible: Spark 3.4/3.5, Scala 2.12/2.13
