---
title: Installation Guide
---

# Installation Guide

How to add generalized-kmeans-clustering to your project.

---

## SBT

Add to `build.sbt`:

```scala
libraryDependencies += "com.massivedatascience" %% "clusterer" % "0.7.0"
```

For specific Scala/Spark versions:

```scala
// Scala 2.13 + Spark 3.5
libraryDependencies += "com.massivedatascience" % "clusterer_2.13" % "0.7.0"

// Scala 2.12 + Spark 3.4
libraryDependencies += "com.massivedatascience" % "clusterer_2.12" % "0.7.0"
```

---

## Maven

```xml
<dependency>
    <groupId>com.massivedatascience</groupId>
    <artifactId>clusterer_2.13</artifactId>
    <version>0.7.0</version>
</dependency>
```

---

## spark-submit

```bash
spark-submit \
  --packages com.massivedatascience:clusterer_2.13:0.7.0 \
  --class com.example.MyApp \
  my-app.jar
```

---

## spark-shell / pyspark

```bash
# Scala
spark-shell --packages com.massivedatascience:clusterer_2.13:0.7.0

# Python
pyspark --packages com.massivedatascience:clusterer_2.13:0.7.0
```

---

## Databricks

### Option 1: Cluster Library

1. Go to **Compute** → Select your cluster → **Libraries**
2. Click **Install New** → **Maven**
3. Enter coordinates: `com.massivedatascience:clusterer_2.13:0.7.0`
4. Click **Install**

### Option 2: Notebook

```python
%pip install massivedatascience-clusterer
```

### Option 3: Init Script

Create `/dbfs/init-scripts/install-clusterer.sh`:

```bash
#!/bin/bash
pip install massivedatascience-clusterer
```

---

## EMR

### Bootstrap Action

```bash
#!/bin/bash
sudo pip3 install massivedatascience-clusterer
```

### Step Configuration

```json
{
  "Name": "Spark Submit with Clusterer",
  "ActionOnFailure": "CONTINUE",
  "HadoopJarStep": {
    "Jar": "command-runner.jar",
    "Args": [
      "spark-submit",
      "--packages", "com.massivedatascience:clusterer_2.13:0.7.0",
      "s3://my-bucket/my-app.jar"
    ]
  }
}
```

---

## Kubernetes / Spark Operator

```yaml
apiVersion: sparkoperator.k8s.io/v1beta2
kind: SparkApplication
spec:
  deps:
    packages:
      - com.massivedatascience:clusterer_2.13:0.7.0
```

---

## Build from Source

```bash
git clone https://github.com/derrickburns/generalized-kmeans-clustering.git
cd generalized-kmeans-clustering

# Build for Scala 2.13
sbt ++2.13.14 publishLocal

# Build for Scala 2.12
sbt ++2.12.18 publishLocal
```

---

## Version Compatibility

| Library | Spark 3.4 | Spark 3.5 | Spark 4.0 |
|---------|-----------|-----------|-----------|
| Scala 2.12 | ✓ | ✓ | — |
| Scala 2.13 | ✓ | ✓ | ✓ |

---

## Verify Installation

```scala
import com.massivedatascience.clusterer.ml.GeneralizedKMeans

val kmeans = new GeneralizedKMeans()
println(s"Library loaded: ${kmeans.getClass.getName}")
```

---

[Back to How-To](index.html) | [Home](../)
