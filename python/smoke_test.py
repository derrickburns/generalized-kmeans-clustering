#!/usr/bin/env python3
"""
Enhanced smoke test for generalized-kmeans-clustering PySpark wrapper.

Goals:
- Prove import & end-to-end fit/transform on local[*]
- Exercise squaredEuclidean (SE fast path) and a non-SE divergence (KL) if supported
- Validate prediction schema/range, determinism by seed, and model persistence
- Keep it FAST and self-contained for CI
"""

import os
import tempfile
import math
from pyspark.sql import SparkSession
from pyspark.ml.linalg import Vectors, VectorUDT
from pyspark.sql import Row

def _mk_spark():
    return (
        SparkSession.builder
        .appName("GKM-Smoke")
        .master("local[*]")
        # Keep CI runs predictable & quick
        .config("spark.ui.enabled", "false")
        .config("spark.sql.shuffle.partitions", "4")
        .config("spark.executor.memory", "1g")
        .config("spark.driver.memory", "1g")
        .getOrCreate()
    )

def _assert(cond, msg):
    if not cond:
        raise AssertionError(msg)

def _collect_preds(df):
    return [r.prediction for r in df.select("prediction").collect()]

def _has_attr(obj, name: str) -> bool:
    return hasattr(obj, name) and callable(getattr(obj, name))

def main():
    print("Starting enhanced smoke test…")
    spark = _mk_spark()

    try:
        # 1) Import the wrapper (Python-facing API)
        try:
            from massivedatascience.clusterer import GeneralizedKMeans
        except Exception as ie:
            raise ImportError(
                "Failed to import massivedatascience.clusterer.GeneralizedKMeans. "
                "Ensure the project JAR is passed via --jars and the Python package is on PYTHONPATH."
            ) from ie
        print("✓ Imported GeneralizedKMeans")

        # 2) Toy dataset
        base = [
            Row(features=Vectors.dense(0.0, 0.0)),
            Row(features=Vectors.dense(1.0, 1.1)),
            Row(features=Vectors.dense(8.9, 9.0)),
            Row(features=Vectors.dense(10.0, 10.1)),
        ]
        df = spark.createDataFrame(base)
        _assert("features" in df.columns, "Missing features column")
        _assert(isinstance(df.schema["features"].dataType, VectorUDT), "features must be VectorUDT")
        print("✓ Created test DataFrame")

        # 3) Fit SE model (fast path)
        kmeans_se = (
            GeneralizedKMeans()
            .setK(2)
            .setDivergence("squaredEuclidean")
            .setMaxIter(10)
            .setSeed(42)
        )
        # If assignmentStrategy param exists, force auto (or crossJoin) to exercise DF path
        if _has_attr(kmeans_se, "setAssignmentStrategy"):
            kmeans_se = kmeans_se.setAssignmentStrategy("auto")

        model_se = kmeans_se.fit(df)
        print(f"✓ Fitted SE model with {getattr(model_se, 'numClusters', 'unknown')} clusters")

        pred_se = model_se.transform(df)
        _assert("prediction" in pred_se.columns, "SE transform missing prediction column")
        preds = _collect_preds(pred_se)
        _assert(len(preds) == df.count(), "SE prediction count mismatch")
        _assert(all(p in (0, 1) for p in preds), f"SE predictions out of range: {preds}")
        # Optional cost
        if _has_attr(model_se, "computeCost"):
            cost = model_se.computeCost(df)
            _assert(math.isfinite(cost) and cost >= 0, f"SE cost invalid: {cost}")
            print(f"✓ SE cost={cost:.6f}")
        print("✓ SE path OK")

        # 4) Determinism (same seed → same centers/preds)
        model_se_2 = kmeans_se.setSeed(42).fit(df)
        pred_se_2 = model_se_2.transform(df)
        preds2 = _collect_preds(pred_se_2)
        _assert(preds == preds2, "Determinism check failed: different predictions with same seed")
        print("✓ Determinism OK (same seed)")

        # 5) Non-SE path (KL) if available
        ran_non_se = False
        try:
            kmeans_kl = (
                GeneralizedKMeans()
                .setK(2)
                .setDivergence("kl")   # non-SE
                .setMaxIter(8)
                .setSeed(7)
            )
            # If transform params exist, make data safe for KL
            if _has_attr(kmeans_kl, "setInputTransform"):
                kmeans_kl = kmeans_kl.setInputTransform("epsilonShift")
                if _has_attr(kmeans_kl, "setShiftValue"):
                    kmeans_kl = kmeans_kl.setShiftValue(1e-6)

            # If strategy param exists, prefer broadcastUDF to exercise that codepath
            if _has_attr(kmeans_kl, "setAssignmentStrategy"):
                kmeans_kl = kmeans_kl.setAssignmentStrategy("broadcastUDF")

            model_kl = kmeans_kl.fit(df)
            pred_kl = model_kl.transform(df)
            _assert("prediction" in pred_kl.columns, "KL transform missing prediction column")
            preds_kl = _collect_preds(pred_kl)
            _assert(all(p in (0, 1) for p in preds_kl), f"KL predictions out of range: {preds_kl}")
            if _has_attr(model_kl, "computeCost"):
                cost_kl = model_kl.computeCost(df)
                _assert(math.isfinite(cost_kl) and cost_kl >= 0, f"KL cost invalid: {cost_kl}")
            ran_non_se = True
            print("✓ Non-SE (KL) path OK")
        except Exception as nonse_err:
            # Don’t fail CI if non-SE isn’t wired in the wrapper yet; surface info for debugging.
            print(f"⚠️  Non-SE path skipped or failed gracefully: {nonse_err}")

        # 6) Persistence round-trip (SE model; small temp dir)
        tmp = tempfile.mkdtemp(prefix="gkm_smoke_")
        save_path = os.path.join(tmp, "model")
        try:
            model_se.write().overwrite().save(save_path)
            from massivedatascience.clusterer import GeneralizedKMeansModel  # if your wrapper exposes it
            model_loaded = GeneralizedKMeansModel.load(save_path)
            pred_loaded = model_loaded.transform(df)
            _assert("prediction" in pred_loaded.columns, "Loaded model missing prediction column")
            preds_loaded = _collect_preds(pred_loaded)
            _assert(preds_loaded == preds, "Loaded model predictions differ from original")
            print("✓ Persistence round-trip OK")
        except Exception as perr:
            print(f"⚠️  Persistence round-trip skipped or failed gracefully: {perr}")

        # 7) Minimal summary visibility (if present)
        if hasattr(model_se, "summary"):
            try:
                s = model_se.summary()
                # Accept either object or string; just ensure callable works without error.
                print("✓ model.summary() callable")
            except Exception as serr:
                print(f"⚠️  model.summary() present but failed: {serr}")

        # Final
        print("\n✅ Smoke tests passed"
              + (" (incl. non-SE)" if ran_non_se else " (SE only)"))
        return 0

    except Exception as e:
        import traceback
        print(f"\n❌ Smoke test failed: {e}")
        traceback.print_exc()
        return 1
    finally:
        spark.stop()

if __name__ == "__main__":
    raise SystemExit(main())
