# Release Notes

## 1.1.0

The big new feature is the addition of the ```RE-SEED``` implementation of ```MultiKMeansClusterer```
that re-seeds clusters that become empty during rounds of Lloyd's algorithm.

With the addition of numerous unit tests, we fixed a few newly discovered bugs.

We also added support for code coverage evaluation.

- Added new ```RE-SEED MultiKMeansClusterer``` implementation
- Added ```DenseEmbedding```
- Deprecated old ```SIMPLE``` and ```TRACKING MultiKMeansClusterer``` implementations
- Fixed bug in ```KMeans.iterativelyTrain```
- Fixed bug in  ```GeneralizedIDivergence```
- Moved ```ConvergenceDetector``` class from ```ColumnTrackingKMeans``` object
- Changed ```RunConfig.toString``` to show names of parameters
- Purged extra log messages
- In-lined vector iterators
- Eliminated ```Logging``` trait from ```SparkHelper```
- Modified log4j test configuration to increase log level on Spark logger output
- Added support for code coverage evaluation with Coveralls and ```sbt```
- Added Coveralls Badge to README
- Updated README
- Added ```organizationName``` to build.sbt
- Added sample logger output in ```SAMPLE_LOG.md```
- Added numerous unit tests

## 1.0.2

- Fixed url in pol.xml

## 1.0.1

- Fixed broken pom.xml. License section was duplicated.

## 1.0.0

- Initial release of K-Means clusterer that provides general distance functions and
other important features that the Spark 1.2.0 clusterer does not.
