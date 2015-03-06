# Release Notes

## 1.1.0

- Added new ```RE-SEED``` MultiKMeansClusterer implementation
- Added Dense Embedding to Embedding.apply
- Deprecated old ```SIMPLE``` and ```TRACKING``` MultiKMeansClusterer implementations
- Fixed bug in ```KMeans.iterativelyTrain```
- Fixed bug in Dense Embedding
- Fixed bug in GeneralizedI Divergence
- Moved ConvergenceDetector class from ColumnTrackingKMeans object
- Changed ```RunConfig.toString``` to show names of parameters
- Purged extra log messages
- In-lined vector iterators
- Eliminated Logging trait from SparkHelper
- Modified log4j test configuration to increase log level on Spark logger output
- Added support for code coverage evaluation with Coveralls and sbt
- Added Coveralls Badge to README
- Updated README
- Added organizationName to build.sbt
- Added sample logger output
- Added numerous unit tests

## 1.0.2

- Fixed url in pol.xml

## 1.0.1

- Fixed broken pom.xml. License section was duplicated.

## 1.0.0 Initial Release
