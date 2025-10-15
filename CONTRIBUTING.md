# Contributing

## Dev setup
- Java 11 (Temurin)
- sbt 1.10.x
- Scala 2.12/2.13 (cross-build)
- Spark 3.4/3.5 (provided)

## Building
```bash
sbt +compile
sbt +test
```

## Releasing
- Tag `vX.Y.Z` and let CI publish (if configured).
- Keep CHANGELOG up to date.

## Code style
- Scalafmt recommended; enforce in CI if added.
- PRs must pass CI and tests.
