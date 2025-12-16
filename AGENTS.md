# Repository Guidelines

Use this guide to make concise, high-signal contributions to the generalized k-means clustering library.

## Project Structure & Module Organization
- Scala sources live in `src/main/scala` (DataFrame/ML API under `com.massivedatascience.clusterer.ml`), with version-specific shims in `src/main/scala-2.12` and `src/main/scala-2.13`. Legacy RDD code remains in `com.massivedatascience.clusterer`.
- Tests use ScalaTest under `src/test/scala` with Spark-local fixtures; shared data is in `src/test/resources`. Executable examples sit in `src/main/scala/examples`.
- Python wrapper lives in `python/` (`massivedatascience` package, examples, and tests). Docs and release notes are in `docs/`, `release-notes/`, `ARCHITECTURE.md`, and `DATAFRAME_API_EXAMPLES.md`.

## Build, Test, and Development Commands
- `sbt compile` — compile against the default Scala/Spark matrix; use `sbt ++2.13.14` or `sbt ++2.12.18` to pin versions.
- `sbt test` — full JVM suite (ScalaTest, Spark local[2]); CI mirrors this with multiple Scala/Spark combos.
- `sbt scalafmtAll` then `sbt scalastyle` — required format/lint gates (`.scalafmt.conf`, `scalastyle-config.xml`).
- `sbt coverage test coverageReport` — generate coverage; keep kernels and persistence paths covered.
- Python: `cd python && pip install -e .[dev] && pytest` (see `python/TESTING.md`).

## Coding Style & Naming Conventions
- Scalafmt enforces 2-space indent and 100-col limit; keep trailing commas and aligned parameters. Prefer immutable vals, small helpers, and Spark ML `Estimator/Model` patterns (`set*`/`get*`).
- Naming: PascalCase classes/objects, camelCase methods/vals/params. Document public APIs with Scaladoc and mirror existing parameter docs.
- In tests, disable the Spark UI and keep partitions small (follow existing suites) to avoid flakiness.

## Testing Guidelines
- Add ScalaTest `AnyFunSuite` cases under `src/test/scala`; keep seeds deterministic and assert numerical tolerances for divergences. Reuse existing fixtures/utilities.
- Include persistence round-trips when adding models/params; CI validates cross-version save/load.
- For Python changes, update `python/tests/test_generalized_kmeans.py` and run `pytest --cov=massivedatascience tests/`.

## Commit & Pull Request Guidelines
- Use conventional commits (`feat|fix|docs|style|refactor|perf|test|build|ci|chore`, optional scope): `type(scope): subject`.
- PRs should summarize behavior changes, list executed commands (e.g., `sbt ++2.13.14 test`, `sbt scalafmtAll`, `pytest`), and link issues (`Closes #123`). Provide before/after snippets for API or doc updates; screenshots only when user-facing outputs change.
- CI runs lint, Scala/Spark matrix tests, python smoke, and CodeQL; align local runs to reduce iteration.

## Security & Configuration Notes
- Target Java 17; avoid committing large datasets or credentials. Report vulnerabilities via `SECURITY.md`.
- When modifying dependencies or persistence formats, consult `DEPENDENCY_MANAGEMENT.md` and `PERSISTENCE_COMPATIBILITY.md` to preserve cross-version compatibility.
