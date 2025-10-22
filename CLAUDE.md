Perfect—here’s a PR-ready replacement for CLAUDE.md that merges your project-specific copilot guide with the Scala standards you provided. I’ve woven the guidelines into sections Claude can act on for this repo (Spark/DF-first, deterministic, persistence-safe), and I’ve called out where we’re Scala 2.13 today while still steering contributions to be Scala-3-friendly.

Drop this in as CLAUDE.md:

⸻


# CLAUDE.md — Maintainer Copilot for *generalized-kmeans-clustering*

**Purpose.** This file tells Claude (or any LLM assistant) exactly how to help on this repo without wasting maintainer time. It encodes project norms, API/persistence guarantees, review rubrics, performance boundaries, and Scala engineering standards.

> TL;DR  
> **Primary surface:** Spark **DataFrame/ML API** (`GeneralizedKMeans`, etc.).  
> **Versions:** Scala **2.13** (primary), Spark **3.5.x** (default; 3.4.x compat).  
> **Math:** Bregman family — divergences include `squaredEuclidean`, `kl`, `itakuraSaito`, `l1`, `generalizedI`, `logistic`.  
> **Variants:** Bisecting, X-Means, Soft/Fuzzy, Streaming, K-Medians, K-Medoids.  
> **Determinism + persistence** are non-negotiable; RDD API is **archived** (reference only).

---

## 0) Operating Principles (do these every time)

1. **Prefer the DataFrame/ML API.** Code and examples use Estimator/Model patterns and Params from this codebase.  
2. **No silent API breaks.** If you touch params, model JSON, or persistence schemas, include migration/round-trip tests.  
3. **Mathematical fidelity first.** Correct Bregman formulations beat micro-perf. Perf changes must not alter semantics.  
4. **Determinism matters.** Same seed ⇒ identical results. Avoid nondeterministic ops in core loops.  
5. **Tight PRs.** Small, test-backed, CI-friendly. No speculative abstractions.

---

## 1) Project Snapshot (facts Claude must anchor to)

- **Surface:** `GeneralizedKMeans` (and friends) via DataFrame API; prediction via `transform`.  
- **Spark:** 3.5.x default; 3.4.x tested.  
- **Scala:** 2.13.x primary (keep code Scala-3-friendly where feasible).  
- **Java:** 17.  
- **Divergences:** `squaredEuclidean | kl | itakuraSaito | l1 | generalizedI | logistic`.  
- **Assignment strategies:** `auto | crossJoin (SE fast path) | broadcastUDF (general Bregman)`.  
- **Input transforms:** `none | log1p | epsilonShift(shiftValue)`; ensure domain validity for KL/IS.  
- **Persistence:** Models round-trip across Spark 3.4↔3.5, Scala 2.12↔2.13.  
- **RDD API:** archived; do not add features.

---

## 2) What Good Help Looks Like

Claude should deliver one of:
- **Small PR plan** with diff outline + tests.  
- **Minimal patch** that compiles locally and within the CI matrix.  
- **Review comments** (actionable, line-anchored).  
- **Doc/example edits** with runnable snippets.

Always include: **test list**, **risk matrix (API/persistence)**, **rollback note**.

---

## 3) Common Tasks & Prompts (ready to paste)

### (A) Add/adjust a DF Param
> “Propose a minimal patch to add `<paramName: type>` to `GeneralizedKMeans` with default `<value>`, update model persistence JSON, validate in `ExamplesSuite`, and write a property test. Include compatibility notes and a unified diff.”

### (B) SE fast path perf (crossJoin)
> “Audit the squared-Euclidean fast path for shuffles/sorts. Propose a plan using `groupBy(rowId).min(distance)` + `join` for argmin where beneficial. Provide an explain plan and a micro-benchmark kept under 30s in CI.”

### (C) Streaming snapshots
> “Make snapshot load idempotent, reuse seeds, and add explicit state schema tests. Include a recovery scenario test (partial write) and document operational steps.”

### (D) KL/IS domain safety
> “Enforce positivity assumptions. Add guardrails at `fit` with actionable error messages and doc the `inputTransform` + `shiftValue` guidance in scaladoc/README.”

### (E) X-Means scoring
> “Validate BIC/AIC selection on a synthetic dataset; ensure it picks the correct `k`. Add an example and a property test.”

---

## 4) Scala Engineering Standards (adopted and tailored)

### Architecture & Design
- **Pure core, effects at edges.** Core clustering logic is referentially transparent; I/O/clock/logging/FS confined to adapters/tests.  
- **Model the domain with types.** Prefer enums/ADTs, opaque types/small value classes over raw primitives; encode invalid states as unrepresentable.  
- **Composition over inheritance.** Use typeclasses and tiny capability traits (e.g., `trait Clock { def now: Instant }`).  
- **Stable public APIs.** Expose interfaces, not implementations; version data crossing module boundaries.  
- **Configuration is data.** Strongly-typed config loaded once, validated early; avoid scattered `sys.env` access.

### Language & Style (Scala-3 friendly, Scala-2.13 today)
- Use Scala-3 features **deliberately** when writing examples/tools that may cross-compile later (ADTs via `enum` style in 3; encode as sealed traits in 2.13 now).  
- **Immutability by default.** `val` everywhere; local `var` only in measured perf hotspots.  
- **Explicitness > magic.** Keep implicits local; prefer `given/using` style in 3; in 2.13, constrain implicit scope via local imports.  
- Avoid implicit conversions; if unavoidable, keep them local and explicit.  
- Collections: immutable `List/Vector/Map/Set`; use `Array`/mutable only with measured perf need.  
- Errors: prefer `Option`/`Either[E, A]`/typed effects; use exceptions for unrecoverable faults.  
- Pattern matching: exhaustive; enable compiler flags.  
- Naming: precise and boring > clever; methods = verbs, ADTs = nouns.

### Effects, Concurrency, Resource Safety
- Choose **one** modern effect stack for tooling/tests/examples (cats-effect or ZIO). Keep core algorithm pure.  
- **Structured concurrency** (fibers + scopes). No ad-hoc threads/futures. Timeouts and cancellation first-class.  
- **Resource safety** via `Resource` (cats-effect) or `ZLayer`/`Scope` (ZIO). No naked open/close.  
- Streaming: prefer fs2/ZIO Stream; avoid hand-rolled queues.  
- Abstract time/randomness (`Clock`, `Random`) for deterministic tests.

### Modules, Wiring, DI
- Constructor injection / compile-time wiring. Avoid runtime reflection DI.  
- Keep module graphs **acyclic**; one module = one responsibility.

### Persistence & I/O
- JSON codecs are explicit and round-tripped in tests; define in companion objects.  
- Retries/circuit breakers centralized; no scattered ad-hoc retry loops.

### Testing Strategy
- **Test pyramid.**  
  - Unit: fast, pure, property-based (ScalaCheck).  
  - Component/integration: adapters (FS/streaming); Testcontainers when needed.  
  - Contract tests for external boundaries.  
  - Few E2E tests, only for critical paths.  
- Determinism: no real time/random/network in unit tests; use `TestClock`, `TestRandom`, in-memory stubs.  
- Golden tests for codecs under `src/test/resources`.  
- Mutation testing (stryker4s) to catch weak tests.  
- Coverage (scoverage) is a **signal**, not a KPI.

### Error Handling & Observability
- **Typed errors.** Domain errors are values; map once at the edge to statuses.  
- **Structured logs** (JSON) with correlation IDs; never log from pure code.  
- Metrics and tracing (OpenTelemetry) where applicable; propagate context through effects.  
- Fail fast on misconfiguration.

### Performance & Reliability
- Prefer correct **algorithms/data structures** over micro-tweaks. Profile with realistic workloads.  
- **Bound everything.** Pools/queues/timeouts/budgets.  
- Batch/chunk streams; avoid per-element I/O.  
- Design idempotent, especially for streaming and snapshot updates.

### Build, Tooling & CI
- **sbt hygiene.** One source of truth for versions; enable version scheme and MiMa where publishing applies.  
- **Compiler flags:** treat warnings as errors; include `-Wunused:*`, `-Wnonunit-statement`, `-Wvalue-discard`.  
- **Formatting & linting:** Scalafmt enforced; Scalafix rules for pitfalls; wartremover if justified.  
- Reproducible builds: pinned toolchain; cache coursier/ivy in CI.  
- Packaging: layered images (distroless) if/when we ship services; health checks; consider native image (GraalVM) only if it pays.

### API & Schema Discipline
- **Schema-first** at external boundaries; generate clients/servers, keep domain logic hand-written.  
- **Compatibility:** Backward-compatible on minors; gate breaking changes behind flags/migrations.  
- Validate at the edge; domain types are already validated (smart constructors).

### Folder & Project Layout (example baseline)

modules/
core/          // pure domain + clustering math
adapters/      // e.g., streaming snapshots, persistence
examples/      // runnable examples & docs-as-tests

### Code Review Checklist (apply every PR)
- Types encode invariants; no invalid states exposed.  
- Effects isolated; pure core stays pure.  
- Functions are small, single-responsibility; no hidden I/O.  
- Exhaustive matches; no `get`/`head`/unsafe casts.  
- Concurrency bounded; timeouts/cancellation tested.  
- Structured logging at edges; no printlns.  
- Tests have meaningful properties; mutation tests pass.  
- No flakes: clocks/randomness controlled.  
- Format/lint clean; **0 warnings**.

### Anti-Patterns (hard NOs)
- God services, wide mutable state, runtime DI via deep reflection.  
- Exceptions as control flow; swallowed failures.  
- Shared global ECs without sizing/labels; unbounded retries; blocking on compute pools.  
- Ad-hoc JSON in handlers; business logic inside controllers or SQL strings.  
- “Write a test to hit coverage” mentality.

---

## 5) Code Review Rubric (Spark/DF specifics)

- **API surface:** clear names, consistent Params, Java/Scala boundary sane.  
- **Math:** divergence domain checks; gradients correct; no mixing transformed/original spaces.  
- **Spark plans:** avoid unnecessary wide shuffles; respect broadcast thresholds; document tradeoffs.  
- **Determinism:** seeded ops; no time-based seeds.  
- **Persistence:** schema versioned; DF & medoid models round-trip.  
- **Docs:** README/examples mirror Params exactly; examples are executable tests.  
- **Legacy:** do not expand RDD API.

---

## 6) Minimal Example (use in docs/tests)

```scala
val df = spark.createDataFrame(Seq(
  Tuple1(Vectors.dense(0.0, 0.0)),
  Tuple1(Vectors.dense(1.0, 1.0)),
  Tuple1(Vectors.dense(9.0, 8.0)),
  Tuple1(Vectors.dense(8.0, 9.0))
)).toDF("features")

val gkm = new GeneralizedKMeans()
  .setK(2)
  .setDivergence("kl")              // "squaredEuclidean" | "itakuraSaito" | "l1" | "generalizedI" | "logistic"
  .setAssignmentStrategy("auto")    // "auto" | "crossJoin" | "broadcastUDF"
  .setMaxIter(20)

val model = gkm.fit(df)
val pred  = model.transform(df)


⸻

7) CI Expectations
	•	Stay within existing matrix (Scala 2.12/2.13 × Spark 3.4/3.5).
	•	Update ExamplesSuite when editing docs; examples are executable documentation.
	•	Keep perf sanity checks ≤ 30s wall clock.

⸻

8) Quick PR Template

### Summary
One-liner of the change and why.

### Changes
- [ ] Code (files…)
- [ ] Tests (suites/classes)
- [ ] Docs/Examples updated
- [ ] Persistence/compatibility notes

### Risk / Compatibility
- API surface: <none | details>
- Persistence schema: <unchanged | change + migration>

### Validation
- Local: spark 3.5.x / scala 2.13.x
- CI: matrix runs


⸻

9) Backlog Suggestions (safe, high-ROI)
	•	Validation helpers for KL/IS domain with actionable messages and suggested transforms.
	•	Better broadcast-threshold diagnostics (include k × dim and configured threshold).
	•	X-Means end-to-end “auto-k” example notebook + doc.

⸻

10) Hallucination Traps (pin these truths)
	•	General Bregman ≠ SE: Do not reuse SE-only vectorized shortcuts in the general (UDF) path.
	•	Transformed space: When transforms apply, centers live in transformed space; inverse is for interpretation only.
	•	RDD API is archived: Do not propose new RDD features.

---

### Drop-in Patch (unified diff)

```diff
--- a/CLAUDE.md
+++ b/CLAUDE.md
@@ -1,0 +1,999 @@
+<paste the entire file above>

If you want, I can also add a short “Using an AI Copilot on this repo” section in README.md that links to CLAUDE.md, plus a scalafmt.conf/compiler-flag stanza to lock in the style.
