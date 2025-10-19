# Dependency Management

This repository uses automated dependency updates to keep libraries current and secure.

## Scala Dependencies - Scala Steward

**Tool**: [Scala Steward](https://github.com/scala-steward-org/scala-steward)

Scala Steward automatically creates pull requests to update Scala and sbt dependencies.

### Configuration

- **File**: `.scala-steward.conf`
- **Workflow**: `.github/workflows/scala-steward.yml`
- **Schedule**: Weekly (Mondays at 00:00 UTC)
- **Update limit**: 5 dependencies per run
- **PR labels**: `dependencies`, `scala-steward`

### Pinned Dependencies

The following dependencies are pinned to maintain compatibility:

- **Spark**: Pinned to `3.5.x` to ensure compatibility across minor versions
  - Major version updates require manual review and testing
  - Configuration: `{ groupId = "org.apache.spark", version = "3.5." }`

### Ignored Dependencies

- **SNAPSHOT versions**: Never auto-update to snapshot/pre-release versions
- **Major version changes**: Require manual review (handled via pinning)

### How It Works

1. **Weekly scan**: Scala Steward checks for updates every Monday
2. **PR creation**: Creates grouped PRs with up to 5 dependency updates
3. **CI validation**: All PRs must pass CI tests before merge
4. **Manual review**: Review PRs to ensure compatibility
5. **Merge**: Merge when CI passes and changes look reasonable

### Manual Trigger

You can manually trigger Scala Steward via GitHub Actions:

```bash
# Via GitHub UI
Actions → Scala Steward → Run workflow

# Via GitHub CLI
gh workflow run scala-steward.yml
```

## Python Dependencies - Dependabot

**Tool**: [Dependabot](https://docs.github.com/en/code-security/dependabot)

Dependabot manages Python dependencies in the `python/` directory.

### Configuration

- **File**: `.github/dependabot.yml`
- **Directory**: `/python`
- **Schedule**: Weekly (Mondays)
- **Update limit**: 5 dependencies per run
- **PR labels**: `dependencies`, `python`

### Requirements Files

Dependabot monitors:
- `python/pyproject.toml` - Core dependencies
- `python/requirements.txt` (if exists) - Additional dependencies

## GitHub Actions - Dependabot

**Tool**: [Dependabot](https://docs.github.com/en/code-security/dependabot)

Dependabot keeps GitHub Actions up to date for security.

### Configuration

- **File**: `.github/dependabot.yml`
- **Directory**: `/`
- **Schedule**: Weekly (Mondays)
- **Update limit**: 5 actions per run
- **PR labels**: `dependencies`, `github-actions`

### Monitored Actions

All workflow files in `.github/workflows/`:
- `actions/checkout`
- `actions/setup-java`
- `sbt/setup-sbt`
- `actions/cache`
- Custom actions

## Security Updates

### Vulnerability Scanning

- **Scala**: Scala Steward flags known CVEs in dependencies
- **Python**: Dependabot security alerts for Python packages
- **GitHub Actions**: Dependabot security alerts for action versions

### Response Process

1. **Alert received**: GitHub/Scala Steward creates security PR
2. **Priority review**: Security updates get immediate attention
3. **Testing**: Run full test suite to verify compatibility
4. **Merge**: Fast-track merge after CI passes
5. **Deploy**: Create patch release if in production

## Dependency Update Best Practices

### Before Merging Dependency PRs

✅ **Check CI status**: All tests must pass
✅ **Review changelog**: Look for breaking changes
✅ **Check compatibility**: Ensure cross-version compatibility (Spark 3.4 ↔ 3.5, Scala 2.12 ↔ 2.13)
✅ **Run examples**: Verify examples still work
✅ **Check performance**: Run perf sanity tests

### Major Version Updates

For major version bumps (e.g., Spark 3.5 → 4.0):

1. Create feature branch for testing
2. Update all related dependencies together
3. Run full test matrix (all Scala/Spark versions)
4. Update documentation for breaking changes
5. Create migration guide if needed
6. Tag as major release (e.g., v1.0 → v2.0)

### Version Pinning Strategy

**When to pin**:
- Spark version (cross-version compatibility critical)
- Test framework versions (to avoid test flakiness)
- Build tool versions (SBT, Scalafmt)

**When NOT to pin**:
- Transitive dependencies (let SBT resolve)
- Utility libraries (logging, etc.)
- Development dependencies (unless causing issues)

## Viewing Dependency Status

### Scala Dependencies

```bash
# List all dependencies
sbt dependencyTree

# Check for updates (manual)
sbt dependencyUpdates

# Check Scala Steward PRs
gh pr list --label scala-steward
```

### Python Dependencies

```bash
# Check Dependabot PRs
gh pr list --label python

# Manual check for updates
cd python
pip list --outdated
```

### GitHub Actions

```bash
# Check Dependabot PRs
gh pr list --label github-actions
```

## Troubleshooting

### Scala Steward Not Creating PRs

1. **Check workflow runs**: `gh run list --workflow=scala-steward.yml`
2. **Check logs**: `gh run view <run-id> --log`
3. **Manual trigger**: `gh workflow run scala-steward.yml`
4. **Verify config**: Check `.scala-steward.conf` syntax

### Dependency Conflicts

If Scala Steward creates a PR that breaks CI:

1. **Don't merge**: Close the PR with a comment explaining why
2. **Update config**: Add to `updates.ignore` or `updates.pin` if needed
3. **Create issue**: Document the incompatibility for future reference

### Stale Dependency PRs

Automatically close stale dependency PRs after 30 days if not merged.

## Related Documentation

- [Scala Steward Docs](https://github.com/scala-steward-org/scala-steward/blob/main/docs/repo-specific-configuration.md)
- [Dependabot Docs](https://docs.github.com/en/code-security/dependabot)
- [SBT Dependency Management](https://www.scala-sbt.org/1.x/docs/Library-Dependencies.html)

---

**Last Updated**: October 19, 2025
