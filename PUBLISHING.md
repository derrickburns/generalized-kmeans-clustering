# Publishing Guide

How to publish releases to Maven Central and PyPI.

## Prerequisites

### Maven Central (Sonatype)

1. **Create Sonatype Account**
   - Register at https://issues.sonatype.org
   - Create a new project ticket for `com.massivedatascience` namespace
   - Wait for approval (usually 1-2 business days)

2. **Generate GPG Key**
   ```bash
   # Generate key
   gpg --full-generate-key
   # Choose: RSA and RSA, 4096 bits, key does not expire

   # List keys to get key ID
   gpg --list-secret-keys --keyid-format LONG

   # Export private key (for GitHub Actions)
   gpg --armor --export-secret-keys YOUR_KEY_ID > private-key.asc

   # Upload public key to keyserver
   gpg --keyserver keyserver.ubuntu.com --send-keys YOUR_KEY_ID
   ```

3. **Add GitHub Secrets**
   Go to: Repository → Settings → Secrets and variables → Actions

   Add these secrets:
   | Secret Name | Value |
   |-------------|-------|
   | `SONATYPE_USERNAME` | Your Sonatype username |
   | `SONATYPE_PASSWORD` | Your Sonatype password |
   | `GPG_PRIVATE_KEY` | Contents of `private-key.asc` |
   | `GPG_PASSPHRASE` | Your GPG key passphrase |

### PyPI

1. **Create PyPI Account**
   - Register at https://pypi.org/account/register/

2. **Create API Token**
   - Go to https://pypi.org/manage/account/token/
   - Create token with scope "Entire account" or project-specific

3. **Add GitHub Secret**
   | Secret Name | Value |
   |-------------|-------|
   | `PYPI_API_TOKEN` | Your PyPI API token (starts with `pypi-`) |

---

## Publishing a Release

### Option 1: GitHub Release (Recommended)

1. **Update version numbers**
   ```bash
   # Scala version
   echo 'version := "0.7.0"' > version.sbt

   # Python version (in python/massivedatascience/__init__.py)
   # __version__ = "0.7.0"

   # Python version (in python/pyproject.toml)
   # version = "0.7.0"
   ```

2. **Commit and tag**
   ```bash
   git add -A
   git commit -m "Release v0.7.0"
   git tag v0.7.0
   git push origin master --tags
   ```

3. **Create GitHub Release**
   - Go to Releases → Draft a new release
   - Choose the tag `v0.7.0`
   - Add release notes
   - Click "Publish release"

4. **Automatic publishing**
   - The `publish.yml` workflow publishes to Maven Central
   - The `publish-pypi.yml` workflow publishes to PyPI

### Option 2: Manual Trigger

Go to Actions → Select workflow → Run workflow

---

## Verifying Publication

### Maven Central

After release (may take 10-30 minutes to sync):

```bash
# Check Maven Central
curl -s "https://search.maven.org/solrsearch/select?q=g:com.massivedatascience+AND+a:massivedatascience-clusterer*&rows=5&wt=json"
```

Users can then add:
```scala
// build.sbt
libraryDependencies += "com.massivedatascience" %% "massivedatascience-clusterer" % "0.7.0"
```

```xml
<!-- Maven -->
<dependency>
  <groupId>com.massivedatascience</groupId>
  <artifactId>massivedatascience-clusterer_2.13</artifactId>
  <version>0.7.0</version>
</dependency>
```

### PyPI

```bash
# Check PyPI
pip index versions massivedatascience-clusterer
```

Users can then install:
```bash
pip install massivedatascience-clusterer
```

---

## Troubleshooting

### Maven Central Issues

**"PGP signature verification failed"**
- Ensure GPG key is uploaded to keyserver
- Wait a few minutes for keyserver propagation
- Check `GPG_PRIVATE_KEY` secret is complete (including headers)

**"Missing required metadata"**
- Check `build.sbt` has all required POM fields:
  - `homepage`
  - `scmInfo`
  - `developers`
  - `description`
  - `licenses`

**"Unauthorized"**
- Verify `SONATYPE_USERNAME` and `SONATYPE_PASSWORD` secrets
- Ensure account has permission for `com.massivedatascience` namespace

### PyPI Issues

**"Invalid API token"**
- Regenerate token at https://pypi.org/manage/account/token/
- Ensure token starts with `pypi-`

**"Version already exists"**
- PyPI doesn't allow re-uploading same version
- Bump version number and try again

---

## Local Testing

### Test Maven Publishing Locally

```bash
# Publish to local Maven cache
sbt publishLocal

# Check local artifacts
ls ~/.ivy2/local/com.massivedatascience/
```

### Test PyPI Packaging Locally

```bash
cd python

# Build package
python -m build

# Check package
twine check dist/*

# Test install
pip install dist/massivedatascience_clusterer-0.7.0-py3-none-any.whl
```

---

## Release Checklist

- [ ] All tests pass (`sbt test`)
- [ ] Version updated in `version.sbt`
- [ ] Version updated in `python/massivedatascience/__init__.py`
- [ ] Version updated in `python/pyproject.toml`
- [ ] CHANGELOG.md updated
- [ ] Git tag created
- [ ] GitHub release created
- [ ] Maven Central publication verified
- [ ] PyPI publication verified
- [ ] Documentation updated
