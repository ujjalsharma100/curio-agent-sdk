# Publishing `curio-agent-sdk` (Python)

This document describes how to publish the Python SDK to [PyPI](https://pypi.org/). The published distribution name is **`curio-agent-sdk`** (see `[project]` → `name` in `pyproject.toml` in this directory). The repository folder may be named `curio-agent-sdk-python`; PyPI only cares about the package metadata in `pyproject.toml`.

## Prerequisites

- **Python** `>=3.11` (matches `requires-python` in `pyproject.toml`).
- A **[PyPI](https://pypi.org/) account** with permission to publish this project name (first upload claims the name; coordinate with your org if you use a shared account or org-owned projects).
- **Build tools** in a clean virtual environment (install when you need to ship):

  ```bash
  pip install build twine
  ```

- **API token** (recommended): In PyPI account settings, create a token scoped to this project (or the whole account for the first publish). You will use username `__token__` and the token as the password when uploading with Twine.

## What gets published

The standard workflow produces a **source distribution** (`.tar.gz`) and a **wheel** (`.whl`) under `dist/`:

- Package code from `src/curio_agent_sdk/` (see `[tool.setuptools.packages.find]` in `pyproject.toml`).
- `README.md` and license metadata as declared in `[project]`.
- Optional dependency extras (`openai`, `anthropic`, `all`, etc.) are metadata only; they are not bundled into the wheel beyond `requires` / `extras` in the built metadata.

Clean old artifacts before a new build:

```bash
rm -rf dist/ build/ src/*.egg-info
```

## Pre-publish checklist

1. **Branch and CI**: Merge intended changes; ensure CI (if any) is green.
2. **Version**: Bump **`version`** in `pyproject.toml` under `[project]` using [semver](https://semver.org/) (`patch` / `minor` / `major`). Keep the README version line in sync if you advertise it there (see the top of `README.md`).
3. **Quality gates** (typical for this repo):

   ```bash
   pip install -e ".[dev]"
   ruff check src tests
   black --check src tests
   isort --check src tests
   mypy src
   make test              # or: make test-cov
   ```

   Run `make test-live` only when you intend to validate against real APIs and have keys configured.

4. **Smoke the build locally**:

   ```bash
   python -m build
   ```

5. **Inspect the artifacts** (optional but useful):

   ```bash
   tar -tzf dist/curio_agent_sdk-*.tar.gz | head
   unzip -l dist/curio_agent_sdk-*.whl
   ```

   Built file names use the normalized distribution name (`curio_agent_sdk`); PyPI still lists the project as `curio-agent-sdk`.

## Publish commands

From the `curio-agent-sdk-python` directory (repo root of this package):

```bash
python -m build
twine check dist/*
twine upload dist/*
```

`twine check` validates metadata before upload. `twine upload` prompts for credentials unless you configure them (see below).

**Dry run (no upload to PyPI):**

- Build only: `python -m build` and inspect `dist/`.
- **TestPyPI**: upload to the test index first (see [Test PyPI](#test-pypi-optional)).

## Version bumps

There is no built-in equivalent to `npm version` in this repo. Common approaches:

1. **Manual**: Edit `version = "x.y.z"` in `[project]` in `pyproject.toml`, then commit.
2. **Automation**: Use a release tool or script your team prefers (`bump2version`, `tbump`, GitHub “Release” workflow, etc.), as long as `pyproject.toml` ends up with the correct `version`.

After tagging releases:

```bash
git tag v0.7.0
git push origin main --follow-tags
```

(Adjust branch name to match your default branch.)

## Test PyPI (optional)

Use [TestPyPI](https://test.pypi.org/) to verify uploads without affecting the real index:

1. Register on TestPyPI and create an API token there.
2. Build as usual, then:

   ```bash
   twine upload --repository testpypi dist/*
   ```

3. Install from TestPyPI (version and index URL as appropriate):

   ```bash
   pip install --index-url https://test.pypi.org/simple/ --extra-index-url https://pypi.org/simple/ curio-agent-sdk==<version>
   ```

The `--extra-index-url` line helps resolve dependencies that only exist on the main index.

## CI / automation

- Store **`PYPI_API_TOKEN`** (or the token string) in CI secrets.
- Configure Twine via environment variables (typical for GitHub Actions):

  ```bash
  TWINE_USERNAME=__token__
  TWINE_PASSWORD=${PYPI_API_TOKEN}
  twine upload dist/*
  ```

- Prefer **project-scoped or narrowly scoped API tokens**; rotate periodically.
- Build in CI with the same Python version you support (e.g. 3.11+), then `python -m build` and `twine upload`.

## After publish

- Confirm the release on `https://pypi.org/project/curio-agent-sdk/` (URL reflects the normalized project name).
- Verify install: `pip install curio-agent-sdk==<version>`.
- Publish release notes or a changelog if your process requires it.

## Troubleshooting

| Issue | What to check |
|--------|----------------|
| **403 / name already taken** | The PyPI project name is global; you need maintainer access on that project or a different `name` in `pyproject.toml`. |
| **Invalid or duplicate version** | That version may already exist on PyPI; versions are immutable—bump `version` and rebuild. |
| **Missing files in wheel** | Confirm `[tool.setuptools.packages.find]` and `include`; rebuild after changes. |
| **Upload works but `pip install` fails** | Check `requires-python` and dependency pins; test in a fresh venv. |
| **Optional extras not found** | Users install with `pip install curio-agent-sdk[openai]` etc.; extras are declared under `[project.optional-dependencies]`. |
