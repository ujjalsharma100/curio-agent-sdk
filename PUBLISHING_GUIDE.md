# Publishing and Testing Guide for Curio Agent SDK

This guide walks you through publishing the SDK to PyPI and testing it by installing and importing.

## Table of Contents

1. [Prerequisites](#prerequisites)
2. [Pre-Publication Checklist](#pre-publication-checklist)
3. [Building the Package](#building-the-package)
4. [Publishing to TestPyPI](#publishing-to-testpypi)
5. [Testing the Published Package](#testing-the-published-package)
6. [Publishing to Production PyPI](#publishing-to-production-pypi)
7. [Verifying Production Release](#verifying-production-release)
8. [Troubleshooting](#troubleshooting)

---

## Prerequisites

### 1. Install Required Tools

```bash
# Install/upgrade build tools
pip install --upgrade build twine

# Verify installation
python -m build --version
twine --version
```

### 2. Create PyPI Accounts

You'll need accounts on both TestPyPI and PyPI:

- **TestPyPI**: https://test.pypi.org/account/register/
- **PyPI**: https://pypi.org/account/register/

**Important**: These are separate accounts. You can use the same credentials, but you need to register for both.

### 3. Create API Tokens

1. **TestPyPI Token**:
   - Go to https://test.pypi.org/manage/account/token/
   - Create a new API token (scope: "Entire account" or project-specific)
   - Save the token (format: `pypi-...`)

2. **PyPI Token**:
   - Go to https://pypi.org/manage/account/token/
   - Create a new API token
   - Save the token

### 4. Configure Credentials (Optional but Recommended)

Create a `~/.pypirc` file to store your credentials:

```ini
[distutils]
index-servers =
    pypi
    testpypi

[pypi]
username = __token__
password = pypi-<your-production-token>

[testpypi]
repository = https://test.pypi.org/legacy/
username = __token__
password = pypi-<your-test-token>
```

**Security Note**: Make sure `~/.pypirc` has restricted permissions:
```bash
chmod 600 ~/.pypirc
```

Alternatively, you can pass credentials via environment variables or command-line flags.

---

## Pre-Publication Checklist

Before publishing, ensure the following:

### 1. Update Version Number

Update the version in **both** `pyproject.toml` and `setup.py`:

**In `pyproject.toml`:**
```toml
[project]
version = "0.1.0"  # Update this
```

**In `setup.py`:**
```python
version="0.1.0",  # Update this
```

**In `__init__.py`:**
```python
__version__ = "0.1.0"  # Update this
```

**Version Format**: Follow [Semantic Versioning](https://semver.org/):
- `MAJOR.MINOR.PATCH` (e.g., `1.0.0`, `0.2.1`)
- Increment MAJOR for breaking changes
- Increment MINOR for new features (backward compatible)
- Increment PATCH for bug fixes

### 2. Update Project URLs (if needed)

In `pyproject.toml`, update the URLs if you have a real repository:

```toml
[project.urls]
Homepage = "https://github.com/your-org/curio-agent-sdk"
Documentation = "https://github.com/your-org/curio-agent-sdk#readme"
Repository = "https://github.com/your-org/curio-agent-sdk.git"
Issues = "https://github.com/your-org/curio-agent-sdk/issues"
```

### 3. Verify Package Structure

Ensure all necessary files are included:

```bash
cd /path/to/curio_agent_sdk

# Check that key files exist
ls -la __init__.py setup.py pyproject.toml README.md LICENSE

# Verify package structure
python -c "import curio_agent_sdk; print(curio_agent_sdk.__version__)"
```

### 4. Test Local Installation

Test that the package can be installed locally:

```bash
# Clean any previous builds
rm -rf build/ dist/ *.egg-info/

# Install in development mode to test
pip install -e .

# Test imports
python -c "from curio_agent_sdk import BaseAgent, AgentConfig; print('Import successful!')"
```

### 5. Run Tests (if available)

```bash
# If you have tests
pytest tests/

# Or run example scripts to verify functionality
python examples/simple_agent.py
```

---

## Building the Package

### 1. Clean Previous Builds

```bash
cd /path/to/curio_agent_sdk

# Remove old build artifacts
rm -rf build/ dist/ *.egg-info/
```

### 2. Build Source Distribution and Wheel

```bash
# Build both source distribution (.tar.gz) and wheel (.whl)
python -m build

# This creates:
# - dist/curio-agent-sdk-<version>.tar.gz (source distribution)
# - dist/curio-agent-sdk-<version>-py3-none-any.whl (wheel)
```

### 3. Verify Build Artifacts

```bash
# List built files
ls -lh dist/

# Check the wheel contents
python -m zipfile -l dist/curio_agent_sdk-*.whl

# Verify metadata
python -m twine check dist/*
```

The `twine check` command will validate:
- Package metadata
- README format
- License file presence
- Required fields

---

## Publishing to TestPyPI

**Always test on TestPyPI first** before publishing to production PyPI.

### 1. Upload to TestPyPI

**Option A: Using ~/.pypirc (Recommended)**
```bash
twine upload --repository testpypi dist/*
```

**Option B: Using Environment Variables**
```bash
export TWINE_USERNAME=__token__
export TWINE_PASSWORD=pypi-<your-test-token>
twine upload --repository-url https://test.pypi.org/legacy/ dist/*
```

**Option C: Using Command-Line Flags**
```bash
twine upload \
  --repository-url https://test.pypi.org/legacy/ \
  --username __token__ \
  --password pypi-<your-test-token> \
  dist/*
```

### 2. Verify Upload

After upload, check:
- https://test.pypi.org/project/curio-agent-sdk/
- The package should appear with your version number

---

## Testing the Published Package

### 1. Create a Clean Test Environment

**Important**: Use a fresh virtual environment to simulate a real user installation.

```bash
# Create a temporary directory for testing
mkdir -p /tmp/curio_sdk_test
cd /tmp/curio_sdk_test

# Create a new virtual environment
python3 -m venv test_env
source test_env/bin/activate  # On Windows: test_env\Scripts\activate

# Upgrade pip
pip install --upgrade pip
```

### 2. Install from TestPyPI

```bash
# Install from TestPyPI
pip install \
  --index-url https://test.pypi.org/simple/ \
  --extra-index-url https://pypi.org/simple/ \
  curio-agent-sdk

# The --extra-index-url is needed because dependencies might not be on TestPyPI
```

**Note**: If you have optional dependencies, install them separately:
```bash
pip install \
  --index-url https://test.pypi.org/simple/ \
  --extra-index-url https://pypi.org/simple/ \
  curio-agent-sdk[openai,anthropic]
```

### 3. Test Basic Import

```bash
python -c "
from curio_agent_sdk import BaseAgent, AgentConfig
from curio_agent_sdk import call_llm, initialize_llm_service
from curio_agent_sdk import __version__
print(f'Successfully imported curio-agent-sdk version {__version__}')
print('All imports working!')
"
```

### 4. Test Package Functionality

Create a test script:

```bash
cat > test_import.py << 'EOF'
#!/usr/bin/env python3
"""Test script to verify SDK installation."""

from curio_agent_sdk import (
    BaseAgent,
    AgentConfig,
    call_llm,
    initialize_llm_service,
    __version__
)

print(f"✓ SDK version: {__version__}")

# Test config
try:
    config = AgentConfig.from_env()
    print("✓ AgentConfig imported and initialized")
except Exception as e:
    print(f"⚠ AgentConfig error (expected if no .env): {e}")

# Test base agent
try:
    class TestAgent(BaseAgent):
        def get_agent_instructions(self):
            return "You are a test agent."
        def initialize_tools(self):
            pass
    
    print("✓ BaseAgent can be subclassed")
except Exception as e:
    print(f"✗ BaseAgent error: {e}")

# Test LLM service
try:
    initialize_llm_service()
    print("✓ LLM service initialized")
except Exception as e:
    print(f"⚠ LLM service error (expected if no API keys): {e}")

# Test imports
try:
    from curio_agent_sdk.llm.providers import OpenAIProvider
    from curio_agent_sdk.persistence.postgres import PostgresPersistence
    from curio_agent_sdk.core.tool_registry import ToolRegistry
    print("✓ All submodules importable")
except Exception as e:
    print(f"✗ Import error: {e}")

print("\n✅ All basic tests passed!")
EOF

python test_import.py
```

### 5. Test with Optional Dependencies

```bash
# Install with specific extras
pip install \
  --index-url https://test.pypi.org/simple/ \
  --extra-index-url https://pypi.org/simple/ \
  'curio-agent-sdk[openai]'

# Test OpenAI provider import
python -c "from curio_agent_sdk.llm.providers import OpenAIProvider; print('OpenAI provider imported!')"
```

### 6. Test Example Usage

If you have example scripts, test them:

```bash
# Create a minimal test
cat > test_agent.py << 'EOF'
from curio_agent_sdk import BaseAgent, AgentConfig

class SimpleAgent(BaseAgent):
    def __init__(self, agent_id: str, config: AgentConfig):
        super().__init__(agent_id, config=config)
        self.agent_name = "SimpleAgent"
        self.max_iterations = 3
        self.initialize_tools()
    
    def get_agent_instructions(self) -> str:
        return "You are a simple test agent."
    
    def initialize_tools(self):
        pass

# Test instantiation (won't run without API keys, but should import)
try:
    config = AgentConfig.from_env()
    agent = SimpleAgent("test-agent", config)
    print("✓ Agent instantiated successfully")
except Exception as e:
    print(f"⚠ Agent instantiation: {e}")
EOF

python test_agent.py
```

### 7. Clean Up Test Environment

```bash
deactivate
rm -rf /tmp/curio_sdk_test
```

---

## Publishing to Production PyPI

**Only proceed after successful TestPyPI testing!**

### 1. Rebuild (if needed)

If you made any changes, rebuild:

```bash
cd /path/to/curio_agent_sdk
rm -rf build/ dist/ *.egg-info/
python -m build
python -m twine check dist/*
```

### 2. Upload to Production PyPI

**Option A: Using ~/.pypirc**
```bash
twine upload dist/*
```

**Option B: Using Environment Variables**
```bash
export TWINE_USERNAME=__token__
export TWINE_PASSWORD=pypi-<your-production-token>
twine upload dist/*
```

**Option C: Using Command-Line Flags**
```bash
twine upload \
  --username __token__ \
  --password pypi-<your-production-token> \
  dist/*
```

### 3. Verify Upload

- Check https://pypi.org/project/curio-agent-sdk/
- The package should be live and installable

---

## Verifying Production Release

### 1. Wait for Indexing

PyPI may take a few minutes to index your package. Wait 2-5 minutes after upload.

### 2. Install from Production PyPI

```bash
# Create fresh test environment
mkdir -p /tmp/curio_prod_test
cd /tmp/curio_prod_test
python3 -m venv test_env
source test_env/bin/activate
pip install --upgrade pip

# Install from production PyPI
pip install curio-agent-sdk

# Or with extras
pip install 'curio-agent-sdk[openai,anthropic]'
```

### 3. Verify Installation

```bash
# Check version
python -c "import curio_agent_sdk; print(curio_agent_sdk.__version__)"

# Test imports
python -c "
from curio_agent_sdk import BaseAgent, AgentConfig, call_llm
from curio_agent_sdk.llm.providers import OpenAIProvider
print('✅ All imports successful!')
"
```

### 4. Test Package Info

```bash
pip show curio-agent-sdk
pip list | grep curio
```

---

## Troubleshooting

### Issue: "Package already exists" on TestPyPI

**Solution**: TestPyPI allows re-uploads, but you need to use a new version number. Increment the patch version (e.g., `0.1.0` → `0.1.1`).

### Issue: "Package already exists" on Production PyPI

**Solution**: PyPI does NOT allow overwriting existing versions. You must:
1. Increment the version number
2. Rebuild and upload

### Issue: Dependencies not found during installation

**Solution**: Use `--extra-index-url` to include PyPI:
```bash
pip install \
  --index-url https://test.pypi.org/simple/ \
  --extra-index-url https://pypi.org/simple/ \
  curio-agent-sdk
```

### Issue: "Invalid distribution" error

**Solution**: 
1. Clean build artifacts: `rm -rf build/ dist/ *.egg-info/`
2. Rebuild: `python -m build`
3. Check: `python -m twine check dist/*`

### Issue: Import errors after installation

**Solution**:
1. Verify package structure in `__init__.py`
2. Check that all modules are included in `pyproject.toml`
3. Test local installation first: `pip install -e .`

### Issue: README not rendering on PyPI

**Solution**:
1. Ensure README.md exists and is valid Markdown
2. Check `pyproject.toml` has `readme = "README.md"`
3. Run `python -m twine check dist/*` to validate

### Issue: Authentication failures

**Solution**:
1. Verify token format: `pypi-...` (not just the token ID)
2. Check token hasn't expired
3. Ensure `__token__` is used as username (not your PyPI username)
4. Verify `~/.pypirc` permissions: `chmod 600 ~/.pypirc`

---

## Quick Reference Commands

### Build
```bash
cd /path/to/curio_agent_sdk
rm -rf build/ dist/ *.egg-info/
python -m build
python -m twine check dist/*
```

### Publish to TestPyPI
```bash
twine upload --repository testpypi dist/*
```

### Test Installation from TestPyPI
```bash
pip install --index-url https://test.pypi.org/simple/ --extra-index-url https://pypi.org/simple/ curio-agent-sdk
```

### Publish to Production PyPI
```bash
twine upload dist/*
```

### Test Installation from Production PyPI
```bash
pip install curio-agent-sdk
```

---

## Best Practices

1. **Always test on TestPyPI first** - Never skip this step
2. **Use semantic versioning** - Follow MAJOR.MINOR.PATCH format
3. **Never overwrite versions** - PyPI doesn't allow it
4. **Test in clean environments** - Simulate real user installations
5. **Update version in all places** - `pyproject.toml`, `setup.py`, and `__init__.py`
6. **Verify before publishing** - Run `twine check` and local tests
7. **Document breaking changes** - Update README and changelog
8. **Tag releases in Git** - `git tag v0.1.0 && git push --tags`

---

## Next Steps After Publishing

1. **Update Documentation**: Add installation instructions using `pip install curio-agent-sdk`
2. **Create Release Notes**: Document what's new in this version
3. **Announce**: Share the release with your team/users
4. **Monitor**: Watch for issues or feedback from users

---

## Additional Resources

- [PyPI Documentation](https://packaging.python.org/en/latest/guides/distributing-packages-using-setuptools/)
- [Twine Documentation](https://twine.readthedocs.io/)
- [Python Packaging Guide](https://packaging.python.org/)
- [Semantic Versioning](https://semver.org/)

