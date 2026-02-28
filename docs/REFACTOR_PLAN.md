# Staged Refactor Plan: src layout + cleaner structure

This document breaks the refactor into **stages** that can be executed one at a time. Each stage leaves the project in a working state (imports and tests can be verified).

**Target state:**
- **src layout** — package lives under `src/curio_agent_sdk/`
- **Foundation packages** — `base`, `credentials`, `resilience` at top level (moved out of core)
- **Core subpackages** — `core/` split into `agent`, `state`, `context`, `events`, `extensions`, `workflow`, `security`, `loops`, `tools`; **llm** moved inside core as `core/llm/`
- **CLI** — top-level `cli/` (moved out of core)
- **Audit** — under `persistence/audit_hooks.py` (moved out of core)
- **pyproject.toml** — updated for `src` layout and full package list

---

## Stage 0: Pre-flight (no file moves) ✅ COMPLETED

**Goal:** Make the refactor safe and repeatable.

1. **Ensure tests exist and pass**
   - If you have a `tests/` directory or pytest config, run: `pytest` (or `python -m pytest`)
   - If tests live inside the repo, document how to run them (e.g. `pytest testing/` or via a script)
   - Add a brief note in this plan or in README for "how to verify after each stage"

2. **Branch**
   - Create a refactor branch, e.g. `git checkout -b refactor/src-and-structure`

3. **Optional: snapshot current imports**
   - Run a quick grep for `from curio_agent_sdk` across the codebase and save the list (or rely on tests to catch breakage)

**Deliverable:** Green tests on current layout; refactor branch ready.

---

### How to verify after each stage

- **Install (editable):** From repo root, run `pip install -e ".[dev]"` (or `pip install -e .`) so the package is importable. After Stage 1 the root will be the parent of `src/`.
- **Smoke import:** `python3 -c "from curio_agent_sdk import Agent, LLMClient, Component; print('OK')"`. Run from repo root (or with `PYTHONPATH` set to the directory that contains the `curio_agent_sdk` package).
- **Tests:** There is no `tests/` directory or `test_*.py` suite in this repo; the `testing/` package provides test utilities (mocks, harness). If you add tests later, run `pytest` after each stage.
- **Stage 0 note:** No pytest suite present. Branch `refactor/src-and-structure` created. Full smoke import requires dependencies (e.g. `httpx`); use `pip install -e .` before running the smoke command.

### Import snapshot (Stage 0)

For refactor stages that change module paths, the main import dependencies are:

- **core.component** → used by: `core/runtime.py`, `core/state_store.py`, `core/event_bus.py`, `llm/client.py`, `memory/manager.py`, `memory/vector.py`, `memory/file_memory.py`, `persistence/base.py`, `connectors/bridge.py`, `mcp/bridge.py`
- **core.credentials** → used by: `llm/router.py`, `connectors/base.py`, `connectors/bridge.py`
- **core.circuit_breaker** → used by: `llm/router.py`, `mcp/bridge.py`, `connectors/bridge.py`
- **core.audit** → re-exported from root; callers use `register_audit_hooks` (move to persistence in Stage 3)
- **core.cli** → re-exported from root as `AgentCLI` (move to top-level `cli/` in Stage 4)
- **llm** (top-level) → used by: `core/agent.py`, `core/builder.py`, `core/context.py`, `core/runtime.py`, root `__init__.py` (move to `core/llm/` in Stage 5)

---

## Stage 1: Move to src layout (only) ✅ COMPLETED

**Goal:** Package lives under `src/curio_agent_sdk/` with no other structural changes. All existing modules stay where they are relative to the package root.

**Steps:**

1. **Create directory**
   - `mkdir -p src/curio_agent_sdk`

2. **Move the entire package into src**
   - Move every file and directory that is part of the package into `src/curio_agent_sdk/`:
     - `__init__.py`, `config/`, `connectors/`, `core/`, `llm/`, `memory/`, `middleware/`, `models/`, `mcp/`, `persistence/`, `testing/`, `tools/`
   - Leave at repo root: `pyproject.toml`, `README.md`, `LICENSE`, `.gitignore`, `docs/`, etc.

3. **Update pyproject.toml**
   - Remove `package-dir = {"curio_agent_sdk" = "."}` (or set it for src): `package-dir = {"curio_agent_sdk" = "src"}`
   - Alternatively use:
     ```toml
     [tool.setuptools.packages.find]
     where = ["src"]
     ```
   - Ensure `packages` lists all subpackages, or use `packages = find:` with `where = ["src"]`.

4. **Verify**
   - Install in editable mode: `pip install -e .`
   - Run tests: `pytest` (or your test command)
   - Quick smoke: `python -c "from curio_agent_sdk import Agent; print('ok')"`

**Deliverable:** Same code and structure, but under `src/curio_agent_sdk/`. Tests pass. No import path changes for users (still `curio_agent_sdk.*`).

**Stage 1 note:** Package moved to `src/curio_agent_sdk/`. `pyproject.toml` updated with `[tool.setuptools.packages.find] where = ["src"]`. Smoke import verified (`from curio_agent_sdk import Agent, LLMClient, Component`). Root `__init__.py` now re-exports `Component` for the public API.

---

## Stage 2: Extract foundation packages (base, credentials, resilience) ✅ COMPLETED

**Goal:** Move `Component` → `base/`, `credentials` → `credentials/`, `circuit_breaker` → `resilience/`. Update all imports.

**Steps:**

1. **Create new packages under `src/curio_agent_sdk/`**
   - `base/__init__.py` + `base/component.py` (content from `core/component.py`)
   - `credentials/__init__.py` + `credentials/credentials.py` (content from `core/credentials.py`)
   - `resilience/__init__.py` + `resilience/circuit_breaker.py` (content from `core/circuit_breaker.py`)

2. **Update imports everywhere**
   - Replace `curio_agent_sdk.core.component` → `curio_agent_sdk.base.component` (or `curio_agent_sdk.base`)
   - Replace `curio_agent_sdk.core.credentials` → `curio_agent_sdk.credentials`
   - Replace `curio_agent_sdk.core.circuit_breaker` → `curio_agent_sdk.resilience.circuit_breaker` (or `curio_agent_sdk.resilience`)
   - Files to touch (from earlier analysis): `core/agent.py`, `core/runtime.py`, `core/state_store.py`, `core/event_bus.py`, `llm/client.py`, `llm/router.py`, `memory/manager.py`, `memory/vector.py`, `memory/file_memory.py`, `persistence/base.py`, `connectors/base.py`, `connectors/bridge.py`, `mcp/bridge.py`, `core/__init__.py`, root `__init__.py`

3. **Re-exports**
   - In `core/__init__.py`: remove `Component`; re-export from base if you want `from curio_agent_sdk.core import Component` to still work: `from curio_agent_sdk.base import Component` (optional).
   - In root `__init__.py`: import `Component` from `curio_agent_sdk.base`, and credentials/circuit breaker from new locations; keep `__all__` unchanged so public API is stable.

4. **Delete from core**
   - Remove `core/component.py`, `core/credentials.py`, `core/circuit_breaker.py`

5. **pyproject.toml**
   - Add to `packages`: `curio_agent_sdk.base`, `curio_agent_sdk.credentials`, `curio_agent_sdk.resilience` (if using explicit list).

6. **Verify**
   - `pip install -e .` and run tests; smoke import.

**Deliverable:** `base`, `credentials`, `resilience` exist at top level; core no longer contains them; all imports updated; tests pass.

**Stage 2 note:** Created `base/`, `credentials/`, `resilience/` under `src/curio_agent_sdk/`; moved `core/component.py` → `base/component.py`, `core/credentials.py` → `credentials/credentials.py`, `core/circuit_breaker.py` → `resilience/circuit_breaker.py`. Added package `__init__.py` re-exports. Updated all imports; `core/__init__.py` and root `__init__.py` re-export from new locations. Smoke import verified (`from curio_agent_sdk import Agent, LLMClient, Component, CircuitBreaker, CircuitState`).

---

## Stage 3: Move audit into persistence

**Goal:** `core/audit.py` → `persistence/audit_hooks.py`; persistence owns “wire hooks to persistence.”

**Steps:**

1. **Create `persistence/audit_hooks.py`**
   - Move content from `core/audit.py` into `persistence/audit_hooks.py`.
   - Update internal imports: `curio_agent_sdk.core.hooks` stays (audit_hooks will import from core.hooks); `curio_agent_sdk.persistence.base` is already correct.

2. **Update `persistence/__init__.py`**
   - Export `register_audit_hooks` (and any other public names from the old audit module).

3. **Update callers**
   - Any code that did `from curio_agent_sdk.core.audit import register_audit_hooks` should use `from curio_agent_sdk.persistence.audit_hooks import register_audit_hooks` (or re-export from `persistence/__init__.py`).

4. **Root and core re-exports**
   - Root `__init__.py`: if you currently re-export audit, change to import from `curio_agent_sdk.persistence` (e.g. `persistence.audit_hooks` or `persistence`).
   - Remove audit from `core/__init__.py`.

5. **Delete `core/audit.py`**

6. **Verify**
   - Tests and smoke import.

**Deliverable:** Audit lives under persistence; core no longer has audit; tests pass.

---

## Stage 4: Move CLI out of core to top-level cli/

**Goal:** `core/cli.py` → top-level `cli/cli.py` (or `cli/__init__.py` exposing `AgentCLI`).

**Steps:**

1. **Create `cli/` under `src/curio_agent_sdk/`**
   - `cli/__init__.py` — export `AgentCLI`
   - `cli/cli.py` — move content from `core/cli.py`

2. **Update imports in `cli/cli.py`**
   - Change `core.agent` → same (agent stays in core for now) or `core.agent.agent` when you restructure core in Stage 6.
   - Change `core.session` → same or `core.state.session` later.

3. **Re-exports**
   - Root `__init__.py`: import `AgentCLI` from `curio_agent_sdk.cli`.
   - Remove CLI from `core/__init__.py`.

4. **Delete `core/cli.py`**

5. **pyproject.toml**
   - Add `curio_agent_sdk.cli` to packages.

6. **Verify**
   - Tests and smoke: `from curio_agent_sdk import AgentCLI`.

**Deliverable:** CLI is at top-level `cli/`; core no longer contains CLI; tests pass.

---

## Stage 5: Move llm into core

**Goal:** Top-level `llm/` → `core/llm/` (including `core/llm/providers/`).

**Steps:**

1. **Create `core/llm/`**
   - Move entire `llm/` directory contents into `core/llm/` (so you have `core/llm/__init__.py`, `core/llm/client.py`, `core/llm/providers/`, etc.).

2. **Update all imports from `curio_agent_sdk.llm` → `curio_agent_sdk.core.llm`**
   - In core: `core/agent.py`, `core/builder.py`, `core/runtime.py`, `core/context.py` (if it uses token_counter), etc.
   - In middleware, testing, or any other internal code that imports from `curio_agent_sdk.llm`.
   - In root `__init__.py`: change to import from `curio_agent_sdk.core.llm` and re-export the same names so public API stays `from curio_agent_sdk import LLMClient`, etc.

3. **Update internal imports inside the moved llm code**
   - `core/llm/client.py`: may import `curio_agent_sdk.core.component` → should already be `curio_agent_sdk.base.component` after Stage 2.
   - `core/llm/router.py`: `curio_agent_sdk.core.circuit_breaker` / `curio_agent_sdk.core.credentials` → `curio_agent_sdk.resilience`, `curio_agent_sdk.credentials`.

4. **Remove top-level `llm/` directory** (it’s now under core).

5. **pyproject.toml**
   - Remove `curio_agent_sdk.llm` and `curio_agent_sdk.llm.providers`; add `curio_agent_sdk.core.llm` and `curio_agent_sdk.core.llm.providers`.

6. **Verify**
   - Tests and smoke: `from curio_agent_sdk import LLMClient`, `from curio_agent_sdk import Agent`.

**Deliverable:** LLM lives under `core/llm/`; root re-exports unchanged; tests pass.

---

## Stage 6: Restructure core into subpackages

**Goal:** Split the remaining flat modules in `core/` into subpackages: `agent`, `state`, `context`, `events`, `extensions`, `workflow`, `security`, and keep `loops` and `tools` as they are.

**Mapping:**

| Current (core/) | New (core/) |
|-----------------|-------------|
| `agent.py`, `builder.py`, `runtime.py` | `agent/agent.py`, `agent/builder.py`, `agent/runtime.py` |
| `state.py`, `state_store.py`, `checkpoint.py`, `session.py` | `state/state.py`, `state/state_store.py`, `state/checkpoint.py`, `state/session.py` |
| `context.py`, `instructions.py` | `context/context.py`, `context/instructions.py` |
| `hooks.py`, `event_bus.py` | `events/hooks.py`, `events/event_bus.py` |
| `plugins.py`, `skills.py`, `subagent.py` | `extensions/plugins.py`, `extensions/skills.py`, `extensions/subagent.py` |
| `plan_mode.py`, `task_manager.py`, `structured_output.py` | `workflow/plan_mode.py`, `workflow/task_manager.py`, `workflow/structured_output.py` |
| `permissions.py`, `human_input.py` | `security/permissions.py`, `security/human_input.py` |
| `loops/` | `loops/` (unchanged) |
| `tools/` | `tools/` (unchanged) |

**Steps:**

1. **Create subpackage dirs and move files**
   - Create `core/agent/`, `core/state/`, `core/context/`, `core/events/`, `core/extensions/`, `core/workflow/`, `core/security/`.
   - Move files as in the table above (one subpackage at a time to avoid mistakes).

2. **Add `__init__.py` per subpackage**
   - Each subpackage’s `__init__.py` should re-export the public names from its modules (so `from curio_agent_sdk.core.agent import Agent` still works, and so do internal cross-references).

3. **Update internal imports within core**
   - Replace `curio_agent_sdk.core.agent` → `curio_agent_sdk.core.agent.agent` (or from `core.agent` import Agent, etc.) — but the preferred approach is: keep importing from `curio_agent_sdk.core.X` by re-exporting from subpackage `__init__.py`. So `core/__init__.py` should do:
     - `from curio_agent_sdk.core.agent import Agent, AgentBuilder, Runtime`
     - etc.
   - Within core, use relative imports where possible (e.g. in `core/agent/runtime.py`: `from curio_agent_sdk.core.state import AgentState` or `from ..state import AgentState`).

4. **Update imports from outside core**
   - Any file outside core that imports from `curio_agent_sdk.core.agent` (e.g. `core.agent` for the Agent class) should still work if `core/__init__.py` re-exports everything. So no change needed for external code if you re-export fully from `core/__init__.py`.
   - Internal core files: after moves, update to use new paths (e.g. `from curio_agent_sdk.core.state import AgentState` or relative `from ..state import AgentState`).

5. **Dependency order within core**
   - `agent` depends on: state, context, events (hooks), extensions (skills), llm, loops, tools, runtime.
   - `state` depends on: models, core.tools (Tool).
   - `context` depends on: core.llm (token_counter), models.
   - `events` depends on: base (Component for EventBusBridge).
   - etc. After moving, fix any circular imports by using TYPE_CHECKING or lazy imports if needed.

6. **pyproject.toml**
   - Add: `curio_agent_sdk.core.agent`, `curio_agent_sdk.core.state`, `curio_agent_sdk.core.context`, `curio_agent_sdk.core.events`, `curio_agent_sdk.core.extensions`, `curio_agent_sdk.core.workflow`, `curio_agent_sdk.core.security`.

7. **Verify**
   - Full test run; smoke imports for Agent, Runtime, hooks, session, permissions, etc.

**Deliverable:** Core is organized into subpackages; no more flat modules in core (except what’s inside subpackages); public API unchanged; tests pass.

---

## Stage 7: Finalize pyproject.toml and docs

**Goal:** Correct package list, optional src layout tweaks, and update README/docs if needed.

**Steps:**

1. **pyproject.toml**
   - Use `[tool.setuptools.packages.find] where = ["src"]` and either `include = ["curio_agent_sdk*"]` or list every package explicitly (as in the plan intro).
   - Ensure all of these are discoverable or listed: `curio_agent_sdk`, `base`, `credentials`, `resilience`, `config`, `cli`, `core` (and core subpackages: agent, state, context, events, extensions, workflow, security, cli removed from core, loops, tools, llm), `memory`, `middleware`, `models`, `persistence`, `connectors`, `mcp`, `tools`, `testing`.

2. **README**
   - If any import paths were documented (e.g. “import from curio_agent_sdk.llm”), update to “import from curio_agent_sdk” (re-exports) or “curio_agent_sdk.core.llm” if you document internal structure.
   - Update “Project structure” or “Directory layout” if you have one, to reflect src layout and new structure.

3. **.gitignore**
   - Add `src/curio_agent_sdk/**/__pycache__` if you want to be explicit; usually `**/__pycache__` is enough.

4. **Optional: tests directory**
   - If you want a separate `tests/` at repo root (outside src), create `tests/`, move or add test files there, and run them with `pytest tests/`. The in-package `testing/` can remain for test *utilities* (mocks, harnesses); actual tests can live under `tests/`.

**Deliverable:** pyproject.toml and docs match the new layout; tests and install work; README accurate.

---

## Summary: stage order and dependencies

| Stage | Description                    | Depends on |
|-------|--------------------------------|------------|
| 0     | Pre-flight, branch, tests green | —          |
| 1     | src layout only               | 0          |
| 2     | base, credentials, resilience  | 1          |
| 3     | audit → persistence           | 1          |
| 4     | CLI → top-level cli/          | 1          |
| 5     | llm → core/llm                | 2          |
| 6     | core subpackages              | 2, 3, 4, 5 |
| 7     | pyproject + docs              | 6          |

Stages 2, 3, and 4 can be done in parallel after Stage 1 (they don’t depend on each other). Stage 5 should follow Stage 2 (llm/router uses resilience and credentials). Stage 6 should be last among code moves, then Stage 7.

---

## Optional: single-commit vs multi-commit

- **One commit per stage:** Keeps history clear and allows reverting by stage.
- **Single large commit:** Simpler if you prefer one “refactor” commit; run all stages locally and commit once after Stage 7.

Recommendation: one commit per stage so that “move to src” and “extract base/credentials/resilience” are easy to review and revert if needed.

---

## Final directory layout (reference)

```
project_root/
├── src/
│   └── curio_agent_sdk/
│       ├── __init__.py
│       ├── base/
│       │   ├── __init__.py
│       │   └── component.py
│       ├── credentials/
│       │   ├── __init__.py
│       │   └── credentials.py
│       ├── resilience/
│       │   ├── __init__.py
│       │   └── circuit_breaker.py
│       ├── models/
│       ├── config/
│       ├── cli/
│       │   ├── __init__.py
│       │   └── cli.py
│       ├── core/
│       │   ├── __init__.py
│       │   ├── agent/
│       │   ├── llm/
│       │   │   └── providers/
│       │   ├── state/
│       │   ├── context/
│       │   ├── events/
│       │   ├── extensions/
│       │   ├── workflow/
│       │   ├── security/
│       │   ├── loops/
│       │   └── tools/
│       ├── memory/
│       ├── middleware/
│       ├── persistence/
│       │   └── audit_hooks.py
│       ├── connectors/
│       ├── mcp/
│       ├── tools/
│       └── testing/
├── docs/
│   └── REFACTOR_PLAN.md
├── pyproject.toml
├── README.md
├── LICENSE
└── .gitignore
```
