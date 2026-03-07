# Curio Agent SDK — Improvement Scope

> Gaps and enhancements identified while planning the Curio Code CLI tool.
> These are things that would ideally be in the SDK but currently require workarounds
> or application-level implementation. **None of these block the coding tool** — they
> are improvements that would make the SDK more capable for this class of application.

---

## Critical Improvements (High Impact)

### 1. Google / Gemini Provider
**Current state**: SDK has providers for OpenAI, Anthropic, Groq, Ollama.
**Gap**: No Google Gemini provider. Gemini 2.5 Pro/Flash are competitive models used by many coding tools.
**Workaround**: Use OpenAI-compatible endpoint (Google supports this) or implement as custom provider at application level.
**Recommendation**: Add `GeminiProvider` to `core/llm/providers/`.

### 2. OpenRouter / Generic OpenAI-Compatible Provider
**Current state**: Providers are hardcoded (OpenAI, Anthropic, Groq, Ollama).
**Gap**: No way to add a custom provider that speaks OpenAI-compatible API without modifying SDK source.
**Workaround**: Use `OpenAIProvider` with custom `base_url`, but model discovery and routing don't know about it.
**Recommendation**: Add a `GenericOpenAIProvider` or make `OpenAIProvider` more configurable for custom endpoints. Allow registering custom providers via `LLMClient.register_provider()`.

### 3. Streaming + Tool Calling Interleave
**Current state**: `astream()` emits `StreamEvent`s but the exact interleave behavior during tool execution is not well-documented.
**Gap**: For a coding CLI, we need fine-grained streaming: text tokens interleaved with tool-call-start/tool-call-end events, progress updates during tool execution, and the ability to render each incrementally.
**Workaround**: Hook into `tool.call.before`/`tool.call.after` events alongside stream events.
**Recommendation**: Ensure `astream()` emits events for every phase transition, including `tool_execution_progress` for long-running tools.

### 4. Extended Thinking / Reasoning Tokens
**Current state**: No explicit support for extended thinking (Anthropic's `thinking` blocks, OpenAI's o1 reasoning).
**Gap**: Advanced models expose thinking/reasoning tokens that should be streamed separately from response text.
**Workaround**: Parse provider-specific response fields at application level.
**Recommendation**: Add `thinking` field to `LLMResponse` and `thinking` stream event type. Detect and route based on model capabilities.

### 5. CLI Harness Enhancements
**Current state**: `AgentCLI` provides basic REPL with slash commands.
**Gap**: The CLI is minimal — no rich rendering, no markdown, no syntax highlighting, no input history, no multi-line editing. Not sufficient for a production coding tool.
**Workaround**: Build custom CLI layer using ink/blessed on top of the SDK's agent (bypass `AgentCLI` entirely, use `agent.astream()` directly).
**Recommendation**: Either enhance `AgentCLI` significantly or explicitly document it as a reference/development tool, not for production CLIs. The SDK's strength is in the agent layer, not the CLI — and that's okay.

---

## Important Improvements (Medium Impact)

### 6. Tool Result Streaming
**Current state**: Tool results are returned as complete strings.
**Gap**: For long-running tools (Bash commands, web fetch), we want to stream partial results back to the UI.
**Workaround**: Use hooks (`tool.call.after`) and implement streaming at the tool level with callbacks.
**Recommendation**: Add `ToolResultStream` — tools can optionally yield partial results via an async generator.

### 7. Cancellation / Abort Support
**Current state**: No explicit cancellation mechanism for in-progress runs.
**Gap**: User presses Ctrl+C → agent should stop current LLM call, abort current tool, and return to prompt.
**Workaround**: Use `AbortController` / `AbortSignal` at application level, pass into LLM calls and tool executions.
**Recommendation**: Add `AbortSignal` support to `agent.arun()` / `agent.astream()` and propagate to LLM client and tool executor. Add `cancel()` method to `Runtime`.

### 8. Diff/Patch Tool Support
**Current state**: Built-in tools include `file_read`, `file_write` but no diff/edit tool.
**Gap**: The most-used tool in coding agents is an "edit" tool that does exact string replacement with uniqueness validation. This is complex enough that SDK-level support would benefit all consumers.
**Workaround**: Implement as custom tool at application level.
**Recommendation**: Add `file_edit` to built-in tools with old_string/new_string replacement, uniqueness validation, and diff output.

### 9. Project-Aware Instruction Loading
**Current state**: `InstructionLoader` loads from hardcoded paths (`AGENT.md`, `.agent/rules.md`).
**Gap**: Different coding tools use different filenames (`CLAUDE.md`, `CURIO.md`, `.cursorrules`). The loader should be configurable.
**Workaround**: Implement custom instruction loading at application level.
**Recommendation**: Make `InstructionLoader` accept configurable filenames and search patterns. Support `include` directives.

### 10. Image/Vision Content in Messages
**Current state**: Message content is string-based.
**Gap**: Modern LLMs accept image inputs. Messages need to support multi-modal content (text + image URLs/base64).
**Workaround**: Construct provider-specific message formats at application level.
**Recommendation**: Add `ContentPart` type (text, image_url, image_base64) to message content. Have providers convert to their native format.

### 11. Token Budget in Agent Builder
**Current state**: `ContextManager` exists but isn't directly wirable from `AgentBuilder`.
**Gap**: Should be a first-class builder option: `.contextWindow(200000).reserveTokens(8192)`.
**Workaround**: Create `ContextManager` separately and inject via hooks.
**Recommendation**: Add `.contextManager()` or `.contextWindow()` to `AgentBuilder`.

---

## Nice-to-Have Improvements (Lower Impact)

### 12. Background Task Management
**Current state**: Subagents can run in background but there's no task management (list, cancel, get status).
**Gap**: Coding tools need to manage background operations (running tests, watching builds).
**Workaround**: Implement task management at application level with `child_process`.
**Recommendation**: Add `BackgroundTaskManager` to SDK for managing long-running child processes.

### 13. File Watcher Integration
**Current state**: No file watching capability.
**Gap**: Useful for auto-reloading context when files change (e.g., test results, build output).
**Workaround**: Use `chokidar` or `fs.watch` at application level.
**Recommendation**: Add optional `FileWatcher` component that emits events through the SDK's event system.

### 14. Git Integration Primitives
**Current state**: No git-specific tools or utilities.
**Gap**: Git operations (status, diff, log, commit) are the most common operations in coding tools. Having SDK-level git utilities would benefit all consumers.
**Workaround**: Implement via Bash tool or custom tools.
**Recommendation**: Add `GitToolkit` to `tools/` with common git operations as pre-built tools.

### 15. Rate Limit Awareness Per Provider
**Current state**: `RateLimitMiddleware` does per-agent rate limiting but doesn't know about provider-specific limits.
**Gap**: Anthropic, OpenAI, etc. have different rate limits. The SDK should track and respect them.
**Workaround**: Handle 429 errors reactively.
**Recommendation**: Add provider-specific rate limit tracking that reads headers (`X-RateLimit-*`, `Retry-After`) and proactively throttles.

### 16. Cost Estimation Before Execution
**Current state**: `CostTracker` tracks actual cost after calls.
**Gap**: Before a large operation, estimate the cost (useful for budget warnings).
**Workaround**: Estimate at application level using token counts.
**Recommendation**: Add `estimate_cost(messages, model)` to `CostTracker`.

### 17. Plugin Discovery via Package Registry
**Current state**: `PluginRegistry.discover_plugins()` scans installed packages.
**Gap**: Should support discovering plugins from npm registry (not just installed).
**Workaround**: Implement npm search at application level.
**Recommendation**: Add remote plugin discovery and installation to `PluginRegistry`.

### 18. Structured Logging Interface
**Current state**: `LoggingMiddleware` logs to Python's logging module.
**Gap**: TypeScript SDK needs structured logging that outputs JSON and integrates with common logging frameworks (pino, winston).
**Workaround**: Configure logging at application level.
**Recommendation**: Use `pino` for structured logging in TypeScript SDK with configurable transports.

---

## TypeScript SDK Specific Notes

These are considerations specifically for the TypeScript port:

### 19. Bun Compatibility
- Ensure all SDK code works with both Node.js and Bun runtimes
- Use Web-standard APIs where possible (`fetch`, `crypto`, `ReadableStream`)
- Avoid Node.js-specific APIs that Bun doesn't support

### 20. Tree-Shaking Support
- Use ESM modules with proper `exports` field in `package.json`
- Avoid side effects in module initialization
- Mark packages as side-effect-free for bundlers
- This is critical for binary size when compiling with Bun

### 21. Type Safety
- Use strict TypeScript types throughout (no `any`)
- Export all types for consumers
- Use discriminated unions for event types
- Use generics for type-safe tool definitions

### 22. Async Iterator Protocol
- Use native `AsyncIterableIterator` for streaming (not custom event emitters)
- Support `for await...of` pattern
- Proper cleanup via `return()` method on iterator

---

## Summary

| Category | Count | Impact |
|----------|-------|--------|
| Critical | 5 | Would significantly improve SDK for coding tools |
| Important | 6 | Would reduce application-level code |
| Nice-to-Have | 7 | Would polish the experience |
| TS-Specific | 4 | Required for TypeScript port quality |
| **Total** | **22** | — |

None of these block the coding tool implementation. All can be worked around at the application level. They represent opportunities to make the SDK better for this class of application.
