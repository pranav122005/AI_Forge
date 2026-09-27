# AIForge Security & Sandboxing Model

AIForge implements defense-in-depth security principles to safely generate, modify, test, and execute AI codebases.

---

## 1. Secret Protection & Sanitization

- **Zero-Storage Keys**: User-provided session API keys are held strictly in server memory (`ProviderRegistry`) and never written to logs, disk, or output ZIP packages.
- **Automated Regex Sanitization**: All exception messages, stdout/stderr streams, and project audit reports pass through `redact_secrets()`, which masks:
  - Google Gemini keys (`AIzaSy...` $\rightarrow$ `[REDACTED_GEMINI_KEY]`)
  - OpenAI keys (`sk-...` $\rightarrow$ `[REDACTED_OPENAI_KEY]`)
  - Anthropic keys (`sk-ant-...` $\rightarrow$ `[REDACTED_ANTHROPIC_KEY]`)
  - Bearer tokens (`Bearer ...` $\rightarrow$ `[REDACTED_TOKEN]`)
  - Private keys (`-----BEGIN PRIVATE KEY-----` $\rightarrow$ `[REDACTED_PRIVATE_KEY]`)

---

## 2. Path Traversal & Filesystem Sandbox

- **Canonical Path Resolution**: Every file read/write request resolves against the project root directory using `pathlib.Path.resolve()`.
- **Root Escape Prevention**: Any path containing `..`, absolute drive letters (`C:\`, `/etc/`), or escaping the project directory triggers a `PathValidationError` (HTTP 400).
- **Protected Files Guard**: The following paths cannot be written or deleted by the agent:
  - `.env`, `.env.*`
  - `.git`, `.gitignore`
  - `credentials.json`, `secrets.json`
  - `.venv`, `node_modules`

---

## 3. Subprocess Execution Isolation

- Generated code tests execute via isolated `pytest` subprocesses.
- **Execution Timeout**: Subprocesses are bounded by strict timeouts (default 60 seconds) to prevent infinite loops or denial-of-service hanging.
- **Resource Constraints**: Failed executions terminate child processes cleanly without leaking background handles.

---

## 4. Atomic Snapshots & Rollback

Before any codebase modification:
1. A compressed snapshot of the target project directory is captured in a quarantine zone.
2. The agent executes code mutations and runs unit tests.
3. If unit tests fail after the maximum configured repair attempts (default 3), the workspace is automatically rolled back to its exact pre-modification state.
4. The modification status is tagged as `ROLLED_BACK`.

---

## 5. ZIP Archive Quarantine

- Project ZIP archives are assembled from validated project roots.
- Hidden files, temporary `.pyc` caches, and local secrets are excluded from generated ZIP archives.
