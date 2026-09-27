# AIForge Architecture Guide

## 1. System Overview

AIForge is designed as a modular, event-driven autonomous AI software engineering system. The platform consists of two primary layers:
1. **Backend Gateway & Autonomous Engine** (FastAPI / Python 3.12)
2. **Scientific Engineering Workstation UI** (React 18 / Vite / CSS Tokens)

```
┌─────────────────────────────────────────────────────────────┐
│                    Workstation Frontend                     │
│  - Command Center          - Reports & Audit                │
│  - 3-Column Workspace      - Telemetry & Event Stream       │
│  - Neural Provider Center  - Playground & ML Manager        │
└──────────────────────────────┬──────────────────────────────┘
                               │ REST / JSON
┌──────────────────────────────▼──────────────────────────────┐
│                    FastAPI Backend Core                     │
│  - LLM Provider Registry (Gemini / OpenAI / Claude)         │
│  - Requirement Analyzer & Planning Engine                   │
│  - Multi-File Code Generation & Dependency Resolver         │
│  - Subprocess Pytest Runner & Stack Trace Parser            │
│  - Auto-Repair & Mutation Loop                              │
│  - Atomic Snapshot & Rollback Coordinator                   │
│  - Security Sandbox & Path Validator                        │
│  - ZIP Packager & Verification Engine                       │
└─────────────────────────────────────────────────────────────┘
```

---

## 2. Core Subsystems

### 2.1 LLM Provider Registry (`app/llm/`)
The Provider Registry manages multiple LLM backends through a unified interface (`LLMProviderInterface`):
- **Google Gemini**: Uses OpenAI-compatible endpoints or Google SDK with structured schema generation (`RequirementSpec`, `ProjectPlan`).
- **OpenAI**: Native OpenAI SDK integration (`gpt-4o`, `gpt-4o-mini`, `gpt-5`).
- **Anthropic**: Claude integration (`claude-3-5-sonnet-20241022`, `claude-3-7-sonnet`).
- **Session Memory Storage**: Dynamically configured keys are stored in encrypted server memory during active sessions and never persisted to disk.

### 2.2 Autonomous Coding Agent Loop (`app/codegen/agent/`)
When a modification or complex requirement is executed, the agent follows a multi-stage lifecycle:
1. **Inspection**: Analyzes repository structure and indexes existing files.
2. **Context Selection**: Identifies relevant files based on natural-language instruction.
3. **Structured Planning**: Generates a step-by-step modification plan (`AgentPlan`).
4. **Execution**: Generates updated/new file contents.
5. **Testing**: Runs `pytest` inside an isolated subprocess.
6. **Auto-Repair**: If tests fail, parses the failure traceback and attempts repair (up to 3 iterations).
7. **Verification**: Checks security compliance, forbidden files, and syntax.
8. **Rollback**: If unrecoverable errors occur, restores the pre-execution snapshot.

### 2.3 Security & Sandbox Guard (`app/codegen/agent/verifier.py`, `path_validator.py`)
- **Path Sanitization**: Prevents `..` traversal and absolute root escapes.
- **Secret Redaction**: Regex filters scrub API keys (`AIza...`, `sk-...`, `Bearer ...`) from test outputs, error responses, and reports.
- **Protected Files**: System files (`.env`, `.git`, `.venv`) cannot be created, modified, or deleted by generated code or agent runs.

### 2.4 Report Intelligence Engine (`app/codegen/service.py`)
Aggregates telemetry from all subsystems into a unified JSON/Markdown format:
- Executive Summary & Project Metadata
- Architecture Component Topology
- Telemetry & Event History
- Test Execution Logs & Repair Statistics
- Security Checklist & Sandbox Compliance
- Evaluation Metrics (Accuracy, Precision, Recall, F1)
