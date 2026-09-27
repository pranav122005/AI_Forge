# AIForge

**Autonomous AI Software Engineering Platform & Neural Code Synthesis Laboratory**

[![Backend CI](https://img.shields.io/badge/backend-FastAPI%20%7C%20Python%203.12-blue.svg)](https://fastapi.tiangolo.com)
[![Frontend](https://img.shields.io/badge/frontend-React%2018%20%7C%20Vite-cyan.svg)](https://vitejs.dev)
[![LLM Support](https://img.shields.io/badge/AI%20Providers-Gemini%20%7C%20OpenAI%20%7C%20Claude-8b5cf6.svg)](https://deepmind.google/technologies/gemini/)
[![Test Suite](https://img.shields.io/badge/tests-501%20passed-green.svg)](backend/tests)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

---

## What is AIForge?

**AIForge** is an autonomous agentic AI software engineering platform. It transforms natural-language system requirements into verified, fully-tested, multi-file codebases, trains task-specific machine learning models, executes automated repair loops, and packages ready-to-deploy software artifacts.

Designed as a **Scientific AI Engineering Workstation**, AIForge provides end-to-end visibility into autonomous software synthesis with real-time agent telemetry, pipeline stage tracking, live code viewing, and automated compliance auditing.

---

## The Problem

Traditional AI application development is fragmented, error-prone, and manual:
1. **Boilerplate Fatigue**: Developers must manually architect file hierarchies, configure dependencies, manage environment variables, and configure API routes.
2. **Untested LLM Code**: Typical AI assistants generate isolated code snippets that frequently contain import errors, hallucinated APIs, and broken syntax.
3. **Manual Repair Loops**: When generated code fails, developers spend hours copying stack traces back and forth to an LLM for debugging.
4. **Lack of Isolation & Safety**: Executing unverified LLM-generated code locally introduces path traversal, security vulnerabilities, and system state corruption.

---

## The AIForge Solution

AIForge automates the entire software engineering lifecycle through deterministic, closed-loop agentic workflows:

```
[ Natural Language Specification ]
               │
               ▼
   [ Requirement Analysis ]
               │
               ▼
   [ Architecture Planning ]
               │
               ▼
     [ Multi-File Codegen ]
               │
               ▼
    [ Automated Unit Tests ] ◄──┐
               │                │ (Auto-Repair Loop
               ▼                │  up to 3 attempts)
     [ Test Evaluation ] ───────┘
               │ (Pass)
               ▼
   [ Security Verification ]
               │
               ▼
   [ Sandboxed ZIP Packaging ]
```

For existing codebases, AIForge executes a structured **Modification & Repair Loop**:
$$\text{Inspect} \longrightarrow \text{Select Files} \longrightarrow \text{Plan} \longrightarrow \text{Apply Changes} \longrightarrow \text{Test} \longrightarrow \text{Auto-Repair} \longrightarrow \text{Verify / Rollback}$$

---

## Key Features

- **Multi-LLM Neural Architecture**: Seamless runtime switching between **Google Gemini (2.5/3.8 Flash & Pro)**, **OpenAI (GPT-4o/GPT-5)**, and **Anthropic (Claude 3.5/3.7 Sonnet)** with live key validation.
- **Autonomous Coding Agent**: Multi-file code synthesizer that identifies dependencies, designs architecture plans, writes code, and repairs defects autonomously.
- **Automated Test & Repair Loop**: Runs isolated `pytest` subprocesses against generated projects, captures exit codes and stack traces, and iteratively fixes errors.
- **Atomic Rollback Engine**: Automatically creates pre-modification workspace snapshots and rolls back project state if agent repairs fail.
- **Report Intelligence Workspace**: Generates comprehensive 6-section technical engineering reports with one-click Markdown export, JSON download, and print-ready PDF layouts.
- **Scientific Cyberpunk UI**: High-density dark interface (`#05070D` base, `#00f0ff` cyber cyan accents) with zero emojis, technical SVG icon system, and live telemetry stream.
- **Machine Learning Pipeline**: Built-in dataset inspection, feature preprocessing, TF-IDF vectorization, scikit-learn classifier training, and real-time evaluation metrics.
- **Vision & OCR Inference**: Optional image document classification and text extraction pipeline with OpenCV and OCR engines.
- **Hardened Security Sandboxing**: Path traversal prevention, secret redaction filters, forbidden file guards, and quarantine isolation for ZIP packaging.

---

## System Architecture

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                          AIForge Frontend (React / Vite)                    │
│                                                                             │
│  ┌──────────────┐  ┌─────────────┐  ┌──────────────┐  ┌──────────────────┐  │
│  │   Command    │  │  Workspace  │  │ AI Provider  │  │     Report       │  │
│  │    Center    │  │ File Viewer │  │Control Center│  │   Intelligence   │  │
│  └──────┬───────┘  └──────┬──────┘  └──────┬───────┘  └────────┬─────────┘  │
└─────────┼─────────────────┼────────────────┼───────────────────┼────────────┘
          │                 │                │                   │
          ▼                 ▼                ▼                   ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                       FastAPI Backend Gateway (:8000)                       │
│                                                                             │
│  ┌───────────────────────┐  ┌───────────────────────────────────────────┐   │
│  │   Provider Registry   │  │             Codegen Service               │   │
│  │  (Gemini/OpenAI/Claude)│  │   ┌─────────────┐     ┌──────────────┐    │   │
│  └──────────┬────────────┘  │   │ Code Parser │ ──► │ File Manager │    │   │
│             │               │   └─────────────┘     └──────┬───────┘    │   │
│             ▼               └──────────────────────────────┼────────────┘   │
│  ┌───────────────────────┐                                 │                │
│  │ Autonomous Dev Loop   │ ◄───────────────────────────────┘                │
│  │ ┌───────────────────┐ │                                                  │
│  │ │  Planning Engine  │ │  ┌─────────────────┐     ┌──────────────────┐   │
│  │ ├───────────────────┤ │  │  Pytest Runner  │ ──► │  Security Guard  │   │
│  │ │  Auto-Repair Loop │ │  │   (Subprocess)  │     │ (Path/Secrets)   │   │
│  │ ├───────────────────┤ │  └─────────────────┘     └────────┬─────────┘   │
│  │ │ Snapshot/Rollback │ │                                   │              │
│  │ └───────────────────┘ │  ┌────────────────────────────────▼──────────┐   │
│  └───────────────────────┘  │         ZIP Packager & Verifier          │   │
│                             └──────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## Technology Stack

| Layer | Technologies |
|---|---|
| **Backend Core** | Python 3.12, FastAPI, Uvicorn, Pydantic v2 |
| **ML & Data** | scikit-learn, NumPy, Pandas, Joblib |
| **Vision & OCR** | OpenCV, Pytesseract |
| **Testing & Quality** | Pytest, Subprocess Isolation, Sandbox Validators |
| **AI Providers** | Google Gemini SDK / OpenAI-compatible API, OpenAI SDK, Anthropic Claude |
| **Frontend UI** | React 18, Vite, Lucide/SVG Icons, CSS3 Modern Tokens |
| **Infrastructure** | Docker, Docker Compose, Nginx (Alpine), Vercel |

---

## Installation & Setup

### Prerequisites
- **Python**: 3.11+ (Python 3.12 recommended)
- **Node.js**: 18.x or 20.x+
- **Git**

### 1. Clone the Repository
```bash
git clone https://github.com/your-username/aiforge.git
cd aiforge
```

### 2. Backend Setup
```bash
cd backend

# Create and activate virtual environment
# Windows:
python -m venv .venv
.venv\Scripts\activate

# Linux / macOS:
# python3 -m venv .venv
# source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### 3. Frontend Setup
```bash
cd ../frontend
npm install
```

---

## Environment Configuration

Create a `.env` file in the project root or configure system environment variables (see [`.env.example`](.env.example)):

```bash
# LLM Provider Configuration
GEMINI_API_KEY=your_gemini_api_key_here
GEMINI_MODEL=gemini-3.8-flash
OPENAI_API_KEY=
ANTHROPIC_API_KEY=

# Server Configuration
HOST=127.0.0.1
PORT=8000
ALLOWED_ORIGINS=http://localhost:5173,http://127.0.0.1:5173

# Execution Safety Limits
GENERATION_TIMEOUT=45
TEST_TIMEOUT=60
MAX_REPAIR_ATTEMPTS=3
```

> **Security Note**: Never commit `.env` files or expose provider API keys in frontend builds. Frontend uses `VITE_API_BASE_URL` to route requests to the secure backend.

---

## Running Locally

### Terminal 1: Backend Service
```bash
cd backend
python -m uvicorn app.main:app --host 127.0.0.1 --port 8000 --reload
```
API Documentation will be available at `http://127.0.0.1:8000/docs`.

### Terminal 2: Frontend Development Server
```bash
cd frontend
npm run dev
```
Workstation UI will be available at `http://localhost:5173`.

---

## Docker Deployment

To launch the full AIForge stack using Docker Compose:

```bash
# Build and start all services in detached mode
docker compose up --build -d

# Check service logs
docker compose logs -f
```

- **Frontend Workstation**: `http://localhost:5173`
- **Backend API**: `http://localhost:8000`
- **Health Check**: `http://localhost:8000/api/health`

---

## Verification & Testing

### Backend Test Suite
```bash
cd backend
python -m pytest -q
```
*Current test suite: **501 passed, 1 skipped**.*

### Frontend Production Build
```bash
cd frontend
npm run build
```
*Build output verification: 100% clean bundle compilation.*

---

## API Reference

| Endpoint | Method | Description |
|---|---|---|
| `/api/health` | `GET` | Health check & system status |
| `/api/capabilities` | `GET` | Runtime engine capabilities & active providers |
| `/api/llm/providers` | `GET` | List configured LLM providers and models |
| `/api/llm/providers/active` | `POST` | Switch active LLM engine |
| `/api/llm/providers/{id}/test`| `POST` | Test live connectivity with a provider |
| `/api/codegen/generate` | `POST` | Synthesize complete multi-file project codebase |
| `/api/codegen/{id}/files` | `GET` | Get generated project file tree |
| `/api/codegen/{id}/file` | `GET` | Read source code of specific file |
| `/api/codegen/{id}/report` | `GET` | Fetch comprehensive project telemetry & audit report |
| `/api/codegen/{id}/download`| `GET` | Download verified project ZIP package |
| `/api/codegen/{id}/agent/run`| `POST` | Execute autonomous coding & repair agent |
| `/api/codegen/{id}/modify` | `POST` | Execute targeted codebase modification |
| `/api/train` | `POST` | Train scikit-learn classification model |
| `/api/predict/text` | `POST` | Execute text inference against trained pipeline |
| `/api/predict/image` | `POST` | Execute vision & OCR prediction |

---

## Security Architecture

1. **Secret Sanitization**: All exception handlers and log formatters automatically redact Gemini (`AIza...`), OpenAI (`sk-...`), and Anthropic (`sk-ant-...`) API keys.
2. **Path Traversal Prevention**: Strict validation on all requested file paths preventing directory traversal (`../`) and unauthorized root escape.
3. **Protected Files**: System files (`.env`, `.git`, `credentials.json`, `venv`) are strictly blocked from modification or deletion.
4. **Subprocess Isolation**: Automated tests execute in isolated worker processes with strict execution timeouts (60s default) to prevent runaway processes.
5. **Atomic Rollback Snapshots**: Full filesystem snapshot created before every agent execution, ensuring deterministic recovery on unexpected failures.
6. **ZIP Quarantine**: Generated ZIP archives are created in quarantine memory spaces and validated before distribution.

---

## Team Agni

AIForge is engineered and maintained by **Team Agni**:

- **Pranav Nimje**
- **Rishi Shahu**
- **Bhagyesh Dedmuthe**
- **Yashraj Talegaonkar**

---

## License

This project is licensed under the **MIT License** — see the [LICENSE](LICENSE) file for details.
