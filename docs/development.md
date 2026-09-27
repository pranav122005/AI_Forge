# AIForge Contributor & Development Guide

Thank you for contributing to AIForge.

---

## 1. Local Development Setup

### Backend (FastAPI)
```bash
cd backend
python -m venv .venv

# Windows
.venv\Scripts\activate

# Linux / macOS
source .venv/bin/activate

pip install -r requirements.txt
python -m uvicorn app.main:app --host 127.0.0.1 --port 8000 --reload
```

### Frontend (React / Vite)
```bash
cd frontend
npm install
npm run dev
```

---

## 2. Running Test Suites

Before opening pull requests or creating commits:

### Backend Tests
```bash
cd backend
python -m pytest -q
```
Ensure all 500+ unit, integration, and security tests pass.

### Frontend Build Verification
```bash
cd frontend
npm run build
```

---

## 3. Engineering Guidelines

- **No Hardcoded Secrets**: Never commit API keys, tokens, or credentials.
- **Path Safety**: Always use `Path.resolve()` and validate paths with `validate_file_path()`.
- **Determinism**: Preserve the distinction between executable components (e.g. text classification, FastAPI codegen) and architecture-only catalog items.
- **Testing**: Add tests in `backend/tests/` for all new endpoints, services, or agent features.
