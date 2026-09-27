# AIForge API Specification

## Base URL
- Local: `http://127.0.0.1:8000`
- Interactive OpenAPI Docs: `http://127.0.0.1:8000/docs`

---

## 1. System & Health

### `GET /api/health`
Checks backend service availability and component readiness.
```json
{
  "status": "ok",
  "version": "1.0.0",
  "service_alive": true
}
```

### `GET /api/capabilities`
Returns available ML, vision, agent, and LLM capabilities.

---

## 2. LLM Providers

### `GET /api/llm/providers`
Lists all supported and configured LLM providers.

### `POST /api/llm/providers/active`
Switches the system-wide active LLM provider and model.
```json
{
  "provider": "gemini",
  "model": "gemini-3.8-flash"
}
```

### `POST /api/llm/providers/{provider_id}/test`
Runs a connectivity test against the specified provider.

---

## 3. Autonomous Code Generation & Workspace

### `POST /api/codegen/generate`
Generates a new multi-file project from natural language.
**Request Body**:
```json
{
  "requirement": "Build a sentiment analysis API with FastAPI and pytest",
  "project_name": "sentiment-service",
  "extra_instructions": "Include comprehensive unit tests"
}
```

### `GET /api/codegen/{project_id}/files`
Returns recursive directory file tree.

### `GET /api/codegen/{project_id}/file?path={relative_path}`
Reads content of a specific project source file.

### `GET /api/codegen/{project_id}/report`
Fetches structured engineering audit and telemetry report.

### `GET /api/codegen/{project_id}/download`
Downloads the verified project archive as a `.zip` file.

### `POST /api/codegen/{project_id}/agent/run`
Executes autonomous agent planning, code writing, and testing loop.

### `POST /api/codegen/{project_id}/modify`
Executes targeted code modification with automatic repair and rollback.

---

## 4. Machine Learning & Inference

### `POST /api/train`
Trains a text classification pipeline from built-in demo or custom uploaded CSV.

### `POST /api/predict/text`
Runs classification inference against trained scikit-learn model.

### `POST /api/predict/image`
Extracts text via OCR and runs classification inference.
