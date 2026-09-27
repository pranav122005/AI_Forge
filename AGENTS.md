# AIForge Project Rules

## Goal
AIForge is an agentic AI engineering factory. A developer provides an AI system specification; AIForge analyzes required capabilities, composes an implementation plan, generates a project scaffold, trains supported task-specific models, evaluates them, and exposes the result through an API.

## Current MVP Scope
- Natural-language requirement analysis
- AI component selection from a curated registry
- Generated project scaffold
- Real text-classification training and evaluation
- Prediction API
- Dashboard showing architecture, training metrics, and API output

## Catalog vs Implemented
OpenCV, OCR, LLM, embeddings, and vector-store components may appear in generated architectures, but the current executable training path is task-specific text classification + FastAPI serving. Do not pretend unsupported components are fully implemented.

## Engineering Rules
- Keep the MVP runnable locally.
- No hardcoded secrets.
- Prefer small, understandable modules.
- Add tests for non-trivial backend logic.
- Preserve the distinction between generated architecture and actually executable components.
