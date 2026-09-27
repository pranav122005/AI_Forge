# IBM Bob 2.0 Task Prompts

Use these in **IBM Bob IDE** on the AIForge repository. Do real work with Bob; do not fabricate task summaries. Capture the official task session consumption summary screenshot for each relevant task and store it in `bob_sessions/`.

## Task 01 — Repository & architecture analysis

> Inspect the entire AIForge repository. Explain the current architecture, identify the implemented versus catalog-only AI components, trace the specification-to-plan-to-training-to-prediction flow, and record concrete architecture risks or gaps. Do not modify application behavior. Save findings to `docs/bob_architecture_review.md`.

## Task 02 — Planner robustness

> Review `backend/app/main.py`, especially `analyze_spec`. Identify natural-language requirements that the current component-selection heuristic misses. Improve the planner without adding external services or inventing unsupported capabilities. Add focused tests for the new cases.

## Task 03 — Training/API code review

> Review the training and prediction workflow for correctness, validation, error handling, and maintainability. Make only justified changes. Ensure the model path remains reproducible and that invalid datasets produce useful errors. Add or improve tests where appropriate.

## Task 04 — Test generation

> Inspect the backend endpoints and existing tests. Generate additional tests for the most important failure modes and the happy path. Run the tests and summarize the results.

## Task 05 — Security/release readiness

> Perform a practical release-readiness review of AIForge. Check for hardcoded secrets, unsafe file handling, obvious injection risks in generated project files, dependency concerns, and missing deployment documentation. Create `docs/bob_release_review.md` with severity, evidence, and concrete remediation suggestions.

## Task 06 — Demo readiness

> Review the complete AIForge user journey from specification through prediction. Identify any confusing UI states or demo-breaking failure modes. Propose or implement only small, high-value fixes that improve the 3-minute hackathon demo.
