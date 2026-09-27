# AIForge Product Definition

## One-line pitch

**AIForge turns a developer's natural-language AI specification into an engineered AI system: it chooses the required components, generates the project, trains supported task-specific models, evaluates them, and exposes the result as an API.**

## Target user

Software developers, ML engineers, and product engineers who repeatedly assemble AI pipelines from separate libraries, models, data-processing tools, serving frameworks, and deployment artifacts.

## Problem

Building an AI application often requires manually deciding which models and libraries are appropriate, wiring preprocessing and inference stages together, creating training/evaluation code, writing tests, and exposing the result through a production API. The workflow is fragmented and error-prone.

## MVP solution

1. Developer describes the desired AI behavior.
2. AIForge maps capabilities in the description to a component registry.
3. AIForge displays the resulting architecture and execution pipeline.
4. AIForge generates a runnable project scaffold.
5. A real task-specific text-classification model can be trained from labeled CSV data.
6. AIForge reports measured evaluation metrics.
7. The trained model is served through a prediction endpoint.

## Product truth

The MVP does **not** claim to train arbitrary LLMs from scratch or fully implement every catalog component. OpenCV, OCR, LLM, embeddings and vector-store entries demonstrate the planned architecture vocabulary. The currently executable path is text classification + FastAPI serving.

## Why this is relevant to the hackathon

The product targets a developer workflow: turning an AI requirement into a structured, testable, deployable implementation. IBM Bob 2.0 is used as the agentic engineering environment for repository understanding, implementation, testing, review and release-readiness tasks.
