# 3-Minute MVP Demo

## Demo specification

> Build an AI system that classifies Indian legal documents into categories such as Contract, Petition, Affidavit, Notice, Deed, Agreement, and Order, and exposes a REST API.

## Exact flow

1. Enter the specification.
2. Click **Analyze requirement**.
3. Show AIForge selecting **Task Classifier + FastAPI** and explaining why.
4. Click **Generate project**.
5. Click **Train model** using the bundled dataset.
6. Show the measured accuracy/F1 metrics.
7. Send the example Petition text.
8. Show the predicted class and confidence.
9. Briefly switch to the component architecture view and explain that the same planner vocabulary can compose vision/OCR/LLM/RAG components; do not claim those catalog components are executable in this MVP.
10. Show the real IBM Bob task evidence/repository briefly.

## What not to claim

- Do not claim arbitrary LLM training.
- Do not claim OCR/OpenCV/LLM execution unless those paths are actually implemented.
- Do not claim IBM Bob is embedded inside the runtime application. It is the agentic engineering environment/workflow used to build, review, test and validate AIForge.
