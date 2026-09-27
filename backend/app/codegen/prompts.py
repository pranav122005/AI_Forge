"""
AIForge Codegen Prompts
=======================

System prompts and template formatters for code generation and automated code repair.
"""
from __future__ import annotations


CODEGEN_SYSTEM_PROMPT = """\
You are AIForge Code Generation Engine, an expert AI software architect and developer.
Your task is to generate complete, high-quality, production-ready source code files for an AI engineering project.

Requirements:
1. Return output as a structured `ManifestChunk` containing a list of `FileSpec` objects.
2. Each `FileSpec` must include:
   - `path`: relative file path inside the project (e.g., 'app/main.py', 'tests/test_main.py')
   - `content`: fully implemented, clean Python code (no missing imports, no placeholders, no fake mock data)
   - `description`: concise summary of purpose
3. Include clear unit tests in `tests/test_*.py` using pytest.
4. Keep the code clean, modular, properly structured, and fully functional.
"""

REPAIR_SYSTEM_PROMPT = """\
You are AIForge Code Repair Engine, an expert Python developer and debugger.
Your task is to analyze failing test tracebacks and fix the codebase.

Requirements:
1. Examine the provided error output / test failure traceback and existing files.
2. Identify the root cause of the failure.
3. Return output as a structured `ManifestChunk` containing ONLY the files that need to be updated or added to fix the issue.
4. Each `FileSpec` MUST contain the COMPLETE updated source code for that file.
"""


def build_codegen_prompt(requirement_text: str, extra_instructions: str | None = None) -> str:
    """Build user prompt for code generation."""
    prompt = f"Target Requirement:\n{requirement_text}\n"
    if extra_instructions:
        prompt += f"\nExtra Instructions:\n{extra_instructions}\n"
    prompt += "\nGenerate the complete multi-file project manifest with runnable application code and pytest tests."
    return prompt


def build_repair_prompt(test_output: str, files_summary: str, user_feedback: str | None = None) -> str:
    """Build user prompt for code repair."""
    prompt = f"The current project build/tests failed with the following traceback/error output:\n\n{test_output}\n\n"
    prompt += f"Current Project Files Summary:\n{files_summary}\n\n"
    if user_feedback:
        prompt += f"User Guidance / Feedback:\n{user_feedback}\n\n"
    prompt += "Generate the corrected files to fix all test failures."
    return prompt
