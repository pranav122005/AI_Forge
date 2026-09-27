"""
AIForge Automated Code Repair Engine
===================================

Analyzes failing test tracebacks and automatically applies LLM code fixes in an iterative repair loop.
"""
from __future__ import annotations

from pathlib import Path
from typing import List, Optional

from app.codegen.errors import CodegenError
from app.codegen.file_writer import read_file_content, write_file_specs
from app.codegen.models import FileSpec, ManifestChunk, RepairResult, TestResult
from app.codegen.prompts import REPAIR_SYSTEM_PROMPT, build_repair_prompt
from app.codegen.tester import run_project_tests
from app.llm.providers.base import BaseLLMProvider


class RepairEngine:
    """
    Automated repair engine managing error analysis and iterative code fixes.
    """

    def __init__(self, provider: Optional[BaseLLMProvider] = None, max_retries: int = 3) -> None:
        self.provider = provider
        self.max_retries = max_retries

    async def repair_project(
        self,
        target_root: Path,
        test_output: Optional[str] = None,
        user_feedback: Optional[str] = None,
    ) -> RepairResult:
        """
        Run iterative repair loop on project at target_root.
        """
        target_resolved = target_root.resolve()
        
        # Initial test execution if output not provided
        test_res = run_project_tests(target_resolved)
        if test_res.success:
            return RepairResult(
                success=True,
                retry_count=0,
                fixed_files=[],
                test_result=test_res,
                error=None,
            )

        current_output = test_output or (test_res.stderr + "\n" + test_res.stdout)
        all_fixed_files: List[str] = []

        for retry in range(1, self.max_retries + 1):
            fixed_chunk = await self._generate_fix(
                target_root=target_resolved,
                test_output=current_output,
                user_feedback=user_feedback,
            )

            if not fixed_chunk or not fixed_chunk.files:
                break

            # Apply fixed files
            written = write_file_specs(target_resolved, fixed_chunk.files)
            all_fixed_files.extend(written)

            # Re-run tests
            test_res = run_project_tests(target_resolved)
            if test_res.success:
                return RepairResult(
                    success=True,
                    retry_count=retry,
                    fixed_files=list(set(all_fixed_files)),
                    test_result=test_res,
                    error=None,
                )

            current_output = test_res.stderr + "\n" + test_res.stdout

        return RepairResult(
            success=False,
            retry_count=self.max_retries,
            fixed_files=list(set(all_fixed_files)),
            test_result=test_res,
            error=f"Repair loop failed to resolve test failures after {self.max_retries} retries.",
        )

    async def _generate_fix(
        self,
        target_root: Path,
        test_output: str,
        user_feedback: Optional[str] = None,
    ) -> Optional[ManifestChunk]:
        """
        Generate fixed files using LLM or heuristic repair.
        """
        if self.provider is not None:
            try:
                # Gather current files summary
                files_summary = self._summarize_files(target_root)
                prompt = build_repair_prompt(test_output, files_summary, user_feedback)
                manifest = await self.provider.generate_structured(
                    prompt=prompt,
                    response_model=ManifestChunk,
                    system_prompt=REPAIR_SYSTEM_PROMPT,
                )
                if manifest and manifest.files:
                    return manifest
            except Exception:
                pass

        # Fallback fix heuristic if LLM provider unavailable
        return self._heuristic_fix(target_root, test_output)

    def _summarize_files(self, target_root: Path) -> str:
        """Collect file paths and contents for LLM repair context."""
        summary = []
        for path in target_root.rglob("*.py"):
            if "__pycache__" in path.parts:
                continue
            rel_path = path.relative_to(target_root).as_posix()
            try:
                content = path.read_text(encoding="utf-8", errors="replace")
                summary.append(f"--- File: {rel_path} ---\n{content}\n")
            except Exception:
                pass
        return "\n".join(summary[:10])

    def _heuristic_fix(self, target_root: Path, test_output: str) -> Optional[ManifestChunk]:
        """
        Basic heuristic repair for common issues when LLM is unconfigured.
        """
        # If health test assertion failed or import missing, ensure app/main.py is clean
        main_path = target_root / "app" / "main.py"
        if main_path.exists():
            content = main_path.read_text(encoding="utf-8", errors="replace")
            # If status endpoint misnamed or missing health
            if "health" in test_output.lower() and "/health" not in content:
                content += "\n\n@app.get('/health')\ndef health(): return {'status': 'ok'}\n"
                return ManifestChunk(
                    files=[FileSpec(path="app/main.py", content=content, description="Add health check endpoint")]
                )
        return None
