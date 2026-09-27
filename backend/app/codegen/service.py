"""
AIForge Codegen Service
=======================

High-level orchestrator service coordinating code generation, safe validation,
filesystem writing, automated testing, repair loops, project modifications, and ZIP packaging.
"""
from __future__ import annotations

import json
import os
import shutil
import time
import uuid
from pathlib import Path
from typing import List, Optional

from app.codegen.errors import CodegenError, GenerationError, PathValidationError
from app.codegen.file_writer import (
    build_file_tree,
    read_file_content,
    write_file_specs,
)
from app.codegen.generator import CodegenEngine
from app.codegen.modifier import ProjectModifierEngine
from app.codegen.models import (
    CodegenModifyRequest,
    CodegenModifyResult,
    CodegenRequest,
    CodegenResult,
    FileTreeNode,
    HistoryItem,
    RepairResult,
    TestResult,
)
from app.codegen.packager import package_project_zip
from app.codegen.repair import RepairEngine
from app.codegen.tester import run_project_tests
from app.codegen.validator import validate_file_path
from app.llm.providers.base import BaseLLMProvider


BASE_DIR = Path(__file__).resolve().parents[2]
GENERATED_PROJECTS_DIR = BASE_DIR / "generated_projects"
PROJECTS_DIR = BASE_DIR / "generated"
ARTIFACTS_DIR = BASE_DIR / "artifacts"

GENERATED_PROJECTS_DIR.mkdir(parents=True, exist_ok=True)
PROJECTS_DIR.mkdir(parents=True, exist_ok=True)
ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)


def resolve_project_dir(project_id: str) -> Path:
    """
    Resolve absolute path to project directory given project_id.
    """
    p1 = GENERATED_PROJECTS_DIR / project_id
    if p1.exists():
        return p1
    p2 = PROJECTS_DIR / project_id
    if p2.exists():
        return p2
    return p1


class CodegenService:
    """
    Orchestrator service for AIForge Code Generation Engine & Iterative Modifier Agent.
    """

    def __init__(self, provider: Optional[BaseLLMProvider] = None) -> None:
        self._custom_provider = provider
        self.repair_engine = RepairEngine(provider=self._resolve_provider(), max_retries=3)

    @property
    def provider(self) -> Optional[BaseLLMProvider]:
        return self._custom_provider

    @provider.setter
    def provider(self, val: Optional[BaseLLMProvider]) -> None:
        self._custom_provider = val
        current_retries = getattr(self.repair_engine, "max_retries", 3)
        self.repair_engine = RepairEngine(provider=val, max_retries=current_retries)

    def _resolve_provider(self, request_provider: Optional[str] = None, request_model: Optional[str] = None) -> Optional[BaseLLMProvider]:
        if self._custom_provider is not None:
            return self._custom_provider
        from app.llm.service import get_provider
        return get_provider(provider_name=request_provider, model=request_model)

    async def generate_project(
        self,
        request: CodegenRequest,
    ) -> CodegenResult:
        """
        Generate, build, test, repair, and package an end-to-end AI project.
        """
        project_id = request.project_id or str(uuid.uuid4())[:8]
        project_name = request.project_name or "aiforge-project"
        req_text = request.requirement_text or (
            request.spec.get("goal") if isinstance(request.spec, dict) else "AI system project"
        )

        active_provider = self._resolve_provider(request.provider, request.model)
        engine = CodegenEngine(provider=active_provider)
        repair_engine = self.repair_engine if self._custom_provider is not None else RepairEngine(provider=active_provider, max_retries=3)

        target_dir = resolve_project_dir(project_id)
        if target_dir.exists():
            shutil.rmtree(target_dir, ignore_errors=True)
        target_dir.mkdir(parents=True, exist_ok=True)

        # 1. Generate code manifest
        manifest = await engine.generate_manifest(
            requirement_text=req_text,
            extra_instructions=request.extra_instructions,
            project_name=project_name,
        )

        # 2. Safe write files to disk
        written_files = write_file_specs(target_dir, manifest.files)

        # Record initial history step
        self._record_history(target_dir, "Initial project generation", "Created project scaffolding and source files")

        # 3. Run automated tests
        test_res = run_project_tests(target_dir)

        # 4. If tests fail, run repair loop
        status = "completed"
        if not test_res.success:
            repair_res = await repair_engine.repair_project(
                target_root=target_dir,
                test_output=test_res.stderr + "\n" + test_res.stdout,
            )
            if repair_res.test_result:
                test_res = repair_res.test_result
            if repair_res.success:
                status = "repaired"
                written_files = list(set(written_files + repair_res.fixed_files))
            else:
                status = "failed"

        # 5. Package project into ZIP artifact
        zip_artifact_path = ARTIFACTS_DIR / project_id / f"{project_name}.zip"
        try:
            package_project_zip(target_dir, zip_artifact_path)
            zip_rel = f"/api/codegen/{project_id}/download"
        except Exception:
            zip_rel = None

        provider_id = getattr(active_provider, "provider_id", "fallback") if active_provider else "fallback"
        model_name = getattr(active_provider, "model", "none") if active_provider else "none"

        return CodegenResult(
            project_id=project_id,
            status=status,
            generated_files=written_files,
            test_result=test_res,
            zip_path=zip_rel,
            error=None if test_res.success else test_res.stderr or "Tests failed",
            provider=provider_id,
            model=model_name,
        )

    def get_file_tree(self, project_id: str) -> FileTreeNode:
        """Get project file directory tree."""
        target_dir = resolve_project_dir(project_id)
        if not target_dir.exists():
            raise FileNotFoundError(f"Project '{project_id}' not found.")
        return build_file_tree(target_dir)

    def read_file(self, project_id: str, relative_path: str) -> str:
        """Read content of a project file."""
        target_dir = resolve_project_dir(project_id)
        if not target_dir.exists():
            raise FileNotFoundError(f"Project '{project_id}' not found.")
        return read_file_content(target_dir, relative_path)

    def run_tests(self, project_id: str) -> TestResult:
        """Run tests on an existing project."""
        target_dir = resolve_project_dir(project_id)
        if not target_dir.exists():
            raise FileNotFoundError(f"Project '{project_id}' not found.")
        return run_project_tests(target_dir)

    async def repair_project(
        self,
        project_id: str,
        test_output: Optional[str] = None,
        user_feedback: Optional[str] = None,
    ) -> RepairResult:
        """Run repair loop on an existing project."""
        target_dir = resolve_project_dir(project_id)
        if not target_dir.exists():
            raise FileNotFoundError(f"Project '{project_id}' not found.")
        return await self.repair_engine.repair_project(
            target_root=target_dir,
            test_output=test_output,
            user_feedback=user_feedback,
        )

    def get_zip_file_path(self, project_id: str, project_name: str = "aiforge-project") -> Path:
        """Get path to packaged ZIP artifact."""
        zip_path = ARTIFACTS_DIR / project_id / f"{project_name}.zip"
        if zip_path.exists():
            return zip_path
        alt_zip = ARTIFACTS_DIR / f"{project_id}.zip"
        if alt_zip.exists():
            return alt_zip

        target_dir = resolve_project_dir(project_id)
        if target_dir.exists():
            return package_project_zip(target_dir, zip_path)

        raise FileNotFoundError(f"ZIP artifact for project '{project_id}' not found.")

    async def modify_project(
        self,
        project_id: str,
        instruction: str,
        user_feedback: Optional[str] = None,
        provider: Optional[str] = None,
        model: Optional[str] = None,
    ) -> CodegenModifyResult:
        """
        Safely modify an existing project using ProjectModifierEngine with snapshot backup & rollback.
        """
        target_dir = resolve_project_dir(project_id)
        if not target_dir.exists():
            raise FileNotFoundError(f"Project '{project_id}' not found.")

        # 1. Create snapshot backup before modification
        snapshot_dir = ARTIFACTS_DIR / project_id / "snapshots" / f"snapshot_{int(time.time())}"
        self._create_snapshot(target_dir, snapshot_dir)

        # 2. Generate modification manifest via ProjectModifierEngine
        active_provider = self._resolve_provider(provider, model)
        modifier_engine = ProjectModifierEngine(provider=active_provider)
        repair_engine = self.repair_engine if self._custom_provider is not None else RepairEngine(provider=active_provider, max_retries=3)
        try:
            mod_manifest = await modifier_engine.generate_modification(
                target_root=target_dir,
                instruction=instruction,
                user_feedback=user_feedback,
            )
        except Exception as exc:
            self._restore_snapshot(target_dir, snapshot_dir)
            return CodegenModifyResult(
                project_id=project_id,
                instruction=instruction,
                summary=f"Modification failed: {exc}",
                files_created=[],
                files_modified=[],
                files_deleted=[],
                tests_passed=0,
                tests_failed=0,
                repair_attempts=0,
                status="ROLLED_BACK",
                zip_available=True,
                zip_path=f"/api/codegen/{project_id}/download",
                history=self.get_project_history(project_id),
                error=str(exc),
            )

        # 3. Apply file changes (create, modify, delete) safely
        files_created: List[str] = []
        files_modified: List[str] = []
        files_deleted: List[str] = []

        for change in mod_manifest.changes:
            rel_path = change.path.strip().replace("\\", "/")
            if rel_path in (".env", ".env.local", "secrets.json", "id_rsa"):
                raise PathValidationError(f"Modification of protected file '{rel_path}' is forbidden.")

            target_file = validate_file_path(rel_path, target_dir)

            if change.action == "delete":
                if target_file.exists():
                    target_file.unlink()
                    files_deleted.append(rel_path)
            else:
                existed = target_file.exists()
                target_file.parent.mkdir(parents=True, exist_ok=True)
                target_file.write_text(change.content or "", encoding="utf-8")
                if existed:
                    files_modified.append(rel_path)
                else:
                    files_created.append(rel_path)

        # 4. Run automated tests
        test_res = run_project_tests(target_dir)
        repair_attempts = 0

        # 5. If tests fail, invoke RepairEngine
        if not test_res.success:
            repair_res = await repair_engine.repair_project(
                target_root=target_dir,
                test_output=test_res.stderr + "\n" + test_res.stdout,
                user_feedback=user_feedback,
            )
            repair_attempts = repair_res.retry_count
            if repair_res.test_result:
                test_res = repair_res.test_result
            if repair_res.fixed_files:
                files_modified = list(set(files_modified + repair_res.fixed_files))

        # 6. If tests still fail after repair, rollback to snapshot
        if not test_res.success:
            self._restore_snapshot(target_dir, snapshot_dir)
            return CodegenModifyResult(
                project_id=project_id,
                instruction=instruction,
                summary=mod_manifest.summary,
                files_created=files_created,
                files_modified=files_modified,
                files_deleted=files_deleted,
                tests_passed=0,
                tests_failed=1,
                repair_attempts=repair_attempts,
                status="ROLLED_BACK",
                zip_available=True,
                zip_path=f"/api/codegen/{project_id}/download",
                history=self.get_project_history(project_id),
                error="Tests failed after modification and repair. Project state restored.",
            )

        # 7. Record step in project history
        history = self._record_history(target_dir, instruction, mod_manifest.summary)

        # 8. Re-package updated project into ZIP artifact
        zip_artifact_path = ARTIFACTS_DIR / project_id / f"{project_id}.zip"
        try:
            package_project_zip(target_dir, zip_artifact_path)
            zip_rel = f"/api/codegen/{project_id}/download"
        except Exception:
            zip_rel = None

        return CodegenModifyResult(
            project_id=project_id,
            instruction=instruction,
            summary=mod_manifest.summary,
            files_created=files_created,
            files_modified=files_modified,
            files_deleted=files_deleted,
            tests_passed=1 if test_res.success else 0,
            tests_failed=0 if test_res.success else 1,
            repair_attempts=repair_attempts,
            status="SUCCESS",
            zip_available=True,
            test_result=test_res,
            zip_path=zip_rel,
            history=history,
            error=None,
        )

    async def run_agent(
        self,
        project_id: str,
        instruction: str,
        user_feedback: Optional[str] = None,
        provider: Optional[str] = None,
        model: Optional[str] = None,
    ):
        """
        Run the autonomous Agentic Development Loop on an existing project.
        """
        from app.codegen.agent import AgentExecutionEngine
        target_dir = resolve_project_dir(project_id)
        if not target_dir.exists():
            raise FileNotFoundError(f"Project '{project_id}' not found.")

        active_provider = self._resolve_provider(provider, model)
        agent_engine = AgentExecutionEngine(provider=active_provider)
        state = await agent_engine.execute_agent_loop(
            target_root=target_dir,
            project_id=project_id,
            instruction=instruction,
            artifacts_dir=ARTIFACTS_DIR,
            user_feedback=user_feedback,
        )
        if state.status == "COMPLETED":
            self._record_history(target_dir, instruction, f"Agentic loop completed: {state.plan.summary if state.plan else instruction}")
        return state

    def get_agent_state(self, run_id: str):
        """
        Retrieve current agent run state by run_id.
        """
        from app.codegen.agent import get_agent_run_state
        return get_agent_run_state(run_id)

    def _create_snapshot(self, target_dir: Path, snapshot_dir: Path) -> Path:
        """Create snapshot copy of target_dir."""
        snapshot_dir.mkdir(parents=True, exist_ok=True)
        ignored = {"__pycache__", ".pytest_cache", ".git", "snapshots"}
        for item in target_dir.iterdir():
            if item.name in ignored:
                continue
            if item.is_dir():
                shutil.copytree(item, snapshot_dir / item.name, dirs_exist_ok=True)
            else:
                shutil.copy2(item, snapshot_dir / item.name)
        return snapshot_dir

    def _restore_snapshot(self, target_dir: Path, snapshot_dir: Path):
        """Restore project directory from snapshot copy."""
        if not snapshot_dir.exists():
            return
        ignored = {"__pycache__", ".pytest_cache", ".git", "snapshots"}
        for item in target_dir.iterdir():
            if item.name in ignored:
                continue
            if item.is_dir():
                shutil.rmtree(item)
            else:
                item.unlink()

        for item in snapshot_dir.iterdir():
            if item.is_dir():
                shutil.copytree(item, target_dir / item.name, dirs_exist_ok=True)
            else:
                shutil.copy2(item, target_dir / item.name)

    def get_project_history(self, project_id: str) -> List[HistoryItem]:
        """Load history list from project.json."""
        target_dir = resolve_project_dir(project_id)
        meta_path = target_dir / "project.json"
        if not meta_path.exists():
            return [HistoryItem(step=1, action="Initial generation", timestamp=time.time())]
        try:
            data = json.loads(meta_path.read_text(encoding="utf-8"))
            raw_h = data.get("history", [])
            return [HistoryItem(**item) for item in raw_h]
        except Exception:
            return [HistoryItem(step=1, action="Initial generation", timestamp=time.time())]

    def _record_history(self, target_dir: Path, instruction: str, summary: str) -> List[HistoryItem]:
        """Record modification step in project.json."""
        meta_path = target_dir / "project.json"
        data = {}
        if meta_path.exists():
            try:
                data = json.loads(meta_path.read_text(encoding="utf-8"))
            except Exception:
                pass

        history = data.get("history", [])
        if not history:
            history.append({
                "step": 1,
                "action": "Initial generation",
                "timestamp": time.time(),
                "summary": "Initial project setup",
            })

        next_step = len(history) + 1
        history.append({
            "step": next_step,
            "action": instruction,
            "timestamp": time.time(),
            "summary": summary,
        })

        data["history"] = history
        meta_path.write_text(json.dumps(data, indent=2), encoding="utf-8")
        return [HistoryItem(**item) for item in history]

    def get_project_report(self, project_id: str) -> dict[str, Any]:
        """Generate comprehensive technical intelligence report for project."""
        from app.codegen.agent.verifier import verify_project
        target_dir = resolve_project_dir(project_id)
        if not target_dir.exists():
            raise FileNotFoundError(f"Project '{project_id}' not found.")

        meta = {}
        meta_file = target_dir / "project.json"
        if meta_file.exists():
            try:
                meta = json.loads(meta_file.read_text(encoding="utf-8"))
            except Exception:
                pass

        history = self.get_project_history(project_id)
        file_tree = self.get_file_tree(project_id)
        all_files = []
        for p in target_dir.rglob("*"):
            if p.is_file() and not any(part in ("__pycache__", ".pytest_cache", ".git", "snapshots") for part in p.parts):
                all_files.append(str(p.relative_to(target_dir)).replace("\\", "/"))

        test_res = self.run_tests(project_id)
        verif = verify_project(target_dir, test_result=test_res)

        model_meta = None
        models_dir = BASE_DIR / "models" / project_id
        if (models_dir / "metrics.json").exists():
            try:
                model_meta = json.loads((models_dir / "metrics.json").read_text(encoding="utf-8"))
            except Exception:
                pass

        zip_p = ARTIFACTS_DIR / project_id / f"{project_id}.zip"
        zip_available = zip_p.exists() or bool(list((ARTIFACTS_DIR / project_id).glob("*.zip"))) if (ARTIFACTS_DIR / project_id).exists() else False

        active_id = meta.get("provider") or (self.provider.provider_id if self.provider else "gemini")
        active_model = meta.get("model") or (self.provider.model if self.provider else "gemini-3.8-flash")

        return {
            "project_id": project_id,
            "project_name": meta.get("project_name", project_id),
            "specification": meta.get("specification") or meta.get("requirement_text") or "Autonomous AI Engineering Service",
            "provider": active_id,
            "model": active_model,
            "status": "COMPLETED" if test_res.success and verif.success else ("FAILED" if not test_res.success else "VERIFIED"),
            "files": sorted(all_files),
            "files_count": len(all_files),
            "file_tree": file_tree.model_dump(),
            "test_result": test_res.model_dump(),
            "verification": verif.model_dump(),
            "history": [h.model_dump() for h in history],
            "metrics": model_meta,
            "zip_available": zip_available,
            "zip_url": f"/api/codegen/{project_id}/download",
            "created_at": meta.get("created_at", time.time()),
        }
