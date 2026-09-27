"""
AIForge Agentic Development Loop Orchestration Engine
======================================================

Orchestrates autonomous software engineering development loops:
1. PROJECT UNDERSTANDING -> 2. RELEVANT FILE SELECTION -> 3. AI PLAN -> 4. MULTI-STEP EXECUTION -> 5. TEST -> 6. REPAIR LOOP -> 7. VERIFICATION -> 8. ZIP PACKAGING.

Features atomic pre-edit snapshot backups, automatic rollback on test/verification failure, bounded repair loops, path security validation, and real-time state tracking.
"""
from __future__ import annotations

import os
import shutil
import time
import uuid
from pathlib import Path
from typing import Dict, List, Optional

from app.codegen.agent.context import (
    build_project_summary,
    read_selected_file_contents,
    select_relevant_files,
)
from app.codegen.agent.models import (
    AgentPlan,
    AgentRunRequest,
    AgentRunState,
    PlanStep,
    VerificationResult,
)
from app.codegen.agent.planner import generate_agent_plan
from app.codegen.agent.verifier import verify_project
from app.codegen.errors import PathValidationError
from app.codegen.modifier import ProjectModifierEngine
from app.codegen.packager import package_project_zip
from app.codegen.repair import RepairEngine
from app.codegen.tester import run_project_tests
from app.codegen.validator import validate_file_path
from app.llm.providers.base import BaseLLMProvider

# Shared in-memory store for active/recent agent run states
_AGENT_RUN_STATES: Dict[str, AgentRunState] = {}


def get_agent_run_state(run_id: str) -> Optional[AgentRunState]:
    """Retrieve an active or completed agent run state by run_id."""
    return _AGENT_RUN_STATES.get(run_id)


class AgentExecutionEngine:
    """
    Autonomous Agentic Development Loop engine.
    """

    def __init__(self, provider: Optional[BaseLLMProvider] = None, max_repair_attempts: int = 3) -> None:
        self.provider = provider
        self.max_repair_attempts = max_repair_attempts
        self.repair_engine = RepairEngine(provider=provider, max_retries=max_repair_attempts)
        self.modifier_engine = ProjectModifierEngine(provider=provider)

    async def execute_agent_loop(
        self,
        target_root: Path,
        project_id: str,
        instruction: str,
        artifacts_dir: Path,
        user_feedback: Optional[str] = None,
    ) -> AgentRunState:
        """
        Execute the full Agentic Development Loop.
        """
        run_id = f"run_{str(uuid.uuid4())[:8]}"
        state = AgentRunState(
            run_id=run_id,
            project_id=project_id,
            instruction=instruction,
            status="ANALYZING",
            created_at=time.time(),
            updated_at=time.time(),
        )
        _AGENT_RUN_STATES[run_id] = state

        # Create snapshot directory for rollback safety
        snapshots_dir = artifacts_dir / project_id / "snapshots"
        snapshots_dir.mkdir(parents=True, exist_ok=True)
        snapshot_dir = snapshots_dir / f"snapshot_{int(time.time())}_{run_id}"

        def _update_state(status: str, step_idx: int = 0, log_summary: Optional[str] = None):
            from app.codegen.agent.models import validate_state_transition
            if not validate_state_transition(state.status, status):
                raise ValueError(f"Invalid state transition from '{state.status}' to '{status}'.")
            state.status = status
            state.current_step_index = step_idx
            state.updated_at = time.time()
            if log_summary:
                state.history.append({
                    "step_index": step_idx,
                    "status": status,
                    "summary": log_summary,
                    "timestamp": time.time(),
                })
            _AGENT_RUN_STATES[run_id] = state

        try:
            # Step 1: Project Understanding & Relevant File Selection
            _update_state("ANALYZING", 0, "Analyzing project structure and selecting relevant files")
            summary = build_project_summary(target_root)
            selected_files = await select_relevant_files(target_root, instruction, summary, self.provider)
            file_contents = read_selected_file_contents(target_root, selected_files)

            # Step 2: Agent Planning
            _update_state("PLANNING", 1, "Generating structured multi-step execution plan")
            plan = await generate_agent_plan(target_root, instruction, summary, selected_files, file_contents, self.provider)
            state.plan = plan
            state.total_steps = len(plan.steps)
            _update_state("PLANNING", 1, f"Plan generated with {len(plan.steps)} steps: {plan.summary}")

            # Step 3: Create Snapshot Backup before modifications
            self._create_snapshot(target_root, snapshot_dir)

            # Step 4: Multi-Step Execution
            _update_state("EXECUTING", 2, "Executing plan steps and applying code changes")

            # Request structured modification manifest from ProjectModifierEngine
            mod_manifest = await self.modifier_engine.generate_modification(
                target_root=target_root,
                instruction=instruction,
                user_feedback=user_feedback,
            )

            # Apply file change actions safely
            for change in mod_manifest.changes:
                rel_path = change.path.strip().replace("\\", "/")
                if rel_path in (".env", ".env.local", "secrets.json", "id_rsa"):
                    raise PathValidationError(f"Modification of protected file '{rel_path}' is forbidden.")

                target_file = validate_file_path(rel_path, target_root)

                if change.action == "delete":
                    if target_file.exists():
                        target_file.unlink()
                        state.files_deleted.append(rel_path)
                else:
                    existed = target_file.exists()
                    target_file.parent.mkdir(parents=True, exist_ok=True)
                    target_file.write_text(change.content or "", encoding="utf-8")
                    if existed:
                        state.files_modified.append(rel_path)
                    else:
                        state.files_created.append(rel_path)

            _update_state("EXECUTING", 2, f"Applied changes: +{len(state.files_created)}, ~{len(state.files_modified)}, -{len(state.files_deleted)}")

            # Step 5: Testing
            _update_state("TESTING", 3, "Running automated test suite")
            test_res = run_project_tests(target_root)
            state.test_result = {
                "success": test_res.success,
                "exit_code": test_res.exit_code,
                "stdout": test_res.stdout,
                "stderr": test_res.stderr,
                "duration": test_res.duration_seconds,
            }

            # Step 6: Automatic Repair Loop if tests fail
            if not test_res.success:
                _update_state("REPAIRING", 4, "Test failure detected. Initiating automated repair loop")
                repair_res = await self.repair_engine.repair_project(
                    target_root=target_root,
                    test_output=test_res.stderr + "\n" + test_res.stdout,
                    user_feedback=user_feedback,
                )
                state.repair_attempts = repair_res.attempts
                if repair_res.test_result:
                    test_res = repair_res.test_result
                    state.test_result = {
                        "success": test_res.success,
                        "exit_code": test_res.exit_code,
                        "stdout": test_res.stdout,
                        "stderr": test_res.stderr,
                        "duration": test_res.duration_seconds,
                    }

                if not repair_res.success:
                    # Repair exhausted -> Rollback snapshot
                    self._restore_snapshot(target_root, snapshot_dir)
                    state.status = "ROLLED_BACK"
                    state.error = f"Automated repair failed after {repair_res.attempts} attempts: {test_res.stderr or 'Tests failed'}"
                    _update_state("ROLLED_BACK", 4, state.error)
                    return state

            # Step 7: Verification Stage
            _update_state("VERIFYING", 5, "Performing final multi-check project verification")
            verif_res = verify_project(target_root, test_result=test_res)
            state.verification_result = verif_res

            if not verif_res.success:
                # Verification failed -> Rollback snapshot
                self._restore_snapshot(target_root, snapshot_dir)
                state.status = "ROLLED_BACK"
                state.error = f"Verification failed: {verif_res.failed_checks}"
                _update_state("ROLLED_BACK", 5, state.error)
                return state

            # Step 8: ZIP Packaging
            _update_state("PACKAGING", 6, "Packaging verified project into ZIP artifact")
            zip_artifact_path = artifacts_dir / project_id / f"aiforge-project.zip"
            package_project_zip(target_root, zip_artifact_path)
            state.zip_available = True
            state.zip_path = f"/api/codegen/{project_id}/download"

            # Final Success State
            _update_state("COMPLETED", 7, "Agentic Development Loop completed successfully")
            return state

        except Exception as exc:
            # Unexpected exception -> Rollback snapshot if created
            if snapshot_dir.exists():
                self._restore_snapshot(target_root, snapshot_dir)
            state.status = "ROLLED_BACK"
            state.error = str(exc)
            _update_state("ROLLED_BACK", 0, f"Error in agent loop: {exc}")
            return state

    def _create_snapshot(self, target_dir: Path, snapshot_dir: Path) -> None:
        """Create a clean snapshot backup of target project files."""
        if snapshot_dir.exists():
            shutil.rmtree(snapshot_dir)
        snapshot_dir.mkdir(parents=True, exist_ok=True)

        ignored = {"__pycache__", ".pytest_cache", ".git", "node_modules", "venv", ".venv", "snapshots"}
        for item in target_dir.iterdir():
            if item.name in ignored:
                continue
            dest = snapshot_dir / item.name
            if item.is_dir():
                shutil.copytree(item, dest, ignore=shutil.ignore_patterns("__pycache__", "*.pyc", ".pytest_cache"))
            else:
                shutil.copy2(item, dest)

    def _restore_snapshot(self, target_dir: Path, snapshot_dir: Path) -> None:
        """Restore target project from snapshot backup."""
        if not snapshot_dir.exists():
            return

        ignored = {"__pycache__", ".pytest_cache", ".git", "node_modules", "venv", ".venv", "snapshots"}
        for item in list(target_dir.iterdir()):
            if item.name in ignored:
                continue
            if item.is_dir():
                shutil.rmtree(item)
            else:
                item.unlink()

        for item in snapshot_dir.iterdir():
            dest = target_dir / item.name
            if item.is_dir():
                shutil.copytree(item, dest)
            else:
                shutil.copy2(item, dest)
