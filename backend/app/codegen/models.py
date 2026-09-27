"""
AIForge Codegen Models
======================

Pydantic data models for structured code generation, file management, test results, repair tracking, packaging, and project modification.
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field


class FileSpec(BaseModel):
    """Represents a single generated code file specification."""

    path: str = Field(..., description="Relative file path within project root, e.g. 'app/main.py'")
    content: str = Field(..., description="Complete source code content for the file")
    description: Optional[str] = Field(default="", description="Brief summary of file purpose")


class ManifestChunk(BaseModel):
    """Chunk of generated file specifications returned by LLM or code generator."""

    chunk_index: int = Field(default=0, description="0-indexed chunk number")
    total_chunks: int = Field(default=1, description="Total planned chunks")
    files: List[FileSpec] = Field(default_factory=list, description="Files included in this chunk")
    rationale: Optional[str] = Field(default="", description="LLM design rationale for this chunk")


class CodegenRequest(BaseModel):
    """Request payload to trigger LLM code generation."""

    project_id: Optional[str] = Field(default=None, description="Optional existing project ID")
    requirement_text: Optional[str] = Field(default=None, description="Natural language user specification")
    spec: Optional[Dict[str, Any]] = Field(default=None, description="RequirementSpec dictionary if parsed")
    extra_instructions: Optional[str] = Field(default=None, description="Optional extra LLM instructions")
    project_name: Optional[str] = Field(default="aiforge-project", description="Project slug/name")
    provider: Optional[str] = Field(default=None, description="Optional LLM provider override (e.g. 'gemini', 'openai', 'anthropic')")
    model: Optional[str] = Field(default=None, description="Optional model name override")


class TestResult(BaseModel):
    """Result of running tests on a project codebase."""

    __test__ = False
    success: bool = Field(..., description="True if test execution succeeded with exit code 0")
    exit_code: int = Field(..., description="Subprocess return code")
    stdout: str = Field(default="", description="Captured standard output")
    stderr: str = Field(default="", description="Captured error output")
    duration_seconds: float = Field(default=0.0, description="Execution duration in seconds")


class RepairRequest(BaseModel):
    """Request payload for automated code repair."""

    project_id: str = Field(..., description="Project ID to repair")
    test_output: Optional[str] = Field(default=None, description="Failed test traceback or error output")
    user_feedback: Optional[str] = Field(default=None, description="Optional user guidance for repair")


class CodegenModifyRequest(BaseModel):
    """Request payload for safe project modification."""

    project_id: Optional[str] = Field(default=None, description="Target project ID to modify")
    instruction: str = Field(..., min_length=3, description="Modification requirement / instructions")
    user_feedback: Optional[str] = Field(default=None, description="Optional extra feedback")
    provider: Optional[str] = Field(default=None, description="Optional LLM provider override")
    model: Optional[str] = Field(default=None, description="Optional model override")


class FileChangeSpec(BaseModel):
    """Specification of a single file change action (create, modify, delete)."""

    action: str = Field(..., description="Action: 'create', 'modify', or 'delete'")
    path: str = Field(..., description="Relative file path within project root")
    content: Optional[str] = Field(default=None, description="Source code content if creating or modifying")
    description: Optional[str] = Field(default="", description="Summary of changes")


class ProjectModificationManifest(BaseModel):
    """Structured modification output returned by LLM or modifier engine."""

    summary: str = Field(..., description="High level summary of modifications applied")
    changes: List[FileChangeSpec] = Field(default_factory=list, description="List of file change actions")
    tests_to_update: List[str] = Field(default_factory=list, description="Test files affected")


class HistoryItem(BaseModel):
    """Single step in project modification history."""

    step: int
    action: str
    timestamp: float
    summary: Optional[str] = None


class CodegenModifyResult(BaseModel):
    """Result object returned after modifying an existing project."""

    project_id: str
    instruction: str
    summary: str
    files_created: List[str] = Field(default_factory=list)
    files_modified: List[str] = Field(default_factory=list)
    files_deleted: List[str] = Field(default_factory=list)
    tests_passed: int = 0
    tests_failed: int = 0
    repair_attempts: int = 0
    status: str  # "SUCCESS", "FAILED", "ROLLED_BACK"
    zip_available: bool = True
    test_result: Optional[TestResult] = None
    zip_path: Optional[str] = None
    history: List[HistoryItem] = Field(default_factory=list)
    error: Optional[str] = None


class RepairResult(BaseModel):
    """Result of code repair process."""

    success: bool = Field(..., description="Whether repair fixed all issues and passed tests")
    retry_count: int = Field(default=0, description="Number of repair retries attempted")
    fixed_files: List[str] = Field(default_factory=list, description="List of file paths that were modified")
    test_result: Optional[TestResult] = Field(default=None, description="Final test execution result")
    error: Optional[str] = Field(default=None, description="Error message if repair failed")

    @property
    def attempts(self) -> int:
        return self.retry_count


class CodegenResult(BaseModel):
    """Complete result object after code generation, build, test, and packaging."""

    project_id: str = Field(..., description="Unique ID of generated project")
    status: str = Field(..., description="Status: 'completed', 'repaired', 'failed', 'pending'")
    generated_files: List[str] = Field(default_factory=list, description="List of generated relative file paths")
    test_result: Optional[TestResult] = Field(default=None, description="Result of running project tests")
    zip_path: Optional[str] = Field(default=None, description="Path or download URL of ZIP artifact")
    error: Optional[str] = Field(default=None, description="Error details if generation failed")
    provider: Optional[str] = Field(default=None, description="LLM provider used for generation")
    model: Optional[str] = Field(default=None, description="LLM model used for generation")


class FileTreeNode(BaseModel):
    """Tree node representing directory/file structure for UI file explorer."""

    name: str = Field(..., description="Directory or file name")
    path: str = Field(..., description="Relative path from project root")
    is_dir: bool = Field(..., description="True if node is a directory")
    size: Optional[int] = Field(default=None, description="File size in bytes if applicable")
    children: Optional[List[FileTreeNode]] = Field(default=None, description="Child nodes if directory")


FileTreeNode.model_rebuild()
