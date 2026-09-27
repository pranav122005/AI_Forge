"""
AIForge Codegen Package
=======================

Full LLM Code Generation, Safe Validation, Test Execution, Automated Repair, and ZIP Packaging Engine.
"""
from __future__ import annotations

from app.codegen.errors import (
    CodegenError,
    GenerationError,
    PackagingError,
    PathValidationError,
    TestExecutionError,
)
from app.codegen.models import (
    CodegenModifyRequest,
    CodegenModifyResult,
    CodegenRequest,
    CodegenResult,
    FileChangeSpec,
    FileSpec,
    FileTreeNode,
    HistoryItem,
    ManifestChunk,
    ProjectModificationManifest,
    RepairRequest,
    RepairResult,
    TestResult,
)
from app.codegen.agent import (
    AgentPlan,
    AgentRunRequest,
    AgentRunState,
    FileSelection,
    PlanStep,
    VerificationResult,
)
from app.codegen.service import CodegenService

__all__ = [
    "CodegenError",
    "PathValidationError",
    "GenerationError",
    "TestExecutionError",
    "PackagingError",
    "FileSpec",
    "ManifestChunk",
    "CodegenRequest",
    "CodegenModifyRequest",
    "CodegenModifyResult",
    "FileChangeSpec",
    "ProjectModificationManifest",
    "HistoryItem",
    "CodegenResult",
    "TestResult",
    "RepairRequest",
    "RepairResult",
    "FileTreeNode",
    "CodegenService",
    "AgentPlan",
    "PlanStep",
    "AgentRunRequest",
    "AgentRunState",
    "FileSelection",
    "VerificationResult",
]
