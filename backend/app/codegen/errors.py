"""
AIForge Codegen Errors
======================

Custom exception hierarchy for code generation, validation, execution, and packaging.
"""
from __future__ import annotations


class CodegenError(Exception):
    """Base exception for all codegen errors."""
    pass


class PathValidationError(CodegenError, ValueError):
    """Raised when a generated file path fails safety/traversal validation."""
    pass


class GenerationError(CodegenError, RuntimeError):
    """Raised when code generation fails or produces invalid manifest structure."""
    pass


class TestExecutionError(CodegenError, RuntimeError):
    """Raised when code test execution fails or encounters systemic runtime error."""
    pass


class PackagingError(CodegenError, RuntimeError):
    """Raised when ZIP packaging fails."""
    pass
