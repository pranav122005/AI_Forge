"""
AIForge Security, Validation, Authentication & Redaction Module
================================================================

Provides:
1. Configurable CORS setup.
2. Input sanitization and regex validation for project_id, run_id, and file paths.
3. Secret redaction for logging and exception output.
4. Tenant isolation and API Key authentication middleware.
"""
from __future__ import annotations

import os
import re
from typing import Dict, List, Optional, Set
from fastapi import Header, HTTPException, Request, Security, status
from fastapi.security import APIKeyHeader

ID_REGEX = re.compile(r"^[a-zA-Z0-9_-]{1,64}$")

FORBIDDEN_PATHS: Set[str] = {
    ".env",
    ".env.local",
    ".env.production",
    ".env.development",
    "secrets.json",
    "id_rsa",
    "id_rsa.pub",
    "credentials.json",
}

SECRET_REDACT_PATTERNS = [
    (re.compile(r"(AIzaSy[0-9A-Za-z-_]{20,50})"), "[REDACTED_GEMINI_KEY]"),
    (re.compile(r"(sk-ant-[0-9A-Za-z-_]{20,})"), "[REDACTED_ANTHROPIC_KEY]"),
    (re.compile(r"(sk-[0-9A-Za-z-_]{20,})"), "[REDACTED_OPENAI_KEY]"),
    (re.compile(r"(AKIA[0-9A-Z]{16})"), "[REDACTED_AWS_KEY]"),
    (re.compile(r"(bearer\s+[a-zA-Z0-9._-]+)", re.IGNORECASE), "Bearer [REDACTED_TOKEN]"),
    (re.compile(r"(password\s*[:=]\s*['\"])[^'\"]+(['\"])", re.IGNORECASE), r"\1[REDACTED_PASSWORD]\2"),
]


def redact_secrets(text: str, extra_secrets: Optional[List[str]] = None) -> str:
    """
    Redact sensitive keys, tokens, and credentials from text output or error logs.
    """
    if not text:
        return ""
    redacted = str(text)
    if extra_secrets:
        for secret in extra_secrets:
            if secret and isinstance(secret, str) and len(secret) >= 4:
                redacted = redacted.replace(secret, "[REDACTED_API_KEY]")
    for pattern, replacement in SECRET_REDACT_PATTERNS:
        redacted = pattern.sub(replacement, redacted)
    return redacted


def validate_id_string(value: str, field_name: str = "ID") -> str:
    """
    Validate that an ID string (project_id or run_id) conforms to safe alphanumeric rules
    and contains no path traversal sequences or special characters.
    """
    if not value or not isinstance(value, str):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Invalid {field_name}: string cannot be empty.",
        )
    value_clean = value.strip()
    if ".." in value_clean or "/" in value_clean or "\\" in value_clean or "\0" in value_clean:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Security error: Path traversal or invalid characters in {field_name}.",
        )
    if not ID_REGEX.match(value_clean):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Invalid {field_name} format: must be 1-64 alphanumeric characters, hyphens, or underscores.",
        )
    return value_clean


def get_cors_origins() -> List[str]:
    """
    Parse allowed CORS origins from environment variable ALLOWED_ORIGINS.
    Defaults to local development origins and the production Vercel frontend.
    """
    env_origins = os.getenv("ALLOWED_ORIGINS", "")
    origins = [
        "http://localhost:5173",
        "http://127.0.0.1:5173",
        "http://localhost:3000",
        "https://aiforge-by-team-agni.vercel.app",
    ]
    if env_origins:
        for o in env_origins.split(","):
            o_clean = o.strip()
            if o_clean and o_clean not in origins:
                origins.append(o_clean)
    return origins


# API Key Security Header
api_key_header = APIKeyHeader(name="X-API-Key", auto_error=False)
user_id_header = APIKeyHeader(name="X-User-ID", auto_error=False)

# Optional project ownership registry for tenant isolation tests
_PROJECT_OWNERSHIP: Dict[str, str] = {}


def register_project_owner(project_id: str, owner_id: str):
    """Register project ownership for tenant isolation."""
    _PROJECT_OWNERSHIP[project_id] = owner_id


def verify_api_access(
    project_id: Optional[str] = None,
    x_api_key: Optional[str] = None,
    x_user_id: Optional[str] = None,
) -> bool:
    """
    Verify API access, API Key authentication, and tenant project ownership.
    Returns True if allowed, raises HTTPException if access is forbidden.
    """
    required_key = os.getenv("AIFORGE_API_KEY", "").strip()
    auth_required = os.getenv("SECURITY_AUTH_REQUIRED", "false").lower() == "true"

    # 1. API Key verification if required
    if required_key or auth_required:
        if not x_api_key or (required_key and x_api_key != required_key):
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Unauthorized: Invalid or missing X-API-Key header.",
            )

    # 2. Tenant isolation check for project_id
    if project_id and project_id in _PROJECT_OWNERSHIP:
        owner = _PROJECT_OWNERSHIP[project_id]
        if x_user_id and x_user_id != owner:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail=f"Forbidden: You do not have permission to access project '{project_id}'.",
            )

    return True
