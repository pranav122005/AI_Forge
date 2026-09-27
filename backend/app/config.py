"""
AIForge Production Configuration & Environment Module
=====================================================

Centralizes production configuration parameters, environment parsing,
defaults, validation, and safe setting exposure.
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import List, Optional
from pydantic import BaseModel, Field

try:
    from dotenv import load_dotenv
    _backend_env = Path(__file__).resolve().parents[1] / ".env"
    _root_env = Path(__file__).resolve().parents[2] / ".env"
    if _backend_env.exists():
        load_dotenv(_backend_env)
    if _root_env.exists():
        load_dotenv(_root_env)
    load_dotenv()
except Exception:
    pass


class Settings(BaseModel):
    # Gemini API Settings
    gemini_api_key: Optional[str] = Field(default_factory=lambda: os.getenv("GEMINI_API_KEY"))
    gemini_model: str = Field(default_factory=lambda: os.getenv("GEMINI_MODEL", "gemini-3.8-flash"))
    gemini_base_url: str = Field(
        default_factory=lambda: os.getenv(
            "GEMINI_BASE_URL", "https://generativelanguage.googleapis.com/v1beta/openai/"
        )
    )

    # Server Configuration
    host: str = Field(default_factory=lambda: os.getenv("HOST", "127.0.0.1"))
    port: int = Field(default_factory=lambda: int(os.getenv("PORT", "8000")))

    # CORS & Security Configuration
    allowed_origins: str = Field(
        default_factory=lambda: os.getenv("ALLOWED_ORIGINS", "http://localhost:5173,http://127.0.0.1:5173")
    )
    security_auth_required: bool = Field(
        default_factory=lambda: os.getenv("SECURITY_AUTH_REQUIRED", "false").lower() in ("true", "1", "yes")
    )
    aiforge_api_key: Optional[str] = Field(default_factory=lambda: os.getenv("AIFORGE_API_KEY"))

    # Execution Timeouts & Limits
    generation_timeout: int = Field(default_factory=lambda: int(os.getenv("GENERATION_TIMEOUT", "45")))
    test_timeout: int = Field(default_factory=lambda: int(os.getenv("TEST_TIMEOUT", "60")))
    max_repair_attempts: int = Field(default_factory=lambda: int(os.getenv("MAX_REPAIR_ATTEMPTS", "3")))

    @property
    def cors_origins_list(self) -> List[str]:
        defaults = [
            "http://localhost:5173",
            "http://127.0.0.1:5173",
            "http://localhost:3000",
            "https://aiforge-by-team-agni.vercel.app",
        ]
        if not self.allowed_origins:
            return defaults
        custom = [origin.strip() for origin in self.allowed_origins.split(",") if origin.strip()]
        for d in defaults:
            if d not in custom:
                custom.append(d)
        return custom

    def is_gemini_configured(self) -> bool:
        return bool(self.gemini_api_key and len(self.gemini_api_key.strip()) > 0)


settings = Settings()
