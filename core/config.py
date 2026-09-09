"""
core/config.py — reads all settings from .env
"""
import os
from dotenv import load_dotenv

load_dotenv()


class Settings:
    ANTHROPIC_API_KEY:    str  = os.getenv("ANTHROPIC_API_KEY", "")
    ANTHROPIC_MODEL:      str  = os.getenv("ANTHROPIC_MODEL", "claude-sonnet-4-6")
    HOST:                 str  = os.getenv("HOST", "0.0.0.0")
    PORT:                 int  = int(os.getenv("PORT", "8080"))
    RELOAD:               bool = os.getenv("RELOAD", "true").lower() == "true"
    # URLs for factory ↔ runtime communication
    FACTORY_URL:          str  = os.getenv("FACTORY_URL", "http://localhost:8080")
    RUNTIME_CALLBACK_URL: str  = os.getenv("RUNTIME_CALLBACK_URL", "http://localhost:8081")


settings = Settings()
