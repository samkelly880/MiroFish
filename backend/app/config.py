"""
Configuration management.
Load settings from the project root `.env` file.
"""

import os
import shutil
from dotenv import load_dotenv

# Load the project-root `.env` file.
# Path: MiroFish/.env (relative to backend/app/config.py)
project_root_env = os.path.join(os.path.dirname(__file__), '../../.env')

if os.path.exists(project_root_env):
    load_dotenv(project_root_env, override=True)
else:
    # If root `.env` is missing, load from the process environment (production).
    load_dotenv(override=True)


class Config:
    """Flask configuration."""

    # Flask
    SECRET_KEY = os.environ.get('SECRET_KEY', 'mirofish-secret-key')
    DEBUG = os.environ.get('FLASK_DEBUG', 'False').lower() == 'true'

    # JSON: disable ASCII escaping so non-ASCII text is preserved
    JSON_AS_ASCII = False

    # LLM provider: grok-cli (primary) or openai-compatible (optional)
    LLM_PROVIDER = os.environ.get('LLM_PROVIDER', 'grok-cli')

    # Optional OpenAI-compatible HTTP API (secondary provider)
    LLM_API_KEY = os.environ.get('LLM_API_KEY')
    LLM_BASE_URL = os.environ.get('LLM_BASE_URL', 'https://api.openai.com/v1')
    LLM_MODEL_NAME = os.environ.get('LLM_MODEL_NAME', 'gpt-4o-mini')

    # Grok CLI settings (primary; no API key required when locally authenticated)
    GROK_CLI_BIN = os.environ.get('GROK_CLI_BIN', 'grok')
    GROK_CLI_MODEL = os.environ.get('GROK_CLI_MODEL')
    GROK_CLI_TIMEOUT_SECONDS = int(os.environ.get('GROK_CLI_TIMEOUT_SECONDS', '300'))

    # Zep
    ZEP_API_KEY = os.environ.get('ZEP_API_KEY')

    # Uploads
    MAX_CONTENT_LENGTH = 50 * 1024 * 1024  # 50MB
    UPLOAD_FOLDER = os.path.join(os.path.dirname(__file__), '../uploads')
    ALLOWED_EXTENSIONS = {'pdf', 'md', 'txt', 'markdown'}

    # Text processing
    DEFAULT_CHUNK_SIZE = 500  # default chunk size
    DEFAULT_CHUNK_OVERLAP = 50  # default chunk overlap

    # OASIS simulation
    OASIS_DEFAULT_MAX_ROUNDS = int(os.environ.get('OASIS_DEFAULT_MAX_ROUNDS', '10'))
    OASIS_SIMULATION_DATA_DIR = os.path.join(os.path.dirname(__file__), '../uploads/simulations')

    # OASIS platform actions
    OASIS_TWITTER_ACTIONS = [
        'CREATE_POST', 'LIKE_POST', 'REPOST', 'FOLLOW', 'DO_NOTHING', 'QUOTE_POST'
    ]
    OASIS_REDDIT_ACTIONS = [
        'LIKE_POST', 'DISLIKE_POST', 'CREATE_POST', 'CREATE_COMMENT',
        'LIKE_COMMENT', 'DISLIKE_COMMENT', 'SEARCH_POSTS', 'SEARCH_USER',
        'TREND', 'REFRESH', 'DO_NOTHING', 'FOLLOW', 'MUTE'
    ]

    # Report Agent
    REPORT_AGENT_MAX_TOOL_CALLS = int(os.environ.get('REPORT_AGENT_MAX_TOOL_CALLS', '5'))
    REPORT_AGENT_MAX_REFLECTION_ROUNDS = int(os.environ.get('REPORT_AGENT_MAX_REFLECTION_ROUNDS', '2'))
    REPORT_AGENT_TEMPERATURE = float(os.environ.get('REPORT_AGENT_TEMPERATURE', '0.5'))

    @classmethod
    def resolved_llm_provider(cls) -> str:
        from .providers.factory import resolve_provider_name

        return resolve_provider_name(cls.LLM_PROVIDER)

    @classmethod
    def validate(cls) -> list[str]:
        """Validate required configuration for the active LLM provider."""
        errors: list[str] = []
        try:
            provider = cls.resolved_llm_provider()
        except ValueError as exc:
            errors.append(str(exc))
            provider = None

        if provider == "grok-cli":
            binary = cls.GROK_CLI_BIN or "grok"
            found = (
                os.path.isfile(binary) and os.access(binary, os.X_OK)
                if (os.path.isabs(binary) or os.sep in binary)
                else shutil.which(binary)
            )
            if not found:
                errors.append(
                    f"Grok CLI binary not found ({binary!r}). "
                    "Install Grok CLI and run `grok login`. "
                    "No xAI API key is required for the normal Grok CLI workflow."
                )
        elif provider == "openai-compatible":
            if not cls.LLM_API_KEY:
                errors.append(
                    "LLM_API_KEY is not configured "
                    "(required for openai-compatible provider)"
                )

        if not cls.ZEP_API_KEY:
            errors.append("ZEP_API_KEY is not configured")
        if os.environ.get("ZEP_API_URL"):
            errors.append(
                "ZEP_API_URL is not supported; MiroFish only connects to Zep Cloud"
            )
        if cls.DEBUG:
            import warnings

            warnings.warn(
                "Flask DEBUG mode is enabled. Do not use in production.",
                RuntimeWarning,
            )
        return errors
