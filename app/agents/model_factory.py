from __future__ import annotations

import os
import logging
from typing import Any
from dotenv import load_dotenv

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
load_dotenv(os.path.join(PROJECT_ROOT, ".env"))

logger = logging.getLogger(__name__)

def get_agent_model(env_var_name: str, fallback_groq_model: str = "groq:llama-3.3-70b-versatile") -> Any:
    """
    Returns an optimized model instance for Pydantic-AI agents.
    Prioritizes OpenRouter (meta-llama/llama-3.3-70b-instruct) for rock-solid stability,
    falling back to Groq or configured environment variable if needed.
    """
    openrouter_key = os.getenv("OPENROUTER_API_KEY")
    override_model = os.getenv(env_var_name)
    
    if openrouter_key:
        try:
            from pydantic_ai.models.openai import OpenAIChatModel
            from pydantic_ai.providers.openai import OpenAIProvider
            provider = OpenAIProvider(
                base_url="https://openrouter.ai/api/v1",
                api_key=openrouter_key
            )
            # Use Llama 3.3 70B via OpenRouter
            return OpenAIChatModel("meta-llama/llama-3.3-70b-instruct", provider=provider)
        except Exception as e:
            logger.warning(f"Failed to initialize OpenRouter model provider: {e}")

    # Fallback to model string
    return override_model or fallback_groq_model
