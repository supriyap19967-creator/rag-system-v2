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
    Prioritizes fast execution via Groq (~0.8s) or Gemini 2.5 Flash on OpenRouter (~1.2s),
    falling back to meta-llama/llama-3.3-70b-instruct only when necessary.
    """
    override_model = os.getenv(env_var_name)
    groq_key = os.getenv("GROQ_API_KEY")
    openrouter_key = os.getenv("OPENROUTER_API_KEY")
    target_model_name = override_model or fallback_groq_model

    # 1. If Groq API key is present and model is a groq model, use ultra-fast native Groq
    if str(target_model_name).startswith("groq:") and groq_key:
        return target_model_name

    # 2. If OpenRouter is available, use fast Google Gemini 2.5 Flash (~1.0s)
    if openrouter_key:
        try:
            from pydantic_ai.models.openai import OpenAIChatModel
            from pydantic_ai.providers.openai import OpenAIProvider
            provider = OpenAIProvider(
                base_url="https://openrouter.ai/api/v1",
                api_key=openrouter_key
            )
            or_model = os.getenv("OPENROUTER_AGENT_MODEL", "google/gemini-2.5-flash")
            return OpenAIChatModel(or_model, provider=provider)
        except Exception as e:
            logger.warning(f"Failed to initialize OpenRouter model provider: {e}")

    # Fallback to model string
    return target_model_name
