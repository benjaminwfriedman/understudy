from typing import Dict, Any
from app.providers.base import BaseLLMProvider
from app.providers.openai_provider import OpenAIProvider
from app.providers.anthropic_provider import AnthropicProvider
import logging

logger = logging.getLogger(__name__)


class ProviderFactory:
    """Factory for creating LLM provider instances."""
    
    _providers = {
        'openai': OpenAIProvider,
        'anthropic': AnthropicProvider,
    }
    
    @classmethod
    def get_provider(cls, provider_name: str, **config) -> BaseLLMProvider:
        """Create and return a provider instance."""
        provider_class = cls._providers.get(provider_name.lower())
        if not provider_class:
            raise ValueError(f"Unknown provider: {provider_name}")
        
        try:
            return provider_class(**config)
        except Exception as e:
            logger.error(f"Failed to create provider {provider_name}: {e}")
            raise
    
    @classmethod
    def list_providers(cls) -> list:
        """List available provider names."""
        return list(cls._providers.keys())
    
    @classmethod
    def register_provider(cls, name: str, provider_class: type):
        """Register a new provider."""
        if not issubclass(provider_class, BaseLLMProvider):
            raise ValueError(f"{provider_class} must inherit from BaseLLMProvider")
        cls._providers[name.lower()] = provider_class