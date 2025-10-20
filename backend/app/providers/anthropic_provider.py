from typing import List, Dict, Any, Optional
from app.providers.base import BaseLLMProvider
from app.core.config import settings
import logging

try:
    from langchain_anthropic import ChatAnthropic
    from langchain.schema import BaseMessage, HumanMessage, AIMessage, SystemMessage
    from anthropic import AsyncAnthropic
    ANTHROPIC_AVAILABLE = True
except ImportError:
    ANTHROPIC_AVAILABLE = False
    # Dummy classes for type hints
    class ChatAnthropic: pass
    class BaseMessage: pass
    class HumanMessage: pass  
    class AIMessage: pass
    class SystemMessage: pass
    class AsyncAnthropic: pass

logger = logging.getLogger(__name__)


class AnthropicProvider(BaseLLMProvider):
    """Anthropic Claude API provider implementation."""
    
    PRICING = {
        'claude-3-opus': {'input': 0.015, 'output': 0.075},
        'claude-3-sonnet': {'input': 0.003, 'output': 0.015},
        'claude-3-haiku': {'input': 0.00025, 'output': 0.00125},
        'claude-2.1': {'input': 0.008, 'output': 0.024},
        'claude-2': {'input': 0.008, 'output': 0.024},
    }
    
    def __init__(self, model: str = "claude-3-sonnet", api_key: Optional[str] = None):
        if not ANTHROPIC_AVAILABLE:
            raise ImportError("Anthropic dependencies not available. Install with: pip install anthropic langchain-anthropic")
            
        self.model = model
        self.api_key = api_key or settings.ANTHROPIC_API_KEY
        if not self.api_key:
            raise ValueError("Anthropic API key not provided")
        
        self.client = AsyncAnthropic(api_key=self.api_key)
        self._langchain_llm = None
    
    async def generate(self, prompt: str, **kwargs) -> str:
        """Generate text from a prompt."""
        try:
            response = await self.client.messages.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=kwargs.get("max_tokens", 256),
                temperature=kwargs.get("temperature", 0.7),
                top_p=kwargs.get("top_p", 1.0),
                stop_sequences=kwargs.get("stop", None)
            )
            return response.content[0].text
        except Exception as e:
            logger.error(f"Anthropic generation error: {e}")
            raise
    
    async def generate_with_messages(
        self, 
        messages: List[BaseMessage], 
        **kwargs
    ) -> str:
        """Generate text from LangChain message format."""
        formatted_messages = []
        system_prompt = None
        
        for msg in messages:
            if isinstance(msg, SystemMessage):
                system_prompt = msg.content
            elif isinstance(msg, HumanMessage):
                formatted_messages.append({"role": "user", "content": msg.content})
            elif isinstance(msg, AIMessage):
                formatted_messages.append({"role": "assistant", "content": msg.content})
            else:
                formatted_messages.append({"role": "user", "content": str(msg.content)})
        
        try:
            params = {
                "model": self.model,
                "messages": formatted_messages,
                "max_tokens": kwargs.get("max_tokens", 256),
                "temperature": kwargs.get("temperature", 0.7),
                "top_p": kwargs.get("top_p", 1.0),
                "stop_sequences": kwargs.get("stop", None)
            }
            
            if system_prompt:
                params["system"] = system_prompt
            
            response = await self.client.messages.create(**params)
            return response.content[0].text
        except Exception as e:
            logger.error(f"Anthropic generation with messages error: {e}")
            raise
    
    def estimate_cost(self, input_tokens: int, output_tokens: int) -> float:
        """Estimate the cost of generation in USD."""
        model_costs = self.PRICING.get(self.model, self.PRICING['claude-3-sonnet'])
        input_cost = (input_tokens * model_costs['input']) / 1000
        output_cost = (output_tokens * model_costs['output']) / 1000
        return input_cost + output_cost
    
    def count_tokens(self, text: str) -> int:
        """Count the number of tokens in text (approximation for Claude)."""
        # Approximate token count for Claude (roughly 1 token per 4 characters)
        return len(text) // 4
    
    def to_langchain_llm(self):
        """Return a LangChain-compatible LLM wrapper."""
        if not self._langchain_llm:
            self._langchain_llm = ChatAnthropic(
                model=self.model,
                anthropic_api_key=self.api_key,
                temperature=0.7
            )
        return self._langchain_llm
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get information about the model."""
        base_info = super().get_model_info()
        base_info.update({
            "provider": "Anthropic",
            "model": self.model,
            "supports_streaming": True,
            "supports_functions": False,
            "max_tokens": 100000 if "claude-3" in self.model else 200000
        })
        return base_info