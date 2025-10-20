from typing import List, Dict, Any, Optional
from app.providers.base import BaseLLMProvider
from app.core.config import settings
import logging

try:
    from langchain_openai import ChatOpenAI
    from langchain.schema import BaseMessage, HumanMessage, AIMessage, SystemMessage
    from openai import AsyncOpenAI
    import tiktoken
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False
    # Dummy classes for type hints
    class ChatOpenAI: pass
    class BaseMessage: pass
    class HumanMessage: pass
    class AIMessage: pass  
    class SystemMessage: pass
    class AsyncOpenAI: pass
    class tiktoken: 
        @staticmethod
        def encoding_for_model(model): return None
        @staticmethod
        def get_encoding(name): return None

logger = logging.getLogger(__name__)


class OpenAIProvider(BaseLLMProvider):
    """OpenAI API provider implementation."""
    
    PRICING = {
        'gpt-4': {'input': 0.03, 'output': 0.06},
        'gpt-4-turbo-preview': {'input': 0.01, 'output': 0.03},
        'gpt-3.5-turbo': {'input': 0.0015, 'output': 0.002},
        'gpt-3.5-turbo-16k': {'input': 0.003, 'output': 0.004}
    }
    
    def __init__(self, model: str = "gpt-3.5-turbo", api_key: Optional[str] = None):
        if not OPENAI_AVAILABLE:
            raise ImportError("OpenAI dependencies not available. Install with: pip install openai langchain-openai tiktoken")
            
        self.model = model
        self.api_key = api_key or settings.OPENAI_API_KEY
        if not self.api_key:
            raise ValueError("OpenAI API key not provided")
        
        self.client = AsyncOpenAI(api_key=self.api_key)
        self._langchain_llm = None
        
        # Initialize tokenizer
        try:
            self.encoding = tiktoken.encoding_for_model(model)
        except KeyError:
            self.encoding = tiktoken.get_encoding("cl100k_base")
    
    async def generate(self, prompt: str, **kwargs) -> str:
        """Generate text from a prompt."""
        try:
            response = await self.client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                temperature=kwargs.get("temperature", 0.7),
                max_tokens=kwargs.get("max_tokens", 256),
                top_p=kwargs.get("top_p", 1.0),
                frequency_penalty=kwargs.get("frequency_penalty", 0),
                presence_penalty=kwargs.get("presence_penalty", 0),
                stop=kwargs.get("stop", None)
            )
            return response.choices[0].message.content
        except Exception as e:
            logger.error(f"OpenAI generation error: {e}")
            raise
    
    async def generate_with_messages(
        self, 
        messages: List[BaseMessage], 
        **kwargs
    ) -> str:
        """Generate text from LangChain message format."""
        formatted_messages = []
        for msg in messages:
            if isinstance(msg, SystemMessage):
                formatted_messages.append({"role": "system", "content": msg.content})
            elif isinstance(msg, HumanMessage):
                formatted_messages.append({"role": "user", "content": msg.content})
            elif isinstance(msg, AIMessage):
                formatted_messages.append({"role": "assistant", "content": msg.content})
            else:
                formatted_messages.append({"role": "user", "content": str(msg.content)})
        
        try:
            response = await self.client.chat.completions.create(
                model=self.model,
                messages=formatted_messages,
                temperature=kwargs.get("temperature", 0.7),
                max_tokens=kwargs.get("max_tokens", 256),
                top_p=kwargs.get("top_p", 1.0),
                frequency_penalty=kwargs.get("frequency_penalty", 0),
                presence_penalty=kwargs.get("presence_penalty", 0),
                stop=kwargs.get("stop", None)
            )
            return response.choices[0].message.content
        except Exception as e:
            logger.error(f"OpenAI generation with messages error: {e}")
            raise
    
    def estimate_cost(self, input_tokens: int, output_tokens: int) -> float:
        """Estimate the cost of generation in USD."""
        model_costs = self.PRICING.get(self.model, self.PRICING['gpt-3.5-turbo'])
        input_cost = (input_tokens * model_costs['input']) / 1000
        output_cost = (output_tokens * model_costs['output']) / 1000
        return input_cost + output_cost
    
    def count_tokens(self, text: str) -> int:
        """Count the number of tokens in text."""
        return len(self.encoding.encode(text))
    
    def to_langchain_llm(self):
        """Return a LangChain-compatible LLM wrapper."""
        if not self._langchain_llm:
            self._langchain_llm = ChatOpenAI(
                model=self.model,
                openai_api_key=self.api_key,
                temperature=0.7
            )
        return self._langchain_llm
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get information about the model."""
        base_info = super().get_model_info()
        base_info.update({
            "provider": "OpenAI",
            "model": self.model,
            "supports_streaming": True,
            "supports_functions": True,
            "max_tokens": 4096 if "gpt-3.5" in self.model else 8192
        })
        return base_info