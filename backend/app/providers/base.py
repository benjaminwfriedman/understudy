from abc import ABC, abstractmethod
from typing import Optional, Dict, Any, List
from langchain.schema import BaseMessage


class BaseLLMProvider(ABC):
    """Abstract base class for LLM providers."""
    
    @abstractmethod
    async def generate(self, prompt: str, **kwargs) -> str:
        """Generate text from a prompt."""
        pass
    
    @abstractmethod
    async def generate_with_messages(
        self, 
        messages: List[BaseMessage], 
        **kwargs
    ) -> str:
        """Generate text from LangChain message format."""
        pass
    
    @abstractmethod
    def estimate_cost(self, input_tokens: int, output_tokens: int) -> float:
        """Estimate the cost of generation in USD."""
        pass
    
    @abstractmethod
    def count_tokens(self, text: str) -> int:
        """Count the number of tokens in text."""
        pass
    
    @abstractmethod
    def to_langchain_llm(self):
        """Return a LangChain-compatible LLM wrapper."""
        pass
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get information about the model."""
        return {
            "provider": self.__class__.__name__,
            "supports_streaming": False,
            "supports_functions": False,
            "max_tokens": 4096
        }