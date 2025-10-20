from typing import Optional, List, Any, Dict, Iterator
from langchain_core.language_models.llms import LLM
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import BaseMessage, AIMessage, HumanMessage, SystemMessage
from langchain_core.callbacks.manager import CallbackManagerForLLMRun
from langchain_core.outputs import ChatResult, ChatGeneration, LLMResult, Generation
from pydantic import Field
import httpx
import logging

logger = logging.getLogger(__name__)


class UnderstudyLLM(LLM):
    """LangChain LLM wrapper that routes through Understudy."""
    
    endpoint_id: str = Field(description="Understudy endpoint ID")
    base_url: str = Field(default="http://localhost:8000", description="Understudy API base URL")
    api_key: Optional[str] = Field(default=None, description="API key for authentication")
    max_tokens: int = Field(default=256, description="Maximum tokens to generate")
    temperature: float = Field(default=0.7, description="Sampling temperature")
    top_p: float = Field(default=1.0, description="Top-p sampling parameter")
    timeout: float = Field(default=30.0, description="Request timeout in seconds")
    
    def __init__(self, **kwargs):
        """Initialize the UnderstudyLLM with proper field handling."""
        super().__init__(**kwargs)
        # Ensure fields are properly initialized (not FieldInfo objects)
        if hasattr(self.max_tokens, 'default'):
            self.max_tokens = 256
        if hasattr(self.temperature, 'default'):
            self.temperature = 0.7
        if hasattr(self.top_p, 'default'):
            self.top_p = 1.0
        if hasattr(self.timeout, 'default'):
            self.timeout = 30.0
        if hasattr(self.api_key, 'default'):
            self.api_key = None
    
    @property
    def _llm_type(self) -> str:
        return "understudy"
    
    @property
    def _identifying_params(self) -> Dict[str, Any]:
        """Get the identifying parameters."""
        return {
            "endpoint_id": self.endpoint_id,
            "base_url": self.base_url,
            "max_tokens": self.max_tokens,
            "temperature": self.temperature,
            "top_p": self.top_p,
        }
    
    def _call(
        self,
        prompt: str,
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> str:
        """Call the Understudy inference endpoint."""
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        
        # Filter kwargs to only include JSON-serializable values
        serializable_kwargs = {}
        for key, value in kwargs.items():
            try:
                import json
                json.dumps(value)
                serializable_kwargs[key] = value
            except (TypeError, ValueError):
                # Skip non-serializable values
                continue
        
        # Get actual values from fields (handle FieldInfo objects)
        max_tokens_value = self.max_tokens.default if hasattr(self.max_tokens, 'default') else self.max_tokens
        temperature_value = self.temperature.default if hasattr(self.temperature, 'default') else self.temperature
        top_p_value = self.top_p.default if hasattr(self.top_p, 'default') else self.top_p
        
        # Prepare request data
        request_data = {
            "prompt": prompt,
            "max_tokens": kwargs.get("max_tokens", max_tokens_value),
            "temperature": kwargs.get("temperature", temperature_value),
            "top_p": kwargs.get("top_p", top_p_value),
            "stop": stop,
            "langchain_metadata": {
                "run_id": str(run_manager.run_id) if run_manager else None,
                "llm_type": self._llm_type,
                **serializable_kwargs
            }
        }
        
        with httpx.Client(timeout=self.timeout) as client:
            try:
                response = client.post(
                    f"{self.base_url}/api/v1/inference/{self.endpoint_id}",
                    json=request_data,
                    headers=headers
                )
                response.raise_for_status()
                result = response.json()
                
                # Add callback information if available
                if run_manager:
                    run_manager.on_llm_end(
                        LLMResult(
                            generations=[[Generation(text=result["output"])]],
                            llm_output={
                                "model_used": result.get("model_used"),
                                "latency_ms": result.get("latency_ms"),
                                "cost_usd": result.get("cost_usd"),
                                "carbon_emissions": result.get("carbon_emissions")
                            }
                        )
                    )
                
                return result["output"]
                
            except httpx.HTTPStatusError as e:
                error_msg = f"HTTP {e.response.status_code}: {e.response.text}"
                logger.error(f"Understudy API error: {error_msg}")
                raise RuntimeError(error_msg)
            except Exception as e:
                logger.error(f"Understudy request failed: {e}")
                raise
    
    async def _acall(
        self,
        prompt: str,
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> str:
        """Async call to the Understudy inference endpoint."""
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        
        # Filter kwargs to only include JSON-serializable values
        serializable_kwargs = {}
        for key, value in kwargs.items():
            try:
                import json
                json.dumps(value)
                serializable_kwargs[key] = value
            except (TypeError, ValueError):
                # Skip non-serializable values
                continue
        
        # Get actual values from fields (handle FieldInfo objects)
        max_tokens_value = self.max_tokens.default if hasattr(self.max_tokens, 'default') else self.max_tokens
        temperature_value = self.temperature.default if hasattr(self.temperature, 'default') else self.temperature
        top_p_value = self.top_p.default if hasattr(self.top_p, 'default') else self.top_p
        
        request_data = {
            "prompt": prompt,
            "max_tokens": kwargs.get("max_tokens", max_tokens_value),
            "temperature": kwargs.get("temperature", temperature_value),
            "top_p": kwargs.get("top_p", top_p_value),
            "stop": stop,
            "langchain_metadata": {
                "run_id": str(run_manager.run_id) if run_manager else None,
                "llm_type": self._llm_type,
                **serializable_kwargs
            }
        }
        
        async with httpx.AsyncClient(timeout=self.timeout) as client:
            try:
                response = await client.post(
                    f"{self.base_url}/api/v1/inference/{self.endpoint_id}",
                    json=request_data,
                    headers=headers
                )
                response.raise_for_status()
                result = response.json()
                
                if run_manager:
                    await run_manager.on_llm_end(
                        LLMResult(
                            generations=[[Generation(text=result["output"])]],
                            llm_output={
                                "model_used": result.get("model_used"),
                                "latency_ms": result.get("latency_ms"),
                                "cost_usd": result.get("cost_usd"),
                                "carbon_emissions": result.get("carbon_emissions")
                            }
                        )
                    )
                
                return result["output"]
                
            except httpx.HTTPStatusError as e:
                error_msg = f"HTTP {e.response.status_code}: {e.response.text}"
                logger.error(f"Understudy API error: {error_msg}")
                raise RuntimeError(error_msg)
            except Exception as e:
                logger.error(f"Understudy async request failed: {e}")
                raise


class UnderstudyChatModel(BaseChatModel):
    """LangChain Chat Model wrapper for Understudy."""
    
    endpoint_id: str = Field(description="Understudy endpoint ID")
    base_url: str = Field(default="http://localhost:8000", description="Understudy API base URL")
    api_key: Optional[str] = Field(default=None, description="API key for authentication")
    max_tokens: int = Field(default=256, description="Maximum tokens to generate")
    temperature: float = Field(default=0.7, description="Sampling temperature")
    top_p: float = Field(default=1.0, description="Top-p sampling parameter")
    timeout: float = Field(default=30.0, description="Request timeout in seconds")
    
    def __init__(self, **kwargs):
        """Initialize the UnderstudyChatModel with proper field handling."""
        super().__init__(**kwargs)
        # Ensure fields are properly initialized (not FieldInfo objects)
        if hasattr(self.max_tokens, 'default'):
            self.max_tokens = 256
        if hasattr(self.temperature, 'default'):
            self.temperature = 0.7
        if hasattr(self.top_p, 'default'):
            self.top_p = 1.0
        if hasattr(self.timeout, 'default'):
            self.timeout = 30.0
        if hasattr(self.api_key, 'default'):
            self.api_key = None
    
    @property
    def _llm_type(self) -> str:
        return "understudy-chat"
    
    @property
    def _identifying_params(self) -> Dict[str, Any]:
        """Get the identifying parameters."""
        return {
            "endpoint_id": self.endpoint_id,
            "base_url": self.base_url,
            "max_tokens": self.max_tokens,
            "temperature": self.temperature,
            "top_p": self.top_p,
        }
    
    def _generate(
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> ChatResult:
        """Generate response from messages."""
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        
        # Convert messages to API format
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
        
        # Filter kwargs to only include JSON-serializable values
        serializable_kwargs = {}
        for key, value in kwargs.items():
            try:
                import json
                json.dumps(value)
                serializable_kwargs[key] = value
            except (TypeError, ValueError):
                # Skip non-serializable values
                continue
        
        # Get actual values from fields (handle FieldInfo objects)
        max_tokens_value = self.max_tokens.default if hasattr(self.max_tokens, 'default') else self.max_tokens
        temperature_value = self.temperature.default if hasattr(self.temperature, 'default') else self.temperature
        top_p_value = self.top_p.default if hasattr(self.top_p, 'default') else self.top_p
        
        request_data = {
            "messages": formatted_messages,
            "max_tokens": kwargs.get("max_tokens", max_tokens_value),
            "temperature": kwargs.get("temperature", temperature_value),
            "top_p": kwargs.get("top_p", top_p_value),
            "stop": stop,
            "langchain_metadata": {
                "run_id": str(run_manager.run_id) if run_manager else None,
                "llm_type": self._llm_type,
                **serializable_kwargs
            }
        }
        
        with httpx.Client(timeout=self.timeout) as client:
            try:
                response = client.post(
                    f"{self.base_url}/api/v1/inference/{self.endpoint_id}",
                    json=request_data,
                    headers=headers
                )
                response.raise_for_status()
                result = response.json()
                
                # Create response message
                message = AIMessage(content=result["output"])
                generation = ChatGeneration(message=message)
                
                chat_result = ChatResult(
                    generations=[generation],
                    llm_output={
                        "model_used": result.get("model_used"),
                        "latency_ms": result.get("latency_ms"),
                        "cost_usd": result.get("cost_usd"),
                        "carbon_emissions": result.get("carbon_emissions")
                    }
                )
                
                return chat_result
                
            except httpx.HTTPStatusError as e:
                error_msg = f"HTTP {e.response.status_code}: {e.response.text}"
                logger.error(f"Understudy API error: {error_msg}")
                raise RuntimeError(error_msg)
            except Exception as e:
                logger.error(f"Understudy request failed: {e}")
                raise
    
    async def _agenerate(
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> ChatResult:
        """Generate response from messages (async)."""
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        
        # Convert messages to API format
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
        
        # Filter kwargs to only include JSON-serializable values
        serializable_kwargs = {}
        for key, value in kwargs.items():
            try:
                import json
                json.dumps(value)
                serializable_kwargs[key] = value
            except (TypeError, ValueError):
                # Skip non-serializable values
                continue
        
        # Get actual values from fields (handle FieldInfo objects)
        max_tokens_value = self.max_tokens.default if hasattr(self.max_tokens, 'default') else self.max_tokens
        temperature_value = self.temperature.default if hasattr(self.temperature, 'default') else self.temperature
        top_p_value = self.top_p.default if hasattr(self.top_p, 'default') else self.top_p
        
        request_data = {
            "messages": formatted_messages,
            "max_tokens": kwargs.get("max_tokens", max_tokens_value),
            "temperature": kwargs.get("temperature", temperature_value),
            "top_p": kwargs.get("top_p", top_p_value),
            "stop": stop,
            "langchain_metadata": {
                "run_id": str(run_manager.run_id) if run_manager else None,
                "llm_type": self._llm_type,
                **serializable_kwargs
            }
        }
        
        async with httpx.AsyncClient(timeout=self.timeout) as client:
            try:
                response = await client.post(
                    f"{self.base_url}/api/v1/inference/{self.endpoint_id}",
                    json=request_data,
                    headers=headers
                )
                response.raise_for_status()
                result = response.json()
                
                # Create response message
                message = AIMessage(content=result["output"])
                generation = ChatGeneration(message=message)
                
                chat_result = ChatResult(
                    generations=[generation],
                    llm_output={
                        "model_used": result.get("model_used"),
                        "latency_ms": result.get("latency_ms"),
                        "cost_usd": result.get("cost_usd"),
                        "carbon_emissions": result.get("carbon_emissions")
                    }
                )
                
                return chat_result
                
            except httpx.HTTPStatusError as e:
                error_msg = f"HTTP {e.response.status_code}: {e.response.text}"
                logger.error(f"Understudy API error: {error_msg}")
                raise RuntimeError(error_msg)
            except Exception as e:
                logger.error(f"Understudy async request failed: {e}")
                raise