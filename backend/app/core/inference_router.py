from typing import Optional, List, Dict, Any
from app.models import Endpoint, InferenceLog, EndpointConfig, Metric
from app.models.database import AsyncSessionLocal
from sqlalchemy import select
from datetime import datetime
import time
import logging
import json

try:
    from app.providers.factory import ProviderFactory
    PROVIDERS_AVAILABLE = True
except ImportError:
    PROVIDERS_AVAILABLE = False
    class ProviderFactory:
        @staticmethod
        def get_provider(*args, **kwargs):
            raise ImportError("Provider dependencies not available")

try:
    from app.slm.inference import SLMInferenceEngine
    SLM_AVAILABLE = True
except ImportError:
    SLM_AVAILABLE = False
    class SLMInferenceEngine:
        def __init__(self):
            pass
        async def generate(*args, **kwargs):
            raise ImportError("SLM dependencies not available")
        async def generate_with_messages(*args, **kwargs):
            raise ImportError("SLM dependencies not available")

try:
    from langchain.schema import BaseMessage
    LANGCHAIN_AVAILABLE = True
except ImportError:
    LANGCHAIN_AVAILABLE = False
    class BaseMessage:
        def __init__(self, content="", type=""):
            self.content = content
            self.type = type

logger = logging.getLogger(__name__)


class InferenceRouter:
    """Routes inference requests to either LLM or SLM based on configuration."""
    
    def __init__(self):
        self.slm_engine = SLMInferenceEngine()
        self.providers_cache = {}
    
    async def route_request(
        self,
        endpoint_id: str,
        prompt: Optional[str] = None,
        messages: Optional[List[BaseMessage]] = None,
        langchain_metadata: Optional[Dict] = None,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Route inference request to either LLM or SLM.
        Supports both prompt strings and LangChain message format.
        """
        start_time = time.time()
        
        # Get endpoint and config from database
        async with AsyncSessionLocal() as session:
            endpoint = await session.get(Endpoint, endpoint_id)
            if not endpoint:
                raise ValueError(f"Endpoint {endpoint_id} not found")
            
            config = await session.get(EndpointConfig, endpoint_id)
            if not config:
                # Create default config if not exists
                config = EndpointConfig(endpoint_id=endpoint_id)
                session.add(config)
                await session.commit()
        
        # Determine if we should use SLM
        use_slm = (
            endpoint.status == 'active' and 
            endpoint.slm_model_path is not None and
            config.auto_switchover
        )
        
        output = None
        model_used = None
        cost = 0.0
        carbon_emissions = None
        
        if use_slm:
            # Use trained SLM
            try:
                if messages:
                    result = await self.slm_engine.generate_with_messages(
                        endpoint.slm_model_path,
                        messages,
                        **kwargs
                    )
                else:
                    result = await self.slm_engine.generate(
                        endpoint.slm_model_path,
                        prompt,
                        **kwargs
                    )
                
                output = result["output"]
                model_used = "slm"
                cost = 0.0001  # Estimated local inference cost
                carbon_emissions = result.get("carbon_emissions")
                
            except Exception as e:
                logger.error(f"SLM inference failed, falling back to LLM: {e}")
                use_slm = False
        
        if not use_slm:
            # Use LLM provider
            provider = await self._get_provider(endpoint.llm_provider, endpoint.llm_model)
            
            try:
                if messages:
                    output = await provider.generate_with_messages(messages, **kwargs)
                else:
                    output = await provider.generate(prompt, **kwargs)
                
                model_used = "llm"
                
                # Estimate cost
                input_tokens = provider.count_tokens(prompt or self._messages_to_text(messages))
                output_tokens = provider.count_tokens(output)
                cost = provider.estimate_cost(input_tokens, output_tokens)
                
            except Exception as e:
                logger.error(f"LLM inference failed: {e}")
                raise
        
        latency_ms = int((time.time() - start_time) * 1000)
        
        # Log inference
        await self._log_inference(
            endpoint_id=endpoint_id,
            input_text=prompt or self._messages_to_text(messages),
            output=output,
            model_used=model_used,
            latency_ms=latency_ms,
            cost_usd=cost,
            langchain_metadata=langchain_metadata
        )
        
        # If using LLM, check if we should evaluate SLM
        if model_used == "llm" and endpoint.slm_model_path and not config.auto_switchover:
            await self._evaluate_slm_readiness(endpoint, prompt or self._messages_to_text(messages), output)
        
        return {
            "output": output,
            "model_used": model_used,
            "latency_ms": latency_ms,
            "cost_usd": cost,
            "carbon_emissions": carbon_emissions
        }
    
    async def _get_provider(self, provider_name: str, model: str):
        """Get or create a provider instance."""
        cache_key = f"{provider_name}:{model}"
        if cache_key not in self.providers_cache:
            self.providers_cache[cache_key] = ProviderFactory.get_provider(
                provider_name,
                model=model
            )
        return self.providers_cache[cache_key]
    
    async def _log_inference(
        self,
        endpoint_id: str,
        input_text: str,
        output: str,
        model_used: str,
        latency_ms: int,
        cost_usd: float,
        langchain_metadata: Optional[Dict] = None
    ):
        """Log inference to database."""
        async with AsyncSessionLocal() as session:
            log = InferenceLog(
                endpoint_id=endpoint_id,
                input_text=input_text,
                llm_output=output if model_used == "llm" else None,
                slm_output=output if model_used == "slm" else None,
                model_used=model_used,
                latency_ms=latency_ms,
                cost_usd=cost_usd,
                langchain_metadata=langchain_metadata
            )
            session.add(log)
            await session.commit()
    
    async def _evaluate_slm_readiness(self, endpoint: Endpoint, input_text: str, llm_output: str):
        """Evaluate if SLM is ready for switchover."""
        try:
            # Generate SLM output for comparison
            result = await self.slm_engine.generate(
                endpoint.slm_model_path,
                input_text
            )
            slm_output = result["output"]
            
            # Calculate similarity
            similarity = await self.slm_engine.evaluate_similarity(
                endpoint.slm_model_path,
                llm_output,
                slm_output
            )
            
            # Log metric
            async with AsyncSessionLocal() as session:
                metric = Metric(
                    endpoint_id=endpoint.id,
                    metric_type="semantic_similarity",
                    value=similarity
                )
                session.add(metric)
                
                # Check if we should activate SLM
                config = await session.get(EndpointConfig, endpoint.id)
                if similarity >= config.similarity_threshold:
                    # Get average similarity over last 100 inferences
                    recent_metrics = await session.execute(
                        select(Metric)
                        .where(Metric.endpoint_id == endpoint.id)
                        .where(Metric.metric_type == "semantic_similarity")
                        .order_by(Metric.calculated_at.desc())
                        .limit(100)
                    )
                    metrics = recent_metrics.scalars().all()
                    
                    if len(metrics) >= 50:  # Need at least 50 evaluations
                        avg_similarity = sum(m.value for m in metrics) / len(metrics)
                        if avg_similarity >= config.similarity_threshold:
                            endpoint.status = "ready"
                            logger.info(f"Endpoint {endpoint.id} SLM is ready for activation")
                
                await session.commit()
                
        except Exception as e:
            logger.error(f"Failed to evaluate SLM readiness: {e}")
    
    def _messages_to_text(self, messages: List[BaseMessage]) -> str:
        """Convert LangChain messages to text for storage."""
        return "\n".join([f"{msg.type}: {msg.content}" for msg in messages])