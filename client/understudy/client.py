import httpx
from typing import Optional, Dict, Any, List
import asyncio
from datetime import datetime
import logging

logger = logging.getLogger(__name__)


class Understudy:
    """Main client for Understudy service."""
    
    def __init__(
        self,
        base_url: str = "http://localhost:8000",
        api_key: Optional[str] = None,
        timeout: float = 30.0
    ):
        """
        Initialize Understudy client.
        
        Args:
            base_url: Base URL of the Understudy API
            api_key: Optional API key for authentication
            timeout: Request timeout in seconds
        """
        self.base_url = base_url.rstrip('/')
        self.api_key = api_key
        self.timeout = timeout
        
        headers = {"Content-Type": "application/json"}
        if api_key:
            headers["Authorization"] = f"Bearer {api_key}"
        
        self.client = httpx.Client(
            base_url=base_url,
            headers=headers,
            timeout=timeout
        )
        
        self.async_client = httpx.AsyncClient(
            base_url=base_url,
            headers=headers,
            timeout=timeout
        )
    
    def create_endpoint(
        self,
        name: str,
        llm_provider: str,
        llm_model: str,
        description: Optional[str] = None,
        training_batch_size: int = 100,
        similarity_threshold: float = 0.95,
        auto_switchover: bool = False,
        **kwargs
    ) -> Dict[str, Any]:
        """Create a new Understudy endpoint."""
        data = {
            "name": name,
            "description": description,
            "llm_provider": llm_provider,
            "llm_model": llm_model,
            "config": {
                "training_batch_size": training_batch_size,
                "similarity_threshold": similarity_threshold,
                "auto_switchover": auto_switchover,
                **kwargs
            }
        }
        
        response = self.client.post("/api/v1/endpoints", json=data)
        response.raise_for_status()
        return response.json()
    
    def list_endpoints(self, skip: int = 0, limit: int = 100) -> List[Dict[str, Any]]:
        """List all endpoints."""
        response = self.client.get(
            "/api/v1/endpoints",
            params={"skip": skip, "limit": limit}
        )
        response.raise_for_status()
        return response.json()
    
    def get_endpoint(self, endpoint_id: str) -> Dict[str, Any]:
        """Get a specific endpoint."""
        response = self.client.get(f"/api/v1/endpoints/{endpoint_id}")
        response.raise_for_status()
        return response.json()
    
    def delete_endpoint(self, endpoint_id: str) -> Dict[str, Any]:
        """Delete an endpoint."""
        response = self.client.delete(f"/api/v1/endpoints/{endpoint_id}")
        response.raise_for_status()
        return response.json()
    
    def generate(
        self,
        endpoint_id: str,
        prompt: str,
        max_tokens: int = 256,
        temperature: float = 0.7,
        **kwargs
    ) -> Dict[str, Any]:
        """Generate a response from an endpoint."""
        data = {
            "prompt": prompt,
            "max_tokens": max_tokens,
            "temperature": temperature,
            **kwargs
        }
        
        response = self.client.post(
            f"/api/v1/inference/{endpoint_id}",
            json=data
        )
        response.raise_for_status()
        return response.json()
    
    async def generate_async(
        self,
        endpoint_id: str,
        prompt: str,
        max_tokens: int = 256,
        temperature: float = 0.7,
        **kwargs
    ) -> Dict[str, Any]:
        """Generate a response from an endpoint (async)."""
        data = {
            "prompt": prompt,
            "max_tokens": max_tokens,
            "temperature": temperature,
            **kwargs
        }
        
        response = await self.async_client.post(
            f"/api/v1/inference/{endpoint_id}",
            json=data
        )
        response.raise_for_status()
        return response.json()
    
    def generate_with_messages(
        self,
        endpoint_id: str,
        messages: List[Dict[str, str]],
        max_tokens: int = 256,
        temperature: float = 0.7,
        **kwargs
    ) -> Dict[str, Any]:
        """Generate a response using message format."""
        data = {
            "messages": messages,
            "max_tokens": max_tokens,
            "temperature": temperature,
            **kwargs
        }
        
        response = self.client.post(
            f"/api/v1/inference/{endpoint_id}",
            json=data
        )
        response.raise_for_status()
        return response.json()
    
    def start_training(
        self,
        endpoint_id: str,
        num_examples: Optional[int] = None,
        force_retrain: bool = False
    ) -> Dict[str, Any]:
        """Start training for an endpoint."""
        data = {
            "num_examples": num_examples,
            "force_retrain": force_retrain
        }
        
        response = self.client.post(
            f"/api/v1/training/{endpoint_id}",
            json=data
        )
        response.raise_for_status()
        return response.json()
    
    def get_training_runs(self, endpoint_id: str) -> List[Dict[str, Any]]:
        """Get training runs for an endpoint."""
        response = self.client.get(f"/api/v1/training/{endpoint_id}/runs")
        response.raise_for_status()
        return response.json()
    
    def get_metrics(self, endpoint_id: str, days: int = 30) -> Dict[str, Any]:
        """Get metrics summary for an endpoint."""
        response = self.client.get(
            f"/api/v1/metrics/{endpoint_id}",
            params={"days": days}
        )
        response.raise_for_status()
        return response.json()
    
    def get_carbon_summary(self, endpoint_id: str) -> Dict[str, Any]:
        """Get carbon emissions summary."""
        response = self.client.get(f"/api/v1/carbon/{endpoint_id}/summary")
        response.raise_for_status()
        return response.json()
    
    def get_carbon_timeline(
        self,
        endpoint_id: str,
        days: int = 30
    ) -> Dict[str, Any]:
        """Get carbon emissions timeline."""
        response = self.client.get(
            f"/api/v1/carbon/{endpoint_id}/timeline",
            params={"days": days}
        )
        response.raise_for_status()
        return response.json()
    
    def activate_slm(self, endpoint_id: str) -> Dict[str, Any]:
        """Manually activate SLM for an endpoint."""
        response = self.client.post(f"/api/v1/endpoints/{endpoint_id}/activate")
        response.raise_for_status()
        return response.json()
    
    def health_check(self) -> Dict[str, Any]:
        """Check API health."""
        response = self.client.get("/api/v1/health")
        response.raise_for_status()
        return response.json()
    
    def wait_for_training(
        self,
        endpoint_id: str,
        max_wait_time: int = 3600,
        check_interval: int = 30
    ) -> Dict[str, Any]:
        """Wait for training to complete."""
        start_time = datetime.now()
        
        while (datetime.now() - start_time).seconds < max_wait_time:
            runs = self.get_training_runs(endpoint_id)
            if runs:
                latest_run = runs[0]
                if latest_run["status"] in ["completed", "failed"]:
                    return latest_run
            
            import time
            time.sleep(check_interval)
        
        raise TimeoutError(f"Training did not complete within {max_wait_time} seconds")
    
    def __enter__(self):
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.client.close()
    
    async def __aenter__(self):
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        await self.async_client.aclose()
    
    def close(self):
        """Close the HTTP clients."""
        self.client.close()
    
    async def aclose(self):
        """Close the async HTTP client."""
        await self.async_client.aclose()