"""
Training Service Client

HTTP client for communicating with the Training Service.
Replaces direct RunPod API calls with service-to-service communication.
"""

import os
import logging
import httpx
from typing import Dict, Any, Optional, List
from datetime import datetime

logger = logging.getLogger(__name__)


class TrainingServiceClient:
    """Client for Training Service API"""
    
    def __init__(self):
        self.training_service_url = os.getenv("TRAINING_SERVICE_URL", "http://training-service:8002")
        self.timeout = 30.0
        
    async def start_training(
        self,
        train_id: str,
        endpoint_id: str,
        version: int,
        training_pairs_count: int,
        slm_type: str,
        source_llm: str,
        provider: str = "runpod",
        training_data: Optional[List[Dict[str, str]]] = None
    ) -> Dict[str, Any]:
        """Start a training job on the training service"""
        try:
            async with httpx.AsyncClient(timeout=self.timeout) as client:
                response = await client.post(
                    f"{self.training_service_url}/api/v1/training/start",
                    json={
                        "train_id": train_id,
                        "endpoint_id": endpoint_id,
                        "version": version,
                        "training_pairs_count": training_pairs_count,
                        "slm_type": slm_type,
                        "source_llm": source_llm,
                        "provider": provider,
                        "training_data": training_data
                    }
                )
                
                if response.status_code == 200:
                    result = response.json()
                    logger.info(f"Training started successfully for {train_id}: {result.get('runpod_job_id')}")
                    return {
                        "success": True,
                        "runpod_job_id": result.get("runpod_job_id"),
                        "status": result.get("status"),
                        "message": result.get("message")
                    }
                else:
                    logger.error(f"Failed to start training for {train_id}: {response.text}")
                    return {
                        "success": False,
                        "error": f"HTTP {response.status_code}: {response.text}"
                    }
                    
        except Exception as e:
            logger.error(f"Error starting training for {train_id}: {e}")
            return {
                "success": False,
                "error": str(e)
            }
    
    async def get_training_status(self, train_id: str) -> Dict[str, Any]:
        """Get status of a training job"""
        try:
            async with httpx.AsyncClient(timeout=self.timeout) as client:
                response = await client.get(
                    f"{self.training_service_url}/api/v1/training/{train_id}/status"
                )
                
                if response.status_code == 200:
                    result = response.json()
                    return {
                        "success": True,
                        "train_id": result.get("train_id"),
                        "status": result.get("status"),
                        "phase": result.get("phase"),
                        "progress": result.get("progress"),
                        "message": result.get("message")
                    }
                elif response.status_code == 404:
                    return {
                        "success": False,
                        "error": "Training job not found"
                    }
                else:
                    logger.error(f"Failed to get training status for {train_id}: {response.text}")
                    return {
                        "success": False,
                        "error": f"HTTP {response.status_code}: {response.text}"
                    }
                    
        except Exception as e:
            logger.error(f"Error getting training status for {train_id}: {e}")
            return {
                "success": False,
                "error": str(e)
            }
    
    async def cancel_training(self, train_id: str) -> Dict[str, Any]:
        """Cancel a training job"""
        try:
            async with httpx.AsyncClient(timeout=self.timeout) as client:
                response = await client.post(
                    f"{self.training_service_url}/api/v1/training/{train_id}/cancel"
                )
                
                if response.status_code == 200:
                    result = response.json()
                    logger.info(f"Training cancelled successfully for {train_id}")
                    return {
                        "success": True,
                        "message": result.get("message")
                    }
                elif response.status_code == 404:
                    return {
                        "success": False,
                        "error": "Training job not found"
                    }
                else:
                    logger.error(f"Failed to cancel training for {train_id}: {response.text}")
                    return {
                        "success": False,
                        "error": f"HTTP {response.status_code}: {response.text}"
                    }
                    
        except Exception as e:
            logger.error(f"Error cancelling training for {train_id}: {e}")
            return {
                "success": False,
                "error": str(e)
            }
    
    async def mark_training_completed(
        self,
        train_id: str,
        status: str = "completed",
        model_path: Optional[str] = None
    ) -> Dict[str, Any]:
        """Mark a training job as completed (for backend use)"""
        try:
            async with httpx.AsyncClient(timeout=self.timeout) as client:
                response = await client.post(
                    f"{self.training_service_url}/api/v1/training/{train_id}/completed",
                    json={
                        "status": status,
                        "model_path": model_path or ""
                    }
                )
                
                if response.status_code == 200:
                    result = response.json()
                    logger.info(f"Training marked as {status} for {train_id}")
                    return {
                        "success": True,
                        "message": result.get("message")
                    }
                else:
                    logger.error(f"Failed to mark training completed for {train_id}: {response.text}")
                    return {
                        "success": False,
                        "error": f"HTTP {response.status_code}: {response.text}"
                    }
                    
        except Exception as e:
            logger.error(f"Error marking training completed for {train_id}: {e}")
            return {
                "success": False,
                "error": str(e)
            }
    
    async def health_check(self) -> bool:
        """Check if training service is healthy"""
        try:
            async with httpx.AsyncClient(timeout=5.0) as client:
                response = await client.get(f"{self.training_service_url}/health")
                return response.status_code == 200
        except Exception as e:
            logger.warning(f"Training service health check failed: {e}")
            return False


# Global client instance
training_client = None

def get_training_client() -> TrainingServiceClient:
    """Get the global training client instance"""
    global training_client
    if training_client is None:
        training_client = TrainingServiceClient()
    return training_client