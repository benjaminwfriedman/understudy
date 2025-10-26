"""
Model Lifecycle Manager

Manages the phases of model lifecycle:
- training: Model training in progress on RunPod
- downloading: Training Service downloading model weights from RunPod  
- llm_evaluation: Evaluating semantic similarity on unseen data
- available: Evaluation complete, weights in storage, not deployed
- deploying: K8s deployment being created
- deployed: Currently serving production traffic (K8s deployment running)
- failed: Training or deployment failed
"""

import os
import logging
from typing import Dict, Any, Optional, List
from datetime import datetime
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, update
import httpx
import redis.asyncio as redis

from app.models import TrainingRun
from app.core.k8s_manager import get_k8s_manager
from app.core.config import settings

logger = logging.getLogger(__name__)


class ModelLifecycleManager:
    """Manages model lifecycle phases and transitions."""
    
    def __init__(self):
        self.training_service_url = os.getenv("TRAINING_SERVICE_URL", "http://training-service:8002")
        self.evaluation_service_url = os.getenv("EVALUATION_SERVICE_URL", "http://evaluation-service:8001")
        self.model_broker_url = os.getenv("MODEL_BROKER_SERVICE_URL", "http://model-broker-service:8003")
        self.redis_url = os.getenv("REDIS_URL", "redis://redis-service:6379")
        self.deployment_threshold = float(os.getenv("DEPLOYMENT_THRESHOLD", "0.85"))
        
        # Initialize Redis client
        self.redis_client = None
        
        # Initialize K8s manager
        self.k8s_manager = get_k8s_manager()
    
    async def initialize(self):
        """Initialize async components."""
        self.redis_client = redis.from_url(self.redis_url)
        await self.redis_client.ping()
        logger.info("Model lifecycle manager initialized")
    
    async def start_training(
        self,
        db: AsyncSession,
        train_id: str,
        endpoint_id: str,
        version: int,
        training_pairs_count: int,
        slm_type: str,
        source_llm: str
    ) -> Dict[str, Any]:
        """Start a new training job and set phase to 'training'."""
        try:
            # Update database phase
            await self._update_phase(db, train_id, "training")
            
            # Send request to Training Service
            async with httpx.AsyncClient() as client:
                response = await client.post(
                    f"{self.training_service_url}/api/v1/training/start",
                    json={
                        "train_id": train_id,
                        "endpoint_id": endpoint_id,
                        "version": version,
                        "training_pairs_count": training_pairs_count,
                        "slm_type": slm_type,
                        "source_llm": source_llm
                    }
                )
                
                if response.status_code == 200:
                    result = response.json()
                    logger.info(f"Training started for {train_id}: {result}")
                    return result
                else:
                    logger.error(f"Failed to start training: {response.status_code} {response.text}")
                    await self._update_phase(db, train_id, "failed")
                    raise Exception(f"Training service error: {response.text}")
        
        except Exception as e:
            logger.error(f"Error starting training for {train_id}: {e}")
            await self._update_phase(db, train_id, "failed")
            raise
    
    async def handle_training_completion(
        self,
        db: AsyncSession,
        train_id: str,
        phase: str
    ) -> Dict[str, Any]:
        """Handle training completion notification from Training Service."""
        try:
            logger.info(f"Training completed for {train_id} with phase: {phase}")
            
            if phase == "available":
                # Start evaluation workflow
                await self._start_evaluation_workflow(db, train_id)
            elif phase == "failed":
                await self._update_phase(db, train_id, "failed")
            
            return {"status": "handled", "phase": phase}
            
        except Exception as e:
            logger.error(f"Error handling training completion for {train_id}: {e}")
            await self._update_phase(db, train_id, "failed")
            raise
    
    async def handle_similarity_score(
        self,
        db: AsyncSession,
        train_id: str,
        similarity_score: float,
        evaluation_count: int
    ) -> Dict[str, Any]:
        """Handle semantic similarity score from Evaluation Service."""
        try:
            logger.info(f"Received similarity score for {train_id}: {similarity_score}")
            
            # Update database with similarity score
            stmt = update(TrainingRun).where(
                TrainingRun.train_id == train_id
            ).values(
                semantic_similarity_score=similarity_score,
                phase="available"
            )
            await db.execute(stmt)
            await db.commit()
            
            # Check if we should deploy
            if similarity_score >= self.deployment_threshold:
                logger.info(f"Similarity score {similarity_score} >= {self.deployment_threshold}, deploying model")
                await self._deploy_model(db, train_id)
            else:
                logger.info(f"Similarity score {similarity_score} < {self.deployment_threshold}, model remains available")
            
            return {
                "status": "processed",
                "similarity_score": similarity_score,
                "should_deploy": similarity_score >= self.deployment_threshold
            }
            
        except Exception as e:
            logger.error(f"Error handling similarity score for {train_id}: {e}")
            await self._update_phase(db, train_id, "failed")
            raise
    
    async def deploy_model(
        self,
        db: AsyncSession,
        train_id: str
    ) -> Dict[str, Any]:
        """Manually deploy a model (for rollbacks or manual deployments)."""
        return await self._deploy_model(db, train_id)
    
    async def undeploy_model(
        self,
        db: AsyncSession,
        endpoint_id: str
    ) -> Dict[str, Any]:
        """Undeploy the currently deployed model for an endpoint."""
        try:
            # Find currently deployed model
            stmt = select(TrainingRun).where(
                TrainingRun.endpoint_id == endpoint_id,
                TrainingRun.phase == "deployed"
            )
            result = await db.execute(stmt)
            current_model = result.scalar_one_or_none()
            
            if not current_model:
                return {"status": "no_deployed_model"}
            
            # Delete K8s deployment
            deployment_name = current_model.k8s_deployment_name
            if deployment_name:
                success = self.k8s_manager.delete_slm_deployment(deployment_name)
                if success:
                    logger.info(f"Deleted deployment: {deployment_name}")
            
            # Update phase to available
            await self._update_phase(db, current_model.train_id, "available")
            
            # Clear deployment name
            stmt = update(TrainingRun).where(
                TrainingRun.train_id == current_model.train_id
            ).values(
                k8s_deployment_name=None,
                inference_mode=None
            )
            await db.execute(stmt)
            await db.commit()
            
            return {
                "status": "undeployed",
                "train_id": current_model.train_id,
                "deployment_name": deployment_name
            }
            
        except Exception as e:
            logger.error(f"Error undeploying model for {endpoint_id}: {e}")
            raise
    
    async def get_model_status(
        self,
        db: AsyncSession,
        train_id: str
    ) -> Dict[str, Any]:
        """Get the current status of a model."""
        try:
            stmt = select(TrainingRun).where(TrainingRun.train_id == train_id)
            result = await db.execute(stmt)
            model = result.scalar_one_or_none()
            
            if not model:
                return {"status": "not_found"}
            
            status = {
                "train_id": train_id,
                "endpoint_id": model.endpoint_id,
                "version": model.version,
                "phase": model.phase,
                "similarity_score": model.semantic_similarity_score,
                "created_at": model.created_at.isoformat() if model.created_at else None,
                "k8s_deployment_name": model.k8s_deployment_name,
                "inference_mode": model.inference_mode
            }
            
            # If deployed, get K8s status
            if model.phase == "deployed" and model.k8s_deployment_name:
                k8s_status = self.k8s_manager.get_deployment_status(model.k8s_deployment_name)
                if k8s_status:
                    status["k8s_status"] = k8s_status
            
            return status
            
        except Exception as e:
            logger.error(f"Error getting model status for {train_id}: {e}")
            raise
    
    async def list_models_by_endpoint(
        self,
        db: AsyncSession,
        endpoint_id: str
    ) -> List[Dict[str, Any]]:
        """List all models for an endpoint."""
        try:
            stmt = select(TrainingRun).where(
                TrainingRun.endpoint_id == endpoint_id,
                TrainingRun.is_deleted == False
            ).order_by(TrainingRun.version.desc())
            
            result = await db.execute(stmt)
            models = result.scalars().all()
            
            model_list = []
            for model in models:
                model_info = {
                    "train_id": model.train_id,
                    "version": model.version,
                    "phase": model.phase,
                    "similarity_score": model.semantic_similarity_score,
                    "created_at": model.created_at.isoformat() if model.created_at else None,
                    "slm_type": model.slm_type,
                    "source_llm": model.source_llm,
                    "is_deployed": model.phase == "deployed"
                }
                model_list.append(model_info)
            
            return model_list
            
        except Exception as e:
            logger.error(f"Error listing models for {endpoint_id}: {e}")
            raise
    
    async def _start_evaluation_workflow(self, db: AsyncSession, train_id: str):
        """Start the evaluation workflow for a trained model."""
        try:
            # Update phase to llm_evaluation
            await self._update_phase(db, train_id, "llm_evaluation")
            
            # Get model info
            stmt = select(TrainingRun).where(TrainingRun.train_id == train_id)
            result = await db.execute(stmt)
            model = result.scalar_one_or_none()
            
            if not model:
                raise Exception(f"Model not found: {train_id}")
            
            # TODO: Collect evaluation pairs from recent LLM calls
            # For now, create dummy evaluation pairs
            evaluation_pairs = [
                {
                    "llm_response": "This is a sample LLM response for evaluation purposes.",
                    "slm_response": "This will be replaced with actual SLM response."
                }
                # Add more evaluation pairs...
            ]
            
            # Create SLM batch job for evaluation
            batch_job = self.k8s_manager.create_slm_batch_job(
                train_id=train_id,
                endpoint_id=model.endpoint_id,
                version=model.version,
                model_path=f"/models/{model.endpoint_id}/v{model.version}",
                evaluation_batch_id=f"eval_{train_id}"
            )
            
            logger.info(f"Created evaluation batch job: {batch_job['job_name']}")
            
            # TODO: Monitor job completion and trigger evaluation service
            
        except Exception as e:
            logger.error(f"Error starting evaluation workflow for {train_id}: {e}")
            await self._update_phase(db, train_id, "failed")
            raise
    
    async def _deploy_model(self, db: AsyncSession, train_id: str) -> Dict[str, Any]:
        """Deploy a model to K8s."""
        try:
            # Get model info
            stmt = select(TrainingRun).where(TrainingRun.train_id == train_id)
            result = await db.execute(stmt)
            model = result.scalar_one_or_none()
            
            if not model:
                raise Exception(f"Model not found: {train_id}")
            
            # Update phase to deploying
            await self._update_phase(db, train_id, "deploying")
            
            # Undeploy current model if exists
            await self.undeploy_model(db, model.endpoint_id)
            
            # Create K8s deployment
            deployment = self.k8s_manager.create_slm_deployment(
                endpoint_id=model.endpoint_id,
                version=model.version,
                model_path=f"/models/{model.endpoint_id}/v{model.version}"
            )
            
            # Update database
            stmt = update(TrainingRun).where(
                TrainingRun.train_id == train_id
            ).values(
                phase="deployed",
                k8s_deployment_name=deployment["deployment_name"],
                inference_mode="endpoint"
            )
            await db.execute(stmt)
            await db.commit()
            
            logger.info(f"Model deployed: {train_id} -> {deployment['deployment_name']}")
            
            return {
                "status": "deployed",
                "deployment_name": deployment["deployment_name"],
                "service_name": deployment["service_name"]
            }
            
        except Exception as e:
            logger.error(f"Error deploying model {train_id}: {e}")
            await self._update_phase(db, train_id, "failed")
            raise
    
    async def _update_phase(self, db: AsyncSession, train_id: str, phase: str):
        """Update the phase of a model in the database."""
        try:
            stmt = update(TrainingRun).where(
                TrainingRun.train_id == train_id
            ).values(phase=phase)
            await db.execute(stmt)
            await db.commit()
            
            logger.info(f"Updated phase for {train_id}: {phase}")
            
        except Exception as e:
            logger.error(f"Error updating phase for {train_id}: {e}")
            raise


# Global instance
lifecycle_manager = None

def get_lifecycle_manager() -> ModelLifecycleManager:
    """Get the global lifecycle manager instance."""
    global lifecycle_manager
    if lifecycle_manager is None:
        lifecycle_manager = ModelLifecycleManager()
    return lifecycle_manager