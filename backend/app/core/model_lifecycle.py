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
    """
    Manages model lifecycle phases and transitions.
    
    TODO: Implement drift detection workflow for deployed SLMs:
    - When SLM is deployed and serving traffic (phase='deployed')
    - Every N SLM inferences (configurable via DRIFT_CHECK_INTERVAL, e.g., 100)
    - Sample the last N SLM inference inputs and run them through the source LLM
    - Compare LLM vs SLM outputs using evaluation service
    - If similarity drops below DRIFT_THRESHOLD (e.g., 0.75):
      * Alert/log drift detection
      * Optionally auto-retrain or fall back to LLM mode
      * Create CarbonEmission record for drift check LLM calls
    - This prevents SLM performance degradation over time due to:
      * Input distribution shift
      * Model staleness
      * Changing user patterns
    - Should be triggered from inference endpoint when model_used='slm'
    """
    
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
        
        # Validate environment variables
        self._validate_config()
        
        logger.info("Model lifecycle manager initialized")
    
    def _validate_config(self):
        """Validate configuration settings."""
        # Validate DEFAULT_EVALUATION_SAMPLE_SIZE
        eval_sample_size = os.getenv("DEFAULT_EVALUATION_SAMPLE_SIZE", "50")
        try:
            size = int(eval_sample_size)
            if size < 1:
                logger.warning(f"Invalid DEFAULT_EVALUATION_SAMPLE_SIZE: {size}. Must be >= 1. Using default of 50.")
            elif size > 1000:
                logger.warning(f"Large DEFAULT_EVALUATION_SAMPLE_SIZE: {size}. This may slow down evaluations.")
            else:
                logger.info(f"Using DEFAULT_EVALUATION_SAMPLE_SIZE: {size}")
        except ValueError:
            logger.error(f"Invalid DEFAULT_EVALUATION_SAMPLE_SIZE: '{eval_sample_size}'. Must be an integer. Using default of 50.")
        
        # Validate DEPLOYMENT_THRESHOLD
        threshold = os.getenv("DEPLOYMENT_THRESHOLD", "0.85")
        try:
            thresh = float(threshold)
            if thresh < 0.0 or thresh > 1.0:
                logger.warning(f"DEPLOYMENT_THRESHOLD {thresh} is outside valid range [0.0, 1.0]. Using default of 0.85.")
            else:
                logger.info(f"Using DEPLOYMENT_THRESHOLD: {thresh}")
        except ValueError:
            logger.error(f"Invalid DEPLOYMENT_THRESHOLD: '{threshold}'. Must be a float. Using default of 0.85.")
    
    async def training_ready_or_needed(
            self,
            db:AsyncSession,
            endpoint_id:str,
        ):

        try:


            from app.models.models import InferenceLog, Endpoint, EndpointConfig
            from sqlalchemy import func

            endpoint = await db.get(Endpoint, endpoint_id)
            endpoint_config = await db.get(EndpointConfig, endpoint_id)

            training_cutoff = endpoint.created_at
            training_data_size_requirement = endpoint_config.training_batch_size

            sample_count = await db.scalar(
                select(func.count(InferenceLog.id)).where(
                    InferenceLog.endpoint_id == endpoint_id,
                    InferenceLog.model_used == "llm",
                    InferenceLog.created_at > training_cutoff  # Only unseen data
                )
            )

            ## TODO smart cadance like [10, 100, 1000]
            if sample_count % training_data_size_requirement == 0:
                
                await self.queue_training(
                    db=db,
                    endpoint_id=endpoint_id,
                    training_pairs_count=training_data_size_requirement,
                    epochs=3,
                    batch_size=training_data_size_requirement,
                    learning_rate=endpoint_config.learning_rate,
                )

                return True



            return False
        except Exception as e:
            logger.error(f"Error checking training readiness for {endpoint_id}: {e}")
            raise



    async def start_training(
        self,
        db: AsyncSession,
        train_id: str,
        endpoint_id: str,
        version: int,
        training_pairs_count: int,
        slm_type: str,
        source_llm: str,
        training_data: Optional[List[Dict[str, str]]] = None
    ) -> Dict[str, Any]:
        """Start a new training job and set phase to 'training'."""
        try:
            # Create training record in database
            from app.models.models import TrainingRun
            from sqlalchemy import func
            
            # Auto-increment version if not provided
            if version is None:
                # Get the next version number for this endpoint
                max_version = await db.scalar(
                    select(func.max(TrainingRun.version))
                    .where(TrainingRun.endpoint_id == endpoint_id)
                    .where(TrainingRun.is_deleted == False)
                ) or 0
                version = max_version + 1
                logger.info(f"Auto-assigned version {version} for endpoint {endpoint_id} (previous max: {max_version})")
            
            training_record = TrainingRun(
                train_id=train_id,
                endpoint_id=endpoint_id,
                version=version,
                training_pairs_count=training_pairs_count,
                slm_type=slm_type,
                source_llm=source_llm,
                phase="training",
                is_deleted=False,
                start_time=datetime.now(),
                examples_used=training_pairs_count  # Legacy field that maps to training_pairs_count
            )
            
            db.add(training_record)
            await db.commit()
            
            logger.info(f"Created training record for {train_id} with phase: training")
            
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
                        "source_llm": source_llm,
                        "training_data": training_data
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

    async def queue_training(
            self,
            db:AsyncSession,
            endpoint_id:str,
            training_pairs_count:int,
            epochs:int,
            batch_size:int,
            learning_rate=float
    ):
        try:
            from fastapi import APIRouter, Depends, HTTPException, BackgroundTasks
            from app.models import (
                get_db, Endpoint, EndpointConfig, InferenceLog,
                TrainingRun, Metric, CarbonEmission
            )
            from sqlalchemy import select, func
            from app.training.gpu_queue_manager import gpu_queue_manager

            # Check if endpoint exists
            endpoint = await db.get(Endpoint, endpoint_id)
            if not endpoint:
                raise HTTPException(status_code=404, detail="Endpoint not found")
            
            # Check if we have enough data
            inference_count = await db.scalar(
                select(func.count(InferenceLog.id))
                .where(InferenceLog.endpoint_id == endpoint_id)
                .where(InferenceLog.model_used == "llm")
            )
            
            if inference_count < 10:
                raise HTTPException(
                    status_code=400,
                    detail=f"Insufficient training data. Need at least 10 examples, have {inference_count}"
                )
            
            # Determine training provider - use from request or check environment config
            provider = 'runpod'
            if not provider:
                # Check environment for default cloud provider
                import os
                if os.getenv("RUNPOD_TRAINING_ENABLED", "false").lower() == "true":
                    provider = "runpod"
                elif os.getenv("LAMBDA_TRAINING_ENABLED", "false").lower() == "true":
                    provider = "lambda"
                elif os.getenv("AZURE_TRAINING_ENABLED", "false").lower() == "true":
                    provider = "azure"
            
            # If provider is specified and GPU training is available, use cloud training
            if provider and gpu_queue_manager:
                # Initialize GPU queue manager if needed
                if not gpu_queue_manager.redis_client:
                    await gpu_queue_manager.initialize()
                
                # Get training examples
                examples = await db.execute(
                    select(InferenceLog.input_text, InferenceLog.llm_output)
                    .where(InferenceLog.endpoint_id == endpoint_id)
                    .where(InferenceLog.model_used == "llm")
                    .limit(training_pairs_count if training_pairs_count else 1000)
                )
                
                # Format training data
                training_data_parts = []
                for input_text, output_text in examples:
                    training_data_parts.append(f"Q: {input_text}\nA: {output_text}")
                
                training_data = "\n\n".join(training_data_parts)
                
                # Create training configuration
                training_config = {
                    "epochs": epochs,
                    "batch_size": batch_size,
                    "learning_rate": learning_rate,
                    "training_data": training_data
                }
                
                # Add job to GPU queue
                job_id = await gpu_queue_manager.add_job(
                    endpoint_id=endpoint_id,
                    training_config=training_config,
                    priority=1,
                    provider=provider
                )
        except Exception as e:
            logger.error(f"Error starting training for {endpoint_id}: {e}")
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
            
            # Collect evaluation data from NEW LLM inferences (unseen data only)
            evaluation_sample_size = int(os.getenv("DEFAULT_EVALUATION_SAMPLE_SIZE", "50"))
            
            # Validate evaluation sample size
            if evaluation_sample_size < 1:
                logger.warning(f"Invalid DEFAULT_EVALUATION_SAMPLE_SIZE: {evaluation_sample_size}. Using default of 50.")
                evaluation_sample_size = 50
            elif evaluation_sample_size > 1000:
                logger.warning(f"Large DEFAULT_EVALUATION_SAMPLE_SIZE: {evaluation_sample_size}. Consider reducing for faster evaluation.")
            
            logger.info(f"Starting evaluation workflow for {train_id} - need {evaluation_sample_size} unseen samples")
            
            from app.models.models import InferenceLog
            from sqlalchemy import func
            
            # Only use inference logs created AFTER the training was completed
            training_cutoff = model.created_at or datetime.now()
            
            # Get NEW LLM inferences for evaluation
            recent_llm_inferences_stmt = select(InferenceLog).where(
                InferenceLog.endpoint_id == model.endpoint_id,
                InferenceLog.model_used == "llm",
                InferenceLog.created_at > training_cutoff  # Only unseen data
            ).order_by(InferenceLog.created_at.desc()).limit(evaluation_sample_size)
            
            recent_inferences_result = await db.execute(recent_llm_inferences_stmt)
            recent_inferences = recent_inferences_result.scalars().all()
            
            if len(recent_inferences) < evaluation_sample_size:
                # Not enough unseen data yet - set up monitoring for new inferences
                still_needed = evaluation_sample_size - len(recent_inferences)
                logger.info(f"Insufficient unseen samples for {train_id}: {len(recent_inferences)}/{evaluation_sample_size} available. Still need {still_needed} more LLM inferences after {training_cutoff}.")
                
                # Store evaluation request in Redis for monitoring
                await self._queue_evaluation_request(train_id, model.endpoint_id, evaluation_sample_size, training_cutoff)
                return
            
            # We have enough unseen data - prepare evaluation inputs
            logger.info(f"Found {len(recent_inferences)} unseen LLM inferences for evaluation of {train_id}")
            
            evaluation_inputs = []
            llm_outputs = []
            
            for inference in recent_inferences:
                evaluation_inputs.append(inference.input_text)
                llm_outputs.append(inference.llm_output)
            
            # Store evaluation context for when SLM batch job completes
            evaluation_context = {
                "train_id": train_id,
                "endpoint_id": model.endpoint_id,
                "version": model.version,
                "evaluation_inputs": evaluation_inputs,
                "llm_outputs": llm_outputs,
                "created_at": datetime.now().isoformat()
            }
            
            # Store evaluation context in Redis for SLM batch job completion
            await self.redis_client.setex(
                f"evaluation_context:{train_id}",
                3600,  # 1 hour TTL
                str(evaluation_context).encode()
            )
            
            # Create SLM batch job for evaluation
            batch_job_result = self.k8s_manager.create_slm_batch_job(
                train_id=train_id,
                endpoint_id=model.endpoint_id,
                version=model.version,
                model_path=f"{model.endpoint_id}/v{model.version}",
                evaluation_batch_id=train_id
            )
            
            logger.info(f"Created evaluation batch job: {batch_job_result['job_name']} for {train_id}")
            
            # The SLM batch job will call back when complete with SLM outputs
            
        except Exception as e:
            logger.error(f"Error starting evaluation workflow for {train_id}: {e}")
            await self._update_phase(db, train_id, "failed")
            raise
    
    async def _queue_evaluation_request(self, train_id: str, endpoint_id: str, required_samples: int, training_cutoff: datetime):
        """Queue an evaluation request to be processed when enough unseen samples are available."""
        evaluation_request = {
            "train_id": train_id,
            "endpoint_id": endpoint_id,
            "required_samples": required_samples,
            "training_cutoff": training_cutoff.isoformat(),
            "queued_at": datetime.now().isoformat()
        }
        
        # Store in Redis with endpoint-specific key
        await self.redis_client.setex(
            f"pending_evaluation:{endpoint_id}",
            86400,  # 24 hours TTL
            str(evaluation_request).encode()
        )
        
        logger.info(f"Queued evaluation request for {train_id} - waiting for {required_samples} unseen samples after {training_cutoff}")
    
    async def check_pending_evaluations(self, db: AsyncSession, endpoint_id: str):
        """Check if any pending evaluations can now be started (called after new LLM inference)."""
        try:
            # Check if there's a pending evaluation for this endpoint
            pending_key = f"pending_evaluation:{endpoint_id}"
            pending_data = await self.redis_client.get(pending_key)
            
            if not pending_data:
                return  # No pending evaluation
            
            import ast
            evaluation_request = ast.literal_eval(pending_data.decode())
            train_id = evaluation_request["train_id"]
            required_samples = evaluation_request["required_samples"]
            # Parse ISO format datetime string - compatible with older Python versions
            cutoff_str = evaluation_request["training_cutoff"]
            # Handle both with and without microseconds
            try:
                training_cutoff = datetime.strptime(cutoff_str, "%Y-%m-%dT%H:%M:%S.%f")
            except ValueError:
                training_cutoff = datetime.strptime(cutoff_str, "%Y-%m-%dT%H:%M:%S")
            
            # Check if we now have enough unseen samples
            from app.models.models import InferenceLog
            from sqlalchemy import func
            
            sample_count = await db.scalar(
                select(func.count(InferenceLog.id)).where(
                    InferenceLog.endpoint_id == endpoint_id,
                    InferenceLog.model_used == "llm",
                    InferenceLog.created_at > training_cutoff  # Only unseen data
                )
            )
            
            if sample_count >= required_samples:
                logger.info(f"Sufficient unseen samples ({sample_count}/{required_samples}) available for {train_id}. Starting evaluation.")
                
                # Remove from pending queue
                # await self.redis_client.delete(pending_key)
                
                # Start evaluation workflow
                await self._start_evaluation_workflow(db, train_id)
            else:
                still_needed = required_samples - sample_count
                logger.debug(f"Still waiting for unseen samples: {sample_count}/{required_samples} for {train_id}. Need {still_needed} more.")
                
        except Exception as e:
            logger.error(f"Error checking pending evaluations for {endpoint_id}: {e}")

    async def handle_slm_batch_completion(
        self,
        db: AsyncSession,
        train_id: str,
        slm_outputs: List[str]
    ) -> Dict[str, Any]:
        """Handle completion of SLM batch job and trigger evaluation service."""
        try:
            # Get evaluation context from Redis
            context_data = await self.redis_client.get(f"evaluation_context:{train_id}")
            if not context_data:
                raise Exception(f"Evaluation context not found for {train_id}")
            
            import ast
            evaluation_context = ast.literal_eval(context_data.decode())
            
            llm_outputs = evaluation_context["llm_outputs"]
            
            if len(slm_outputs) != len(llm_outputs):
                raise Exception(f"Mismatched output lengths: {len(slm_outputs)} SLM vs {len(llm_outputs)} LLM")
            
            # Create evaluation pairs
            evaluation_pairs = []
            for llm_output, slm_output in zip(llm_outputs, slm_outputs):
                evaluation_pairs.append({
                    "llm_response": llm_output,
                    "slm_response": slm_output
                })
            
            # Send to evaluation service
            async with httpx.AsyncClient() as client:
                response = await client.post(
                    f"{self.evaluation_service_url}/api/v1/evaluate",
                    json={
                        "train_id": train_id,
                        "evaluation_pairs": evaluation_pairs
                    }
                )
                
                if response.status_code == 200:
                    result = response.json()
                    logger.info(f"Evaluation job queued for {train_id}: {result}")
                    
                    # Clean up Redis context
                    await self.redis_client.delete(f"evaluation_context:{train_id}")
                    
                    return {"status": "evaluation_queued", "result": result}
                else:
                    raise Exception(f"Evaluation service error: {response.status_code} {response.text}")
                    
        except Exception as e:
            logger.error(f"Error handling SLM batch completion for {train_id}: {e}")
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
                model_path=f"{model.endpoint_id}/v{model.version}"
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