"""Lambda Cloud Training API Endpoints

Provides endpoints for managing Lambda Cloud GPU training, monitoring progress,
and retrieving training metrics.
"""

from fastapi import APIRouter, HTTPException, BackgroundTasks, Query
from typing import Dict, Any, List, Optional
from datetime import datetime
import logging
from pydantic import BaseModel

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/lambda", tags=["Lambda Cloud"])


class TrainingJobRequest(BaseModel):
    """Request model for starting a training job"""
    endpoint_id: str
    training_data: str  # Training data as string
    priority: int = 0
    epochs: int = 3
    batch_size: int = 4
    learning_rate: float = 2e-4


class TrainingProgressResponse(BaseModel):
    """Response model for training progress"""
    job_id: str
    status: str
    progress_percent: float
    current_epoch: Optional[int] = None
    total_epochs: Optional[int] = None
    train_loss: Optional[List[float]] = None
    val_loss: Optional[List[float]] = None
    logs: Optional[List[str]] = None
    estimated_time_remaining: Optional[str] = None


@router.get("/status")
async def get_lambda_status() -> Dict[str, Any]:
    """Get Lambda Cloud infrastructure status"""
    try:
        from app.training.lambda_cloud_provisioner import lambda_cloud_provisioner
        
        status = await lambda_cloud_provisioner.get_infrastructure_status()
        readiness = await lambda_cloud_provisioner.validate_training_readiness()
        
        return {
            "provider": "lambda_cloud",
            "initialized": status.get("initialized", False),
            "ready": readiness.get("ready", False),
            "selected_instance_type": status.get("selected_instance_type"),
            "available_instance_types": status.get("available_instance_types", 0),
            "active_instances": status.get("active_instances", []),
            "estimated_hourly_cost": status.get("estimated_hourly_cost", 0),
            "issues": readiness.get("issues", []),
            "recommendations": readiness.get("recommendations", [])
        }
    except Exception as e:
        logger.error(f"Failed to get Lambda Cloud status: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/instance-types")
async def list_available_instance_types() -> Dict[str, Any]:
    """List available Lambda Cloud GPU instance types"""
    try:
        from app.training.lambda_cloud_trainer import LambdaCloudTrainer
        
        trainer = LambdaCloudTrainer()
        instance_types = await trainer.get_available_instance_types()
        
        return {
            "available_count": len(instance_types),
            "instance_types": instance_types
        }
    except Exception as e:
        logger.error(f"Failed to list instance types: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/train")
async def start_training(request: TrainingJobRequest) -> Dict[str, Any]:
    """Start a new training job on Lambda Cloud"""
    try:
        from app.training.gpu_queue_manager import gpu_queue_manager
        from app.training.lambda_cloud_provisioner import lambda_cloud_provisioner
        
        # Validate infrastructure readiness
        readiness = await lambda_cloud_provisioner.validate_training_readiness()
        if not readiness["ready"]:
            return {
                "error": "Lambda Cloud not ready",
                "issues": readiness["issues"],
                "recommendations": readiness["recommendations"]
            }
        
        # Prepare training configuration
        training_config = {
            "epochs": request.epochs,
            "batch_size": request.batch_size,
            "learning_rate": request.learning_rate,
            "training_data": request.training_data  # Use provided training data
        }
        
        # Initialize queue manager if needed
        if not gpu_queue_manager.redis_client:
            await gpu_queue_manager.initialize()
        
        # Add job to queue
        job_id = await gpu_queue_manager.add_job(
            endpoint_id=request.endpoint_id,
            training_config=training_config,
            priority=request.priority,
            provider="lambda"
        )
        
        # Estimate cost
        estimated_hours = request.epochs * 0.5  # Rough estimate
        cost_estimate = await lambda_cloud_provisioner.estimate_training_cost(estimated_hours)
        
        return {
            "job_id": job_id,
            "status": "queued",
            "provider": "lambda",
            "estimated_cost": cost_estimate,
            "message": "Training job queued for Lambda Cloud GPU processing"
        }
        
    except Exception as e:
        logger.error(f"Failed to start Lambda Cloud training: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/jobs/{job_id}/progress")
async def get_training_progress(job_id: str) -> TrainingProgressResponse:
    """Get real-time training progress for a job"""
    try:
        from app.training.gpu_queue_manager import gpu_queue_manager
        from app.training.lambda_cloud_trainer import LambdaCloudTrainer
        
        # Get job status from queue
        job_status = await gpu_queue_manager.get_job_status(job_id)
        if not job_status:
            raise HTTPException(status_code=404, detail="Job not found")
        
        # Get training logs
        trainer = LambdaCloudTrainer()
        logs = await trainer.get_training_logs(job_id)
        
        # Parse logs for metrics
        train_loss = []
        val_loss = []
        current_epoch = None
        total_epochs = job_status.get("training_config", {}).get("epochs", 3)
        
        for log_entry in logs:
            message = log_entry.get("message", "")
            
            # Parse epoch information
            if "Epoch" in message:
                try:
                    parts = message.split("Epoch")[1].strip().split("/")
                    if len(parts) >= 2:
                        current_epoch = int(parts[0])
                except:
                    pass
            
            # Parse loss values
            if "train" in message.lower() and "loss:" in message.lower():
                try:
                    loss_str = message.split("loss:")[-1].strip().split()[0]
                    train_loss.append(float(loss_str))
                except:
                    pass
            
            if ("val" in message.lower() or "eval" in message.lower()) and "loss:" in message.lower():
                try:
                    loss_str = message.split("loss:")[-1].strip().split()[0]
                    val_loss.append(float(loss_str))
                except:
                    pass
        
        # Calculate progress
        progress_percent = 0.0
        if current_epoch and total_epochs:
            progress_percent = (current_epoch / total_epochs) * 100
        elif job_status["status"] == "completed":
            progress_percent = 100.0
        
        # Extract recent log messages
        recent_logs = [log["message"] for log in logs[-10:]]  # Last 10 messages
        
        return TrainingProgressResponse(
            job_id=job_id,
            status=job_status["status"],
            progress_percent=progress_percent,
            current_epoch=current_epoch,
            total_epochs=total_epochs,
            train_loss=train_loss if train_loss else None,
            val_loss=val_loss if val_loss else None,
            logs=recent_logs if recent_logs else None,
            estimated_time_remaining=None  # TODO: Calculate based on progress
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get training progress: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/jobs")
async def list_training_jobs(
    status: Optional[str] = Query(None, description="Filter by job status"),
    limit: int = Query(10, description="Number of jobs to return")
) -> Dict[str, Any]:
    """List Lambda Cloud training jobs"""
    try:
        from app.training.gpu_queue_manager import gpu_queue_manager
        
        all_jobs = await gpu_queue_manager.list_jobs()
        
        # Filter for Lambda jobs
        lambda_jobs = [j for j in all_jobs if j.get("provider") == "lambda"]
        
        # Apply status filter if provided
        if status:
            lambda_jobs = [j for j in lambda_jobs if j["status"] == status]
        
        # Limit results
        lambda_jobs = lambda_jobs[:limit]
        
        return {
            "total_count": len(lambda_jobs),
            "jobs": lambda_jobs
        }
        
    except Exception as e:
        logger.error(f"Failed to list Lambda jobs: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/jobs/{job_id}/cancel")
async def cancel_training_job(job_id: str) -> Dict[str, Any]:
    """Cancel a Lambda Cloud training job"""
    try:
        from app.training.gpu_queue_manager import gpu_queue_manager
        
        success = await gpu_queue_manager.cancel_job(job_id)
        
        if success:
            return {
                "job_id": job_id,
                "status": "cancelled",
                "message": "Training job cancelled successfully"
            }
        else:
            return {
                "job_id": job_id,
                "status": "error",
                "message": "Could not cancel job (may be already running or completed)"
            }
            
    except Exception as e:
        logger.error(f"Failed to cancel Lambda job: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/instances")
async def list_active_instances() -> Dict[str, Any]:
    """List active Lambda Cloud instances"""
    try:
        from app.training.lambda_cloud_trainer import LambdaCloudTrainer
        
        trainer = LambdaCloudTrainer()
        instances = []
        
        for job_id, instance in trainer.active_instances.items():
            instances.append({
                "job_id": job_id,
                "instance_id": instance.id,
                "instance_type": instance.instance_type,
                "ip_address": instance.ip_address,
                "region": instance.region,
                "status": instance.status
            })
        
        return {
            "count": len(instances),
            "instances": instances
        }
        
    except Exception as e:
        logger.error(f"Failed to list active instances: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/instances/{instance_id}/terminate")
async def terminate_instance(instance_id: str) -> Dict[str, Any]:
    """Terminate a Lambda Cloud instance"""
    try:
        from app.training.lambda_cloud_trainer import LambdaCloudTrainer
        
        trainer = LambdaCloudTrainer()
        success = await trainer.terminate_instance(instance_id)
        
        if success:
            return {
                "instance_id": instance_id,
                "status": "terminated",
                "message": "Instance terminated successfully"
            }
        else:
            return {
                "instance_id": instance_id,
                "status": "error",
                "message": "Failed to terminate instance"
            }
            
    except Exception as e:
        logger.error(f"Failed to terminate instance: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/initialize")
async def initialize_lambda_cloud(background_tasks: BackgroundTasks) -> Dict[str, Any]:
    """Initialize Lambda Cloud infrastructure"""
    try:
        from app.training.lambda_cloud_provisioner import lambda_cloud_provisioner
        
        # Run initialization in background
        background_tasks.add_task(lambda_cloud_provisioner.initialize)
        
        return {
            "status": "initializing",
            "message": "Lambda Cloud infrastructure initialization started"
        }
        
    except Exception as e:
        logger.error(f"Failed to initialize Lambda Cloud: {e}")
        raise HTTPException(status_code=500, detail=str(e))