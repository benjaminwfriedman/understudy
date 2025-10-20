"""RunPod GPU Training API Endpoints

Provides API endpoints for managing RunPod GPU training jobs and instances.
Similar to Lambda Cloud endpoints but using RunPod infrastructure.
"""

import logging
from typing import Dict, Any, List, Optional
from datetime import datetime

from fastapi import APIRouter, HTTPException, BackgroundTasks, Query
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)

# Create router
router = APIRouter(prefix="/api/v1/runpod", tags=["runpod"])

# Request/Response models
class RunPodTrainingJobRequest(BaseModel):
    """Request model for starting a RunPod training job"""
    endpoint_id: str = Field(..., description="Endpoint ID for this training job")
    training_data: str = Field(..., description="Training data as text")
    epochs: int = Field(default=3, ge=1, le=100, description="Number of training epochs")
    batch_size: int = Field(default=4, ge=1, le=32, description="Training batch size")
    learning_rate: float = Field(default=2e-4, gt=0, lt=1, description="Learning rate")
    priority: int = Field(default=0, ge=0, le=10, description="Job priority (0=lowest, 10=highest)")
    gpu_type: Optional[str] = Field(default=None, description="Specific GPU type to use")


class RunPodTrainingProgressResponse(BaseModel):
    """Response model for training progress"""
    job_id: str
    status: str
    progress: Optional[float] = None
    current_epoch: Optional[int] = None
    total_epochs: Optional[int] = None
    current_loss: Optional[float] = None
    estimated_time_remaining: Optional[int] = None  # seconds
    logs: List[str] = Field(default_factory=list)
    metrics: Optional[Dict[str, Any]] = None


class RunPodInstanceInfo(BaseModel):
    """RunPod instance information"""
    id: str
    name: str
    status: str
    gpu_type: str
    created_at: datetime
    uptime_seconds: Optional[int] = None
    cost_per_hour: Optional[float] = None


# Initialize RunPod trainer (lazy loading)
runpod_trainer = None


def get_runpod_trainer():
    """Get RunPod trainer instance (lazy initialization)"""
    global runpod_trainer
    if runpod_trainer is None:
        from app.training.runpod_trainer import RunPodTrainer
        runpod_trainer = RunPodTrainer()
    return runpod_trainer


@router.get("/gpu-types")
async def list_gpu_types() -> Dict[str, Any]:
    """List available GPU types and their pricing"""
    try:
        trainer = get_runpod_trainer()
        gpu_types = await trainer.get_available_gpu_types()
        
        return {
            "gpu_types": gpu_types,
            "total_available": len(gpu_types),
            "recommended": "NVIDIA GeForce RTX 4090"  # Good price/performance ratio
        }
        
    except Exception as e:
        logger.error(f"Failed to list GPU types: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/train")
async def start_training(request: RunPodTrainingJobRequest) -> Dict[str, Any]:
    """Start a new training job on RunPod"""
    try:
        from app.training.gpu_queue_manager import gpu_queue_manager
        
        # Prepare training configuration
        training_config = {
            "epochs": request.epochs,
            "batch_size": request.batch_size,
            "learning_rate": request.learning_rate,
            "training_data": request.training_data,
            "gpu_type": request.gpu_type
        }
        
        # Initialize queue manager if needed
        if not gpu_queue_manager.redis_client:
            await gpu_queue_manager.initialize()
        
        # Add job to queue
        job_id = await gpu_queue_manager.add_job(
            endpoint_id=request.endpoint_id,
            training_config=training_config,
            priority=request.priority,
            provider="runpod"
        )
        
        # Estimate cost (rough calculation)
        estimated_hours = request.epochs * 0.3  # RunPod is typically faster
        estimated_cost = estimated_hours * 0.34  # RTX 4090 pricing
        
        return {
            "job_id": job_id,
            "status": "queued",
            "provider": "runpod",
            "estimated_cost": round(estimated_cost, 2),
            "estimated_runtime_minutes": int(estimated_hours * 60),
            "message": "Training job queued for RunPod GPU processing"
        }
        
    except Exception as e:
        logger.error(f"Failed to start RunPod training: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/jobs/{job_id}/progress")
async def get_training_progress(job_id: str) -> RunPodTrainingProgressResponse:
    """Get real-time training progress for a job"""
    try:
        from app.training.gpu_queue_manager import gpu_queue_manager
        
        # Get job details
        job = await gpu_queue_manager.get_job(job_id)
        if not job:
            raise HTTPException(status_code=404, detail="Job not found")
        
        # Get training logs if available
        trainer = get_runpod_trainer()
        logs = await trainer.get_training_logs(job_id)
        
        # Extract progress information from logs
        progress_info = _extract_progress_from_logs(logs)
        
        return RunPodTrainingProgressResponse(
            job_id=job_id,
            status=job["status"],
            progress=progress_info.get("progress"),
            current_epoch=progress_info.get("current_epoch"),
            total_epochs=job.get("training_config", {}).get("epochs"),
            current_loss=progress_info.get("current_loss"),
            logs=[log["message"] for log in logs[-20:]] if logs else [],  # Last 20 logs
            metrics=job.get("result", {}).get("metrics") if job.get("result") else None
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get training progress: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/jobs")
async def list_training_jobs(
    status: Optional[str] = Query(None, description="Filter by job status"),
    limit: int = Query(50, ge=1, le=200, description="Maximum number of jobs to return")
) -> Dict[str, Any]:
    """List RunPod training jobs"""
    try:
        from app.training.gpu_queue_manager import gpu_queue_manager
        
        # Get all jobs and filter by provider
        all_jobs = await gpu_queue_manager.list_jobs()
        runpod_jobs = [job for job in all_jobs if job.get("provider") == "runpod"]
        
        # Filter by status if provided
        if status:
            runpod_jobs = [job for job in runpod_jobs if job.get("status") == status]
        
        # Limit results
        runpod_jobs = runpod_jobs[:limit]
        
        return {
            "jobs": runpod_jobs,
            "total": len(runpod_jobs),
            "provider": "runpod"
        }
        
    except Exception as e:
        logger.error(f"Failed to list training jobs: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.delete("/jobs/{job_id}")
async def cancel_training_job(job_id: str) -> Dict[str, Any]:
    """Cancel a RunPod training job"""
    try:
        from app.training.gpu_queue_manager import gpu_queue_manager
        
        # Get job details
        job = await gpu_queue_manager.get_job(job_id)
        if not job:
            raise HTTPException(status_code=404, detail="Job not found")
        
        if job.get("provider") != "runpod":
            raise HTTPException(status_code=400, detail="Job is not a RunPod job")
        
        # Cancel the job
        success = await gpu_queue_manager.cancel_job(job_id)
        
        if success:
            return {
                "message": f"Training job {job_id} cancelled successfully",
                "job_id": job_id,
                "status": "cancelled"
            }
        else:
            raise HTTPException(status_code=400, detail="Failed to cancel job")
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to cancel training job: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/instances")
async def list_runpod_instances() -> Dict[str, Any]:
    """List all RunPod instances"""
    try:
        trainer = get_runpod_trainer()
        pods = await trainer.get_existing_pods()
        
        instances = []
        for pod in pods:
            runtime = pod.get("runtime", {})
            uptime = runtime.get("uptimeInSeconds") if runtime else None
            
            instances.append(RunPodInstanceInfo(
                id=pod["id"],
                name=pod["name"],
                status=pod.get("desiredStatus", "unknown"),
                gpu_type=pod.get("gpuCount", "1") + "x GPU",  # Simplified
                created_at=datetime.utcnow(),  # Would need actual creation time
                uptime_seconds=uptime
            ))
        
        return {
            "instances": [instance.dict() for instance in instances],
            "total": len(instances),
            "provider": "runpod"
        }
        
    except Exception as e:
        logger.error(f"Failed to list RunPod instances: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.delete("/instances/{instance_id}")
async def terminate_instance(instance_id: str) -> Dict[str, Any]:
    """Terminate a specific RunPod instance"""
    try:
        trainer = get_runpod_trainer()
        success = await trainer.terminate_pod(instance_id)
        
        if success:
            return {
                "message": f"RunPod instance {instance_id} terminated successfully",
                "instance_id": instance_id,
                "status": "terminated"
            }
        else:
            raise HTTPException(status_code=400, detail="Failed to terminate instance")
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to terminate instance: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/status")
async def get_runpod_status() -> Dict[str, Any]:
    """Get RunPod training infrastructure status"""
    try:
        trainer = get_runpod_trainer()
        
        # Get available GPU types
        gpu_types = await trainer.get_available_gpu_types()
        
        # Get existing instances
        pods = await trainer.get_existing_pods()
        
        return {
            "status": "ready",
            "provider": "runpod",
            "available_gpu_types": len(gpu_types),
            "active_instances": len(pods),
            "recommended_gpu": "NVIDIA GeForce RTX 4090",
            "features": [
                "Per-second billing",
                "Fast 30-second deployment",
                "Wide GPU selection",
                "SSH access",
                "Container-based execution"
            ]
        }
        
    except Exception as e:
        logger.error(f"Failed to get RunPod status: {e}")
        return {
            "status": "error",
            "provider": "runpod",
            "error": str(e)
        }


def _extract_progress_from_logs(logs: List[Dict]) -> Dict[str, Any]:
    """Extract training progress information from logs"""
    if not logs:
        return {}
    
    progress_info = {}
    
    for log in reversed(logs):  # Check most recent logs first
        message = log.get("message", "").lower()
        
        # Extract epoch information
        if "epoch" in message and "/" in message:
            try:
                # Look for patterns like "Epoch 2/5" or "epoch: 2/5"
                parts = message.split("epoch")[-1].strip()
                if "/" in parts:
                    epoch_part = parts.split()[0] if " " in parts else parts
                    if "/" in epoch_part:
                        current, total = epoch_part.split("/")
                        progress_info["current_epoch"] = int(current.strip(": "))
                        progress_info["total_epochs"] = int(total.strip())
                        progress_info["progress"] = progress_info["current_epoch"] / progress_info["total_epochs"]
            except (ValueError, IndexError):
                pass
        
        # Extract loss information
        if "loss:" in message:
            try:
                loss_part = message.split("loss:")[-1].strip()
                loss_value = float(loss_part.split()[0])
                progress_info["current_loss"] = loss_value
            except (ValueError, IndexError):
                pass
    
    return progress_info