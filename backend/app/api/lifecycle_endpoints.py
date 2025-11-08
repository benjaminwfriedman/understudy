"""
Model Lifecycle API Endpoints

Endpoints for managing model lifecycle phases and distributed service coordination.
"""

from fastapi import APIRouter, Depends, HTTPException, BackgroundTasks
from sqlalchemy.ext.asyncio import AsyncSession
from typing import Dict, Any, List, Optional
from pydantic import BaseModel

from app.models import get_db
from app.core.model_lifecycle import get_lifecycle_manager
from app.core.config import settings
import logging

logger = logging.getLogger(__name__)

router = APIRouter(prefix=f"{settings.API_V1_STR}/lifecycle", tags=["model-lifecycle"])


class TrainingCompletionRequest(BaseModel):
    phase: str
    training_loss: Optional[float] = None
    model_weights_path: Optional[str] = None
    training_metrics: Optional[Dict[str, Any]] = None
    carbon_emissions_kg: Optional[float] = None
    energy_consumed_kwh: Optional[float] = None
    training_time_wall: Optional[float] = None


class SimilarityScoreRequest(BaseModel):
    train_id: str
    semantic_similarity_score: float
    evaluation_count: int
    processing_time: float


class DeploymentRequest(BaseModel):
    train_id: str


@router.post("/training/{train_id}/complete")
async def training_complete(
    train_id: str,
    request: TrainingCompletionRequest,
    db: AsyncSession = Depends(get_db)
):
    """
    Handle training completion notification from Training Service.
    This endpoint is called when the Training Service completes model download.
    """
    try:
        lifecycle_manager = get_lifecycle_manager()
        result = await lifecycle_manager.handle_training_completion(
            db=db,
            train_id=train_id,
            phase=request.phase,
            training_loss=request.training_loss,
            model_weights_path=request.model_weights_path,
            carbon_emissions_kg=request.carbon_emissions_kg,
            energy_consumed_kwh=request.energy_consumed_kwh,
            training_time_wall=request.training_time_wall
        )
        
        return {
            "message": "Training completion handled",
            "train_id": train_id,
            "result": result
        }
        
    except Exception as e:
        logger.error(f"Error handling training completion for {train_id}: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/models/{train_id}/similarity-score")
async def receive_similarity_score(
    train_id: str,
    request: SimilarityScoreRequest,
    db: AsyncSession = Depends(get_db)
):
    """
    Receive semantic similarity score from Evaluation Service.
    This endpoint is called when evaluation is complete.
    """
    try:
        lifecycle_manager = get_lifecycle_manager()
        result = await lifecycle_manager.handle_similarity_score(
            db=db,
            train_id=train_id,
            similarity_score=request.semantic_similarity_score,
            evaluation_count=request.evaluation_count
        )
        
        return {
            "message": "Similarity score processed",
            "train_id": train_id,
            "result": result
        }
        
    except Exception as e:
        logger.error(f"Error processing similarity score for {train_id}: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/models/{train_id}/deploy")
async def deploy_model(
    train_id: str,
    db: AsyncSession = Depends(get_db)
):
    """
    Manually deploy a model to Kubernetes.
    This can be used for rollbacks or manual deployments.
    """
    try:
        lifecycle_manager = get_lifecycle_manager()
        result = await lifecycle_manager.deploy_model(db=db, train_id=train_id)
        
        return {
            "message": "Model deployment initiated",
            "train_id": train_id,
            "result": result
        }
        
    except Exception as e:
        logger.error(f"Error deploying model {train_id}: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.delete("/endpoints/{endpoint_id}/deployment")
async def undeploy_model(
    endpoint_id: str,
    db: AsyncSession = Depends(get_db)
):
    """
    Undeploy the currently deployed model for an endpoint.
    """
    try:
        lifecycle_manager = get_lifecycle_manager()
        result = await lifecycle_manager.undeploy_model(db=db, endpoint_id=endpoint_id)
        
        return {
            "message": "Model undeployed",
            "endpoint_id": endpoint_id,
            "result": result
        }
        
    except Exception as e:
        logger.error(f"Error undeploying model for {endpoint_id}: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/models/{train_id}/status")
async def get_model_status(
    train_id: str,
    db: AsyncSession = Depends(get_db)
):
    """
    Get the current status of a model including its lifecycle phase.
    """
    try:
        lifecycle_manager = get_lifecycle_manager()
        status = await lifecycle_manager.get_model_status(db=db, train_id=train_id)
        
        if status.get("status") == "not_found":
            raise HTTPException(status_code=404, detail="Model not found")
        
        return status
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting model status for {train_id}: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/endpoints/{endpoint_id}/models")
async def list_endpoint_models(
    endpoint_id: str,
    db: AsyncSession = Depends(get_db)
):
    """
    List all models for an endpoint with their lifecycle status.
    """
    try:
        lifecycle_manager = get_lifecycle_manager()
        models = await lifecycle_manager.list_models_by_endpoint(
            db=db, 
            endpoint_id=endpoint_id
        )
        
        return {
            "endpoint_id": endpoint_id,
            "models": models,
            "total_count": len(models)
        }
        
    except Exception as e:
        logger.error(f"Error listing models for {endpoint_id}: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/models")
async def list_all_models(
    db: AsyncSession = Depends(get_db),
    phase: str = None,
    endpoint_id: str = None
):
    """
    List all models with optional filtering by phase or endpoint.
    """
    try:
        from sqlalchemy import select
        from app.models.models import TrainingRun
        
        # Build query with optional filters
        stmt = select(TrainingRun).where(TrainingRun.is_deleted == False)
        
        if phase:
            stmt = stmt.where(TrainingRun.phase == phase)
        if endpoint_id:
            stmt = stmt.where(TrainingRun.endpoint_id == endpoint_id)
            
        stmt = stmt.order_by(TrainingRun.created_at.desc())
        
        result = await db.execute(stmt)
        models = result.scalars().all()
        
        model_list = []
        for model in models:
            model_info = {
                "train_id": model.train_id,
                "endpoint_id": model.endpoint_id,
                "version": model.version,
                "phase": model.phase,
                "similarity_score": model.semantic_similarity_score,
                "created_at": model.created_at.isoformat() if model.created_at else None,
                "slm_type": model.slm_type,
                "source_llm": model.source_llm,
                "is_deployed": model.phase == "deployed",
                "k8s_deployment_name": model.k8s_deployment_name
            }
            model_list.append(model_info)
        
        return {
            "models": model_list,
            "total_count": len(model_list),
            "filters": {
                "phase": phase,
                "endpoint_id": endpoint_id
            }
        }
        
    except Exception as e:
        logger.error(f"Error listing models: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/k8s/deployments")
async def list_k8s_deployments():
    """
    List all current SLM deployments in Kubernetes.
    """
    try:
        from app.core.k8s_manager import get_k8s_manager
        
        k8s_manager = get_k8s_manager()
        deployments = k8s_manager.list_slm_deployments()
        
        return {
            "deployments": deployments,
            "total_count": len(deployments)
        }
        
    except Exception as e:
        logger.error(f"Error listing K8s deployments: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/k8s/deployments/{deployment_name}/status")
async def get_deployment_status(deployment_name: str):
    """
    Get the status of a specific Kubernetes deployment.
    """
    try:
        from app.core.k8s_manager import get_k8s_manager
        
        k8s_manager = get_k8s_manager()
        status = k8s_manager.get_deployment_status(deployment_name)
        
        if status is None:
            raise HTTPException(status_code=404, detail="Deployment not found")
        
        return status
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting deployment status for {deployment_name}: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/services/health")
async def check_services_health():
    """
    Check the health of all distributed services.
    """
    import httpx
    import os
    
    services = {
        "training_service": os.getenv("TRAINING_SERVICE_URL", "http://training-service:8002"),
        "evaluation_service": os.getenv("EVALUATION_SERVICE_URL", "http://evaluation-service:8001"),
        "model_broker": os.getenv("MODEL_BROKER_SERVICE_URL", "http://model-broker-service:8003")
    }
    
    health_status = {}
    
    for service_name, service_url in services.items():
        try:
            async with httpx.AsyncClient(timeout=5.0) as client:
                response = await client.get(f"{service_url}/health")
                health_status[service_name] = {
                    "status": "healthy" if response.status_code == 200 else "unhealthy",
                    "url": service_url,
                    "response_time": response.elapsed.total_seconds() if hasattr(response, 'elapsed') else None
                }
        except Exception as e:
            health_status[service_name] = {
                "status": "unreachable",
                "url": service_url,
                "error": str(e)
            }
    
    overall_healthy = all(s["status"] == "healthy" for s in health_status.values())
    
    return {
        "overall_status": "healthy" if overall_healthy else "degraded",
        "services": health_status
    }