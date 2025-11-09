from fastapi import APIRouter, Depends, HTTPException, BackgroundTasks
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, func
from typing import List, Optional
from datetime import datetime, timedelta

from app.models import (
    get_db, Endpoint, EndpointConfig, InferenceLog,
    TrainingRun, Metric, CarbonEmission, Comparison
)
from app.api.schemas import (
    EndpointCreate, EndpointResponse, InferenceRequest, InferenceResponse,
    TrainingRequest, TrainingResponse, TrainingRunResponse, MetricResponse,
    MetricsSummary, CarbonSummary, CarbonTimeline, HealthResponse,
    ExampleResponse, ExamplesListResponse, ComparisonResponse
)
try:
    from app.core.inference_router import InferenceRouter
    INFERENCE_AVAILABLE = True
except ImportError:
    INFERENCE_AVAILABLE = False
    class InferenceRouter:
        async def route_request(self, *args, **kwargs):
            raise ImportError("Inference dependencies not available")

try:
    from app.training.trainer import TrainingScheduler
    TRAINING_AVAILABLE = True
except ImportError:
    TRAINING_AVAILABLE = False

try:
    from app.training.gpu_queue_manager import gpu_queue_manager
    GPU_TRAINING_AVAILABLE = True
except ImportError:
    GPU_TRAINING_AVAILABLE = False
    gpu_queue_manager = None
    class TrainingScheduler:
        def get_active_jobs(self):
            return []
        async def schedule_training(self, endpoint_id):
            raise ImportError("Training dependencies not available")
from app.core.config import settings
import logging
import os
import redis.asyncio as redis
import ast
import json

logger = logging.getLogger(__name__)

router = APIRouter(prefix=settings.API_V1_STR, tags=["endpoints"])
inference_router = InferenceRouter()
training_scheduler = TrainingScheduler()

# Import Azure endpoints if enabled
if os.getenv("AZURE_TRAINING_ENABLED", "false").lower() == "true":
    from app.api.azure_endpoints import router as azure_router
    router.include_router(azure_router)


# Endpoints CRUD
@router.post("/endpoints", response_model=EndpointResponse)
async def create_endpoint(
    endpoint_data: EndpointCreate,
    db: AsyncSession = Depends(get_db)
):
    """Create a new Understudy endpoint."""
    # Create endpoint
    endpoint = Endpoint(
        name=endpoint_data.name,
        description=endpoint_data.description,
        llm_provider=endpoint_data.llm_provider,
        llm_model=endpoint_data.llm_model,
        status="training"
    )
    db.add(endpoint)
    await db.flush()
    
    # Create config
    from app.api.schemas import EndpointConfigCreate
    config_data = endpoint_data.config or EndpointConfigCreate()
    config = EndpointConfig(
        endpoint_id=endpoint.id,
        **config_data.dict()
    )
    db.add(config)
    
    await db.commit()
    await db.refresh(endpoint)
    
    # Load the config relationship
    await db.refresh(endpoint, ["config"])
    
    return endpoint


@router.get("/endpoints", response_model=List[EndpointResponse])
async def list_endpoints(
    skip: int = 0,
    limit: int = 100,
    db: AsyncSession = Depends(get_db)
):
    """List all endpoints with configs."""
    result = await db.execute(
        select(Endpoint)
        .offset(skip)
        .limit(limit)
        .order_by(Endpoint.created_at.desc())
    )
    endpoints = result.scalars().all()
    
    # Load config for each endpoint
    for endpoint in endpoints:
        await db.refresh(endpoint, ["config"])
    
    return endpoints


@router.get("/endpoints/{endpoint_id}", response_model=EndpointResponse)
async def get_endpoint(
    endpoint_id: str,
    db: AsyncSession = Depends(get_db)
):
    """Get a specific endpoint with config."""
    stmt = select(Endpoint).where(Endpoint.id == endpoint_id)
    result = await db.execute(stmt)
    endpoint = result.scalar_one_or_none()
    
    if not endpoint:
        raise HTTPException(status_code=404, detail="Endpoint not found")
    
    # Load the config
    await db.refresh(endpoint, ["config"])
    
    return endpoint


@router.delete("/endpoints/{endpoint_id}")
async def delete_endpoint(
    endpoint_id: str,
    db: AsyncSession = Depends(get_db)
):
    """Delete an endpoint and all associated data."""
    endpoint = await db.get(Endpoint, endpoint_id)
    if not endpoint:
        raise HTTPException(status_code=404, detail="Endpoint not found")
    
    await db.delete(endpoint)
    await db.commit()
    
    return {"message": "Endpoint deleted successfully"}


# Inference
@router.post("/inference/{endpoint_id}", response_model=InferenceResponse)
async def inference(
    endpoint_id: str,
    request: InferenceRequest,
    db: AsyncSession = Depends(get_db)
):
    """Generate text using the endpoint."""
    # Check if endpoint exists
    endpoint = await db.get(Endpoint, endpoint_id)
    if not endpoint:
        raise HTTPException(status_code=404, detail="Endpoint not found")
    
    # Route inference request
    if not INFERENCE_AVAILABLE:
        raise HTTPException(
            status_code=503,
            detail="Inference not available. Install ML dependencies: pip install -r requirements.txt"
        )
    
    try:
        result = await inference_router.route_request(
            endpoint_id=endpoint_id,
            prompt=request.prompt,
            messages=request.messages,
            langchain_metadata=request.langchain_metadata,
            max_tokens=request.max_tokens,
            temperature=request.temperature,
            top_p=request.top_p,
            frequency_penalty=request.frequency_penalty,
            presence_penalty=request.presence_penalty,
            stop=request.stop
        )
        
        # Check for pending evaluations if this was an LLM inference
        # (Only trigger when we're collecting unseen data for evaluation)
        if result.get("model_used") == "llm":

            

            try:
                from app.core.model_lifecycle import get_lifecycle_manager
                lifecycle_manager = get_lifecycle_manager()
                await lifecycle_manager.check_pending_evaluations(db, endpoint_id)
                await lifecycle_manager.training_ready_or_needed(db, endpoint_id)

            except Exception as eval_check_error:
                # Don't fail the inference if evaluation check fails
                logger.warning(f"Error checking pending evaluations for {endpoint_id}: {eval_check_error}")
        
        return InferenceResponse(**result)
    except Exception as e:
        logger.error(f"Inference error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# Training
@router.post("/training/{endpoint_id}", response_model=TrainingResponse)
async def start_training(
    endpoint_id: str,
    request: TrainingRequest,
    background_tasks: BackgroundTasks,
    db: AsyncSession = Depends(get_db)
):
    """Start training for an endpoint."""
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
    provider = request.provider
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
    if provider and GPU_TRAINING_AVAILABLE and gpu_queue_manager:
        # Initialize GPU queue manager if needed
        if not gpu_queue_manager.redis_client:
            await gpu_queue_manager.initialize()
        
        # Get training examples
        examples = await db.execute(
            select(InferenceLog.input_text, InferenceLog.llm_output)
            .where(InferenceLog.endpoint_id == endpoint_id)
            .where(InferenceLog.model_used == "llm")
            .limit(request.num_examples if request.num_examples else 1000)
        )
        
        # Format training data
        training_data_parts = []
        for input_text, output_text in examples:
            training_data_parts.append(f"Q: {input_text}\nA: {output_text}")
        
        training_data = "\n\n".join(training_data_parts)
        actual_examples_count = len(training_data_parts)
        
        # Create training configuration
        training_config = {
            "epochs": request.epochs,
            "batch_size": request.batch_size,
            "learning_rate": request.learning_rate,
            "training_data": training_data,
            "training_pairs_count": actual_examples_count
        }
        
        # Add job to GPU queue
        job_id = await gpu_queue_manager.add_job(
            endpoint_id=endpoint_id,
            training_config=training_config,
            provider=provider
        )
        
        return TrainingResponse(
            training_run_id=job_id,
            status="queued",
            message=f"Training job queued for {provider.upper()} GPU processing"
        )
    
    # Fall back to local training
    if not TRAINING_AVAILABLE:
        raise HTTPException(
            status_code=503,
            detail="Training not available. Install ML dependencies: pip install -r requirements.txt"
        )
    
    # Check for active local training
    active_jobs = training_scheduler.get_active_jobs()
    if endpoint_id in active_jobs:
        return TrainingResponse(
            training_run_id="",
            status="already_running",
            message="Training is already in progress for this endpoint"
        )
    
    # Schedule local training
    message = await training_scheduler.schedule_training(endpoint_id)
    
    return TrainingResponse(
        training_run_id="",
        status="scheduled",
        message=message
    )


@router.get("/training/{endpoint_id}/runs", response_model=List[TrainingRunResponse])
async def get_training_runs(
    endpoint_id: str,
    db: AsyncSession = Depends(get_db)
):
    """Get training runs for an endpoint."""
    result = await db.execute(
        select(TrainingRun)
        .where(TrainingRun.endpoint_id == endpoint_id)
        .order_by(TrainingRun.start_time.desc())
    )
    return result.scalars().all()


@router.get("/endpoints/{endpoint_id}/comparisons", response_model=List[ComparisonResponse])
async def get_comparisons(
    endpoint_id: str,
    skip: int = 0,
    limit: int = 20,
    db: AsyncSession = Depends(get_db)
):
    """Get LLM vs SLM comparisons for an endpoint."""
    result = await db.execute(
        select(Comparison)
        .where(Comparison.endpoint_id == endpoint_id)
        .order_by(Comparison.created_at.desc())
        .offset(skip)
        .limit(limit)
    )
    return result.scalars().all()


# Metrics
@router.get("/metrics/{endpoint_id}", response_model=MetricsSummary)
async def get_metrics_summary(
    endpoint_id: str,
    days: int = 30,
    db: AsyncSession = Depends(get_db)
):
    """Get metrics summary for an endpoint."""
    start_date = datetime.utcnow() - timedelta(days=days)
    
    # Get average similarity
    avg_similarity = await db.scalar(
        select(func.avg(Metric.value))
        .where(Metric.endpoint_id == endpoint_id)
        .where(Metric.metric_type == "semantic_similarity")
        .where(Metric.calculated_at >= start_date)
    ) or 0.0
    
    # Get inference counts
    total_inferences = await db.scalar(
        select(func.count(InferenceLog.id))
        .where(InferenceLog.endpoint_id == endpoint_id)
        .where(InferenceLog.created_at >= start_date)
    ) or 0
    
    llm_inferences = await db.scalar(
        select(func.count(InferenceLog.id))
        .where(InferenceLog.endpoint_id == endpoint_id)
        .where(InferenceLog.model_used == "llm")
        .where(InferenceLog.created_at >= start_date)
    ) or 0
    
    slm_inferences = total_inferences - llm_inferences
    
    # Calculate average costs for LLM and SLM
    llm_avg_cost = await db.scalar(
        select(func.avg(InferenceLog.cost_usd))
        .where(InferenceLog.endpoint_id == endpoint_id)
        .where(InferenceLog.model_used == "llm")
        .where(InferenceLog.created_at >= start_date)
    ) or 0.0
    
    slm_avg_cost = await db.scalar(
        select(func.avg(InferenceLog.cost_usd))
        .where(InferenceLog.endpoint_id == endpoint_id)
        .where(InferenceLog.model_used == "slm")
        .where(InferenceLog.created_at >= start_date)
    ) or 0.0001  # Default to small value if no SLM data yet
    
    # Calculate total cost saved (for backward compatibility)
    total_cost_saved = (llm_avg_cost - slm_avg_cost) * slm_inferences if slm_inferences > 0 else 0
    
    # Calculate latency reduction
    llm_avg_latency = await db.scalar(
        select(func.avg(InferenceLog.latency_ms))
        .where(InferenceLog.endpoint_id == endpoint_id)
        .where(InferenceLog.model_used == "llm")
        .where(InferenceLog.created_at >= start_date)
    ) or 0
    
    slm_avg_latency = await db.scalar(
        select(func.avg(InferenceLog.latency_ms))
        .where(InferenceLog.endpoint_id == endpoint_id)
        .where(InferenceLog.model_used == "slm")
        .where(InferenceLog.created_at >= start_date)
    ) or 0
    
    avg_latency_reduction = llm_avg_latency - slm_avg_latency if slm_avg_latency > 0 else 0
    
    return MetricsSummary(
        endpoint_id=endpoint_id,
        avg_similarity=float(avg_similarity),
        total_inferences=total_inferences,
        llm_inferences=llm_inferences,
        slm_inferences=slm_inferences,
        total_cost_saved=float(total_cost_saved),
        avg_latency_reduction_ms=float(avg_latency_reduction),
        llm_avg_cost=float(llm_avg_cost) if llm_avg_cost else None,
        slm_avg_cost=float(slm_avg_cost) if slm_avg_cost else None
    )


# Carbon emissions
@router.get("/carbon/{endpoint_id}/summary", response_model=CarbonSummary)
async def get_carbon_summary(
    endpoint_id: str,
    db: AsyncSession = Depends(get_db)
):
    """Get carbon emissions summary for an endpoint."""
    # Training emissions
    training_emissions = await db.scalar(
        select(func.sum(TrainingRun.carbon_emissions_kg))
        .where(TrainingRun.endpoint_id == endpoint_id)
    ) or 0.0
    
    # Inference emissions (SLM only) - using new CarbonEmission structure
    inference_emissions = await db.scalar(
        select(func.sum(CarbonEmission.emissions_kg))
        .where(CarbonEmission.endpoint_id == endpoint_id)
        .where(CarbonEmission.event_type == "inference")
    ) or 0.0
    
    # Calculate avoided emissions
    # Estimate: LLM inference is ~100x more carbon intensive than SLM
    slm_inference_count = await db.scalar(
        select(func.count(InferenceLog.id))
        .where(InferenceLog.endpoint_id == endpoint_id)
        .where(InferenceLog.model_used == "slm")
    ) or 0
    
    avoided_emissions = slm_inference_count * 0.001  # Rough estimate
    net_emissions = avoided_emissions - (training_emissions + inference_emissions)
    carbon_payback = net_emissions > 0
    
    # Estimate inferences to payback
    inferences_to_payback = None
    if not carbon_payback and slm_inference_count > 0:
        emissions_per_inference = 0.001
        remaining_emissions = (training_emissions + inference_emissions) - avoided_emissions
        inferences_to_payback = int(remaining_emissions / emissions_per_inference)
    
    return CarbonSummary(
        total_training_emissions_kg=float(training_emissions),
        total_inference_emissions_kg=float(inference_emissions),
        avoided_emissions_kg=float(avoided_emissions),
        net_emissions_saved_kg=float(net_emissions),
        carbon_payback_achieved=carbon_payback,
        estimated_inferences_to_payback=inferences_to_payback
    )


@router.get("/carbon/{endpoint_id}/timeline", response_model=CarbonTimeline)
async def get_carbon_timeline(
    endpoint_id: str,
    days: int = 30,
    db: AsyncSession = Depends(get_db)
):
    """Get carbon emissions timeline."""
    start_date = datetime.utcnow() - timedelta(days=days)
    
    # Get emissions data
    result = await db.execute(
        select(CarbonEmission)
        .where(CarbonEmission.timestamp >= start_date)
        .order_by(CarbonEmission.timestamp)
    )
    emissions = result.scalars().all()
    
    # Group by day
    timeline_data = {}
    for emission in emissions:
        day = emission.timestamp.date().isoformat()
        if day not in timeline_data:
            timeline_data[day] = {
                "date": day,
                "training_emissions_kg": 0.0,
                "inference_emissions_kg": 0.0,
                "total_kg": 0.0
            }
        
        if emission.training_run_id:
            timeline_data[day]["training_emissions_kg"] += emission.emissions_kg
        else:
            timeline_data[day]["inference_emissions_kg"] += emission.emissions_kg
        
        timeline_data[day]["total_kg"] = (
            timeline_data[day]["training_emissions_kg"] + 
            timeline_data[day]["inference_emissions_kg"]
        )
    
    timeline = list(timeline_data.values())
    timeline.sort(key=lambda x: x["date"])
    
    return CarbonTimeline(timeline=timeline)


# Endpoint activation
@router.post("/endpoints/{endpoint_id}/activate")
async def activate_slm(
    endpoint_id: str,
    db: AsyncSession = Depends(get_db)
):
    """Manually activate SLM for an endpoint."""
    endpoint = await db.get(Endpoint, endpoint_id)
    if not endpoint:
        raise HTTPException(status_code=404, detail="Endpoint not found")
    
    if not endpoint.slm_model_path:
        raise HTTPException(status_code=400, detail="No trained SLM model available")
    
    # Update config to enable auto-switchover
    config = await db.get(EndpointConfig, endpoint_id)
    if config:
        config.auto_switchover = True
    
    endpoint.status = "active"
    await db.commit()
    
    return {"message": "SLM activated successfully"}


# Examples
@router.get("/endpoints/{endpoint_id}/examples", response_model=ExamplesListResponse)
async def get_examples(
    endpoint_id: str,
    skip: int = 0,
    limit: int = 50,
    filter_trained: Optional[bool] = None,
    search: Optional[str] = None,
    db: AsyncSession = Depends(get_db)
):
    """Get training examples for an endpoint."""
    # Check if endpoint exists
    endpoint = await db.get(Endpoint, endpoint_id)
    if not endpoint:
        raise HTTPException(status_code=404, detail="Endpoint not found")
    
    # Build query
    query = select(InferenceLog).where(InferenceLog.endpoint_id == endpoint_id)
    
    # Apply filters
    if filter_trained is True:
        # Only examples that have been used in training (have SLM output)
        query = query.where(InferenceLog.slm_output.isnot(None))
    elif filter_trained is False:
        # Only examples that haven't been used in training yet
        query = query.where(InferenceLog.slm_output.is_(None))
    
    if search:
        query = query.where(InferenceLog.input_text.ilike(f"%{search}%"))
    
    # Get total count for this query
    count_query = select(func.count(InferenceLog.id)).where(InferenceLog.endpoint_id == endpoint_id)
    if filter_trained is True:
        count_query = count_query.where(InferenceLog.slm_output.isnot(None))
    elif filter_trained is False:
        count_query = count_query.where(InferenceLog.slm_output.is_(None))
    if search:
        count_query = count_query.where(InferenceLog.input_text.ilike(f"%{search}%"))
    
    total_count = await db.scalar(count_query) or 0
    
    # Get examples with pagination
    query = query.order_by(InferenceLog.created_at.desc()).offset(skip).limit(limit)
    result = await db.execute(query)
    examples = result.scalars().all()
    
    # Get trained and pending counts
    trained_count = await db.scalar(
        select(func.count(InferenceLog.id))
        .where(InferenceLog.endpoint_id == endpoint_id)
        .where(InferenceLog.slm_output.isnot(None))
    ) or 0
    
    pending_count = await db.scalar(
        select(func.count(InferenceLog.id))
        .where(InferenceLog.endpoint_id == endpoint_id)
        .where(InferenceLog.slm_output.is_(None))
    ) or 0
    
    return ExamplesListResponse(
        examples=examples,
        total_count=total_count,
        trained_count=trained_count,
        pending_count=pending_count
    )


# Health check
@router.get("/health", response_model=HealthResponse)
async def health_check(db: AsyncSession = Depends(get_db)):
    """Health check endpoint."""
    try:
        # Check database
        await db.execute(select(1))
        db_status = True
    except:
        db_status = False
    
    return HealthResponse(
        status="healthy" if db_status else "unhealthy",
        version=settings.VERSION,
        timestamp=datetime.utcnow(),
        database=db_status,
        carbon_tracking=settings.CARBON_TRACKING_ENABLED
    )


# Evaluation endpoints for SLM batch processing
@router.get("/evaluation/{batch_id}/prompts")
async def get_evaluation_prompts(batch_id: str):
    """Get evaluation prompts for SLM batch processing."""
    try:
        # Initialize Redis client
        redis_url = os.getenv("REDIS_URL", "redis://redis-service:6379")
        redis_client = redis.from_url(redis_url)
        
        # Get evaluation context from Redis
        context_data = await redis_client.get(f"evaluation_context:{batch_id}")
        if not context_data:
            raise HTTPException(status_code=404, detail="Evaluation batch not found")
        
        # Parse evaluation context
        evaluation_context = ast.literal_eval(context_data.decode())
        
        await redis_client.close()
        
        return {
            "prompts": evaluation_context["evaluation_inputs"],
            "train_id": evaluation_context["train_id"]
        }
        
    except Exception as e:
        logger.error(f"Error getting evaluation prompts for {batch_id}: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to get evaluation prompts: {str(e)}")


@router.post("/evaluation/{batch_id}/results")
async def submit_evaluation_results(
    batch_id: str,
    results: dict
):
    """Receive evaluation results from SLM batch processing."""
    try:
        # Extract data from results
        train_id = results.get("train_id")
        slm_results = results.get("results", [])
        
        if not train_id or not slm_results:
            raise HTTPException(status_code=400, detail="Missing required fields: train_id and results")
        
        # Extract just the response text from each result
        slm_outputs = [result["response"] for result in slm_results]
        
        # Initialize lifecycle manager to handle completion
        try:
            from app.core.model_lifecycle import ModelLifecycleManager
            from app.models.database import AsyncSessionLocal
            
            lifecycle_manager = ModelLifecycleManager()
            await lifecycle_manager.initialize()
            
            # Create a database session directly
            async with AsyncSessionLocal() as db:
                result = await lifecycle_manager.handle_slm_batch_completion(
                    db=db,
                    train_id=train_id,
                    slm_outputs=slm_outputs
                )
                
                return {"status": "success", "result": result}
                
        except Exception as e:
            logger.error(f"Error in lifecycle manager: {e}")
            raise HTTPException(status_code=500, detail=f"Failed to process results: {str(e)}")
        
    except Exception as e:
        logger.error(f"Error submitting evaluation results for {batch_id}: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to submit results: {str(e)}")