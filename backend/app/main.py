from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager
import logging

from app.core.config import settings
from app.models import init_db
from app.api.endpoints import router

# Configure logging
logging.basicConfig(
    level=getattr(logging, settings.LOG_LEVEL),
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler(settings.LOG_FILE),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Startup and shutdown events."""
    # Startup
    logger.info("Starting Understudy API...")
    await init_db()
    logger.info("Database initialized")
    
    # Initialize cloud GPU training if enabled
    await _initialize_cloud_training()
    
    # Initialize model lifecycle manager
    from app.core.model_lifecycle import get_lifecycle_manager
    lifecycle_manager = get_lifecycle_manager()
    await lifecycle_manager.initialize()
    logger.info("Model lifecycle manager initialized")
    
    yield
    
    # Shutdown
    logger.info("Shutting down Understudy API...")


async def _initialize_cloud_training():
    """Initialize cloud GPU training resources (Azure, Lambda Cloud, or RunPod)"""
    import os
    
    azure_enabled = os.getenv("AZURE_TRAINING_ENABLED", "false").lower() == "true"
    lambda_enabled = os.getenv("LAMBDA_TRAINING_ENABLED", "false").lower() == "true"
    runpod_enabled = os.getenv("RUNPOD_TRAINING_ENABLED", "false").lower() == "true"
    
    if not azure_enabled and not lambda_enabled and not runpod_enabled:
        logger.info("Cloud GPU training is disabled")
        return
    
    # Initialize GPU queue manager (shared by both providers)
    from app.training.gpu_queue_manager import gpu_queue_manager
    logger.info("Initializing GPU training queue...")
    await gpu_queue_manager.initialize()
    
    # Initialize Lambda Cloud if enabled
    if lambda_enabled:
        await _initialize_lambda_training()
    
    # Initialize Azure if enabled
    if azure_enabled:
        await _initialize_azure_training()
    
    logger.info("Cloud GPU training system ready")


async def _initialize_lambda_training():
    """Initialize Lambda Cloud GPU training"""
    import os
    
    api_key = os.getenv("LAMBDA_CLOUD_API_KEY")
    if not api_key:
        logger.warning(
            "Lambda Cloud training enabled but LAMBDA_CLOUD_API_KEY not set. "
            "Lambda Cloud training will be unavailable."
        )
        return
    
    try:
        logger.info("Initializing Lambda Cloud GPU training...")
        
        from app.training.lambda_cloud_provisioner import lambda_cloud_provisioner
        
        # Initialize Lambda Cloud infrastructure
        success = await lambda_cloud_provisioner.initialize()
        
        if success:
            logger.info("Lambda Cloud GPU training initialized successfully")
        else:
            logger.warning("Lambda Cloud initialization incomplete - check configuration")
            
    except Exception as e:
        logger.error(f"Failed to initialize Lambda Cloud training: {e}")
        logger.warning("Lambda Cloud GPU training will be unavailable")


async def _initialize_azure_training():
    """Initialize Azure GPU training resources"""
    import os
    
    # Check required configuration
    subscription_id = os.getenv("AZURE_SUBSCRIPTION_ID")
    storage_account = os.getenv("AZURE_STORAGE_ACCOUNT")
    
    if not subscription_id or not storage_account:
        logger.warning(
            "Azure GPU training enabled but configuration incomplete. "
            "Required: AZURE_SUBSCRIPTION_ID and AZURE_STORAGE_ACCOUNT"
        )
        return
    
    try:
        logger.info("Initializing Azure GPU training...")
        
        # Initialize provisioner and validate/create resources
        from app.training.azure_provisioner import azure_provisioner
        
        logger.info("Checking Azure resource status...")
        validation = await azure_provisioner.validate_resources()
        
        if not validation.get("all_resources_valid"):
            logger.info("Provisioning Azure resources...")
            result = await azure_provisioner.provision_resources()
            logger.info(f"Azure resources provisioned: {result.get('status')}")
        else:
            logger.info("All Azure resources are ready")
        
        logger.info("Azure GPU training system ready")
        
    except Exception as e:
        logger.error(f"Failed to initialize Azure GPU training: {e}")
        logger.warning("Azure GPU training will be unavailable")


# Create FastAPI app
app = FastAPI(
    title=settings.APP_NAME,
    version=settings.VERSION,
    lifespan=lifespan
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.BACKEND_CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include routers
app.include_router(router)

# Include lifecycle management endpoints
from app.api.lifecycle_endpoints import router as lifecycle_router
app.include_router(lifecycle_router)

# Include Lambda Cloud endpoints if enabled
import os
if os.getenv("LAMBDA_TRAINING_ENABLED", "false").lower() == "true":
    from app.api.lambda_endpoints import router as lambda_router
    app.include_router(lambda_router)

# Include Azure endpoints if enabled  
if os.getenv("AZURE_TRAINING_ENABLED", "false").lower() == "true":
    from app.api.azure_endpoints import router as azure_router
    app.include_router(azure_router)

# Include RunPod endpoints if enabled
if os.getenv("RUNPOD_TRAINING_ENABLED", "false").lower() == "true":
    from app.api.runpod_endpoints import router as runpod_router
    app.include_router(runpod_router)


@app.get("/")
async def root():
    """Root endpoint."""
    return {
        "name": settings.APP_NAME,
        "version": settings.VERSION,
        "description": "LLM to SLM Training & Transition Platform with LangChain Integration"
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "app.main:app",
        host=settings.HOST,
        port=settings.PORT,
        reload=settings.RELOAD
    )