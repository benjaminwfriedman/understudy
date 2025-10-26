#!/usr/bin/env python3
"""
Model Broker Service

A service that manages model weights storage and distribution.
Acts as the single point of access to the model storage PVC,
streaming model files to SLM inference pods on demand.
"""

import os
import logging
import shutil
from pathlib import Path
from typing import Optional
import aiofiles
from fastapi import FastAPI, HTTPException, UploadFile, File, Response
from fastapi.responses import StreamingResponse
import uvicorn

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="Understudy Model Broker",
    description="Service for managing and distributing model weights",
    version="1.0.0"
)

# Configuration
MODEL_STORAGE_PATH = Path(os.getenv("MODEL_STORAGE_PATH", "/app/model_weights"))
CHUNK_SIZE = 8192 * 1024  # 8MB chunks for streaming


@app.on_event("startup")
async def startup_event():
    """Initialize the model broker service."""
    logger.info("Starting Model Broker Service...")
    
    # Ensure model storage directory exists
    MODEL_STORAGE_PATH.mkdir(parents=True, exist_ok=True)
    logger.info(f"Model storage path: {MODEL_STORAGE_PATH}")
    
    # Log existing models
    existing_models = list_all_models()
    logger.info(f"Found {len(existing_models)} existing models:")
    for model in existing_models:
        logger.info(f"  - {model}")


def get_model_path(endpoint_id: str, version: str) -> Path:
    """Get the file system path for a model."""
    return MODEL_STORAGE_PATH / endpoint_id / f"v{version}" / "model.safetensors"


def get_model_metadata_path(endpoint_id: str, version: str) -> Path:
    """Get the metadata file path for a model."""
    return MODEL_STORAGE_PATH / endpoint_id / f"v{version}" / "metadata.json"


def list_all_models() -> list[str]:
    """List all available models in the storage."""
    models = []
    try:
        for endpoint_dir in MODEL_STORAGE_PATH.iterdir():
            if endpoint_dir.is_dir():
                for version_dir in endpoint_dir.iterdir():
                    if version_dir.is_dir() and version_dir.name.startswith("v"):
                        model_file = version_dir / "model.safetensors"
                        if model_file.exists():
                            models.append(f"{endpoint_dir.name}/{version_dir.name}")
    except Exception as e:
        logger.error(f"Error listing models: {e}")
    return models


async def stream_file(file_path: Path):
    """Stream a file in chunks."""
    try:
        async with aiofiles.open(file_path, 'rb') as file:
            while chunk := await file.read(CHUNK_SIZE):
                yield chunk
    except Exception as e:
        logger.error(f"Error streaming file {file_path}: {e}")
        raise


@app.get("/")
async def root():
    """Health check endpoint."""
    return {
        "service": "Understudy Model Broker",
        "status": "healthy",
        "storage_path": str(MODEL_STORAGE_PATH),
        "available_models": list_all_models()
    }


@app.get("/health")
async def health():
    """Health check endpoint."""
    return {"status": "healthy"}


@app.head("/model-exists/{endpoint_id}/{version}")
async def model_exists(endpoint_id: str, version: str):
    """Check if a model exists without downloading it."""
    model_path = get_model_path(endpoint_id, version)
    
    if not model_path.exists():
        raise HTTPException(status_code=404, detail="Model not found")
    
    # Return model metadata in headers
    file_size = model_path.stat().st_size
    return Response(
        status_code=200,
        headers={
            "X-Model-Size": str(file_size),
            "X-Model-Path": str(model_path),
            "X-Endpoint-ID": endpoint_id,
            "X-Model-Version": version
        }
    )


@app.get("/stream-model/{endpoint_id}/{version}")
async def stream_model(endpoint_id: str, version: str):
    """Stream model weights to requesting pod."""
    model_path = get_model_path(endpoint_id, version)
    
    if not model_path.exists():
        logger.error(f"Model not found: {model_path}")
        raise HTTPException(status_code=404, detail=f"Model {endpoint_id}/v{version} not found")
    
    file_size = model_path.stat().st_size
    logger.info(f"Streaming model {endpoint_id}/v{version} ({file_size} bytes) to client")
    
    return StreamingResponse(
        stream_file(model_path),
        media_type="application/octet-stream",
        headers={
            "Content-Length": str(file_size),
            "Content-Disposition": f"attachment; filename=model-{endpoint_id}-v{version}.safetensors",
            "X-Model-Size": str(file_size),
            "X-Endpoint-ID": endpoint_id,
            "X-Model-Version": version
        }
    )


@app.put("/store-model/{endpoint_id}/{version}")
async def store_model(endpoint_id: str, version: str, file: UploadFile = File(...)):
    """Store a new model (typically called by training service)."""
    model_path = get_model_path(endpoint_id, version)
    
    # Create directory if it doesn't exist
    model_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Store the model file
    try:
        async with aiofiles.open(model_path, 'wb') as f:
            while chunk := await file.read(CHUNK_SIZE):
                await f.write(chunk)
        
        file_size = model_path.stat().st_size
        logger.info(f"Stored model {endpoint_id}/v{version} ({file_size} bytes)")
        
        return {
            "message": "Model stored successfully",
            "endpoint_id": endpoint_id,
            "version": version,
            "file_size": file_size,
            "path": str(model_path)
        }
        
    except Exception as e:
        logger.error(f"Error storing model {endpoint_id}/v{version}: {e}")
        # Clean up partial file
        if model_path.exists():
            model_path.unlink()
        raise HTTPException(status_code=500, detail=f"Failed to store model: {str(e)}")


@app.put("/store-model-directory/{endpoint_id}/{version}")
async def store_model_directory(endpoint_id: str, version: str, files: list[UploadFile] = File(...)):
    """Store a complete model directory with multiple files (called by training service)."""
    model_dir = MODEL_STORAGE_PATH / endpoint_id / f"v{version}"
    
    # Create directory if it doesn't exist
    model_dir.mkdir(parents=True, exist_ok=True)
    
    stored_files = []
    total_size = 0
    
    try:
        # Store each file in the model directory
        for file in files:
            if not file.filename:
                continue
                
            file_path = model_dir / file.filename
            
            async with aiofiles.open(file_path, 'wb') as f:
                while chunk := await file.read(CHUNK_SIZE):
                    await f.write(chunk)
            
            file_size = file_path.stat().st_size
            total_size += file_size
            stored_files.append({
                "filename": file.filename,
                "size": file_size
            })
            
            logger.info(f"Stored file {file.filename} ({file_size} bytes)")
        
        logger.info(f"Stored model directory {endpoint_id}/v{version} with {len(stored_files)} files ({total_size} bytes total)")
        
        return {
            "message": "Model directory stored successfully",
            "endpoint_id": endpoint_id,
            "version": version,
            "files": stored_files,
            "total_size": total_size,
            "path": str(model_dir)
        }
        
    except Exception as e:
        logger.error(f"Error storing model directory {endpoint_id}/v{version}: {e}")
        # Clean up partial files
        if model_dir.exists():
            shutil.rmtree(model_dir, ignore_errors=True)
        raise HTTPException(status_code=500, detail=f"Failed to store model directory: {str(e)}")


@app.delete("/delete-model/{endpoint_id}/{version}")
async def delete_model(endpoint_id: str, version: str):
    """Delete a model and its metadata."""
    model_path = get_model_path(endpoint_id, version)
    metadata_path = get_model_metadata_path(endpoint_id, version)
    
    if not model_path.exists():
        raise HTTPException(status_code=404, detail="Model not found")
    
    try:
        # Delete model file
        model_path.unlink()
        
        # Delete metadata if exists
        if metadata_path.exists():
            metadata_path.unlink()
        
        # Remove directory if empty
        if not any(model_path.parent.iterdir()):
            model_path.parent.rmdir()
        
        # Remove endpoint directory if empty
        endpoint_dir = model_path.parent.parent
        if not any(endpoint_dir.iterdir()):
            endpoint_dir.rmdir()
        
        logger.info(f"Deleted model {endpoint_id}/v{version}")
        
        return {
            "message": "Model deleted successfully",
            "endpoint_id": endpoint_id,
            "version": version
        }
        
    except Exception as e:
        logger.error(f"Error deleting model {endpoint_id}/v{version}: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to delete model: {str(e)}")


@app.get("/list-models")
async def list_models():
    """List all available models."""
    models = []
    
    try:
        for endpoint_dir in MODEL_STORAGE_PATH.iterdir():
            if endpoint_dir.is_dir():
                endpoint_id = endpoint_dir.name
                
                for version_dir in endpoint_dir.iterdir():
                    if version_dir.is_dir() and version_dir.name.startswith("v"):
                        version = version_dir.name[1:]  # Remove 'v' prefix
                        model_file = version_dir / "model.safetensors"
                        
                        if model_file.exists():
                            file_size = model_file.stat().st_size
                            models.append({
                                "endpoint_id": endpoint_id,
                                "version": version,
                                "file_size": file_size,
                                "path": str(model_file)
                            })
    
    except Exception as e:
        logger.error(f"Error listing models: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to list models: {str(e)}")
    
    return {
        "models": models,
        "total_count": len(models)
    }


@app.get("/model-info/{endpoint_id}/{version}")
async def model_info(endpoint_id: str, version: str):
    """Get detailed information about a specific model."""
    model_path = get_model_path(endpoint_id, version)
    
    if not model_path.exists():
        raise HTTPException(status_code=404, detail="Model not found")
    
    try:
        stat = model_path.stat()
        
        return {
            "endpoint_id": endpoint_id,
            "version": version,
            "file_size": stat.st_size,
            "path": str(model_path),
            "modified_time": stat.st_mtime,
            "exists": True
        }
        
    except Exception as e:
        logger.error(f"Error getting model info for {endpoint_id}/v{version}: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to get model info: {str(e)}")


if __name__ == "__main__":
    port = int(os.getenv("PORT", 8003))
    host = os.getenv("HOST", "0.0.0.0")
    
    uvicorn.run(
        "main:app",
        host=host,
        port=port,
        reload=False,
        log_level="info"
    )