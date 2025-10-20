"""Azure Management API Endpoints"""

from fastapi import APIRouter, Depends, HTTPException
from typing import Dict, Any
import os
import logging

from app.training.azure_provisioner import azure_provisioner
from app.training.gpu_queue_manager import gpu_queue_manager
from app.core.config import settings

logger = logging.getLogger(__name__)

router = APIRouter(prefix=f"{settings.API_V1_STR}/azure", tags=["azure"])


@router.get("/status")
async def get_azure_status() -> Dict[str, Any]:
    """Check Azure configuration and resource status"""
    
    azure_enabled = os.getenv("AZURE_TRAINING_ENABLED", "false").lower() == "true"
    
    if not azure_enabled:
        return {
            "enabled": False,
            "message": "Azure GPU training is disabled. Set AZURE_TRAINING_ENABLED=true to enable."
        }
    
    # Check configuration
    config_status = {
        "subscription_id": bool(os.getenv("AZURE_SUBSCRIPTION_ID")),
        "storage_account": bool(os.getenv("AZURE_STORAGE_ACCOUNT")),
        "resource_group": os.getenv("AZURE_RESOURCE_GROUP", "understudy-training"),
        "location": os.getenv("AZURE_LOCATION", "eastus"),
        "vm_size": os.getenv("AZURE_GPU_VM_SIZE", "Standard_NC6s_v3"),
        "max_concurrent_vms": int(os.getenv("AZURE_MAX_CONCURRENT_VMS", "1")),
        "use_spot_instances": os.getenv("AZURE_USE_SPOT", "true").lower() == "true"
    }
    
    # Validate resources
    try:
        validation = await azure_provisioner.validate_resources()
    except Exception as e:
        validation = {
            "all_resources_valid": False,
            "error": str(e)
        }
    
    # Get queue stats
    queue_stats = await gpu_queue_manager.get_queue_stats()
    
    return {
        "enabled": True,
        "configuration": config_status,
        "resources": validation,
        "queue": queue_stats
    }


@router.post("/provision")
async def provision_azure_resources() -> Dict[str, Any]:
    """Manually trigger Azure resource provisioning"""
    
    azure_enabled = os.getenv("AZURE_TRAINING_ENABLED", "false").lower() == "true"
    
    if not azure_enabled:
        raise HTTPException(
            status_code=400,
            detail="Azure GPU training is disabled. Set AZURE_TRAINING_ENABLED=true to enable."
        )
    
    if not os.getenv("AZURE_SUBSCRIPTION_ID") or not os.getenv("AZURE_STORAGE_ACCOUNT"):
        raise HTTPException(
            status_code=400,
            detail="Azure configuration incomplete. Required: AZURE_SUBSCRIPTION_ID and AZURE_STORAGE_ACCOUNT"
        )
    
    try:
        result = await azure_provisioner.provision_resources()
        return result
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to provision Azure resources: {str(e)}"
        )


@router.get("/queue")
async def get_training_queue() -> Dict[str, Any]:
    """Get current training queue status"""
    
    azure_enabled = os.getenv("AZURE_TRAINING_ENABLED", "false").lower() == "true"
    
    if not azure_enabled:
        return {
            "enabled": False,
            "message": "Azure GPU training is disabled"
        }
    
    # Get queue statistics
    stats = await gpu_queue_manager.get_queue_stats()
    
    # Get job list
    jobs = await gpu_queue_manager.list_jobs()
    
    return {
        "statistics": stats,
        "jobs": jobs
    }


@router.delete("/queue/{job_id}")
async def cancel_training_job(job_id: str) -> Dict[str, Any]:
    """Cancel a queued training job"""
    
    success = await gpu_queue_manager.cancel_job(job_id)
    
    if not success:
        raise HTTPException(
            status_code=404,
            detail="Job not found or cannot be cancelled"
        )
    
    return {
        "job_id": job_id,
        "status": "cancelled"
    }


@router.post("/cleanup")
async def cleanup_idle_resources() -> Dict[str, Any]:
    """Clean up idle Azure VMs"""
    
    azure_enabled = os.getenv("AZURE_TRAINING_ENABLED", "false").lower() == "true"
    
    if not azure_enabled:
        raise HTTPException(
            status_code=400,
            detail="Azure GPU training is disabled"
        )
    
    cleaned_vms = []
    for vm_name, vm_info in list(gpu_queue_manager.active_vms.items()):
        if vm_info['status'] == 'idle':
            try:
                if gpu_queue_manager.azure_trainer:
                    await gpu_queue_manager.azure_trainer.cleanup_vm(vm_name)
                del gpu_queue_manager.active_vms[vm_name]
                cleaned_vms.append(vm_name)
            except Exception as e:
                logger.error(f"Failed to cleanup VM {vm_name}: {e}")
    
    return {
        "cleaned_vms": cleaned_vms,
        "remaining_vms": list(gpu_queue_manager.active_vms.keys())
    }