"""Lambda Cloud Infrastructure Provisioner

Handles initial setup and management of Lambda Cloud resources for training.
"""

import os
import logging
import asyncio
from typing import Dict, Any, Optional, List
from datetime import datetime
import aiofiles
import json

logger = logging.getLogger(__name__)


class LambdaCloudProvisioner:
    """Manages Lambda Cloud infrastructure setup and validation"""
    
    def __init__(self):
        self.api_key = os.getenv("LAMBDA_CLOUD_API_KEY")
        self.config_file = "/app/config/lambda_cloud_config.json"
        self.initialized = False
        self.config = {}
        
    async def initialize(self) -> bool:
        """Initialize Lambda Cloud infrastructure on backend startup"""
        try:
            logger.info("Initializing Lambda Cloud infrastructure...")
            
            # Check API key
            if not self.api_key:
                logger.warning("LAMBDA_CLOUD_API_KEY not set. Lambda Cloud training disabled.")
                return False
            
            # Load or create configuration
            await self._load_config()
            
            # Import trainer to validate API access
            from app.training.lambda_cloud_trainer import LambdaCloudTrainer
            trainer = LambdaCloudTrainer()
            
            # Ensure SSH key exists
            logger.info("Ensuring SSH key configuration...")
            ssh_key_ready = await trainer.ensure_ssh_key()
            if not ssh_key_ready:
                logger.error("Failed to setup SSH key")
                return False
            
            # Check available instance types
            logger.info("Checking available GPU instance types...")
            available_types = await trainer.get_available_instance_types()
            
            if not available_types:
                logger.warning("No GPU instances available")
                self.config["available_instances"] = []
            else:
                self.config["available_instances"] = available_types
                logger.info(f"Found {len(available_types)} available GPU instance types")
                
                # Log available options
                for instance in available_types[:3]:  # Show top 3
                    logger.info(f"  - {instance['name']}: {instance['description']} "
                              f"(${instance['price_cents_per_hour']/100:.2f}/hr)")
            
            # Select optimal instance type
            self._select_optimal_instance()
            
            # Save configuration
            await self._save_config()
            
            self.initialized = True
            logger.info("Lambda Cloud infrastructure initialized successfully")
            return True
            
        except Exception as e:
            logger.error(f"Failed to initialize Lambda Cloud infrastructure: {e}")
            return False
    
    async def validate_training_readiness(self) -> Dict[str, Any]:
        """Validate that infrastructure is ready for training"""
        status = {
            "ready": False,
            "issues": [],
            "recommendations": []
        }
        
        try:
            # Check initialization
            if not self.initialized:
                status["issues"].append("Lambda Cloud not initialized")
                status["recommendations"].append("Run provisioner.initialize() first")
                return status
            
            # Check API key
            if not self.api_key:
                status["issues"].append("Lambda Cloud API key not configured")
                status["recommendations"].append("Set LAMBDA_CLOUD_API_KEY environment variable")
                return status
            
            # Check instance availability
            if not self.config.get("available_instances"):
                status["issues"].append("No GPU instances available")
                status["recommendations"].append("Check Lambda Cloud capacity or try different regions")
                return status
            
            # Check selected instance type
            if not self.config.get("selected_instance_type"):
                status["issues"].append("No instance type selected")
                status["recommendations"].append("Configure LAMBDA_INSTANCE_TYPE or use auto-selection")
                return status
            
            # Check SSH configuration
            ssh_key_name = os.getenv("LAMBDA_SSH_KEY_NAME", "understudy-key")
            if not ssh_key_name:
                status["issues"].append("SSH key not configured")
                status["recommendations"].append("Set LAMBDA_SSH_KEY_NAME environment variable")
                return status
            
            # All checks passed
            status["ready"] = True
            status["selected_instance"] = self.config.get("selected_instance_type")
            status["estimated_cost_per_hour"] = self.config.get("estimated_cost_per_hour")
            
            logger.info("Lambda Cloud training infrastructure is ready")
            
        except Exception as e:
            status["issues"].append(f"Validation error: {str(e)}")
            logger.error(f"Training readiness validation failed: {e}")
        
        return status
    
    async def estimate_training_cost(self, estimated_hours: float) -> Dict[str, Any]:
        """Estimate the cost of a training job"""
        try:
            if not self.config.get("selected_instance_type"):
                return {
                    "error": "No instance type selected",
                    "estimated_cost": 0
                }
            
            instance_info = None
            for inst in self.config.get("available_instances", []):
                if inst["name"] == self.config["selected_instance_type"]:
                    instance_info = inst
                    break
            
            if not instance_info:
                return {
                    "error": "Selected instance type not found",
                    "estimated_cost": 0
                }
            
            price_per_hour = instance_info["price_cents_per_hour"] / 100
            estimated_cost = price_per_hour * estimated_hours
            
            return {
                "instance_type": instance_info["name"],
                "gpu_type": instance_info["gpu"],
                "price_per_hour": price_per_hour,
                "estimated_hours": estimated_hours,
                "estimated_cost": round(estimated_cost, 2),
                "currency": "USD"
            }
            
        except Exception as e:
            logger.error(f"Failed to estimate training cost: {e}")
            return {
                "error": str(e),
                "estimated_cost": 0
            }
    
    async def get_infrastructure_status(self) -> Dict[str, Any]:
        """Get current infrastructure status"""
        try:
            from app.training.lambda_cloud_trainer import LambdaCloudTrainer
            trainer = LambdaCloudTrainer()
            
            # Get active instances
            active_instances = []
            if hasattr(trainer, 'active_instances'):
                for job_id, instance in trainer.active_instances.items():
                    active_instances.append({
                        "job_id": job_id,
                        "instance_id": instance.id,
                        "instance_type": instance.instance_type,
                        "ip_address": instance.ip_address,
                        "status": instance.status
                    })
            
            return {
                "initialized": self.initialized,
                "api_configured": bool(self.api_key),
                "selected_instance_type": self.config.get("selected_instance_type"),
                "available_instance_types": len(self.config.get("available_instances", [])),
                "active_instances": active_instances,
                "estimated_hourly_cost": self.config.get("estimated_cost_per_hour", 0)
            }
            
        except Exception as e:
            logger.error(f"Failed to get infrastructure status: {e}")
            return {
                "initialized": False,
                "error": str(e)
            }
    
    def _select_optimal_instance(self):
        """Select the optimal instance type based on availability and cost"""
        try:
            # Get configured preference
            preferred_type = os.getenv("LAMBDA_INSTANCE_TYPE")
            
            available = self.config.get("available_instances", [])
            if not available:
                logger.warning("No instances available for selection")
                return
            
            selected = None
            
            # Check if preferred type is available
            if preferred_type:
                for inst in available:
                    if inst["name"] == preferred_type:
                        selected = inst
                        break
                
                if not selected:
                    logger.warning(f"Preferred instance type {preferred_type} not available")
            
            # If no preferred or not available, select based on criteria
            if not selected:
                # Filter for A10 or similar mid-range GPUs
                a10_instances = [i for i in available if "a10" in i["name"].lower()]
                if a10_instances:
                    # Select cheapest A10
                    selected = min(a10_instances, key=lambda x: x["price_cents_per_hour"])
                else:
                    # Select cheapest available
                    selected = min(available, key=lambda x: x["price_cents_per_hour"])
            
            if selected:
                self.config["selected_instance_type"] = selected["name"]
                self.config["estimated_cost_per_hour"] = selected["price_cents_per_hour"] / 100
                logger.info(f"Selected instance type: {selected['name']} "
                          f"({selected['gpu']}) at ${selected['price_cents_per_hour']/100:.2f}/hr")
            
        except Exception as e:
            logger.error(f"Failed to select optimal instance: {e}")
    
    async def _load_config(self):
        """Load configuration from file"""
        try:
            if os.path.exists(self.config_file):
                async with aiofiles.open(self.config_file, 'r') as f:
                    content = await f.read()
                    self.config = json.loads(content)
                    logger.info("Loaded Lambda Cloud configuration from file")
            else:
                self.config = {
                    "created_at": datetime.utcnow().isoformat(),
                    "version": "1.0.0"
                }
                logger.info("Created new Lambda Cloud configuration")
        except Exception as e:
            logger.error(f"Failed to load configuration: {e}")
            self.config = {}
    
    async def _save_config(self):
        """Save configuration to file"""
        try:
            os.makedirs(os.path.dirname(self.config_file), exist_ok=True)
            
            self.config["updated_at"] = datetime.utcnow().isoformat()
            
            async with aiofiles.open(self.config_file, 'w') as f:
                await f.write(json.dumps(self.config, indent=2))
            
            logger.info("Saved Lambda Cloud configuration")
            
        except Exception as e:
            logger.error(f"Failed to save configuration: {e}")
    
    async def cleanup(self):
        """Cleanup resources on shutdown"""
        try:
            logger.info("Cleaning up Lambda Cloud resources...")
            
            # Terminate any orphaned instances
            from app.training.lambda_cloud_trainer import LambdaCloudTrainer
            trainer = LambdaCloudTrainer()
            
            if hasattr(trainer, 'active_instances'):
                for job_id, instance in list(trainer.active_instances.items()):
                    logger.warning(f"Terminating orphaned instance {instance.id} for job {job_id}")
                    await trainer.terminate_instance(instance.id)
            
            logger.info("Lambda Cloud cleanup completed")
            
        except Exception as e:
            logger.error(f"Failed to cleanup Lambda Cloud resources: {e}")


# Global provisioner instance
lambda_cloud_provisioner = LambdaCloudProvisioner()