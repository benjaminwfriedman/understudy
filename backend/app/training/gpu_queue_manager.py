"""GPU Queue Manager for Cloud Training

Manages a queue of training jobs that execute sequentially on cloud GPU instances.
Supports Azure, Lambda Cloud, and RunPod providers.
Accepts training requests asynchronously and processes them in order.
"""

import os
import json
import asyncio
import logging
from typing import Dict, Any, Optional, List
from datetime import datetime, timedelta
from enum import Enum
import redis.asyncio as redis
from dataclasses import dataclass, asdict
import pickle

logger = logging.getLogger(__name__)


class JobStatus(Enum):
    QUEUED = "queued"
    PREPARING = "preparing"
    TRAINING = "training"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


@dataclass
class TrainingJob:
    """Training job data structure"""
    job_id: str
    endpoint_id: str
    status: JobStatus
    priority: int = 0
    created_at: datetime = None
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    training_config: Dict[str, Any] = None
    result: Optional[Dict[str, Any]] = None
    error: Optional[str] = None
    carbon_emissions: Optional[float] = None
    provider: str = "azure"  # 'azure', 'lambda', or 'runpod'
    
    def __post_init__(self):
        if self.created_at is None:
            self.created_at = datetime.utcnow()
    
    def to_dict(self):
        data = asdict(self)
        data['status'] = self.status.value
        data['created_at'] = self.created_at.isoformat() if self.created_at else None
        data['started_at'] = self.started_at.isoformat() if self.started_at else None
        data['completed_at'] = self.completed_at.isoformat() if self.completed_at else None
        return data


class GPUQueueManager:
    """Manages training job queue and GPU resource allocation"""
    
    def __init__(self):
        # Redis configuration for distributed queue
        self.redis_host = os.getenv("REDIS_HOST", "localhost")
        self.redis_port = int(os.getenv("REDIS_PORT", "6379"))
        self.redis_password = os.getenv("REDIS_PASSWORD", "")
        self.redis_db = int(os.getenv("REDIS_DB", "0"))
        
        # Cloud provider selection
        self.default_provider = os.getenv("GPU_TRAINING_PROVIDER", "runpod").lower()
        
        # Azure GPU configuration
        self.azure_max_concurrent_vms = int(os.getenv("AZURE_MAX_CONCURRENT_VMS", "1"))
        self.azure_vm_reuse_enabled = os.getenv("AZURE_VM_REUSE", "true").lower() == "true"
        self.azure_vm_idle_timeout = int(os.getenv("AZURE_VM_IDLE_MINUTES", "15"))
        
        # Lambda Cloud configuration
        self.lambda_max_concurrent_instances = int(os.getenv("LAMBDA_MAX_CONCURRENT_INSTANCES", "1"))
        self.lambda_instance_reuse_enabled = os.getenv("LAMBDA_INSTANCE_REUSE", "false").lower() == "true"
        
        # RunPod configuration
        self.runpod_training_enabled = os.getenv("RUNPOD_TRAINING_ENABLED", "false").lower() == "true"
        self.runpod_max_concurrent_pods = int(os.getenv("RUNPOD_MAX_CONCURRENT_PODS", "1"))
        self.runpod_pod_reuse_enabled = os.getenv("RUNPOD_POD_REUSE", "false").lower() == "true"
        self.runpod_pod_idle_timeout = int(os.getenv("RUNPOD_POD_IDLE_MINUTES", "5"))
        self.runpod_pod_auto_shutdown = os.getenv("RUNPOD_POD_AUTO_SHUTDOWN", "false").lower() == "true"
        self.lambda_instance_idle_timeout = int(os.getenv("LAMBDA_INSTANCE_IDLE_MINUTES", "5"))
        
        self.queue_prefix = "understudy:training:"
        
        # Initialize connections
        self.redis_client: Optional[redis.Redis] = None
        self.azure_trainer = None  # Will be initialized lazily
        self.lambda_trainer = None  # Will be initialized lazily
        self.processing_task = None
        self.cleanup_task = None
        self.active_vms: Dict[str, Dict] = {}  # Track active VMs/instances
        
    async def initialize(self):
        """Initialize queue manager connections"""
        logger.info("Starting GPU queue manager initialization...")
        try:
            # Initialize Redis connection
            redis_kwargs = {
                "host": self.redis_host,
                "port": self.redis_port,
                "db": self.redis_db,
                "decode_responses": False  # We'll handle encoding/decoding
            }
            # Only add password if it's not empty
            if self.redis_password:
                redis_kwargs["password"] = self.redis_password
            
            self.redis_client = redis.Redis(**redis_kwargs)
            
            await self.redis_client.ping()
            logger.info("Redis connection established for training queue")
            
            # Start queue processor
            if not self.processing_task:
                self.processing_task = asyncio.create_task(self._process_queue())
                logger.info("GPU training queue processor started")
                
        except Exception as e:
            logger.error(f"Failed to initialize queue manager: {e}")
            # Fall back to in-memory queue if Redis not available
            self.redis_client = None
            self.local_queue = asyncio.Queue()
            logger.warning("Using in-memory queue as fallback")
            
            # Start queue processor for local queue
            if not self.processing_task:
                self.processing_task = asyncio.create_task(self._process_queue())
                logger.info("Local GPU training queue processor started")
        
        # Start cleanup task for auto-shutdown
        if self.runpod_pod_auto_shutdown:
            if not hasattr(self, 'cleanup_task') or not self.cleanup_task:
                self.cleanup_task = asyncio.create_task(self._cleanup_idle_pods())
                logger.info("Pod auto-shutdown cleanup task started")
        
        # Check for existing Lambda instances if reuse is enabled
        logger.info("About to check Lambda instance reuse configuration...")
        logger.info(f"Lambda instance reuse enabled: {self.lambda_instance_reuse_enabled}")
        if self.lambda_instance_reuse_enabled:
            logger.info("Checking for existing Lambda instances to reuse...")
            await self._check_existing_lambda_instances()
            
        # Check for existing RunPod pods if reuse is enabled
        logger.info(f"RunPod pod reuse enabled: {self.runpod_pod_reuse_enabled}")
        if self.runpod_pod_reuse_enabled:
            logger.info("Checking for existing RunPod pods to reuse...")
            await self._check_existing_runpod_pods()
    
    async def add_job(self, endpoint_id: str, training_config: Dict[str, Any], priority: int = 0, provider: str = None) -> str:
        """Add a training job to the queue"""
        job_id = f"{endpoint_id}_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}"
        
        # Use specified provider or default
        if provider is None:
            provider = self.default_provider
        
        job = TrainingJob(
            job_id=job_id,
            endpoint_id=endpoint_id,
            status=JobStatus.QUEUED,
            priority=priority,
            training_config=training_config,
            provider=provider
        )
        
        if self.redis_client:
            # Add to Redis sorted set (priority queue)
            score = -priority * 1000000 + datetime.utcnow().timestamp()  # Higher priority = lower score
            await self.redis_client.zadd(
                f"{self.queue_prefix}queue",
                {pickle.dumps(job): score}
            )
            
            # Store job details
            await self.redis_client.hset(
                f"{self.queue_prefix}jobs",
                job_id,
                json.dumps(job.to_dict())
            )
        else:
            # Fallback to local queue
            await self.local_queue.put((priority, job))
        
        logger.info(f"Training job {job_id} added to queue with priority {priority}")
        return job_id
    
    async def get_job_status(self, job_id: str) -> Optional[Dict[str, Any]]:
        """Get the status of a training job"""
        if self.redis_client:
            job_data = await self.redis_client.hget(f"{self.queue_prefix}jobs", job_id)
            if job_data:
                return json.loads(job_data)
        return None
    
    async def list_jobs(self, status: Optional[JobStatus] = None) -> List[Dict[str, Any]]:
        """List all jobs, optionally filtered by status"""
        jobs = []
        
        if self.redis_client:
            all_jobs = await self.redis_client.hgetall(f"{self.queue_prefix}jobs")
            for job_data in all_jobs.values():
                job = json.loads(job_data)
                if status is None or job['status'] == status.value:
                    jobs.append(job)
        
        return sorted(jobs, key=lambda x: x.get('created_at', ''), reverse=True)
    
    async def cancel_job(self, job_id: str) -> bool:
        """Cancel a queued job"""
        job = await self.get_job_status(job_id)
        if not job:
            return False
        
        if job['status'] in [JobStatus.QUEUED.value, JobStatus.PREPARING.value]:
            job['status'] = JobStatus.CANCELLED.value
            job['completed_at'] = datetime.utcnow().isoformat()
            
            if self.redis_client:
                await self.redis_client.hset(
                    f"{self.queue_prefix}jobs",
                    job_id,
                    json.dumps(job)
                )
            return True
        
        return False
    
    async def _process_queue(self):
        """Main queue processing loop"""
        logger.info("Starting training queue processor")
        
        while True:
            try:
                # Get next job from queue
                job = await self._get_next_job()
                if not job:
                    await asyncio.sleep(10)  # Wait before checking again
                    continue
                
                # Check if we can start a new VM or reuse existing
                vm = await self._get_or_create_vm(job.provider)
                if not vm:
                    # Put job back in queue if no VM available
                    await self._requeue_job(job)
                    await asyncio.sleep(30)
                    continue
                
                # Process the job
                await self._process_job(job, vm)
                
            except Exception as e:
                logger.error(f"Error in queue processor: {e}")
                await asyncio.sleep(30)
    
    async def _get_next_job(self) -> Optional[TrainingJob]:
        """Get next job from priority queue"""
        if self.redis_client:
            # Get highest priority job
            result = await self.redis_client.zpopmin(f"{self.queue_prefix}queue", count=1)
            if result:
                job_data, score = result[0]
                return pickle.loads(job_data)
        else:
            # Fallback to local queue
            if not self.local_queue.empty():
                priority, job = await self.local_queue.get()
                return job
        
        return None
    
    async def _requeue_job(self, job: TrainingJob):
        """Put job back in queue"""
        if self.redis_client:
            score = -job.priority * 1000000 + datetime.utcnow().timestamp()
            await self.redis_client.zadd(
                f"{self.queue_prefix}queue",
                {pickle.dumps(job): score}
            )
    
    async def _get_or_create_vm(self, provider: str = None) -> Optional[Dict[str, Any]]:
        """Training service handles all VM management now"""
        # All training is now handled by the training service
        # No VM management needed in the backend
        return {"provider": "training_service", "status": "available"}
    
    async def _get_or_create_azure_vm(self) -> Optional[Dict[str, Any]]:
        """Get an available Azure VM or create a new one"""
        # Check for idle VMs that can be reused
        if self.azure_vm_reuse_enabled:
            for vm_id, vm_info in self.active_vms.items():
                if vm_info.get('provider') == 'azure' and vm_info['status'] == 'idle':
                    idle_time = datetime.utcnow() - vm_info['last_used']
                    if idle_time.total_seconds() < self.azure_vm_idle_timeout * 60:
                        vm_info['status'] = 'busy'
                        logger.info(f"Reusing Azure VM {vm_id}")
                        return vm_info
        
        # Check if we can create a new VM
        active_count = sum(1 for vm in self.active_vms.values() 
                          if vm.get('provider') == 'azure' and vm['status'] == 'busy')
        if active_count < self.azure_max_concurrent_vms:
            # Initialize Azure trainer if needed
            if not self.azure_trainer:
                from .azure_gpu_trainer import AzureGPUTrainer
                self.azure_trainer = AzureGPUTrainer()
            
            # Create new VM
            vm_name = f"azure-gpu-{datetime.utcnow().strftime('%Y%m%d%H%M%S')}"
            try:
                vm_details = await self.azure_trainer.create_training_vm(vm_name)
                vm_info = {
                    'vm_id': vm_details['vm_id'],
                    'vm_name': vm_name,
                    'details': vm_details,
                    'status': 'busy',
                    'provider': 'azure',
                    'created_at': datetime.utcnow(),
                    'last_used': datetime.utcnow()
                }
                self.active_vms[vm_name] = vm_info
                logger.info(f"Created new Azure VM {vm_name}")
                return vm_info
            except Exception as e:
                logger.error(f"Failed to create Azure VM: {e}")
                return None
        
        return None
    
    async def _check_existing_lambda_instances(self):
        """Check for existing Lambda Cloud instances on startup"""
        try:
            if not self.lambda_trainer:
                from .lambda_cloud_trainer import LambdaCloudTrainer
                self.lambda_trainer = LambdaCloudTrainer()
            
            # Ensure SSH key is set up for reusing instances
            logger.info("Setting up SSH key for Lambda instance reuse...")
            await self.lambda_trainer.ensure_ssh_key()
            
            instances = await self.lambda_trainer.get_existing_instances()
            logger.info(f"Checking {len(instances)} existing Lambda instances for reuse eligibility")
            for instance in instances:
                logger.info(f"Instance: name='{instance.get('name', 'N/A')}', status='{instance.get('status', 'N/A')}', id='{instance.get('id', 'N/A')}'")
                logger.info(f"Instance SSH keys: {instance.get('ssh_key_names', [])}")
                # Add active instances to active_vms (Lambda uses 'active' status, not 'running')
                if instance.get("status") == "active" and instance.get("name", "").startswith("lambda-gpu-"):
                    instance_id = instance["id"]
                    self.active_vms[instance_id] = {
                        "id": instance_id,
                        "name": instance["name"],
                        "ip": instance.get("ip_address"),
                        "provider": "lambda",
                        "status": "idle",
                        "last_used": datetime.utcnow(),
                        "instance": instance
                    }
                    logger.info(f"Found existing Lambda instance: {instance['name']} ({instance_id})")
            
            if self.active_vms:
                logger.info(f"Discovered {len(self.active_vms)} existing Lambda instances available for reuse")
                
        except Exception as e:
            logger.error(f"Failed to check existing Lambda instances: {e}")
    
    async def _check_existing_runpod_pods(self):
        """Check for existing RunPod pods on startup - now delegated to training service"""
        try:
            # Training service now manages RunPod pods directly
            # We no longer need to check existing pods from the backend
            logger.info("RunPod pod management is now handled by Training Service")
                
        except Exception as e:
            logger.error(f"Failed to check existing RunPod pods: {e}")
    
    async def _refresh_lambda_instances(self):
        """Refresh the list of available Lambda instances by checking current status"""
        try:
            if not self.lambda_trainer:
                from .lambda_cloud_trainer import LambdaCloudTrainer
                self.lambda_trainer = LambdaCloudTrainer()
            
            current_instances = await self.lambda_trainer.get_existing_instances()
            current_instance_ids = {instance.get('id') for instance in current_instances}
            
            # Remove instances that no longer exist
            lambda_vms_to_remove = []
            lambda_vms_to_add = []
            for instance_id, instance_info in self.active_vms.items():
                if instance_info.get('provider') == 'lambda':
                    if instance_id not in current_instance_ids:
                        logger.info(f"Removing terminated Lambda instance {instance_id}")
                        lambda_vms_to_remove.append(instance_id)

            # Add instances that currently exist that are not in self.active_vms.items()
            
            
            for instance_id in lambda_vms_to_remove:
                del self.active_vms[instance_id]
                
        except Exception as e:
            logger.error(f"Failed to refresh Lambda instances: {e}")
    
    async def _refresh_runpod_pods(self):
        """Refresh the list of available RunPod pods by checking current status"""
        try:
            if not hasattr(self, 'runpod_trainer') or not self.runpod_trainer:
                from .runpod_trainer import RunPodTrainer
                self.runpod_trainer = RunPodTrainer()
            
            current_pods = await self.runpod_trainer.get_existing_pods()
            current_pod_ids = {pod.get('id') for pod in current_pods}
            
            # Remove pods that no longer exist
            runpod_vms_to_remove = []
            for pod_id, pod_info in list(self.active_vms.items()):
                if pod_info.get('provider') == 'runpod':
                    if pod_id not in current_pod_ids:
                        logger.info(f"Removing terminated RunPod pod {pod_id}")
                        runpod_vms_to_remove.append(pod_id)
            
            for pod_id in runpod_vms_to_remove:
                del self.active_vms[pod_id]
                
        except Exception as e:
            logger.error(f"Failed to refresh RunPod pods: {e}")
    
    async def _cleanup_idle_pods(self):
        """Background task to automatically shutdown idle RunPod pods after timeout"""
        while True:
            try:
                await asyncio.sleep(60)  # Check every minute
                
                if not self.runpod_pod_auto_shutdown:
                    continue
                
                current_time = datetime.utcnow()
                pods_to_shutdown = []
                
                for pod_id, pod_info in list(self.active_vms.items()):
                    if pod_info.get('provider') != 'runpod':
                        continue
                    
                    # Skip if pod is currently busy (training active)
                    if pod_info.get('status') == 'busy':
                        continue
                    
                    # Check if pod has been idle too long
                    last_used = pod_info.get('last_used', current_time)
                    idle_time = current_time - last_used
                    
                    if idle_time.total_seconds() > (self.runpod_pod_idle_timeout * 60):
                        pods_to_shutdown.append((pod_id, pod_info))
                        logger.info(f"RunPod pod {pod_id} has been idle for {idle_time.total_seconds()/60:.1f} minutes, scheduling for shutdown")
                
                # Shutdown idle pods
                for pod_id, pod_info in pods_to_shutdown:
                    await self._shutdown_runpod_pod(pod_id, pod_info)
                    
            except Exception as e:
                logger.error(f"Error in pod cleanup task: {e}")
                await asyncio.sleep(60)  # Wait before retrying
    
    async def _shutdown_runpod_pod(self, pod_id: str, pod_info: Dict[str, Any]):
        """Shutdown a specific RunPod pod"""
        try:
            if not hasattr(self, 'runpod_trainer') or not self.runpod_trainer:
                from .runpod_trainer import RunPodTrainer
                self.runpod_trainer = RunPodTrainer()
            
            pod_name = pod_info.get('vm_name', 'unknown')
            logger.info(f"Shutting down idle RunPod pod: {pod_name} ({pod_id})")
            
            # Terminate the pod
            success = await self.runpod_trainer.terminate_pod(pod_id)
            
            if success:
                # Remove from active VMs
                if pod_id in self.active_vms:
                    del self.active_vms[pod_id]
                logger.info(f"Successfully shut down RunPod pod: {pod_name} ({pod_id})")
            else:
                logger.warning(f"Failed to shutdown RunPod pod: {pod_name} ({pod_id})")
                
        except Exception as e:
            logger.error(f"Error shutting down RunPod pod {pod_id}: {e}")
    
    async def _get_or_create_lambda_instance(self) -> Optional[Dict[str, Any]]:
        """Get an available Lambda Cloud instance or create a new one"""
        # Lambda Cloud instances are not typically reused due to cost
        # but we support it if configured
        if self.lambda_instance_reuse_enabled:
            # Refresh instance list to check current status
            await self._refresh_lambda_instances()
            logger.info(f"Existing instances: {self.active_vms.items()}")
            for instance_id, instance_info in self.active_vms.items():
                if instance_info.get('provider') == 'lambda' and instance_info['status'] == 'idle':
                    idle_time = datetime.utcnow() - instance_info['last_used']
                    if idle_time.total_seconds() < self.lambda_instance_idle_timeout * 60:
                        instance_info['status'] = 'busy'
                        logger.info(f"Reusing Lambda instance {instance_id}")
                        return instance_info
                    else:
                        logger.info(f"Not using instance: {instance_id} because its been idle too long")
        
        # Check if we can create a new instance
        active_count = sum(1 for vm in self.active_vms.values() 
                          if vm.get('provider') == 'lambda' and vm['status'] == 'busy')
        if active_count < self.lambda_max_concurrent_instances:
            # Initialize Lambda trainer if needed
            if not self.lambda_trainer:
                from .lambda_cloud_trainer import LambdaCloudTrainer
                self.lambda_trainer = LambdaCloudTrainer()
            
            # Create new instance
            instance_name = f"lambda-gpu-{datetime.utcnow().strftime('%Y%m%d%H%M%S')}"
            job_id = f"job-{datetime.utcnow().strftime('%Y%m%d%H%M%S')}"
            
            try:
                instance = await self.lambda_trainer.launch_instance(job_id, instance_name)
                if instance:
                    instance_info = {
                        'vm_id': instance.id,
                        'vm_name': instance_name,
                        'details': {
                            'id': instance.id,
                            'ip_address': instance.ip_address,
                            'instance_type': instance.instance_type,
                            'region': instance.region
                        },
                        'instance': instance,
                        'status': 'busy',
                        'provider': 'lambda',
                        'created_at': datetime.utcnow(),
                        'last_used': datetime.utcnow()
                    }
                    self.active_vms[instance_name] = instance_info
                    logger.info(f"Created new Lambda instance {instance_name}")
                    return instance_info
            except Exception as e:
                logger.error(f"Failed to create Lambda instance: {e}")
                return None
        
        return None
    
    async def _get_or_create_runpod_pod(self) -> Optional[Dict[str, Any]]:
        """Create a new RunPod pod for each training run"""
        if not self.runpod_training_enabled:
            logger.error("RunPod training is not enabled")
            return None
        
        # Always create a new pod for each training run (no reuse)
        logger.info("Creating new RunPod pod for training run")
        
        # Check if we can create a new pod
        active_count = sum(1 for vm in self.active_vms.values() 
                          if vm.get('provider') == 'runpod' and vm['status'] == 'busy')
        if active_count < self.runpod_max_concurrent_pods:
            # Initialize RunPod trainer if needed
            if not hasattr(self, 'runpod_trainer') or not self.runpod_trainer:
                from .runpod_trainer import RunPodTrainer
                self.runpod_trainer = RunPodTrainer()
            
            # Create new pod
            pod_name = f"runpod-gpu-{datetime.utcnow().strftime('%Y%m%d%H%M%S')}"
            job_id = f"job-{datetime.utcnow().strftime('%Y%m%d%H%M%S')}"
            
            try:
                pod = await self.runpod_trainer.launch_pod(job_id, pod_name)
                if pod:
                    pod_info = {
                        'vm_id': pod.id,
                        'vm_name': pod_name,
                        'status': 'busy',
                        'created_at': datetime.utcnow(),
                        'last_used': datetime.utcnow(),
                        'provider': 'runpod',
                        'pod': pod,
                        'auto_shutdown': True  # Mark for automatic shutdown after training
                    }
                    self.active_vms[pod.id] = pod_info
                    logger.info(f"Created new RunPod pod {pod_name} for single-use training")
                    return pod_info
            except Exception as e:
                logger.error(f"Failed to create RunPod pod: {e}")
                return None
        
        return None
    
    async def _process_job(self, job: TrainingJob, vm: Dict[str, Any]):
        """Process a training job on the given VM/instance"""
        try:
            # Update job status
            job.status = JobStatus.TRAINING
            job.started_at = datetime.utcnow()
            await self._update_job(job)
            
            # All training is now handled by the training service
            logger.info(f"Starting training for job {job.job_id} via Training Service")
            
            # Use Training Service for all training
            from app.core.training_client import get_training_client
            training_client = get_training_client()
            
            # Extract training parameters from job
            training_config = job.training_config or {}
            
            # Fetch training data from InferenceLog
            training_data = None
            try:
                from app.models.models import InferenceLog
                from app.models.database import AsyncSessionLocal
                from sqlalchemy import select
                
                async with AsyncSessionLocal() as session:
                    # Get training examples from inference logs
                    query = select(InferenceLog).where(
                        InferenceLog.endpoint_id == job.endpoint_id,
                        InferenceLog.model_used == "llm",
                        InferenceLog.llm_output.isnot(None)
                    ).order_by(InferenceLog.created_at.desc())
                    
                    # Limit to requested number of training pairs
                    num_examples = training_config.get('training_pairs_count', 100)
                    query = query.limit(num_examples)
                    
                    result = await session.execute(query)
                    logs = result.scalars().all()
                    
                    if logs:
                        # Format training data with "text" field for training script compatibility
                        training_data = []
                        for log in logs:
                            # Format with special tokens for instruction tuning (matching trainer.py format)
                            formatted_text = f"<|user|>\n{log.input_text}\n<|assistant|>\n{log.llm_output}\n<|end|>"
                            training_data.append({"text": formatted_text})
                        logger.info(f"Fetched {len(training_data)} training examples for job {job.job_id}")
                    else:
                        logger.warning(f"No training data found for endpoint {job.endpoint_id}")
                        
            except Exception as e:
                logger.error(f"Failed to fetch training data: {e}")
                # Continue without training data - let training service handle it
            
            result = await training_client.start_training(
                train_id=job.job_id,
                endpoint_id=job.endpoint_id,
                version=training_config.get('version', 1),
                training_pairs_count=training_config.get('training_pairs_count', 100),
                slm_type=training_config.get('slm_type', 'microsoft/DialoGPT-small'),
                source_llm=training_config.get('source_llm', 'gpt-3.5-turbo'),
                provider=job.provider,
                training_data=training_data
            )
            
            # Update job with results from training service
            if result.get('success'):
                job.status = JobStatus.TRAINING  # Will be updated by training service when complete
                job.result = {'runpod_job_id': result.get('runpod_job_id'), 'message': result.get('message')}
            else:
                job.status = JobStatus.FAILED
                job.completed_at = datetime.utcnow()
                job.error = result.get('error', 'Training service failed to start job')
            
            logger.info(f"Training completed for job {job.job_id}")
            
        except Exception as e:
            logger.error(f"Training failed for job {job.job_id}: {e}")
            job.status = JobStatus.FAILED
            job.completed_at = datetime.utcnow()
            job.error = str(e)
        
        finally:
            # Update job in storage
            await self._update_job(job)
            
            # Handle RunPod auto-shutdown for single-use pods
            if vm.get('provider') == 'runpod' and vm.get('auto_shutdown'):
                vm_name = vm.get('vm_name', 'unknown')
                pod_id = vm.get('vm_id')
                logger.info(f"Auto-shutting down RunPod pod {vm_name} ({pod_id}) after training completion")
                await self._shutdown_runpod_pod(pod_id, vm)
            else:
                # Mark VM as idle for potential reuse (for non-RunPod or non-auto-shutdown VMs)
                vm['status'] = 'idle'
                vm['last_used'] = datetime.utcnow()
                logger.info(f"Marked {vm.get('provider', 'unknown')} instance {vm.get('vm_name', vm.get('name', 'unknown'))} as idle for potential reuse")
                
                # Schedule VM cleanup if idle timeout exceeded
                vm_name = vm.get('vm_name') or vm.get('name', 'unknown')
                asyncio.create_task(self._cleanup_idle_vm(vm_name))
    
    async def _update_job(self, job: TrainingJob):
        """Update job in storage"""
        if self.redis_client:
            await self.redis_client.hset(
                f"{self.queue_prefix}jobs",
                job.job_id,
                json.dumps(job.to_dict())
            )
    
    async def _cleanup_idle_vm(self, vm_name: str):
        """Clean up VM/instance after idle timeout"""
        if vm_name not in self.active_vms:
            return
        
        vm_info = self.active_vms[vm_name]
        provider = vm_info.get('provider', 'azure')
        
        # Wait for idle timeout based on provider
        if provider == 'lambda':
            await asyncio.sleep(self.lambda_instance_idle_timeout * 60)
        elif provider == 'runpod':
            await asyncio.sleep(self.runpod_pod_idle_timeout * 60)
        else:
            await asyncio.sleep(self.azure_vm_idle_timeout * 60)
        
        if vm_name in self.active_vms:
            vm_info = self.active_vms[vm_name]
            if vm_info['status'] == 'idle':
                idle_time = datetime.utcnow() - vm_info['last_used']
                
                # Check timeout based on provider
                if provider == 'lambda':
                    timeout_minutes = self.lambda_instance_idle_timeout
                elif provider == 'runpod':
                    timeout_minutes = self.runpod_pod_idle_timeout
                else:
                    timeout_minutes = self.azure_vm_idle_timeout
                
                if idle_time.total_seconds() >= timeout_minutes * 60:
                    logger.info(f"Cleaning up idle {provider} instance {vm_name}")
                    
                    if provider == 'lambda':
                        if self.lambda_trainer:
                            await self.lambda_trainer.terminate_instance(vm_info['vm_id'])
                    elif provider == 'runpod':
                        if hasattr(self, 'runpod_trainer') and self.runpod_trainer:
                            await self.runpod_trainer.terminate_pod(vm_info['vm_id'])
                    else:
                        if self.azure_trainer:
                            await self.azure_trainer.cleanup_vm(vm_name)
                    
                    del self.active_vms[vm_name]
    
    async def get_queue_stats(self) -> Dict[str, Any]:
        """Get queue statistics"""
        stats = {
            'queued_jobs': 0,
            'running_jobs': 0,
            'completed_jobs': 0,
            'failed_jobs': 0,
            'active_vms': len([vm for vm in self.active_vms.values() if vm['status'] == 'busy']),
            'idle_vms': len([vm for vm in self.active_vms.values() if vm['status'] == 'idle']),
            'max_concurrent_vms': self.max_concurrent_vms
        }
        
        jobs = await self.list_jobs()
        for job in jobs:
            status = job['status']
            if status == JobStatus.QUEUED.value:
                stats['queued_jobs'] += 1
            elif status in [JobStatus.PREPARING.value, JobStatus.TRAINING.value]:
                stats['running_jobs'] += 1
            elif status == JobStatus.COMPLETED.value:
                stats['completed_jobs'] += 1
            elif status == JobStatus.FAILED.value:
                stats['failed_jobs'] += 1
        
        return stats


# Global queue manager instance
gpu_queue_manager = GPUQueueManager()