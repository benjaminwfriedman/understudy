"""RunPod GPU Training Implementation for Training Service

Handles pod creation, training execution, and resource cleanup on RunPod GPU instances.
Moved from backend to training service for proper separation of concerns.
"""

import os
import json
import logging
import asyncio
import aiohttp
import asyncssh
import tempfile
import time
from typing import Dict, Any, Optional, List
from datetime import datetime
from dataclasses import dataclass
import redis.asyncio as redis

logger = logging.getLogger(__name__)


@dataclass
class RunPodInstance:
    """RunPod instance information"""
    id: str
    name: str
    ip_address: str
    gpu_type: str
    status: str
    ssh_port: int
    pod_type: str  # "pod" or "serverless"


class RunPodTrainer:
    """Manages training execution on RunPod GPU instances"""
    
    def __init__(self):
        self.api_key = os.getenv("RUNPOD_API_KEY")
        self.api_base = "https://api.runpod.io/graphql"
        self.gpu_type = os.getenv("RUNPOD_GPU_TYPE", "NVIDIA GeForce RTX 4090")
        self.container_image = os.getenv("RUNPOD_CONTAINER_IMAGE", "runpod/pytorch:2.0.1-py3.10-cuda11.8.0-devel-ubuntu22.04")
        self.disk_size = int(os.getenv("RUNPOD_DISK_SIZE", "50"))  # GB
        self.storage_bucket = os.getenv("AWS_S3_BUCKET", "understudy-training")
        self.hf_token = os.getenv("HF_TOKEN")
        
        # SSH Configuration
        self.ssh_public_key = os.getenv("RUNPOD_SSH_PUBLIC_KEY", "")
        self.ssh_private_key_path = os.getenv("RUNPOD_SSH_PRIVATE_KEY_PATH", "/app/keys/runpod_ssh_key")
        
        # Training monitoring
        self.active_pods = {}
        self.training_logs = {}
        
        # Redis client for status updates
        redis_url = os.getenv("REDIS_URL", "redis://redis-service:6379")
        self.redis_client = redis.from_url(redis_url)
        
    async def _graphql_request(self, query: str, variables: Optional[Dict] = None) -> Dict:
        """Make a GraphQL request to RunPod API"""
        headers = {
            "Content-Type": "application/json",
        }
        
        payload = {
            "query": query,
            "variables": variables or {}
        }
        
        async with aiohttp.ClientSession() as session:
            async with session.post(
                self.api_base,
                headers=headers,
                json=payload,
                params={"api_key": self.api_key}
            ) as response:
                response_data = await response.json()
                
                if "errors" in response_data:
                    raise Exception(f"RunPod API error: {response_data['errors']}")
                
                return response_data.get("data", {})
    
    async def launch_pod(self, job_id: str, pod_name: str) -> Optional[RunPodInstance]:
        """Launch a new RunPod GPU instance"""
        try:
            # Build the pod configuration
            mutation = """
            mutation CreatePod($input: PodFindAndDeployOnDemandInput) {
                podFindAndDeployOnDemand(input: $input) {
                    id
                    imageName
                    desiredStatus
                    lastStatusChange
                    name
                    runtime {
                        uptimeInSeconds
                        ports {
                            ip
                            isIpPublic
                            privatePort  
                            publicPort
                            type
                        }
                    }
                }
            }
            """
            
            variables = {
                "input": {
                    "cloudType": "ALL",
                    "gpuCount": 1,
                    "volumeInGb": self.disk_size,
                    "containerDiskInGb": self.disk_size,
                    "minMemoryInGb": 24,
                    "gpuTypeId": self.gpu_type,
                    "name": pod_name,
                    "imageName": self.container_image,
                    "dockerArgs": "",
                    "ports": "22/tcp,8888/http",
                    "volumeMountPath": "/workspace",
                    "env": [
                        {"key": "PUBLIC_KEY", "value": self.ssh_public_key},
                        {"key": "JUPYTER_PASSWORD", "value": "understudy"}
                    ],
                }
            }
            
            logger.info(f"Launching RunPod instance: {pod_name}")
            result = await self._graphql_request(mutation, variables)
            pod_data = result.get("podFindAndDeployOnDemand")
            
            if not pod_data:
                raise Exception("Failed to create pod - no pod data returned")
            
            # Wait for the pod to become available
            pod_id = pod_data["id"]
            pod_info = await self._wait_for_pod_ready(pod_id)
            
            if pod_info:
                logger.info(f"Pod {pod_id} launched successfully at {pod_info.ip_address}:{pod_info.ssh_port}")
                return pod_info
            else:
                raise Exception("Pod failed to become ready within timeout")
                
        except Exception as e:
            logger.error(f"Failed to launch RunPod instance: {e}")
            return None
    
    async def _wait_for_pod_ready(self, pod_id: str, timeout: int = 300) -> Optional[RunPodInstance]:
        """Wait for pod to be ready and return connection info"""
        start_time = time.time()
        
        while time.time() - start_time < timeout:
            try:
                query = """
                query GetPod($podId: String!) {
                    pod(input: {podId: $podId}) {
                        id
                        name
                        desiredStatus
                        lastStatusChange
                        runtime {
                            uptimeInSeconds
                            ports {
                                ip
                                isIpPublic
                                privatePort
                                publicPort
                                type
                            }
                        }
                    }
                }
                """
                
                result = await self._graphql_request(query, {"podId": pod_id})
                pod = result.get("pod")
                
                if not pod:
                    logger.warning(f"Pod {pod_id} not found")
                    await asyncio.sleep(5)
                    continue
                
                runtime = pod.get("runtime")
                if not runtime:
                    logger.info(f"Pod {pod_id} not yet running, waiting...")
                    await asyncio.sleep(10)
                    continue
                
                # Find SSH port
                ssh_port = 22
                ip_address = None
                
                for port in runtime.get("ports", []):
                    if port.get("privatePort") == 22 and port.get("isIpPublic"):
                        ip_address = port.get("ip")
                        ssh_port = port.get("publicPort", 22)
                        break
                
                if ip_address:
                    return RunPodInstance(
                        id=pod_id,
                        name=pod["name"],
                        ip_address=ip_address,
                        gpu_type=self.gpu_type,
                        status="running",
                        ssh_port=ssh_port,
                        pod_type="pod"
                    )
                else:
                    logger.info(f"Pod {pod_id} running but no public IP yet, waiting...")
                    await asyncio.sleep(10)
                    
            except Exception as e:
                logger.error(f"Error checking pod status: {e}")
                await asyncio.sleep(10)
        
        logger.error(f"Pod {pod_id} failed to become ready within {timeout} seconds")
        return None
    
    async def execute_training(self, pod_instance: RunPodInstance, training_script: str, job_id: str, training_data: Optional[List[Dict[str, str]]] = None) -> bool:
        """Execute training script on the RunPod instance via async SSH"""
        try:
            logger.info(f"Starting training execution on pod {pod_instance.id}")
            logger.info("Generated training script for RunPod")
            
            # Execute training with async SSH
            success = await self._async_ssh_execute_training(
                pod_instance.ip_address,
                pod_instance.ssh_port,
                training_script,
                job_id,
                training_data
            )
            
            if success:
                logger.info(f"Training completed successfully on pod {pod_instance.id}")
                await self._update_training_status(job_id, "completed")
                return True
            else:
                logger.error(f"Training failed on pod {pod_instance.id}")
                await self._update_training_status(job_id, "failed")
                return False
                
        except Exception as e:
            logger.error(f"Error executing training: {e}")
            await self._update_training_status(job_id, "failed", str(e))
            return False
    
    async def _update_training_status(self, job_id: str, status: str, error: str = None):
        """Update training status in Redis"""
        try:
            update_data = {
                "status": status,
                "updated_at": datetime.now().isoformat()
            }
            if error:
                update_data["error"] = error
            
            await self.redis_client.hset(f"training:{job_id}", mapping=update_data)
            logger.info(f"Updated training {job_id} status to {status}")
        except Exception as e:
            logger.error(f"Failed to update training status: {e}")
    
    async def _async_ssh_execute_training(self, host: str, port: int, script: str, job_id: str, training_data: Optional[List[Dict[str, str]]] = None) -> bool:
        """Execute training script via async SSH with proper timeouts and monitoring"""
        max_retries = 10
        base_delay = 10
        
        for attempt in range(1, max_retries + 1):
            try:
                logger.info(f"Attempting SSH connection to {host}:{port} (attempt {attempt}/{max_retries})")
                
                # Check if SSH private key exists
                if not os.path.exists(self.ssh_private_key_path):
                    logger.error(f"SSH private key not found at {self.ssh_private_key_path}")
                    return False
                
                logger.info(f"Using SSH private key from {self.ssh_private_key_path}")
                
                # Connect with asyncssh with proper timeouts
                async with asyncssh.connect(
                    host,
                    port=port,
                    username='root',
                    client_keys=[self.ssh_private_key_path],
                    known_hosts=None,  # Skip host key verification for dynamic IPs
                    connect_timeout=30,
                    login_timeout=30
                ) as conn:
                    logger.info("SSH connection established, setting up environment")
                    await self._update_training_status(job_id, "installing_packages")
                    
                    # Install packages with timeout and progress monitoring
                    ## TODO set the exact versions
                    install_cmd = """
                    cd /workspace && \
                    pip install --upgrade pip && \
                    pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121 && \
                    pip install transformers peft accelerate datasets codecarbon bitsandbytes
                    """
                    
                    logger.info("Installing required packages...")
                    result = await conn.run(install_cmd, timeout=600)  # 10 minute timeout
                    
                    if result.exit_status != 0:
                        logger.error(f"Package installation failed: {result.stderr}")
                        raise Exception(f"Package installation failed: {result.stderr}")
                    
                    # Log package versions after installation
                    logger.info("Logging package versions used in training...")
                    log_versions_cmd = """
                    echo "=== PACKAGE VERSIONS USED IN TRAINING ===" && \
                    pip show torch transformers peft accelerate datasets codecarbon bitsandbytes | grep -E "(Name|Version)" && \
                    echo "==========================================="
                    """
                    
                    version_result = await conn.run(log_versions_cmd, timeout=60)
                    if version_result.exit_status == 0:
                        logger.info(f"Training environment package versions:\n{version_result.stdout}")
                    else:
                        logger.warning(f"Could not log package versions: {version_result.stderr}")
                        return False
                    
                    logger.info("Packages installed successfully")
                    await self._update_training_status(job_id, "running_training")
                    
                    # Create training data file if provided
                    if training_data:
                        import json
                        training_data_json = json.dumps(training_data, indent=2)
                        logger.info("Creating training data file")
                        await conn.run("cat > /workspace/training_data.json", input=training_data_json)
                        logger.info("Training data file created successfully")
                    
                    # Upload and execute training script
                    script_path = f"/tmp/train_{job_id}.py"
                    logger.info("Uploading training script")
                    
                    # Write script to remote file
                    await conn.run(f"cat > {script_path}", input=script)
                    
                    logger.info("Executing training script")
                    
                    # Run training with heartbeat monitoring
                    training_cmd = f"cd /workspace && python {script_path}"
                    
                    # Start training process
                    async with conn.create_process(training_cmd) as process:
                        # Monitor with periodic heartbeats
                        while process.returncode is None:
                            await asyncio.sleep(30)  # Heartbeat every 30 seconds
                            await self._update_training_status(job_id, "training_in_progress", "Training running...")
                            logger.info(f"Training heartbeat for {job_id}")
                    
                    # Get final result
                    if process.returncode == 0:
                        logger.info("Training completed successfully")
                        return True
                    else:
                        stderr_output = ""
                        if process.stderr:
                            stderr_output = await process.stderr.read()
                        logger.error(f"Training failed with exit code {process.returncode}: {stderr_output}")
                        return False
                        
            except asyncssh.Error as e:
                logger.error(f"SSH connection failed: {e}")
                if attempt < max_retries:
                    delay = base_delay * attempt
                    logger.info(f"Retrying in {delay}s...")
                    await asyncio.sleep(delay)
                else:
                    logger.error("SSH connection failed after all retries")
                    return False
                    
            except Exception as e:
                logger.error(f"SSH execution failed: {e}")
                if attempt < max_retries:
                    delay = base_delay * attempt
                    logger.info(f"Retrying in {delay}s...")
                    await asyncio.sleep(delay)
                else:
                    logger.error("SSH execution failed after all retries")
                    return False
        
        return False
    
    def _ssh_execute_training_sync(self, host: str, port: int, script: str, job_id: str) -> bool:
        """Synchronous SSH execution - will be run in thread"""
        """Execute training script via SSH connection"""
        max_retries = 10
        base_delay = 10
        
        for attempt in range(1, max_retries + 1):
            try:
                logger.info(f"Attempting SSH connection to {host}:{port} (attempt {attempt}/{max_retries})")
                
                # Check if SSH private key exists
                if not os.path.exists(self.ssh_private_key_path):
                    logger.warning(f"SSH private key not found at {self.ssh_private_key_path}, trying without authentication")
                
                ssh = paramiko.SSHClient()
                ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
                
                # Try to connect with key authentication first
                if os.path.exists(self.ssh_private_key_path):
                    logger.info(f"Using SSH private key from {self.ssh_private_key_path}")
                    ssh.connect(
                        hostname=host,
                        port=port,
                        username="root",
                        key_filename=self.ssh_private_key_path,
                        timeout=30,
                        banner_timeout=30
                    )
                else:
                    # Try without authentication (will fail but we want the proper error)
                    ssh.connect(
                        hostname=host,
                        port=port,
                        username="root",
                        timeout=30,
                        banner_timeout=30
                    )
                
                # Execute the training script
                logger.info("SSH connection established, setting up environment")
                
                # Install required packages
                install_cmd = """
                cd /workspace && \
                pip install --upgrade pip && \
                pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121 && \
                pip install transformers peft accelerate datasets codecarbon bitsandbytes
                """
                stdin, stdout, stderr = ssh.exec_command(install_cmd)
                
                # Monitor installation progress
                while not stdout.channel.exit_status_ready():
                    if stdout.channel.recv_ready():
                        output = stdout.read().decode('utf-8')
                        if output.strip():
                            logger.info(f"Install output: {output.strip()}")
                    time.sleep(5)
                
                exit_status = stdout.channel.recv_exit_status()
                if exit_status != 0:
                    # Read error output non-blocking
                    error_output = ""
                    if stderr.channel.recv_ready():
                        error_output = stderr.read().decode()
                    logger.error(f"Package installation failed: {error_output}")
                    return {"success": False, "error": f"Package installation failed: {error_output}"}
                else:
                    logger.info("Packages installed successfully")
                
                logger.info("Executing training script")
                
                # Upload and execute script
                stdin, stdout, stderr = ssh.exec_command(f"cat > /tmp/train_{job_id}.py")
                stdin.write(script)
                stdin.close()
                
                # Make script executable and run
                stdin, stdout, stderr = ssh.exec_command(f"cd /workspace && python /tmp/train_{job_id}.py")
                
                # Monitor the execution
                while not stdout.channel.exit_status_ready():
                    if stdout.channel.recv_ready():
                        output = stdout.read().decode('utf-8')
                        logger.info(f"Training output: {output}")
                    time.sleep(5)
                
                exit_status = stdout.channel.recv_exit_status()
                
                # Read remaining output non-blocking
                final_output = ""
                error_output = ""
                
                if stdout.channel.recv_ready():
                    final_output = stdout.read().decode('utf-8')
                if stderr.channel.recv_ready():
                    error_output = stderr.read().decode('utf-8')
                
                logger.info(f"Training script exit status: {exit_status}")
                if final_output:
                    logger.info(f"Final output: {final_output}")
                if error_output:
                    logger.error(f"Error output: {error_output}")
                
                ssh.close()
                
                return exit_status == 0
                
            except paramiko.AuthenticationException:
                logger.warning("SSH connection failed: Authentication failed.")
                if attempt < max_retries:
                    delay = base_delay * attempt
                    logger.info(f"Retrying in {delay}s...")
                    time.sleep(delay)
                else:
                    logger.error("SSH authentication failed after all retries")
                    return False
                    
            except Exception as e:
                logger.error(f"SSH connection failed: {e}")
                if attempt < max_retries:
                    delay = base_delay * attempt
                    logger.info(f"Retrying in {delay}s...")
                    time.sleep(delay)
                else:
                    logger.error("SSH connection failed after all retries")
                    return False
        
        return False
    
    async def download_model_weights(self, pod_instance: RunPodInstance, job_id: str, model_path: str) -> bool:
        """Download trained model weights from RunPod instance"""
        try:
            logger.info(f"Downloading model weights from pod {pod_instance.id}")
            
            # Use SSH to download the model files
            ssh = paramiko.SSHClient()
            ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
            ssh.connect(
                hostname=pod_instance.ip_address,
                port=pod_instance.ssh_port,
                username="root",
                key_filename=self.ssh_private_key_path,
                timeout=30
            )
            
            # Use SCP to download model files
            import scp
            scp_client = scp.SCPClient(ssh.get_transport())
            
            # Download the entire model directory
            remote_model_path = f"/workspace/output/{job_id}"
            scp_client.get(remote_model_path, model_path, recursive=True)
            
            scp_client.close()
            ssh.close()
            
            logger.info(f"Model weights downloaded successfully to {model_path}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to download model weights: {e}")
            return False
    
    async def download_model_weights(self, pod_instance: RunPodInstance, job_id: str, model_path: str) -> bool:
        """Download trained model weights from RunPod instance using asyncssh"""
        try:
            logger.info(f"Downloading model weights from pod {pod_instance.id}")
            
            # Connect with asyncssh
            async with asyncssh.connect(
                pod_instance.ip_address,
                port=pod_instance.ssh_port,
                username='root',
                client_keys=[self.ssh_private_key_path],
                known_hosts=None,
                connect_timeout=30,
                login_timeout=30
            ) as conn:
                
                # Download the trained model directory
                remote_model_path = "/workspace/final_model"
                
                # Check if model directory exists
                result = await conn.run(f"ls -la {remote_model_path}")
                if result.exit_status != 0:
                    logger.error(f"Model directory not found at {remote_model_path}")
                    return False
                
                # Create local directory
                os.makedirs(model_path, exist_ok=True)
                
                # Download files using asyncssh's SCP-like functionality
                try:
                    await asyncssh.scp((conn, remote_model_path), model_path, recurse=True)
                    logger.info(f"Model weights downloaded successfully to {model_path}")
                    return True
                except Exception as e:
                    logger.error(f"Failed to download model files: {e}")
                    return False
                    
        except Exception as e:
            logger.error(f"Failed to download model weights: {e}")
            return False
    
    async def terminate_pod(self, pod_id: str) -> bool:
        """Terminate a RunPod instance"""
        try:
            mutation = """
            mutation TerminatePod($input: PodTerminateInput!) {
                podTerminate(input: $input)
            }
            """
            
            variables = {
                "input": {
                    "podId": pod_id
                }
            }
            
            result = await self._graphql_request(mutation, variables)
            logger.info(f"Pod {pod_id} termination requested")
            return True
            
        except Exception as e:
            logger.error(f"Failed to terminate pod {pod_id}: {e}")
            return False


# Global trainer instance
runpod_trainer = None

def get_runpod_trainer() -> RunPodTrainer:
    """Get the global RunPod trainer instance"""
    global runpod_trainer
    if runpod_trainer is None:
        runpod_trainer = RunPodTrainer()
    return runpod_trainer