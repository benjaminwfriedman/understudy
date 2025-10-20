"""RunPod GPU Training Implementation for Understudy

Handles pod creation, training execution, and resource cleanup on RunPod GPU instances.
Similar to Lambda Cloud implementation but using RunPod's API and infrastructure.
"""

import os
import json
import logging
import asyncio
import aiohttp
import paramiko
import tempfile
from typing import Dict, Any, Optional, List
from datetime import datetime
from dataclasses import dataclass
import time

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
    
    async def get_available_gpu_types(self) -> List[Dict]:
        """Get available GPU types and their pricing"""
        try:
            query = """
            query GpuTypes {
                gpuTypes {
                    id
                    displayName
                    memoryInGb
                    secureCloud
                    communityCloud
                    lowestPrice {
                        minimumBidPrice
                        uninterruptablePrice
                    }
                }
            }
            """
            
            result = await self._graphql_request(query)
            gpu_types = result.get("gpuTypes", [])
            
            available = []
            for gpu in gpu_types:
                if gpu.get("communityCloud", False) or gpu.get("secureCloud", False):
                    price_info = gpu.get("lowestPrice", {})
                    available.append({
                        "id": gpu["id"],
                        "name": gpu["displayName"],
                        "memory_gb": gpu["memoryInGb"],
                        "price_per_hour": price_info.get("uninterruptablePrice", 0) * 3600 if price_info else 0,
                        "available": True
                    })
            
            return available
            
        except Exception as e:
            logger.error(f"Failed to get GPU types: {e}")
            return []
    
    async def get_existing_pods(self) -> List[Dict[str, Any]]:
        """Get list of existing RunPod instances"""
        try:
            query = """
            query GetPods {
                myself {
                    pods {
                        id
                        name
                        desiredStatus
                        imageName
                        gpuCount
                        volumeInGb
                        memoryInGb
                        vcpuCount
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
            }
            """
            
            result = await self._graphql_request(query)
            pods = result.get("myself", {}).get("pods", [])
            logger.info(f"Found {len(pods)} existing RunPod instances")
            return pods
            
        except Exception as e:
            logger.error(f"Error getting existing pods: {e}")
            return []
    
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
                    "supportPublicIp": True,
                    "startSsh": True,
                    "startJupyter": False
                }
            }
            
            logger.info(f"Launching RunPod instance: {pod_name}")
            result = await self._graphql_request(mutation, variables)
            
            pod_data = result.get("podFindAndDeployOnDemand")
            if not pod_data:
                raise Exception("Failed to create pod - no data returned")
            
            pod_id = pod_data["id"]
            
            # Wait for pod to be ready
            pod = await self._wait_for_pod(pod_id)
            
            if pod:
                self.active_pods[job_id] = pod
                logger.info(f"Pod {pod.id} launched successfully at {pod.ip_address}:{pod.ssh_port}")
            
            return pod
            
        except Exception as e:
            logger.error(f"Failed to launch pod: {e}")
            return None
    
    async def _wait_for_pod(self, pod_id: str, timeout: int = 300) -> Optional[RunPodInstance]:
        """Wait for pod to be ready with SSH access"""
        start_time = time.time()
        
        query = """
        query GetPod($podId: String!) {
            pod(input: {podId: $podId}) {
                id
                name
                desiredStatus
                imageName
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
        
        while time.time() - start_time < timeout:
            try:
                result = await self._graphql_request(query, {"podId": pod_id})
                pod = result.get("pod")
                
                if pod and pod.get("runtime") and pod["runtime"].get("ports"):
                    ports = pod["runtime"]["ports"]
                    
                    # Find SSH port
                    ssh_info = None
                    for port in ports:
                        if port["privatePort"] == 22 and port.get("ip") and port.get("publicPort"):
                            ssh_info = port
                            break
                    
                    if ssh_info:
                        return RunPodInstance(
                            id=pod["id"],
                            name=pod["name"],
                            ip_address=ssh_info["ip"],
                            ssh_port=ssh_info["publicPort"],
                            gpu_type=self.gpu_type,
                            status="running",
                            pod_type="pod"
                        )
                
                await asyncio.sleep(5)
                
            except Exception as e:
                logger.error(f"Error checking pod status: {e}")
                await asyncio.sleep(5)
        
        logger.error(f"Pod {pod_id} did not become ready within {timeout} seconds")
        return None
    
    async def execute_training(self, job: Dict[str, Any], pod: RunPodInstance) -> Dict[str, Any]:
        """Execute training on the RunPod instance"""
        try:
            logger.info(f"Starting training execution on pod {pod.id}")
            
            # Generate training script (same as Lambda)
            training_script = self._generate_training_script(job)
            
            # Setup SSH connection
            ssh_client = await self._setup_ssh(pod.ip_address, pod.ssh_port)
            
            # Upload and execute training script
            result = await self._run_training(ssh_client, job["job_id"], training_script, job["training_config"])
            
            # Close SSH connection
            ssh_client.close()
            
            return result
            
        except Exception as e:
            logger.error(f"Training execution failed: {e}")
            raise
    
    async def _setup_ssh(self, ip_address: str, ssh_port: int) -> paramiko.SSHClient:
        """Setup SSH connection to RunPod instance"""
        ssh_client = paramiko.SSHClient()
        ssh_client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
        
        # RunPod uses root user by default
        max_retries = 10
        
        for i in range(max_retries):
            try:
                logger.info(f"Attempting SSH connection to {ip_address}:{ssh_port} (attempt {i+1}/{max_retries})")
                
                # Use SSH key authentication
                if os.path.exists(self.ssh_private_key_path):
                    logger.info(f"Using SSH private key from {self.ssh_private_key_path}")
                    ssh_client.connect(
                        hostname=ip_address,
                        port=ssh_port,
                        username="root",
                        key_filename=self.ssh_private_key_path,
                        timeout=30,
                        allow_agent=False,
                        look_for_keys=False
                    )
                else:
                    # Fallback to no authentication (for testing)
                    logger.warning(f"SSH private key not found at {self.ssh_private_key_path}, trying without authentication")
                    ssh_client.connect(
                        hostname=ip_address,
                        port=ssh_port,
                        username="root",
                        password="",
                        timeout=30,
                        allow_agent=False,
                        look_for_keys=False
                    )
                
                logger.info(f"SSH connection established to {ip_address}:{ssh_port}")
                return ssh_client
                
            except Exception as e:
                if i < max_retries - 1:
                    wait_time = 10 * (i + 1)
                    logger.warning(f"SSH connection failed: {str(e)}")
                    logger.info(f"Retrying in {wait_time}s...")
                    await asyncio.sleep(wait_time)
                else:
                    raise Exception(f"Failed to establish SSH connection after {max_retries} attempts: {e}")
    
    async def _run_training(self, ssh_client: paramiko.SSHClient, job_id: str, 
                          training_script: str, config: Dict) -> Dict[str, Any]:
        """Run training on the RunPod instance"""
        try:
            # Create job directory
            stdin, stdout, stderr = ssh_client.exec_command(f"mkdir -p /workspace/training/{job_id}")
            stdout.read()
            
            # Upload training script
            sftp = ssh_client.open_sftp()
            script_path = f"/workspace/training/{job_id}/train.py"
            with sftp.file(script_path, "w") as f:
                f.write(training_script)
            
            # Upload training data
            data_path = f"/workspace/training/{job_id}/training_data.json"
            with sftp.file(data_path, "w") as f:
                json.dump(config["training_data"], f)
            
            sftp.close()
            
            # Install dependencies  
            logger.info("Installing dependencies on RunPod...")
            install_cmd = """
                cd /workspace && \
                pip install --upgrade pip && \
                pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121 && \
                pip install transformers peft accelerate datasets codecarbon bitsandbytes
            """
            stdin, stdout, stderr = ssh_client.exec_command(install_cmd)
            exit_status = stdout.channel.recv_exit_status()
            if exit_status != 0:
                error_output = stderr.read().decode()
                logger.warning(f"Dependency installation had issues: {error_output}")

            # Execute training
            logger.info("Starting training...")
            train_cmd = f"cd /workspace/training/{job_id} && python train.py 2>&1 | tee training.log"
            stdin, stdout, stderr = ssh_client.exec_command(train_cmd, get_pty=True)
            
            # Monitor training progress
            training_metrics = {
                "epochs": [],
                "train_loss": [],
                "val_loss": [],
                "learning_rate": []
            }
            
            # Read output line by line
            for line in stdout:
                line = line.strip()
                logger.info(f"Training output: {line}")
                
                # Parse training metrics
                if "loss:" in line.lower():
                    if "train" in line.lower():
                        try:
                            loss_value = float(line.split("loss:")[-1].split()[0])
                            training_metrics["train_loss"].append(loss_value)
                        except:
                            pass
                    elif "val" in line.lower() or "eval" in line.lower():
                        try:
                            loss_value = float(line.split("loss:")[-1].split()[0])
                            training_metrics["val_loss"].append(loss_value)
                        except:
                            pass
                
                # Store in training logs
                if job_id not in self.training_logs:
                    self.training_logs[job_id] = []
                self.training_logs[job_id].append({
                    "timestamp": datetime.utcnow().isoformat(),
                    "message": line
                })
            
            exit_status = stdout.channel.recv_exit_status()
            
            if exit_status != 0:
                error_output = stderr.read().decode()
                raise Exception(f"Training failed: {error_output}")
            
            # Download trained model
            logger.info("Downloading trained model...")
            model_path = await self._download_model(ssh_client, job_id)
            
            result = {
                "status": "completed",
                "model_path": model_path,
                "metrics": training_metrics,
                "job_id": job_id,
                "training_time": datetime.utcnow().isoformat()
            }
            
            logger.info(f"Training completed successfully for job {job_id}")
            return result
            
        except Exception as e:
            logger.error(f"Training execution failed: {e}")
            raise
    
    async def _download_model(self, ssh_client: paramiko.SSHClient, job_id: str) -> str:
        """Download trained model from RunPod instance"""
        try:
            local_model_dir = f"/app/models/{job_id}"
            os.makedirs(local_model_dir, exist_ok=True)
            
            sftp = ssh_client.open_sftp()
            
            # Download model files
            remote_model_dir = f"/workspace/training/{job_id}/final_model"
            
            # List files in remote directory
            files_to_download = []
            try:
                files_to_download = sftp.listdir(remote_model_dir)
            except:
                logger.warning(f"Could not list files in {remote_model_dir}")
                return local_model_dir
            
            # Download each file
            for filename in files_to_download:
                if filename not in ['.', '..']:
                    remote_path = f"{remote_model_dir}/{filename}"
                    local_path = f"{local_model_dir}/{filename}"
                    logger.info(f"Downloading {filename}...")
                    sftp.get(remote_path, local_path)
            
            sftp.close()
            
            logger.info(f"Model downloaded to {local_model_dir}")
            return local_model_dir
            
        except Exception as e:
            logger.error(f"Failed to download model: {e}")
            raise
    
    async def terminate_pod(self, pod_id: str) -> bool:
        """Terminate a RunPod instance"""
        try:
            logger.info(f"Terminating pod {pod_id}")
            
            mutation = """
            mutation TerminatePod($podId: String!) {
                podTerminate(input: {podId: $podId})
            }
            """
            
            await self._graphql_request(mutation, {"podId": pod_id})
            
            # Remove from active pods
            for job_id, pod in self.active_pods.items():
                if pod.id == pod_id:
                    del self.active_pods[job_id]
                    break
            
            logger.info(f"Pod {pod_id} terminated successfully")
            return True
            
        except Exception as e:
            logger.error(f"Failed to terminate pod {pod_id}: {e}")
            return False
    
    async def get_training_logs(self, job_id: str) -> List[Dict]:
        """Get training logs for a job"""
        return self.training_logs.get(job_id, [])
    
    async def get_pod_status(self, pod_id: str) -> Optional[Dict]:
        """Get status of a specific pod"""
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
                    }
                }
            }
            """
            
            result = await self._graphql_request(query, {"podId": pod_id})
            return result.get("pod")
            
        except Exception as e:
            logger.error(f"Failed to get pod status: {e}")
            return None
    
    def _generate_training_script(self, job: Dict[str, Any]) -> str:
        """Generate Python training script for the RunPod instance"""
        # Use the same training script as Lambda Cloud
        config = job["training_config"]
        
        script = f'''#!/usr/bin/env python3
# Training script for job: {job["job_id"]}
# Endpoint: {job["endpoint_id"]}
# Generated at: {datetime.utcnow().isoformat()}

import json
import torch
import os
from datetime import datetime

from transformers import (
    AutoTokenizer, 
    AutoModelForCausalLM, 
    TrainingArguments, 
    Trainer,
    DataCollatorForLanguageModeling,
    BitsAndBytesConfig
)
from peft import LoraConfig, get_peft_model, TaskType
from datasets import Dataset
import codecarbon
import accelerate

print("=" * 50)
print(f"Starting training job: {job['job_id']}")
print("Model: meta-llama/Llama-3.2-1B")
print(f"Device: {{torch.cuda.get_device_name() if torch.cuda.is_available() else 'CPU'}}")
print("=" * 50)

try:
    with open("training_data.json", "r") as f:
        raw_training_data = json.load(f)
except Exception as e:
    raise ValueError(f"Failed to load training data: {{e}}")

if isinstance(raw_training_data, str):
    pairs = raw_training_data.strip().split("\\n\\n")
    training_data = []
    for pair in pairs:
        if pair.strip():
            training_data.append({{"text": pair.strip()}})
else:
    training_data = raw_training_data

print(f"Loaded {{len(training_data)}} training examples")

if len(training_data) == 0:
    raise ValueError("No training data found")
if len(training_data) < 5:
    raise ValueError(f"Insufficient training data: {{len(training_data)}} examples (minimum 5 required)")

tracker = codecarbon.EmissionsTracker(
    project_name="understudy-{job['endpoint_id']}",
    measure_power_secs=15
)

tracker.start()

try:
    hf_token = "{self.hf_token}"
    if not hf_token or hf_token == "None":
        raise ValueError("HuggingFace token is required but not provided")
    
    model_name = "meta-llama/Llama-3.2-1B"
    
    print("Loading tokenizer...")
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_name, token=hf_token)
    except Exception as e:
        raise ValueError(f"Failed to load tokenizer: {{e}}")
    
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"
    
    print("Loading base model...")
    try:
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float16,
            device_map="auto",
            trust_remote_code=True,
            token=hf_token
        )
    except Exception as e:
        raise ValueError(f"Failed to load model: {{e}}")

    print(f"Model loaded: {{model}}")
    model.gradient_checkpointing_enable()

    print("Configuring LoRA...")
    lora_config = LoraConfig(
        r={config.get('lora_r', 16)},
        lora_alpha={config.get('lora_alpha', 32)},
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        lora_dropout={config.get('lora_dropout', 0.1)},
        bias="none",
        task_type=TaskType.CAUSAL_LM
    )
    
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()
    
    model.train()
    
    lora_param_count = 0
    for name, param in model.named_parameters():
        if "lora" in name.lower():
            param.requires_grad_(True)
            lora_param_count += param.numel()
            print(f"LoRA parameter: {{name}}, requires_grad: {{param.requires_grad}}")
    
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total trainable parameters: {{trainable_params}}")
    print(f"LoRA parameters: {{lora_param_count}}")
    
    if trainable_params == 0:
        raise ValueError("No trainable parameters found!")
    
    print("Preparing dataset...")
    def tokenize_function(examples):
        return tokenizer(
            examples["text"],
            truncation=True,
            padding="max_length",
            max_length=512,
            return_tensors=None
        )
    
    dataset = Dataset.from_list(training_data)
    tokenized_dataset = dataset.map(
        tokenize_function,
        batched=True,
        remove_columns=dataset.column_names
    )
    
    split_dataset = tokenized_dataset.train_test_split(test_size=0.1, seed=42)
    train_dataset = split_dataset["train"]
    eval_dataset = split_dataset["test"]
    
    print(f"Train samples: {{len(train_dataset)}}, Eval samples: {{len(eval_dataset)}}")
    
    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)
    
    training_args = TrainingArguments(
        output_dir="./results",
        num_train_epochs={config.get('epochs', 3)},
        per_device_train_batch_size={config.get('batch_size', 4)},
        per_device_eval_batch_size=4,
        gradient_accumulation_steps=4,
        warmup_ratio=0.1,
        learning_rate={config.get('learning_rate', 2e-4)},
        fp16=True,
        logging_steps=10,
        eval_strategy="steps",
        eval_steps=50,
        save_strategy="steps",
        save_steps=100,
        save_total_limit=2,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        report_to="none",
        optim="paged_adamw_8bit",
        lr_scheduler_type="cosine",
        gradient_checkpointing=True,
        max_grad_norm=0.3,
        group_by_length=True,
        ddp_find_unused_parameters=False
    )
    
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=data_collator,
        tokenizer=tokenizer
    )
    
    print("Starting training...")
    print("-" * 50)
    try:
        train_result = trainer.train()
    except RuntimeError as e:
        if "does not require grad" in str(e):
            raise ValueError(f"Gradient computation error: {{e}}")
        raise ValueError(f"Training failed: {{e}}")
    except Exception as e:
        raise ValueError(f"Training failed: {{e}}")
    
    print("-" * 50)
    print("Training completed!")
    print(f"Final training loss: {{train_result.training_loss:.4f}}")
    
    print("Saving model...")
    os.makedirs("./final_model", exist_ok=True)
    trainer.save_model("./final_model")
    tokenizer.save_pretrained("./final_model")
    
    metrics = {{
        "train_loss": train_result.training_loss,
        "train_runtime": train_result.metrics["train_runtime"],
        "train_samples_per_second": train_result.metrics["train_samples_per_second"],
        "epoch": train_result.metrics["epoch"]
    }}
    
    with open("./final_model/training_metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)
    
    print("Model saved successfully!")
    
finally:
    emissions = tracker.stop()
    print(f"Carbon emissions: {{emissions:.6f}} kg CO2")
    
    os.makedirs("./final_model", exist_ok=True)
    with open("./final_model/carbon_emissions.json", "w") as f:
        json.dump({{"emissions_kg": emissions, "timestamp": datetime.utcnow().isoformat()}}, f)

print("=" * 50)
print("Training job completed successfully!")
'''
        logger.info("Generated training script for RunPod")
        return script