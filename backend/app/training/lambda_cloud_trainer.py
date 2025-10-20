"""Lambda Cloud GPU Training Implementation for Understudy

Handles instance creation, training execution, and resource cleanup on Lambda Cloud GPU instances.
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
class LambdaInstance:
    """Lambda Cloud instance information"""
    id: str
    name: str
    ip_address: str
    instance_type: str
    region: str
    status: str
    ssh_key_names: List[str]


class LambdaCloudTrainer:
    """Manages training execution on Lambda Cloud GPU instances"""
    
    def __init__(self):
        self.api_key = os.getenv("LAMBDA_CLOUD_API_KEY")
        self.api_base = "https://cloud.lambdalabs.com/api/v1"
        self.ssh_key_name = os.getenv("LAMBDA_SSH_KEY_NAME", "understudy-key-v2")
        self.ssh_private_key_path = os.getenv("LAMBDA_SSH_PRIVATE_KEY_PATH")
        self.instance_type = os.getenv("LAMBDA_INSTANCE_TYPE", "gpu_1x_a10")
        self.region = os.getenv("LAMBDA_REGION", "us-west-1")
        self.storage_bucket = os.getenv("AWS_S3_BUCKET", "understudy-training")
        self.hf_token = os.getenv("HF_TOKEN")
        
        # Training monitoring
        self.active_instances = {}
        self.training_logs = {}
        
    async def _api_request(self, method: str, endpoint: str, data: Optional[Dict] = None) -> Dict:
        """Make an API request to Lambda Cloud"""
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
        
        url = f"{self.api_base}{endpoint}"
        
        async with aiohttp.ClientSession() as session:
            async with session.request(method, url, headers=headers, json=data) as response:
                response_data = await response.json()
                
                if response.status >= 400:
                    error = response_data.get("error", {})
                    raise Exception(f"Lambda Cloud API error: {error.get('message', 'Unknown error')}")
                
                return response_data.get("data", response_data)
    
    async def ensure_ssh_key(self) -> bool:
        """Ensure SSH key exists in Lambda Cloud account"""
        try:
            # List existing SSH keys
            ssh_keys = await self._api_request("GET", "/ssh-keys")
            
            # Check if our key exists
            key_exists_in_cloud = False
            for key in ssh_keys:
                if key["name"] == self.ssh_key_name:
                    logger.info(f"SSH key '{self.ssh_key_name}' already exists in Lambda Cloud")
                    key_exists_in_cloud = True
                    break
            
            # Set up local private key path
            if not self.ssh_private_key_path:
                # Use persistent key storage in /app/keys
                persistent_key_path = f"/app/keys/lambda_cloud_{self.ssh_key_name}.pem"
                
                if os.path.exists(persistent_key_path):
                    logger.info(f"Using existing private key from {persistent_key_path}")
                    self.ssh_private_key_path = persistent_key_path
                elif not key_exists_in_cloud:
                    # Generate new SSH key pair
                    logger.info(f"Generating new SSH key pair '{self.ssh_key_name}'")
                    key_data = await self._api_request("POST", "/ssh-keys", {
                        "name": self.ssh_key_name
                    })
                    
                    # Save private key to persistent storage
                    os.makedirs("/app/keys", exist_ok=True)
                    with open(persistent_key_path, "w") as f:
                        f.write(key_data["private_key"])
                    os.chmod(persistent_key_path, 0o600)
                    self.ssh_private_key_path = persistent_key_path
                    
                    logger.info(f"SSH key pair created and saved to {persistent_key_path}")
                else:
                    logger.warning(f"SSH key exists in cloud but no local private key found at {persistent_key_path}")
                    # Need to regenerate the key pair since we don't have the private key locally
                    logger.info("Deleting existing key in cloud to regenerate with local private key")
                    
                    # Find the key ID and delete existing key
                    ssh_keys = await self._api_request("GET", "/ssh-keys")
                    key_id = None
                    for key in ssh_keys:
                        if key["name"] == self.ssh_key_name:
                            key_id = key["id"]
                            break
                    
                    if key_id:
                        await self._api_request("DELETE", f"/ssh-keys/{key_id}")
                    else:
                        logger.error(f"Could not find key ID for {self.ssh_key_name}")
                    
                    # Generate new key pair
                    logger.info(f"Generating new SSH key pair '{self.ssh_key_name}'")
                    key_data = await self._api_request("POST", "/ssh-keys", {
                        "name": self.ssh_key_name
                    })
                    
                    # Save private key to persistent storage
                    os.makedirs("/app/keys", exist_ok=True)
                    with open(persistent_key_path, "w") as f:
                        f.write(key_data["private_key"])
                    os.chmod(persistent_key_path, 0o600)
                    self.ssh_private_key_path = persistent_key_path
                    
                    logger.info(f"New SSH key pair created and saved to {persistent_key_path}")
            else:
                # Upload existing public key if not already in cloud
                if not key_exists_in_cloud:
                    with open(f"{self.ssh_private_key_path}.pub", "r") as f:
                        public_key = f.read()
                    
                    await self._api_request("POST", "/ssh-keys", {
                        "name": self.ssh_key_name,
                        "public_key": public_key
                    })
                    
                    logger.info(f"SSH key '{self.ssh_key_name}' uploaded")
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to ensure SSH key: {e}")
            return False
    
    async def get_available_instance_types(self) -> List[Dict]:
        """Get available GPU instance types"""
        try:
            instance_types = await self._api_request("GET", "/instance-types")
            
            # Filter for GPU instances with availability
            available = []
            for name, info in instance_types.items():
                if info["regions_with_capacity_available"]:
                    available.append({
                        "name": name,
                        "description": info["instance_type"]["description"],
                        "gpu": info["instance_type"]["gpu_description"],
                        "price_cents_per_hour": info["instance_type"]["price_cents_per_hour"],
                        "available_regions": [r["name"] for r in info["regions_with_capacity_available"]]
                    })
            
            return available
            
        except Exception as e:
            logger.error(f"Failed to get instance types: {e}")
            return []
    
    async def get_existing_instances(self) -> List[Dict[str, Any]]:
        """Get list of existing Lambda Cloud instances"""
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(
                    f"{self.api_base}/instances",
                    auth=aiohttp.BasicAuth(self.api_key, ''),
                    headers={"Accept": "application/json"}
                ) as response:
                    if response.status == 200:
                        data = await response.json()
                        instances = data.get("data", [])
                        logger.info(f"Found {len(instances)} existing Lambda Cloud instances")
                        return instances
                    else:
                        logger.warning(f"Failed to get instances: {response.status}")
                        return []
        except Exception as e:
            logger.error(f"Error getting existing instances: {e}")
            return []
    
    async def launch_instance(self, job_id: str, instance_name: str) -> Optional[LambdaInstance]:
        """Launch a new Lambda Cloud GPU instance"""
        try:
            # Ensure SSH key exists
            if not await self.ensure_ssh_key():
                raise Exception("Failed to ensure SSH key")
            
            launch_data = {
                "region_name": self.region,
                "instance_type_name": self.instance_type,
                "ssh_key_names": [self.ssh_key_name],
                "quantity": 1,
                "name": instance_name
            }
            
            logger.info(f"Launching Lambda Cloud instance: {instance_name}")
            result = await self._api_request("POST", "/instance-operations/launch", launch_data)
            
            # Extract instance ID from result
            instance_ids = result.get("instance_ids", [])
            if not instance_ids:
                raise Exception("No instance ID returned")
            
            instance_id = instance_ids[0]
            
            # Wait for instance to be ready
            instance = await self._wait_for_instance(instance_id)
            
            if instance:
                self.active_instances[job_id] = instance
                logger.info(f"Instance {instance.id} launched successfully at {instance.ip_address}")
            
            return instance
            
        except Exception as e:
            logger.error(f"Failed to launch instance: {e}")
            return None
    
    async def _wait_for_instance(self, instance_id: str, timeout: int = 300) -> Optional[LambdaInstance]:
        """Wait for instance to be ready"""
        start_time = time.time()
        
        while time.time() - start_time < timeout:
            try:
                instances = await self._api_request("GET", "/instances")
                
                for inst in instances:
                    if inst["id"] == instance_id:
                        if inst["status"] == "active" and inst.get("ip"):
                            return LambdaInstance(
                                id=inst["id"],
                                name=inst["name"],
                                ip_address=inst["ip"],
                                instance_type=inst["instance_type"]["name"],
                                region=inst["region"]["name"],
                                status=inst["status"],
                                ssh_key_names=inst["ssh_key_names"]
                            )
                
                await asyncio.sleep(5)
                
            except Exception as e:
                logger.error(f"Error checking instance status: {e}")
                await asyncio.sleep(5)
        
        logger.error(f"Instance {instance_id} did not become ready within {timeout} seconds")
        return None
    
    async def execute_training(self, job: Dict[str, Any], instance: LambdaInstance) -> Dict[str, Any]:
        """Execute training on the Lambda Cloud instance"""
        try:
            logger.info(f"Starting training execution on instance {instance.id}")
            
            # Generate training script
            training_script = self._generate_training_script(job)
            
            # Setup SSH connection
            ssh_client = await self._setup_ssh(instance.ip_address)
            
            # Upload and execute training script
            result = await self._run_training(ssh_client, job["job_id"], training_script, job["training_config"])
            
            # Close SSH connection
            ssh_client.close()
            
            return result
            
        except Exception as e:
            logger.error(f"Training execution failed: {e}")
            raise
    
    async def _setup_ssh(self, ip_address: str) -> paramiko.SSHClient:
        """Setup SSH connection to instance"""
        ssh_client = paramiko.SSHClient()
        ssh_client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
        
        # Retry connection with backoff
        max_retries = 10
        usernames_to_try = ["ubuntu", "lambda", "ec2-user", "admin"]
        
        for i in range(max_retries):
            # Try different usernames on different attempts
            username = usernames_to_try[i % len(usernames_to_try)]
            
            try:
                logger.info(f"Attempting SSH connection to {ip_address} with username '{username}' (attempt {i+1}/{max_retries})")
                logger.info(f"Using SSH key: {self.ssh_private_key_path}")
                
                # Check if key file exists and has correct permissions
                import os
                if not os.path.exists(self.ssh_private_key_path):
                    logger.error(f"SSH private key file not found: {self.ssh_private_key_path}")
                    raise Exception(f"SSH private key file not found: {self.ssh_private_key_path}")
                
                key_stat = os.stat(self.ssh_private_key_path)
                logger.info(f"SSH key permissions: {oct(key_stat.st_mode)[-3:]}")
                
                ssh_client.connect(
                    hostname=ip_address,
                    username=username,
                    key_filename=self.ssh_private_key_path,
                    timeout=30
                )
                logger.info(f"SSH connection established to {ip_address} with username '{username}'")
                return ssh_client
            except Exception as e:
                if i < max_retries - 1:
                    wait_time = 10 * (i + 1)
                    logger.error(f"SSH connection failed with username '{username}': {str(e)}")
                    logger.info(f"Retrying in {wait_time}s...")
                    await asyncio.sleep(wait_time)
                else:
                    raise Exception(f"Failed to establish SSH connection after {max_retries} attempts with usernames {usernames_to_try}: {e}")
    
    async def _run_training(self, ssh_client: paramiko.SSHClient, job_id: str, 
                          training_script: str, config: Dict) -> Dict[str, Any]:
        """Run training on the instance"""
        try:
            # Create job directory
            stdin, stdout, stderr = ssh_client.exec_command(f"mkdir -p /home/ubuntu/training/{job_id}")
            stdout.read()
            
            # Upload training script
            sftp = ssh_client.open_sftp()
            script_path = f"/home/ubuntu/training/{job_id}/train.py"
            with sftp.file(script_path, "w") as f:
                f.write(training_script)
            
            # Upload training data
            data_path = f"/home/ubuntu/training/{job_id}/training_data.json"
            with sftp.file(data_path, "w") as f:
                json.dump(config["training_data"], f)
            
            sftp.close()
            
            # Install dependencies with Lambda Cloud compatible versions  
            logger.info("Installing Lambda Cloud compatible dependencies...")
            install_cmd = """
                export DEBIAN_FRONTEND=noninteractive && \
                python3 -m venv /home/ubuntu/training_env && \
                /home/ubuntu/training_env/bin/pip install --upgrade pip && \
                /home/ubuntu/training_env/bin/pip install "numpy<2.0" && \
                /home/ubuntu/training_env/bin/pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121 && \
                /home/ubuntu/training_env/bin/pip install transformers peft accelerate datasets codecarbon bits bitsandbytes
            """
            stdin, stdout, stderr = ssh_client.exec_command(install_cmd)
            exit_status = stdout.channel.recv_exit_status()
            if exit_status != 0:
                error_output = stderr.read().decode()
                logger.warning(f"Dependency installation had issues: {error_output}")

            # Execute training with progress monitoring
            logger.info("Starting training...")
            train_cmd = f"cd /home/ubuntu/training/{job_id} && /home/ubuntu/training_env/bin/python3 train.py 2>&1 | tee training.log"
            # Start training in background and monitor
            stdin, stdout, stderr = ssh_client.exec_command(train_cmd, get_pty=True)
            
            # Monitor training progress
            training_metrics = {
                "epochs": [],
                "train_loss": [],
                "val_loss": [],
                "learning_rate": []
            }
            
            # Read output line by line for progress monitoring
            for line in stdout:
                line = line.strip()
                logger.info(f"Training output: {line}")
                
                # Parse training metrics from output
                if "loss:" in line.lower():
                    # Extract loss values for visualization
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
                
                # Store in training logs for real-time access
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
        """Download trained model from instance"""
        try:
            local_model_dir = f"/app/models/{job_id}"
            os.makedirs(local_model_dir, exist_ok=True)
            
            sftp = ssh_client.open_sftp()
            
            # Download model files
            remote_model_dir = f"/home/ubuntu/training/{job_id}/final_model"
            
            # List files in remote directory
            files_to_download = []
            stdin, stdout, stderr = ssh_client.exec_command(f"ls -la {remote_model_dir}")
            output = stdout.read().decode()
            
            for line in output.split('\n'):
                if line and not line.startswith('total') and not line.startswith('d'):
                    parts = line.split()
                    if len(parts) >= 9:
                        filename = parts[-1]
                        if filename not in ['.', '..']:
                            files_to_download.append(filename)
            
            # Download each file
            for filename in files_to_download:
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
    
    async def terminate_instance(self, instance_id: str) -> bool:
        """Terminate a Lambda Cloud instance"""
        try:
            logger.info(f"Terminating instance {instance_id}")
            
            result = await self._api_request("POST", "/instance-operations/terminate", {
                "instance_ids": [instance_id]
            })
            
            # Remove from active instances
            for job_id, instance in self.active_instances.items():
                if instance.id == instance_id:
                    del self.active_instances[job_id]
                    break
            
            logger.info(f"Instance {instance_id} terminated successfully")
            return True
            
        except Exception as e:
            logger.error(f"Failed to terminate instance {instance_id}: {e}")
            return False
    
    async def get_training_logs(self, job_id: str) -> List[Dict]:
        """Get training logs for a job"""
        return self.training_logs.get(job_id, [])
    
    async def get_instance_status(self, instance_id: str) -> Optional[Dict]:
        """Get status of a specific instance"""
        try:
            instances = await self._api_request("GET", "/instances")
            
            for inst in instances:
                if inst["id"] == instance_id:
                    return inst
            
            return None
            
        except Exception as e:
            logger.error(f"Failed to get instance status: {e}")
            return None
    
    def _generate_training_script(self, job: Dict[str, Any]) -> str:
        """Generate Python training script for the instance"""
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
        logger.info("SCRIPT:")
        logger.info(script)
        return script