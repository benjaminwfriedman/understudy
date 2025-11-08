# Training Service

The Training Service handles all GPU training operations for Understudy. It manages training jobs on various cloud providers and handles the complete training lifecycle from job submission to model weight storage.

## Architecture

The Training Service follows a provider-agnostic design where the backend submits training requests without needing to know the specifics of each cloud provider. All provider management, VM/instance lifecycle, SSH connections, and model storage are handled within this service.

## Current Provider Support

### ✅ RunPod (Implemented)
- **Status**: Fully implemented
- **Features**: 
  - Pod creation and management
  - SSH-based training execution
  - Model weight download and storage
  - Automatic cleanup
- **Configuration**: Via environment variables and Kubernetes secrets

### 🔄 Future Providers (To Be Implemented)

All future provider implementations should be added to this training service:

#### Azure GPU VMs
- **Implementation needed in**: `azure_trainer.py`
- **Features to implement**:
  - Azure VM lifecycle management
  - SSH key management
  - Training script execution
  - Model storage integration
  - Cost optimization (VM reuse, auto-shutdown)

#### Lambda Cloud 
- **Implementation needed in**: `lambda_trainer.py`
- **Features to implement**:
  - Lambda instance management
  - SSH connections
  - Training execution
  - Instance cleanup

#### AWS (SageMaker/EC2)
- **Implementation needed in**: `aws_trainer.py`
- **Features to implement**:
  - SageMaker training jobs OR EC2 instances
  - S3 integration for model storage
  - IAM role management
  - Cost optimization

#### Google Cloud Platform
- **Implementation needed in**: `gcp_trainer.py`
- **Features to implement**:
  - Compute Engine instance management
  - Cloud Storage integration
  - Service account management

## API Endpoints

### `POST /api/v1/training/start`
Start a new training job
```json
{
  "train_id": "unique_training_id",
  "endpoint_id": "endpoint_uuid",
  "version": 1,
  "training_pairs_count": 100,
  "slm_type": "microsoft/DialoGPT-small",
  "source_llm": "gpt-3.5-turbo",
  "provider": "runpod"
}
```

### `GET /api/v1/training/{train_id}/status`
Get training job status

### `POST /api/v1/training/{train_id}/cancel`
Cancel a training job

### `POST /api/v1/training/{train_id}/completed`
Mark training as completed (internal use)

## Provider Implementation Guide

When adding a new provider:

1. **Create trainer class** (e.g., `azure_trainer.py`)
2. **Implement required methods**:
   - `launch_instance()` - Create VM/instance
   - `execute_training()` - Run training via SSH
   - `download_model_weights()` - Transfer model files
   - `terminate_instance()` - Cleanup resources

3. **Update main.py**:
   - Add provider validation
   - Import new trainer class
   - Add provider-specific logic in `start_training` endpoint

4. **Update configuration**:
   - Add environment variables
   - Update Kubernetes secrets
   - Document required permissions

## Environment Variables

### RunPod Configuration
```env
RUNPOD_API_KEY=your_api_key
RUNPOD_SSH_PUBLIC_KEY=ssh_public_key
RUNPOD_SSH_PRIVATE_KEY_PATH=/app/keys/runpod_ssh_key
RUNPOD_GPU_TYPE="NVIDIA GeForce RTX 4090"
```

### Future Provider Environment Variables
```env
# Azure
AZURE_SUBSCRIPTION_ID=your_subscription_id
AZURE_RESOURCE_GROUP=your_resource_group
AZURE_SSH_KEY_NAME=your_ssh_key

# Lambda Cloud
LAMBDA_API_KEY=your_lambda_api_key
LAMBDA_SSH_KEY_NAME=your_ssh_key

# AWS
AWS_ACCESS_KEY_ID=your_access_key
AWS_SECRET_ACCESS_KEY=your_secret_key
AWS_REGION=us-west-2
```

## Security

- **SSH Keys**: Private keys are mounted as files with 0600 permissions
- **API Keys**: Stored in Kubernetes secrets
- **Network**: All training communication over SSH
- **Isolation**: Each training job runs in isolated instance

## Monitoring

The service provides comprehensive logging and metrics:
- Training job lifecycle events
- Provider-specific metrics
- Error tracking and alerting
- Resource utilization monitoring

## Cost Optimization

Each provider implementation should include:
- Instance reuse capabilities
- Auto-shutdown for idle instances
- Resource right-sizing
- Spot/preemptible instance support where available