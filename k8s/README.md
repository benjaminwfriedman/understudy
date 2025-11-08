# Understudy Kubernetes Deployment

This directory contains Kubernetes manifests and scripts for deploying Understudy on a local Kubernetes cluster (Docker Desktop).

## Directory Structure

```
k8s/
├── base/                     # Base Kubernetes manifests
│   ├── namespace.yaml       # Understudy namespace
│   ├── configmap.yaml       # Application configuration
│   ├── secrets.yaml         # Secrets template
│   ├── pvc.yaml            # Persistent Volume Claims
│   ├── postgres.yaml        # PostgreSQL StatefulSet
│   ├── redis.yaml          # Redis deployment
│   ├── rbac.yaml           # ServiceAccounts and RBAC
│   ├── backend.yaml        # Backend API deployment
│   ├── training-service.yaml # Training service deployment
│   ├── evaluation-service.yaml # Evaluation service deployment
│   ├── frontend.yaml       # Frontend deployment
│   ├── slm-inference-template.yaml # Template for SLM deployments
│   └── slm-batch-job-template.yaml # Template for evaluation jobs
├── local/                   # Local environment overrides
│   ├── ingress.yaml        # Ingress configuration
│   └── secrets-local.yaml  # Local secrets (add your keys here)
└── scripts/                 # Deployment scripts
    ├── build-images.sh     # Build all Docker images
    ├── deploy-local.sh     # Deploy to local K8s
    ├── cleanup-local.sh    # Remove all resources
    └── monitor.sh          # Real-time monitoring dashboard
```

## Prerequisites

1. **Docker Desktop** with Kubernetes enabled
   - Open Docker Desktop preferences
   - Go to Kubernetes tab
   - Check "Enable Kubernetes"
   - Click "Apply & Restart"

2. **kubectl** installed
   ```bash
   # macOS
   brew install kubectl
   
   # Or download directly
   curl -LO "https://dl.k8s.io/release/$(curl -L -s https://dl.k8s.io/release/stable.txt)/bin/darwin/amd64/kubectl"
   ```

3. **Verify setup**
   ```bash
   kubectl cluster-info
   kubectl config current-context  # Should show "docker-desktop"
   ```

## Quick Start

### 1. Build Docker Images
```bash
./k8s/scripts/build-images.sh
```

This builds all required Docker images locally:
- `understudy-backend:latest`
- `understudy-frontend:latest`
- `understudy-evaluation:latest`
- `understudy-training:latest`
- `understudy-slm-inference:latest`

### 2. Configure Secrets
Edit `k8s/local/secrets-local.yaml` and add your API keys:
```yaml
stringData:
  OPENAI_API_KEY: "your-actual-openai-key"
  ANTHROPIC_API_KEY: "your-actual-anthropic-key"
  HF_TOKEN: "your-actual-huggingface-token"
  RUNPOD_API_KEY: "your-actual-runpod-key"
```

### 3. Deploy to Kubernetes
```bash
./k8s/scripts/deploy-local.sh
```

This will:
- Create the understudy namespace
- Deploy PostgreSQL and Redis
- Deploy all application services
- Set up port forwarding
- Configure ingress (optional)

### 4. Monitor Deployment
In a separate terminal:
```bash
./k8s/scripts/monitor.sh
```

This shows real-time status of:
- Pod health and readiness
- Service endpoints
- Persistent volume status
- SLM model deployments
- Running jobs

## Access Points

After deployment, services are available at:

- **Frontend**: http://localhost:3000
- **Backend API**: http://localhost:8000
- **API Documentation**: http://localhost:8000/docs
- **PostgreSQL**: localhost:5432
- **Redis**: localhost:6379

## Architecture Overview

### Core Services (Always Running)

1. **Backend Service** (2 replicas)
   - FastAPI application
   - Manages endpoints and model lifecycle
   - Creates/deletes SLM deployments via K8s API
   - Routes inference requests

2. **PostgreSQL** (StatefulSet)
   - Stores model metadata
   - Training records
   - Evaluation scores

3. **Redis**
   - Job queue for async tasks
   - Caching layer

4. **Frontend** (2 replicas)
   - React/Next.js UI
   - Model management dashboard

5. **Training Service**
   - Manages RunPod GPU training
   - Downloads model weights
   - Updates model lifecycle phases

6. **Evaluation Service** (2 replicas)
   - Calculates semantic similarity
   - CPU-optimized with Sentence Transformers

### Dynamic Services

**SLM Inference Services** (Created on-demand by Backend)
- Deployed when model evaluation passes threshold
- Can scale to zero when not in use
- Two modes:
  - Persistent deployments for serving endpoints
  - Batch jobs for evaluation

## Model Lifecycle

1. **Training** → Model trains on RunPod GPU
2. **Downloading** → Training service downloads weights
3. **LLM Evaluation** → Semantic similarity evaluation
4. **Available** → Model stored, not deployed
5. **Deploying** → K8s deployment being created
6. **Deployed** → Serving production traffic

## Management Commands

### View logs
```bash
kubectl logs -n understudy -l app=backend
kubectl logs -n understudy -l app=frontend
kubectl logs -n understudy deployment/training-service
```

### Scale deployments
```bash
kubectl scale deployment backend -n understudy --replicas=3
```

### Execute commands in pods
```bash
kubectl exec -it -n understudy deployment/backend -- /bin/bash
```

### View SLM deployments
```bash
kubectl get deployments -n understudy -l app=slm-inference
```

### Clean up
```bash
./k8s/scripts/cleanup-local.sh
```

## Troubleshooting

### Pods not starting
```bash
kubectl describe pod <pod-name> -n understudy
kubectl logs <pod-name> -n understudy
```

### Database connection issues
Ensure PostgreSQL is running:
```bash
kubectl get statefulset postgres -n understudy
kubectl logs postgres-0 -n understudy
```

### Port forwarding not working
Kill existing port forwards:
```bash
pkill -f "kubectl port-forward"
```

Then restart:
```bash
kubectl port-forward -n understudy service/backend-service 8000:8000 &
```

### Storage issues
Check PVC status:
```bash
kubectl get pvc -n understudy
```

## Configuration

### Environment Variables
Edit `k8s/base/configmap.yaml` for non-sensitive configs:
- `DEFAULT_SIMILARITY_THRESHOLD`: Deployment threshold (0.85)
- `MAX_DEPLOYMENTS`: Maximum concurrent SLM deployments (10)
- `EVALUATION_BATCH_SIZE`: Batch size for evaluation (100)

### Resource Limits
Adjust in respective deployment files:
- Backend: 2 CPU, 4Gi memory
- SLM Inference: 4 CPU, 8Gi memory
- Evaluation: 2 CPU, 4Gi memory

### Scaling
Configure HPA in `slm-inference-template.yaml`:
- Min replicas: 1
- Max replicas: 5
- CPU threshold: 70%

## Development Tips

1. **Local image updates**: After code changes, rebuild images:
   ```bash
   docker build -t understudy-backend:latest ./backend
   kubectl rollout restart deployment backend -n understudy
   ```

2. **Database access**:
   ```bash
   kubectl port-forward -n understudy postgres-0 5432:5432
   psql -h localhost -U understudy -d understudy
   ```

3. **Watch deployments**:
   ```bash
   watch kubectl get pods -n understudy
   ```

4. **Debug failing pods**:
   ```bash
   kubectl get events -n understudy --sort-by='.lastTimestamp'
   ```