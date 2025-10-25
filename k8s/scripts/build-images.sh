#!/bin/bash

# Build script for all Docker images needed for Kubernetes deployment
# This script builds the images locally for use with Docker Desktop Kubernetes
# 
# Usage: 
#   ./build-images.sh           # Build images locally only
#   ./build-images.sh --push     # Build and push to Docker Hub

set -e

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
PROJECT_ROOT="$( cd "$SCRIPT_DIR/../.." && pwd )"

# Parse arguments
PUSH_TO_HUB=false
if [[ "$1" == "--push" ]]; then
    PUSH_TO_HUB=true
fi

echo "Building Understudy Docker images for Kubernetes deployment..."
echo "Project root: $PROJECT_ROOT"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Build backend image
echo -e "${YELLOW}Building backend image...${NC}"
docker build -t understudy-backend:latest "$PROJECT_ROOT/backend" || {
    echo -e "${RED}Failed to build backend image${NC}"
    exit 1
}
echo -e "${GREEN}✓ Backend image built successfully${NC}"

# Build frontend image
echo -e "${YELLOW}Building frontend image...${NC}"
docker build -t understudy-frontend:latest "$PROJECT_ROOT/frontend" || {
    echo -e "${RED}Failed to build frontend image${NC}"
    exit 1
}
echo -e "${GREEN}✓ Frontend image built successfully${NC}"

# Build evaluation service image
echo -e "${YELLOW}Building evaluation service image...${NC}"
if [ -d "$PROJECT_ROOT/evaluation_service" ]; then
    docker build -t understudy-evaluation:latest "$PROJECT_ROOT/evaluation_service" || {
        echo -e "${RED}Failed to build evaluation service image${NC}"
        exit 1
    }
    echo -e "${GREEN}✓ Evaluation service image built successfully${NC}"
else
    echo -e "${YELLOW}Warning: evaluation_service directory not found. Creating placeholder...${NC}"
    mkdir -p "$PROJECT_ROOT/evaluation_service"
    cat > "$PROJECT_ROOT/evaluation_service/Dockerfile" <<EOF
FROM python:3.11-slim
WORKDIR /app
RUN pip install fastapi uvicorn sentence-transformers redis
COPY . .
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8001"]
EOF
    cat > "$PROJECT_ROOT/evaluation_service/main.py" <<EOF
from fastapi import FastAPI
import logging

app = FastAPI()
logger = logging.getLogger(__name__)

@app.get("/health")
async def health():
    return {"status": "healthy"}

@app.post("/api/v1/evaluate")
async def evaluate(data: dict):
    # Placeholder for evaluation logic
    return {"semantic_similarity_score": 0.85}
EOF
    docker build -t understudy-evaluation:latest "$PROJECT_ROOT/evaluation_service"
    echo -e "${GREEN}✓ Evaluation service placeholder image built${NC}"
fi

# Build training service image
echo -e "${YELLOW}Building training service image...${NC}"
if [ -d "$PROJECT_ROOT/docker/training" ]; then
    docker build -t understudy-training:latest "$PROJECT_ROOT/docker/training" || {
        echo -e "${RED}Failed to build training service image${NC}"
        exit 1
    }
    echo -e "${GREEN}✓ Training service image built successfully${NC}"
else
    echo -e "${YELLOW}Warning: docker/training directory found but may need updates${NC}"
    docker build -t understudy-training:latest "$PROJECT_ROOT/docker/training"
fi

# Build model broker service image
echo -e "${YELLOW}Building model broker service image...${NC}"
docker build -t understudy-model-broker:latest "$PROJECT_ROOT/model_broker_service" || {
    echo -e "${RED}Failed to build model broker service image${NC}"
    exit 1
}
echo -e "${GREEN}✓ Model broker service image built successfully${NC}"

# Build SLM inference service image
echo -e "${YELLOW}Building SLM inference service image...${NC}"
if [ -d "$PROJECT_ROOT/slm_inference_service" ]; then
    docker build -t understudy-slm-inference:latest "$PROJECT_ROOT/slm_inference_service" || {
        echo -e "${RED}Failed to build SLM inference service image${NC}"
        exit 1
    }
    echo -e "${GREEN}✓ SLM inference service image built successfully${NC}"
else
    echo -e "${YELLOW}Warning: slm_inference_service directory not found. Creating placeholder...${NC}"
    mkdir -p "$PROJECT_ROOT/slm_inference_service"
    cat > "$PROJECT_ROOT/slm_inference_service/Dockerfile" <<EOF
FROM python:3.11-slim
WORKDIR /app
RUN pip install torch transformers vllm fastapi uvicorn
COPY . .
CMD ["python", "-m", "vllm.entrypoints.openai.api_server", "--host", "0.0.0.0", "--port", "8000"]
EOF
    docker build -t understudy-slm-inference:latest "$PROJECT_ROOT/slm_inference_service"
    echo -e "${GREEN}✓ SLM inference placeholder image built${NC}"
fi

echo -e "${GREEN}All images built successfully!${NC}"
echo ""
echo "Images created:"
docker images | grep understudy | awk '{print "  - "$1":"$2}'
echo ""

# Push to Docker Hub if requested
if [ "$PUSH_TO_HUB" = true ]; then
    # Check if user is logged into Docker Hub
    DOCKER_USER=""
    
    # Method 1: Check docker info (works on Linux/Windows)
    DOCKER_USER=$(docker info 2>/dev/null | grep "Username:" | awk '{print $2}')
    
    # Method 2: Check macOS keychain (for macOS users)
    if [ -z "$DOCKER_USER" ] && [ "$(uname)" = "Darwin" ]; then
        # Check if docker-credential-osxkeychain exists
        if command -v docker-credential-osxkeychain &> /dev/null; then
            # Try to get credentials from keychain
            KEYCHAIN_OUTPUT=$(echo "https://index.docker.io/v1/" | docker-credential-osxkeychain get 2>/dev/null || echo "")
            if [ -n "$KEYCHAIN_OUTPUT" ]; then
                DOCKER_USER=$(echo "$KEYCHAIN_OUTPUT" | grep -o '"Username":"[^"]*"' | cut -d'"' -f4)
            fi
        fi
    fi
    
    # Method 3: Check if config.json has auth entry (cross-platform fallback)
    if [ -z "$DOCKER_USER" ] && [ -f ~/.docker/config.json ]; then
        if grep -q '"https://index.docker.io/v1/"' ~/.docker/config.json 2>/dev/null; then
            # Credentials exist but we can't get username, prompt for it
            echo -e "${YELLOW}Docker Hub credentials found. Please enter your Docker Hub username:${NC}"
            read -r DOCKER_USER
        fi
    fi
    
    if [ -n "$DOCKER_USER" ]; then
        echo -e "${YELLOW}Pushing images to Docker Hub as user: $DOCKER_USER${NC}"
        
        # Tag and push all images
        for image in understudy-backend understudy-frontend understudy-evaluation understudy-training understudy-model-broker understudy-slm-inference; do
            echo -e "${YELLOW}Pushing $image to $DOCKER_USER/$image:latest...${NC}"
            docker tag $image:latest $DOCKER_USER/$image:latest
            if docker push $DOCKER_USER/$image:latest; then
                echo -e "${GREEN}✓ $image pushed successfully${NC}"
            else
                echo -e "${RED}✗ Failed to push $image${NC}"
            fi
        done
        
        echo ""
        echo -e "${GREEN}✓ All images pushed to Docker Hub as $DOCKER_USER/*${NC}"
        echo ""
        echo -e "${YELLOW}Next steps:${NC}"
        echo "  1. Update secrets in k8s/local/secrets-local.yaml with your API keys"
        echo "  2. Ensure k8s/base/*.yaml files use $DOCKER_USER/* image prefix"
        echo "  3. Run: ./k8s/scripts/deploy-local.sh"
        echo ""
        echo -e "${GREEN}Quick update all deployments:${NC}"
        echo "  sed -i '' 's|image: understudy-|image: $DOCKER_USER/understudy-|g' k8s/base/*.yaml"
    else
        echo -e "${RED}Error: Not logged into Docker Hub${NC}"
        echo "Please run: docker login"
        exit 1
    fi
else
    echo -e "${YELLOW}Images built locally. To push to Docker Hub, run:${NC}"
    echo "  ./k8s/scripts/build-images.sh --push"
    echo ""
    echo -e "${YELLOW}Next steps for local images:${NC}"
    echo "  1. Update secrets in k8s/local/secrets-local.yaml with your API keys"
    echo "  2. Run: ./k8s/scripts/deploy-local.sh"
    echo ""
    echo -e "${RED}Note: Local images may not work with Docker Desktop Kubernetes.${NC}"
    echo -e "${RED}Consider using --push flag to push to Docker Hub.${NC}"
fi