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

# Available services
AVAILABLE_SERVICES=("backend" "frontend" "evaluation" "training" "model-broker" "slm-inference")
BUILD_SERVICES=()

# Parse arguments
PUSH_TO_HUB=false
REBUILD_NO_CACHE=false
FORCE_PLATFORM=""
REMOVE_AFTER_PUSH=false
SELECTIVE_BUILD=false

for arg in "$@"; do
    case $arg in
        --push)
            PUSH_TO_HUB=true
            ;;
        --rebuild)
            REBUILD_NO_CACHE=true
            ;;
        --platform=*)
            FORCE_PLATFORM="${arg#*=}"
            ;;
        --remove)
            REMOVE_AFTER_PUSH=true
            ;;
        --services=*)
            SELECTIVE_BUILD=true
            IFS=',' read -ra BUILD_SERVICES <<< "${arg#*=}"
            ;;
        --services)
            echo "Error: --services requires a value"
            echo "Usage: $0 --services=backend,evaluation"
            exit 1
            ;;
        *)
            echo "Unknown argument: $arg"
            echo "Usage: $0 [--push] [--rebuild] [--platform=<arch>] [--remove] [--services=service1,service2,...]"
            echo "  --push              Push images to Docker Hub"
            echo "  --rebuild           Force rebuild with no cache"
            echo "  --platform=<arch>   Force specific architecture (amd64/arm64)"
            echo "  --remove            Remove local images after push (requires --push)"
            echo "  --services=<list>   Build only specific services (backend,frontend,evaluation,training,model-broker,slm-inference)"
            echo "Available services: ${AVAILABLE_SERVICES[*]}"
            exit 1
            ;;
    esac
done

# If no services specified, build all services
if [ "$SELECTIVE_BUILD" = false ]; then
    BUILD_SERVICES=("${AVAILABLE_SERVICES[@]}")
fi

# Validate build services
for service in "${BUILD_SERVICES[@]}"; do
    if [[ ! " ${AVAILABLE_SERVICES[@]} " =~ " ${service} " ]]; then
        echo "Error: Unknown service '$service'"
        echo "Available services: ${AVAILABLE_SERVICES[*]}"
        exit 1
    fi
done

# Validate --remove flag usage
if [ "$REMOVE_AFTER_PUSH" = true ] && [ "$PUSH_TO_HUB" = false ]; then
    echo -e "${RED}Error: --remove flag requires --push flag${NC}"
    echo "The --remove flag only works when pushing images to Docker Hub"
    exit 1
fi

echo "Building Understudy Docker images for Kubernetes deployment..."
echo "Project root: $PROJECT_ROOT"

# Detect or use forced architecture
DOCKER_ARCH=""
TAG_SUFFIX=""

if [ -n "$FORCE_PLATFORM" ]; then
    case $FORCE_PLATFORM in
        amd64)
            DOCKER_ARCH="linux/amd64"
            TAG_SUFFIX="-amd64"
            ;;
        arm64)
            DOCKER_ARCH="linux/arm64"
            TAG_SUFFIX="-arm64"
            ;;
        *)
            echo "Error: Unsupported platform: $FORCE_PLATFORM"
            echo "Supported platforms: amd64, arm64"
            exit 1
            ;;
    esac
    echo "Forced platform: $FORCE_PLATFORM"
else
    # Auto-detect architecture
    ARCH=$(uname -m)
    case $ARCH in
        x86_64)
            DOCKER_ARCH="linux/amd64"
            TAG_SUFFIX=""
            ;;
        arm64|aarch64)
            DOCKER_ARCH="linux/arm64"
            TAG_SUFFIX=""
            ;;
        *)
            echo "Warning: Unknown architecture: $ARCH, defaulting to multi-arch build"
            DOCKER_ARCH=""
            TAG_SUFFIX=""
            ;;
    esac
fi

if [ -n "$DOCKER_ARCH" ]; then
    echo "Building for architecture: $DOCKER_ARCH"
fi

if [ "$REBUILD_NO_CACHE" = true ]; then
    echo "Building with --no-cache flag for complete rebuild"
    BUILD_ARGS="--no-cache"
else
    BUILD_ARGS=""
fi

# Add platform flag if specified
if [ -n "$DOCKER_ARCH" ]; then
    BUILD_ARGS="$BUILD_ARGS --platform $DOCKER_ARCH"
fi

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Function to build, push, and optionally remove a single service
build_service() {
    local service_name=$1
    local service_dir=$2
    local image_name=$3
    local dockerfile_path=$4
    
    echo -e "${YELLOW}Building $service_name image...${NC}"
    
    if [ -n "$dockerfile_path" ]; then
        docker buildx build $BUILD_ARGS -f "$dockerfile_path" -t "$image_name:latest${TAG_SUFFIX}" "$service_dir" --load || {
            echo -e "${RED}Failed to build $service_name image${NC}"
            exit 1
        }
    else
        docker buildx build $BUILD_ARGS -t "$image_name:latest${TAG_SUFFIX}" "$service_dir" --load || {
            echo -e "${RED}Failed to build $service_name image${NC}"
            exit 1
        }
    fi
    
    echo -e "${GREEN}✓ $service_name image built successfully${NC}"
    
    # Push immediately if requested
    if [ "$PUSH_TO_HUB" = true ]; then
        push_service_image "$image_name"
    fi
}

# Function to push and optionally remove a service image
push_service_image() {
    local image_name=$1
    
    if [ -n "$DOCKER_USER" ]; then
        echo -e "${YELLOW}Pushing $image_name to $DOCKER_USER/$image_name:latest${TAG_SUFFIX}...${NC}"
        docker tag "$image_name:latest${TAG_SUFFIX}" "$DOCKER_USER/$image_name:latest${TAG_SUFFIX}"
        if docker push "$DOCKER_USER/$image_name:latest${TAG_SUFFIX}"; then
            echo -e "${GREEN}✓ $image_name pushed successfully${NC}"
            
            # Remove local images immediately if --remove flag is set
            if [ "$REMOVE_AFTER_PUSH" = true ]; then
                echo -e "${YELLOW}Removing local images for $image_name...${NC}"
                docker rmi "$image_name:latest${TAG_SUFFIX}" "$DOCKER_USER/$image_name:latest${TAG_SUFFIX}" 2>/dev/null || true
                echo -e "${GREEN}✓ Local images removed${NC}"
            fi
        else
            echo -e "${RED}✗ Failed to push $image_name${NC}"
            exit 1
        fi
    fi
}

# Check Docker Hub login if pushing
if [ "$PUSH_TO_HUB" = true ]; then
    # Get Docker Hub username (same logic as before)
    DOCKER_USER=""
    
    # Method 1: Check docker info (works on Linux/Windows)
    DOCKER_USER=$(docker info 2>/dev/null | grep "Username:" | awk '{print $2}')
    
    # Method 2: Check macOS keychain (for macOS users)
    if [ -z "$DOCKER_USER" ] && [ "$(uname)" = "Darwin" ]; then
        if command -v docker-credential-osxkeychain &> /dev/null; then
            KEYCHAIN_OUTPUT=$(echo "https://index.docker.io/v1/" | docker-credential-osxkeychain get 2>/dev/null || echo "")
            if [ -n "$KEYCHAIN_OUTPUT" ]; then
                DOCKER_USER=$(echo "$KEYCHAIN_OUTPUT" | grep -o '"Username":"[^"]*"' | cut -d'"' -f4)
            fi
        fi
    fi
    
    # Method 3: Check if config.json has auth entry (cross-platform fallback)
    if [ -z "$DOCKER_USER" ] && [ -f ~/.docker/config.json ]; then
        if grep -q '"https://index.docker.io/v1/"' ~/.docker/config.json 2>/dev/null; then
            echo -e "${YELLOW}Docker Hub credentials found. Please enter your Docker Hub username:${NC}"
            read -r DOCKER_USER
        fi
    fi
    
    if [ -z "$DOCKER_USER" ]; then
        echo -e "${RED}Error: Not logged into Docker Hub${NC}"
        echo "Please run: docker login"
        exit 1
    fi
    
    echo -e "${YELLOW}Will push images to Docker Hub as user: $DOCKER_USER${NC}"
fi

# Build selected services
for service in "${BUILD_SERVICES[@]}"; do
    case $service in
        "backend")
            build_service "backend" "$PROJECT_ROOT/backend" "understudy-backend"
            ;;
        "frontend")
            build_service "frontend" "$PROJECT_ROOT/frontend" "understudy-frontend"
            ;;
        "evaluation")
            if [ -d "$PROJECT_ROOT/evaluation_service" ]; then
                build_service "evaluation" "$PROJECT_ROOT/evaluation_service" "understudy-evaluation"
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
                build_service "evaluation (placeholder)" "$PROJECT_ROOT/evaluation_service" "understudy-evaluation"
            fi
            ;;
        "training")
            if [ -d "$PROJECT_ROOT/training_service" ]; then
                build_service "training" "$PROJECT_ROOT/training_service" "understudy-training"
            else
                echo -e "${YELLOW}Warning: training_service directory not found${NC}"
                exit 1
            fi
            ;;
        "model-broker")
            build_service "model-broker" "$PROJECT_ROOT/model_broker_service" "understudy-model-broker"
            ;;
        "slm-inference")
            if [ -d "$PROJECT_ROOT/slm_inference_service" ]; then
                build_service "slm-inference" "$PROJECT_ROOT/slm_inference_service" "understudy-slm-inference"
            else
                echo -e "${YELLOW}Warning: slm_inference_service directory not found. Creating placeholder...${NC}"
                mkdir -p "$PROJECT_ROOT/slm_inference_service"
                cat > "$PROJECT_ROOT/slm_inference_service/Dockerfile" <<EOF
FROM python:3.11-slim
WORKDIR /app
RUN pip install torch --index-url https://download.pytorch.org/whl/cpu && \\
    pip install transformers vllm fastapi uvicorn
COPY . .
CMD ["python", "-m", "vllm.entrypoints.openai.api_server", "--host", "0.0.0.0", "--port", "8000"]
EOF
                build_service "slm-inference (placeholder)" "$PROJECT_ROOT/slm_inference_service" "understudy-slm-inference"
            fi
            ;;
    esac
done

echo -e "${GREEN}All images built successfully!${NC}"
echo ""
echo "Images created:"
docker images | grep understudy | awk '{print "  - "$1":"$2}'
echo ""

if [ "$PUSH_TO_HUB" = true ]; then
    echo ""
    echo -e "${GREEN}✓ All images built and pushed to Docker Hub successfully!${NC}"
    if [ "$REMOVE_AFTER_PUSH" = true ]; then
        echo -e "${GREEN}✓ Local images removed to save disk space${NC}"
    fi
    echo ""
    echo -e "${YELLOW}Next steps:${NC}"
    echo "  1. Update secrets in k8s/local/secrets-local.yaml with your API keys"
    echo "  2. Ensure k8s/base/*.yaml files use your Docker Hub username prefix"
    echo "  3. Run: ./k8s/scripts/deploy-local.sh"
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