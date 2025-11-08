#!/bin/bash

# Debug build script - builds images with debugging enabled
# Usage: ./build-debug.sh [--push] [--service <service-name>]

set -e

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
PROJECT_ROOT="$( cd "$SCRIPT_DIR/../.." && pwd )"

# Parse arguments
PUSH_TO_HUB=false
SPECIFIC_SERVICE=""
REBUILD_NO_CACHE=false
FORCE_PLATFORM=""
REMOVE_AFTER_PUSH=false

for arg in "$@"; do
    case $arg in
        --push)
            PUSH_TO_HUB=true
            ;;
        --service)
            SPECIFIC_SERVICE="$2"
            shift 2
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
        *)
            if [[ $arg != --* ]]; then
                continue
            fi
            echo "Unknown argument: $arg"
            echo "Usage: $0 [--push] [--service <service-name>] [--rebuild] [--platform=<arch>] [--remove]"
            echo "  --push               Push images to Docker Hub"
            echo "  --service <name>     Build only specific service (backend, frontend, etc.)"
            echo "  --rebuild            Force rebuild with no cache"
            echo "  --platform=<arch>    Force specific architecture (amd64/arm64)"
            echo "  --remove             Remove local images after push (requires --push)"
            exit 1
            ;;
    esac
done

# Validate --remove flag usage
if [ "$REMOVE_AFTER_PUSH" = true ] && [ "$PUSH_TO_HUB" = false ]; then
    echo -e "${RED}Error: --remove flag requires --push flag${NC}"
    echo "The --remove flag only works when pushing images to Docker Hub"
    exit 1
fi

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

echo -e "${BLUE}============================================${NC}"
echo -e "${BLUE}    Building DEBUG Images for Understudy   ${NC}"
echo -e "${BLUE}============================================${NC}"
echo ""

BUILD_ARGS=""
if [ "$REBUILD_NO_CACHE" = true ]; then
    BUILD_ARGS="--no-cache"
    echo -e "${YELLOW}Building with --no-cache flag${NC}"
fi

# Function to build a service with debug configuration
build_debug_service() {
    local service_name=$1
    local service_dir=$2
    local image_name="understudy-${service_name}-debug"
    
    echo -e "${YELLOW}Building DEBUG ${service_name} image...${NC}"
    
    # Create temporary debug Dockerfile
    local debug_dockerfile="$service_dir/Dockerfile.debug"
    
    case $service_name in
        "backend"|"evaluation"|"training")
            cat > "$debug_dockerfile" <<EOF
FROM python:3.11-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \\
    gcc \\
    g++ \\
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu && \\
    pip install --no-cache-dir -r requirements.txt

# DEBUG: Install debugging tools
RUN pip install debugpy ipdb

# Copy application code
COPY . .

# DEBUG: Set debug environment variables
ENV PYTHONUNBUFFERED=1
ENV LOG_LEVEL=DEBUG
ENV DEBUG_MODE=true

# Expose both app and debug ports
EXPOSE 8000 5678

# DEBUG: Default command includes debugpy but can be overridden
CMD ["python", "-m", "debugpy", "--listen", "0.0.0.0:5678", "--wait-for-client", "-m", "uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000", "--reload"]
EOF
            ;;
        "frontend")
            cat > "$debug_dockerfile" <<EOF
FROM node:18-alpine

WORKDIR /app

# Copy package files
COPY package*.json ./

# Install dependencies
RUN npm install

# Copy source code
COPY . .

# DEBUG: Set development environment
ENV NODE_ENV=development
ENV REACT_APP_DEBUG=true

# Expose port
EXPOSE 3000

# DEBUG: Start in development mode
CMD ["npm", "start"]
EOF
            ;;
        "model-broker")
            cat > "$debug_dockerfile" <<EOF
FROM python:3.11-slim

WORKDIR /app

# Install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# DEBUG: Install debugging tools
RUN pip install debugpy ipdb

# Copy application code
COPY . .

# DEBUG: Set debug environment variables
ENV PYTHONUNBUFFERED=1
ENV LOG_LEVEL=DEBUG
ENV DEBUG_MODE=true

# Expose both app and debug ports
EXPOSE 8003 5681

# DEBUG: Default command with debugging enabled
CMD ["python", "-m", "debugpy", "--listen", "0.0.0.0:5681", "--wait-for-client", "-m", "uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8003", "--reload"]
EOF
            ;;
    esac
    
    # Detect or use forced architecture
    local docker_arch=""
    local tag_suffix=""
    
    if [ -n "$FORCE_PLATFORM" ]; then
        case $FORCE_PLATFORM in
            amd64)
                docker_arch="linux/amd64"
                tag_suffix="amd64"
                ;;
            arm64)
                docker_arch="linux/arm64"
                tag_suffix="arm64"
                ;;
            *)
                echo -e "${RED}Unsupported forced platform: $FORCE_PLATFORM${NC}"
                return 1
                ;;
        esac
    else
        local arch=$(uname -m)
        case $arch in
            x86_64)
                docker_arch="linux/amd64"
                tag_suffix="amd64"
                ;;
            arm64|aarch64)
                docker_arch="linux/arm64"
                tag_suffix="arm64"
                ;;
            *)
                echo -e "${RED}Unsupported architecture: $arch${NC}"
                return 1
                ;;
        esac
    fi
    
    echo -e "${BLUE}Building for architecture: $docker_arch (tag: latest-$tag_suffix)${NC}"
    
    # Build the debug image for current architecture
    if docker buildx build $BUILD_ARGS --platform "$docker_arch" -f "$debug_dockerfile" -t "$image_name:latest-$tag_suffix" "$service_dir" --load; then
        echo -e "${GREEN}✓ DEBUG ${service_name} image built successfully${NC}"
        
        # Clean up temporary Dockerfile
        rm -f "$debug_dockerfile"
        
        # Push if requested
        if [ "$PUSH_TO_HUB" = true ]; then
            push_debug_image "$image_name"
        fi
    else
        echo -e "${RED}✗ Failed to build DEBUG ${service_name} image${NC}"
        rm -f "$debug_dockerfile"
        exit 1
    fi
}

# Function to push debug image
push_debug_image() {
    local image_name=$1
    
    # Get Docker Hub username (cross-platform detection)
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
        # Detect or use forced architecture for tagging
        local tag_suffix=""
        if [ -n "$FORCE_PLATFORM" ]; then
            tag_suffix="$FORCE_PLATFORM"
        else
            local arch=$(uname -m)
            case $arch in
                x86_64) tag_suffix="amd64" ;;
                arm64|aarch64) tag_suffix="arm64" ;;
            esac
        fi
        
        echo -e "${YELLOW}Pushing $image_name to Docker Hub as $DOCKER_USER (architecture: $tag_suffix)...${NC}"
        docker tag "$image_name:latest-$tag_suffix" "$DOCKER_USER/$image_name:latest-$tag_suffix"
        if docker push "$DOCKER_USER/$image_name:latest-$tag_suffix"; then
            echo -e "${GREEN}✓ $image_name pushed successfully${NC}"
            
            # Remove local images if --remove flag is set
            if [ "$REMOVE_AFTER_PUSH" = true ]; then
                echo -e "${YELLOW}Removing local images for $image_name...${NC}"
                docker rmi "$image_name:latest-$tag_suffix" "$DOCKER_USER/$image_name:latest-$tag_suffix" 2>/dev/null || true
                echo -e "${GREEN}✓ Local images removed${NC}"
            fi
        else
            echo -e "${RED}✗ Failed to push $image_name${NC}"
        fi
    else
        echo -e "${RED}Error: Not logged into Docker Hub${NC}"
        echo "Please run: docker login"
        exit 1
    fi
}

# Build specific service or all services
if [ -n "$SPECIFIC_SERVICE" ]; then
    case $SPECIFIC_SERVICE in
        "backend")
            build_debug_service "backend" "$PROJECT_ROOT/backend"
            ;;
        "frontend")
            build_debug_service "frontend" "$PROJECT_ROOT/frontend"
            ;;
        "evaluation")
            build_debug_service "evaluation" "$PROJECT_ROOT/evaluation_service"
            ;;
        "training")
            build_debug_service "training" "$PROJECT_ROOT/training_service"
            ;;
        "model-broker")
            build_debug_service "model-broker" "$PROJECT_ROOT/model_broker_service"
            ;;
        *)
            echo -e "${RED}Unknown service: $SPECIFIC_SERVICE${NC}"
            echo "Available services: backend, frontend, evaluation, training, model-broker"
            exit 1
            ;;
    esac
else
    # Build all services
    build_debug_service "backend" "$PROJECT_ROOT/backend"
    build_debug_service "frontend" "$PROJECT_ROOT/frontend"
    build_debug_service "evaluation" "$PROJECT_ROOT/evaluation_service"
    build_debug_service "training" "$PROJECT_ROOT/training_service"
    build_debug_service "model-broker" "$PROJECT_ROOT/model_broker_service"
fi

echo ""
echo -e "${GREEN}============================================${NC}"
echo -e "${GREEN}      DEBUG Images Built Successfully!     ${NC}"
echo -e "${GREEN}============================================${NC}"
echo ""
echo "DEBUG images created:"
docker images | grep understudy.*debug | awk '{print "  - "$1":"$2}'
echo ""
echo -e "${YELLOW}Next steps:${NC}"
echo "  1. Deploy debug environment: ./k8s/scripts/deploy-debug.sh"
echo "  2. Connect VSCode debugger to pods"
echo "  3. Set breakpoints and debug!"
echo ""
echo -e "${BLUE}VSCode Debug Ports:${NC}"
echo "  - Backend: 5678"
echo "  - Evaluation: 5679"  
echo "  - Training: 5680"
echo "  - Model Broker: 5681"