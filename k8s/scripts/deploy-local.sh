#!/bin/bash

# Local deployment script for Understudy on Docker Desktop Kubernetes
# Prerequisites:
#   - Docker Desktop with Kubernetes enabled
#   - kubectl installed and configured
#   - Docker Hub login (docker login)
#   - Images pushed to Docker Hub with build-images.sh --push

set -e

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
K8S_DIR="$( cd "$SCRIPT_DIR/.." && pwd )"
PROJECT_ROOT="$( cd "$K8S_DIR/.." && pwd )"

# Get Docker Hub username (cross-platform)
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

# Method 3: Check Windows credential helper
if [ -z "$DOCKER_USER" ] && [ "$(uname -s | grep -c MINGW)" -eq 1 ]; then
    if command -v docker-credential-desktop &> /dev/null; then
        CRED_OUTPUT=$(echo "https://index.docker.io/v1/" | docker-credential-desktop get 2>/dev/null || echo "")
        if [ -n "$CRED_OUTPUT" ]; then
            DOCKER_USER=$(echo "$CRED_OUTPUT" | grep -o '"Username":"[^"]*"' | cut -d'"' -f4)
        fi
    fi
fi

# Method 4: Check if config.json has auth entry (fallback - user has creds but we can't get username)
if [ -z "$DOCKER_USER" ] && [ -f ~/.docker/config.json ]; then
    if grep -q '"https://index.docker.io/v1/"' ~/.docker/config.json 2>/dev/null; then
        # Credentials exist but we can't extract username automatically
        echo -e "${YELLOW}Docker Hub credentials found but username not detected.${NC}"
        echo -e "${YELLOW}Please enter your Docker Hub username:${NC}"
        read -r DOCKER_USER
    fi
fi

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

echo -e "${BLUE}============================================${NC}"
echo -e "${BLUE}   Understudy Kubernetes Local Deployment   ${NC}"
echo -e "${BLUE}============================================${NC}"
echo ""

# Function to check prerequisites
check_prerequisites() {
    echo -e "${YELLOW}Checking prerequisites...${NC}"
    
    # Check if Docker is running
    if ! docker info > /dev/null 2>&1; then
        echo -e "${RED}Error: Docker is not running${NC}"
        echo "Please start Docker Desktop"
        exit 1
    fi
    
    # Check Docker Hub login
    if [ -z "$DOCKER_USER" ]; then
        echo -e "${RED}Error: Not logged into Docker Hub${NC}"
        echo ""
        echo "Please login to Docker Hub:"
        echo "  docker login"
        echo ""
        echo "Then build and push images:"
        echo "  ./k8s/scripts/build-images.sh --push"
        echo ""
        exit 1
    else
        echo -e "${GREEN}✓ Docker Hub user: $DOCKER_USER${NC}"
    fi
    
    # Check if kubectl is installed
    if ! command -v kubectl &> /dev/null; then
        echo -e "${RED}Error: kubectl is not installed${NC}"
        echo "Please install kubectl: https://kubernetes.io/docs/tasks/tools/"
        exit 1
    fi
    
    # Check if Kubernetes is enabled in Docker Desktop
    if ! kubectl cluster-info &> /dev/null; then
        echo -e "${RED}Error: Kubernetes is not running${NC}"
        echo "Please enable Kubernetes in Docker Desktop settings"
        exit 1
    fi
    
    # Check current context
    CURRENT_CONTEXT=$(kubectl config current-context)
    echo -e "${GREEN}✓ Using Kubernetes context: $CURRENT_CONTEXT${NC}"
    
    if [[ "$CURRENT_CONTEXT" != "docker-desktop" ]]; then
        echo -e "${YELLOW}Warning: Current context is not 'docker-desktop'${NC}"
        read -p "Do you want to switch to docker-desktop context? (y/n): " -n 1 -r
        echo
        if [[ $REPLY =~ ^[Yy]$ ]]; then
            kubectl config use-context docker-desktop
        fi
    fi
    
    # Check if images exist on Docker Hub (quick check for backend image)
    echo -e "${YELLOW}Verifying Docker Hub images...${NC}"
    if curl -s "https://hub.docker.com/v2/repositories/$DOCKER_USER/understudy-backend/tags" | grep -q "name"; then
        echo -e "${GREEN}✓ Docker Hub images found${NC}"
    else
        echo -e "${YELLOW}Warning: Images may not be on Docker Hub yet${NC}"
        echo "Run: ./k8s/scripts/build-images.sh --push"
    fi
    
    echo -e "${GREEN}✓ All prerequisites checked${NC}"
}

# Function to create namespace
create_namespace() {
    echo -e "${YELLOW}Creating namespace...${NC}"
    kubectl apply -f "$K8S_DIR/base/namespace.yaml"
    echo -e "${GREEN}✓ Namespace created${NC}"
}

# Function to apply base configurations
apply_base_configs() {
    echo -e "${YELLOW}Applying base configurations...${NC}"
    
    # Apply secrets (local version with placeholder keys)
    if [ -f "$K8S_DIR/local/secrets-local.yaml" ]; then
        echo "Applying local secrets..."
        kubectl apply -f "$K8S_DIR/local/secrets-local.yaml"
    else
        echo -e "${YELLOW}Warning: Using base secrets. Please update with your API keys!${NC}"
        kubectl apply -f "$K8S_DIR/base/secrets.yaml"
    fi
    
    # Apply other base configs
    kubectl apply -f "$K8S_DIR/base/configmap.yaml"
    kubectl apply -f "$K8S_DIR/base/rbac.yaml"
    kubectl apply -f "$K8S_DIR/base/pvc.yaml"
    
    echo -e "${GREEN}✓ Base configurations applied${NC}"
}

# Function to deploy databases
deploy_databases() {
    echo -e "${YELLOW}Deploying databases...${NC}"
    
    kubectl apply -f "$K8S_DIR/base/postgres.yaml"
    kubectl apply -f "$K8S_DIR/base/redis.yaml"
    
    echo "Waiting for databases to be ready..."
    kubectl wait --for=condition=ready pod -l app=postgres -n understudy --timeout=120s || true
    kubectl wait --for=condition=ready pod -l app=redis -n understudy --timeout=120s || true
    
    echo -e "${GREEN}✓ Databases deployed${NC}"
}

# Function to deploy services
deploy_services() {
    echo -e "${YELLOW}Deploying application services...${NC}"
    
    # Deploy model broker first (other services depend on it)
    kubectl apply -f "$K8S_DIR/base/model-broker.yaml"
    
    # Wait for model broker to be ready
    echo "Waiting for model broker to be ready..."
    kubectl wait --for=condition=ready pod -l app=model-broker -n understudy --timeout=120s || true
    
    # Deploy backend
    kubectl apply -f "$K8S_DIR/base/backend.yaml"
    
    # Deploy evaluation service
    kubectl apply -f "$K8S_DIR/base/evaluation-service.yaml"
    
    # Deploy training service
    kubectl apply -f "$K8S_DIR/base/training-service.yaml"
    
    # Deploy frontend
    kubectl apply -f "$K8S_DIR/base/frontend.yaml"
    
    echo "Waiting for services to be ready..."
    kubectl wait --for=condition=ready pod -l app=backend -n understudy --timeout=120s || true
    kubectl wait --for=condition=ready pod -l app=frontend -n understudy --timeout=120s || true
    
    echo -e "${GREEN}✓ Application services deployed${NC}"
}

# Function to setup ingress
setup_ingress() {
    echo -e "${YELLOW}Setting up ingress...${NC}"
    
    # Check if nginx ingress controller is installed
    if ! kubectl get namespace ingress-nginx &> /dev/null; then
        echo "Installing NGINX Ingress Controller..."
        kubectl apply -f https://raw.githubusercontent.com/kubernetes/ingress-nginx/controller-v1.8.1/deploy/static/provider/cloud/deploy.yaml
        echo "Waiting for ingress controller to be ready..."
        kubectl wait --namespace ingress-nginx \
            --for=condition=ready pod \
            --selector=app.kubernetes.io/component=controller \
            --timeout=120s || true
    fi
    
    # Apply ingress rules
    kubectl apply -f "$K8S_DIR/local/ingress.yaml"
    
    echo -e "${GREEN}✓ Ingress configured${NC}"
    
    # Add entry to /etc/hosts if not present
    if ! grep -q "understudy.local" /etc/hosts; then
        echo -e "${YELLOW}Adding understudy.local to /etc/hosts (requires sudo)${NC}"
        echo "127.0.0.1 understudy.local" | sudo tee -a /etc/hosts
    fi
}

# Function to port forward services
setup_port_forwards() {
    echo -e "${YELLOW}Setting up port forwards...${NC}"
    
    # Kill any existing port forwards
    pkill -f "kubectl port-forward" || true
    
    # Start port forwards in background
    kubectl port-forward -n understudy service/backend-service 8000:8000 &
    kubectl port-forward -n understudy service/frontend-service 3000:3000 &
    kubectl port-forward -n understudy service/postgres-service 5432:5432 &
    kubectl port-forward -n understudy service/redis-service 6379:6379 &
    kubectl port-forward -n understudy service/model-broker-service 8003:8003 &
    
    echo -e "${GREEN}✓ Port forwards established${NC}"
}

# Function to display status
display_status() {
    echo ""
    echo -e "${BLUE}============================================${NC}"
    echo -e "${BLUE}         Deployment Status                  ${NC}"
    echo -e "${BLUE}============================================${NC}"
    
    kubectl get pods -n understudy
    
    echo ""
    echo -e "${BLUE}============================================${NC}"
    echo -e "${BLUE}         Access Information                  ${NC}"
    echo -e "${BLUE}============================================${NC}"
    echo ""
    echo -e "${GREEN}Frontend:${NC} http://localhost:3000"
    echo -e "${GREEN}Backend API:${NC} http://localhost:8000"
    echo -e "${GREEN}API Docs:${NC} http://localhost:8000/docs"
    echo -e "${GREEN}Model Broker:${NC} http://localhost:8003"
    echo ""
    echo -e "${YELLOW}Port forwards running in background:${NC}"
    echo "  - Backend: localhost:8000"
    echo "  - Frontend: localhost:3000"
    echo "  - Model Broker: localhost:8003"
    echo "  - PostgreSQL: localhost:5432"
    echo "  - Redis: localhost:6379"
    echo ""
    echo -e "${YELLOW}To stop port forwards:${NC}"
    echo "  pkill -f 'kubectl port-forward'"
    echo ""
    echo -e "${YELLOW}To view logs:${NC}"
    echo "  kubectl logs -n understudy -l app=backend"
    echo "  kubectl logs -n understudy -l app=frontend"
    echo ""
    echo -e "${YELLOW}To delete deployment:${NC}"
    echo "  ./k8s/scripts/cleanup-local.sh"
}

# Main deployment flow
main() {
    check_prerequisites
    
    echo ""
    echo -e "${BLUE}Starting deployment...${NC}"
    echo ""
    
    create_namespace
    apply_base_configs
    deploy_databases
    
    # Wait a bit for databases to initialize
    echo "Waiting for database initialization..."
    sleep 10
    
    deploy_services
    setup_ingress
    setup_port_forwards
    
    display_status
    
    echo ""
    echo -e "${GREEN}============================================${NC}"
    echo -e "${GREEN}    Deployment completed successfully!      ${NC}"
    echo -e "${GREEN}============================================${NC}"
}

# Run main function
main "$@"