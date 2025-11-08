#!/bin/bash

# Unified deployment script for Understudy
# Supports both local (Docker Desktop/Kind) and Azure AKS deployments
# Usage: ./deploy.sh --local|--azure [--debug] [--services service1,service2,...]

set -e

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
K8S_DIR="$( cd "$SCRIPT_DIR/.." && pwd )"
PROJECT_ROOT="$( cd "$K8S_DIR/.." && pwd )"

# Deployment configuration
DEPLOYMENT_TYPE=""
DEBUG_MODE=false
SELECTIVE_DEBUG=false
DEBUG_SERVICES=()
AVAILABLE_SERVICES=("backend" "frontend" "evaluation" "training" "model-broker")

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --local)
            if [ -n "$DEPLOYMENT_TYPE" ]; then
                echo -e "${RED}Error: Cannot specify both --local and --azure${NC}"
                exit 1
            fi
            DEPLOYMENT_TYPE="local"
            shift
            ;;
        --azure)
            if [ -n "$DEPLOYMENT_TYPE" ]; then
                echo -e "${RED}Error: Cannot specify both --local and --azure${NC}"
                exit 1
            fi
            DEPLOYMENT_TYPE="azure"
            shift
            ;;
        --debug)
            DEBUG_MODE=true
            shift
            ;;
        --services=*)
            SELECTIVE_DEBUG=true
            IFS=',' read -ra DEBUG_SERVICES <<< "${1#*=}"
            shift
            ;;
        --services)
            echo -e "${RED}Error: --services requires a value${NC}"
            echo "Usage: $0 --local|--azure --debug [--services=backend,evaluation]"
            exit 1
            ;;
        -h|--help)
            echo "Usage: $0 --local|--azure [--debug] [--services=service1,service2,...]"
            echo ""
            echo "Required flags (choose one):"
            echo "  --local    Deploy to local Kubernetes (Docker Desktop/Kind)"
            echo "  --azure    Deploy to Azure AKS"
            echo ""
            echo "Optional flags:"
            echo "  --debug    Deploy in debug mode with VSCode debugging support"
            echo "  --services Deploy specific services in debug mode (requires --debug)"
            echo "             Available services: ${AVAILABLE_SERVICES[*]}"
            echo ""
            echo "Examples:"
            echo "  $0 --local                                # Local deployment in production mode"
            echo "  $0 --azure                                # Azure deployment in production mode"
            echo "  $0 --local --debug                        # Local deployment with all services in debug"
            echo "  $0 --azure --debug --services=backend    # Azure with only backend in debug"
            exit 0
            ;;
        *)
            echo -e "${RED}Unknown argument: $1${NC}"
            echo "Use --help for usage information"
            exit 1
            ;;
    esac
done

# Validate required arguments
if [ -z "$DEPLOYMENT_TYPE" ]; then
    echo -e "${RED}Error: Must specify either --local or --azure${NC}"
    echo "Use --help for usage information"
    exit 1
fi

# If services are specified, require debug mode
if [ "$SELECTIVE_DEBUG" = true ] && [ "$DEBUG_MODE" = false ]; then
    echo -e "${RED}Error: --services requires --debug flag${NC}"
    echo "Usage: $0 --local|--azure --debug --services=service1,service2"
    exit 1
fi

# If debug mode without specific services, debug all
if [ "$DEBUG_MODE" = true ] && [ "$SELECTIVE_DEBUG" = false ]; then
    DEBUG_SERVICES=("${AVAILABLE_SERVICES[@]}")
fi

# Validate debug services if specified
for service in "${DEBUG_SERVICES[@]}"; do
    if [[ ! " ${AVAILABLE_SERVICES[@]} " =~ " ${service} " ]]; then
        echo -e "${RED}Error: Unknown service '$service'${NC}"
        echo "Available services: ${AVAILABLE_SERVICES[*]}"
        exit 1
    fi
done

# Display deployment configuration
echo -e "${BLUE}============================================${NC}"
echo -e "${BLUE}       Understudy Deployment                ${NC}"
echo -e "${BLUE}============================================${NC}"
echo ""
echo -e "${BLUE}Deployment Type: $DEPLOYMENT_TYPE${NC}"
if [ "$DEBUG_MODE" = true ]; then
    echo -e "${YELLOW}Debug Mode: ENABLED${NC}"
    if [ "$SELECTIVE_DEBUG" = true ]; then
        echo -e "${YELLOW}Debug Services:${NC}"
        printf '  - %s\n' "${DEBUG_SERVICES[@]}"
    else
        echo -e "${YELLOW}All services will be deployed in debug mode${NC}"
    fi
else
    echo -e "${GREEN}Production Mode${NC}"
fi
echo ""

# Function to get Docker Hub username
get_docker_user() {
    local DOCKER_USER=""
    
    # Method 1: Check docker info
    DOCKER_USER=$(docker info 2>/dev/null | grep "Username:" | awk '{print $2}')
    
    # Method 2: Check macOS keychain
    if [ -z "$DOCKER_USER" ] && [ "$(uname)" = "Darwin" ]; then
        if command -v docker-credential-osxkeychain &> /dev/null; then
            local KEYCHAIN_OUTPUT=$(echo "https://index.docker.io/v1/" | docker-credential-osxkeychain get 2>/dev/null || echo "")
            if [ -n "$KEYCHAIN_OUTPUT" ]; then
                DOCKER_USER=$(echo "$KEYCHAIN_OUTPUT" | grep -o '"Username":"[^"]*"' | cut -d'"' -f4)
            fi
        fi
    fi
    
    # Method 3: Check Windows credential helper
    if [ -z "$DOCKER_USER" ] && [ "$(uname -s | grep -c MINGW)" -eq 1 ]; then
        if command -v docker-credential-desktop &> /dev/null; then
            local CRED_OUTPUT=$(echo "https://index.docker.io/v1/" | docker-credential-desktop get 2>/dev/null || echo "")
            if [ -n "$CRED_OUTPUT" ]; then
                DOCKER_USER=$(echo "$CRED_OUTPUT" | grep -o '"Username":"[^"]*"' | cut -d'"' -f4)
            fi
        fi
    fi
    
    # Method 4: Check if config.json has auth entry
    if [ -z "$DOCKER_USER" ] && [ -f ~/.docker/config.json ]; then
        if grep -q '"https://index.docker.io/v1/"' ~/.docker/config.json 2>/dev/null; then
            echo -e "${YELLOW}Docker Hub credentials found but username not detected.${NC}"
            echo -e "${YELLOW}Please enter your Docker Hub username:${NC}"
            read -r DOCKER_USER
        fi
    fi
    
    echo "$DOCKER_USER"
}

# Function to check prerequisites
check_prerequisites() {
    echo -e "${YELLOW}Checking prerequisites...${NC}"
    
    # Check Docker
    if ! docker info > /dev/null 2>&1; then
        echo -e "${RED}Error: Docker is not running${NC}"
        exit 1
    fi
    
    # Check kubectl
    if ! command -v kubectl &> /dev/null; then
        echo -e "${RED}Error: kubectl is not installed${NC}"
        exit 1
    fi
    
    # Check Kubernetes connection
    if ! kubectl cluster-info &> /dev/null; then
        echo -e "${RED}Error: Cannot connect to Kubernetes cluster${NC}"
        if [ "$DEPLOYMENT_TYPE" = "local" ]; then
            echo "Please enable Kubernetes in Docker Desktop"
        else
            echo "Please ensure you're connected to your Azure AKS cluster"
        fi
        exit 1
    fi
    
    # Check Docker Hub login
    local DOCKER_USER=$(get_docker_user)
    if [ -z "$DOCKER_USER" ]; then
        echo -e "${RED}Error: Not logged into Docker Hub${NC}"
        echo "Please run: docker login"
        exit 1
    fi
    echo -e "${GREEN}✓ Docker Hub user: $DOCKER_USER${NC}"
    
    # Check context for local deployment
    if [ "$DEPLOYMENT_TYPE" = "local" ]; then
        local CURRENT_CONTEXT=$(kubectl config current-context)
        echo -e "${GREEN}✓ Kubernetes context: $CURRENT_CONTEXT${NC}"
        
        if [[ "$CURRENT_CONTEXT" != "docker-desktop" ]] && [[ "$CURRENT_CONTEXT" != "kind-"* ]]; then
            echo -e "${YELLOW}Warning: Context is not docker-desktop or kind${NC}"
            read -p "Continue anyway? (y/n): " -n 1 -r
            echo
            if [[ ! $REPLY =~ ^[Yy]$ ]]; then
                exit 1
            fi
        fi
    else
        # For Azure, just show the current context
        local CURRENT_CONTEXT=$(kubectl config current-context)
        echo -e "${GREEN}✓ Using Azure context: $CURRENT_CONTEXT${NC}"
    fi
    
    echo -e "${GREEN}✓ Prerequisites checked${NC}"
}

# Function to cleanup existing deployment
cleanup_existing() {
    echo -e "${YELLOW}Cleaning up existing deployment...${NC}"
    
    # Kill port forwards
    pkill -f "kubectl port-forward.*understudy" || true
    
    # Delete existing namespace (will delete all resources)
    if kubectl get namespace understudy &> /dev/null; then
        echo "Deleting existing understudy namespace..."
        kubectl delete namespace understudy --timeout=60s || true
        
        # Wait for namespace to be fully deleted
        while kubectl get namespace understudy &> /dev/null; do
            echo "Waiting for namespace deletion..."
            sleep 5
        done
    fi
    
    echo -e "${GREEN}✓ Cleanup completed${NC}"
}

# Function to determine architecture tag
get_arch_tag() {
    if [ "$DEPLOYMENT_TYPE" = "azure" ]; then
        echo "amd64"
    else
        local arch=$(uname -m)
        case $arch in
            x86_64) echo "amd64" ;;
            arm64|aarch64) echo "arm64" ;;
            *) echo "amd64" ;; # Default fallback
        esac
    fi
}

# Function to deploy with kustomization
deploy_with_kustomization() {
    local overlay_path="$K8S_DIR/overlays/$DEPLOYMENT_TYPE"
    
    if [ ! -f "$overlay_path/kustomization.yaml" ]; then
        echo -e "${RED}Error: No kustomization found for $DEPLOYMENT_TYPE${NC}"
        exit 1
    fi
    
    echo -e "${YELLOW}Deploying with kustomization for $DEPLOYMENT_TYPE...${NC}"
    kubectl apply -k "$overlay_path"
    echo -e "${GREEN}✓ Resources deployed${NC}"
}

# Function to build and push debug images
build_debug_images() {
    local arch_tag=$(get_arch_tag)
    
    for service in "${DEBUG_SERVICES[@]}"; do
        echo -e "${YELLOW}Building debug image for $service...${NC}"
        
        local build_args="--service $service --push"
        if [ "$DEPLOYMENT_TYPE" = "azure" ]; then
            build_args="$build_args --platform=amd64"
        fi
        
        if [ -f "$SCRIPT_DIR/build-debug.sh" ]; then
            "$SCRIPT_DIR/build-debug.sh" $build_args
        else
            echo -e "${YELLOW}Warning: build-debug.sh not found, assuming images exist${NC}"
        fi
    done
}

# Function to deploy service in debug mode
deploy_service_debug() {
    local service=$1
    local arch_tag=$(get_arch_tag)
    local deployment_name=""
    
    case $service in
        "backend") deployment_name="backend" ;;
        "frontend") deployment_name="frontend" ;;
        "evaluation") deployment_name="evaluation-service" ;;
        "training") deployment_name="training-service" ;;
        "model-broker") deployment_name="model-broker" ;;
    esac
    
    echo -e "${YELLOW}Deploying $service in debug mode...${NC}"
    
    # Patch deployment to use debug image
    kubectl set image deployment/"$deployment_name" \
        "$deployment_name=bennyfriedman/understudy-${service}-debug:latest-$arch_tag" \
        -n understudy
    
    # Add debug port and command based on service
    case $service in
        "backend")
            kubectl patch deployment backend -n understudy --type='json' -p='[
                {"op": "add", "path": "/spec/template/spec/containers/0/ports/-", "value": {"containerPort": 5678, "name": "debug"}},
                {"op": "replace", "path": "/spec/template/spec/containers/0/command", "value": ["python", "-m", "debugpy", "--listen", "0.0.0.0:5678", "--wait-for-client", "-m", "uvicorn"]},
                {"op": "replace", "path": "/spec/template/spec/containers/0/args", "value": ["app.main:app", "--host", "0.0.0.0", "--port", "8000", "--reload"]}
            ]'
            ;;
        "evaluation")
            kubectl patch deployment evaluation-service -n understudy --type='json' -p='[
                {"op": "add", "path": "/spec/template/spec/containers/0/ports/-", "value": {"containerPort": 5679, "name": "debug"}},
                {"op": "replace", "path": "/spec/template/spec/containers/0/command", "value": ["python", "-m", "debugpy", "--listen", "0.0.0.0:5679", "--wait-for-client", "-m", "uvicorn"]},
                {"op": "replace", "path": "/spec/template/spec/containers/0/args", "value": ["main:app", "--host", "0.0.0.0", "--port", "8001", "--reload"]}
            ]'
            ;;
        "training")
            kubectl patch deployment training-service -n understudy --type='json' -p='[
                {"op": "add", "path": "/spec/template/spec/containers/0/ports/-", "value": {"containerPort": 5680, "name": "debug"}},
                {"op": "replace", "path": "/spec/template/spec/containers/0/command", "value": ["python", "-m", "debugpy", "--listen", "0.0.0.0:5680", "--wait-for-client", "-m", "uvicorn"]},
                {"op": "replace", "path": "/spec/template/spec/containers/0/args", "value": ["main:app", "--host", "0.0.0.0", "--port", "8002", "--reload"]}
            ]'
            ;;
        "model-broker")
            kubectl patch deployment model-broker -n understudy --type='json' -p='[
                {"op": "add", "path": "/spec/template/spec/containers/0/ports/-", "value": {"containerPort": 5681, "name": "debug"}}
            ]'
            ;;
    esac
}

# Function to wait for deployments
wait_for_deployments() {
    echo -e "${YELLOW}Waiting for pods to be ready...${NC}"
    
    # Wait for databases first
    kubectl wait --for=condition=ready pod -l app=postgres -n understudy --timeout=120s || true
    kubectl wait --for=condition=ready pod -l app=redis -n understudy --timeout=120s || true
    
    # Wait for services
    kubectl wait --for=condition=ready pod -l app=backend -n understudy --timeout=120s || true
    kubectl wait --for=condition=ready pod -l app=frontend -n understudy --timeout=120s || true
    kubectl wait --for=condition=ready pod -l app=model-broker -n understudy --timeout=120s || true
    kubectl wait --for=condition=ready pod -l app=evaluation-service -n understudy --timeout=120s || true
    kubectl wait --for=condition=ready pod -l app=training-service -n understudy --timeout=120s || true
    
    echo -e "${GREEN}✓ All pods ready${NC}"
}

# Function to setup port forwards
setup_port_forwards() {
    echo -e "${YELLOW}Setting up port forwards...${NC}"
    
    # Kill existing port forwards
    pkill -f "kubectl port-forward" || true
    sleep 2
    
    # Application ports
    kubectl port-forward -n understudy service/backend-service 8000:8000 &
    kubectl port-forward -n understudy service/frontend-service 3000:3000 &
    kubectl port-forward -n understudy service/model-broker-service 8003:8003 &
    
    # Database ports (optional for local development)
    kubectl port-forward -n understudy service/postgres-service 5432:5432 &
    kubectl port-forward -n understudy service/redis-service 6379:6379 &
    
    # Debug ports if in debug mode
    if [ "$DEBUG_MODE" = true ]; then
        for service in "${DEBUG_SERVICES[@]}"; do
            case $service in
                "backend")
                    kubectl port-forward -n understudy deployment/backend 5678:5678 &
                    ;;
                "evaluation")
                    kubectl port-forward -n understudy deployment/evaluation-service 5679:5679 &
                    ;;
                "training")
                    kubectl port-forward -n understudy deployment/training-service 5680:5680 &
                    ;;
                "model-broker")
                    kubectl port-forward -n understudy deployment/model-broker 5681:5681 &
                    ;;
            esac
        done
    fi
    
    sleep 3
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
    echo -e "${GREEN}Application URLs (via port-forward):${NC}"
    echo "  Frontend:     http://localhost:3000"
    echo "  Backend API:  http://localhost:8000"
    echo "  API Docs:     http://localhost:8000/docs"
    echo "  Model Broker: http://localhost:8003"
    
    if [ "$DEBUG_MODE" = true ]; then
        echo ""
        echo -e "${GREEN}Debug Ports (for VSCode):${NC}"
        for service in "${DEBUG_SERVICES[@]}"; do
            case $service in
                "backend") echo "  Backend:      localhost:5678" ;;
                "evaluation") echo "  Evaluation:   localhost:5679" ;;
                "training") echo "  Training:     localhost:5680" ;;
                "model-broker") echo "  Model Broker: localhost:5681" ;;
            esac
        done
    fi
    
    echo ""
    echo -e "${GREEN}Database Ports:${NC}"
    echo "  PostgreSQL:   localhost:5432"
    echo "  Redis:        localhost:6379"
    
    if [ "$DEPLOYMENT_TYPE" = "azure" ]; then
        echo ""
        echo -e "${YELLOW}Note for Azure:${NC}"
        echo "  Port forwards are active for local access"
        echo "  For production access, configure ingress or LoadBalancer"
        echo "  Check Azure portal for public IPs if using LoadBalancer"
    fi
    
    echo ""
    echo -e "${YELLOW}Useful Commands:${NC}"
    echo "  View logs:    kubectl logs -n understudy -l app=backend"
    echo "  Get pods:     kubectl get pods -n understudy"
    echo "  Delete:       kubectl delete namespace understudy"
    echo "  Stop ports:   pkill -f 'kubectl port-forward'"
}

# Main deployment flow
main() {
    check_prerequisites
    
    echo ""
    echo -e "${BLUE}Starting deployment...${NC}"
    echo ""
    
    # Clean up if requested (optional, could add --clean flag)
    # cleanup_existing
    
    # Deploy base resources
    deploy_with_kustomization
    
    # If debug mode, apply debug configurations
    if [ "$DEBUG_MODE" = true ]; then
        echo ""
        echo -e "${YELLOW}Configuring debug mode...${NC}"
        
        # Build debug images
        build_debug_images
        
        # Wait for initial deployment
        wait_for_deployments
        
        # Apply debug patches to selected services
        for service in "${DEBUG_SERVICES[@]}"; do
            deploy_service_debug "$service"
        done
        
        # Restart deployments to pick up debug changes
        for service in "${DEBUG_SERVICES[@]}"; do
            local deployment_name
            case $service in
                "backend") deployment_name="backend" ;;
                "frontend") deployment_name="frontend" ;;
                "evaluation") deployment_name="evaluation-service" ;;
                "training") deployment_name="training-service" ;;
                "model-broker") deployment_name="model-broker" ;;
            esac
            kubectl rollout restart deployment "$deployment_name" -n understudy
        done
    fi
    
    # Wait for all deployments to be ready
    wait_for_deployments
    
    # Setup port forwards for both local and Azure
    setup_port_forwards
    
    # Display final status
    display_status
    
    echo ""
    echo -e "${GREEN}============================================${NC}"
    echo -e "${GREEN}    Deployment completed successfully!      ${NC}"
    echo -e "${GREEN}============================================${NC}"
}

# Run main function
main