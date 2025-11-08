#!/bin/bash

# Quick rebuild and redeploy script for debugging
# Usage: ./rebuild-debug.sh <service-name>
# Example: ./rebuild-debug.sh backend

set -e

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
K8S_DIR="$( cd "$SCRIPT_DIR/.." && pwd )"
PROJECT_ROOT="$( cd "$K8S_DIR/.." && pwd )"

SERVICE_NAME="$1"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

if [ -z "$SERVICE_NAME" ]; then
    echo -e "${RED}Error: Service name required${NC}"
    echo ""
    echo "Usage: $0 <service-name>"
    echo ""
    echo "Available services:"
    echo "  backend       - Main FastAPI backend"
    echo "  frontend      - React frontend"
    echo "  evaluation    - Evaluation service"
    echo "  training      - Training service"
    echo "  model-broker  - Model broker service"
    echo "  all           - Rebuild all services"
    echo ""
    exit 1
fi

echo -e "${BLUE}============================================${NC}"
echo -e "${BLUE}    Quick DEBUG Rebuild: $SERVICE_NAME"
echo -e "${BLUE}============================================${NC}"
echo ""

# Function to rebuild and redeploy a specific service
rebuild_service() {
    local service=$1
    
    # Determine deployment name based on debug setup
    local deployment_name
    case $service in
        "backend") deployment_name="backend" ;;
        "frontend") deployment_name="frontend" ;;
        "evaluation") deployment_name="evaluation-service" ;;
        "training") deployment_name="training-service" ;;
        "model-broker") deployment_name="model-broker" ;;
        *)
            echo -e "${RED}Unknown service: $service${NC}"
            exit 1
            ;;
    esac
    
    # Check if we're in full debug mode or selective debug mode
    if kubectl get deployment "debug-${deployment_name}" -n understudy &> /dev/null; then
        # Full debug mode - use debug- prefix
        deployment_name="debug-${deployment_name}"
        echo -e "${YELLOW}Detected full debug mode for $service${NC}"
    elif kubectl get deployment "$deployment_name" -n understudy &> /dev/null; then
        # Selective debug mode or production mode
        # Check if the deployment has debug labels
        if kubectl get deployment "$deployment_name" -n understudy -o jsonpath='{.spec.template.metadata.labels.debug-enabled}' 2>/dev/null | grep -q "true"; then
            echo -e "${YELLOW}Detected selective debug mode for $service${NC}"
        else
            echo -e "${YELLOW}Warning: $service appears to be in production mode. Converting to debug mode...${NC}"
        fi
    else
        echo -e "${RED}Error: No deployment found for $service${NC}"
        echo "Available deployments:"
        kubectl get deployments -n understudy
        exit 1
    fi
    
    echo -e "${YELLOW}Step 1: Building debug image for $service...${NC}"
    "$SCRIPT_DIR/build-debug.sh" --service "$service" --rebuild
    
    echo -e "${YELLOW}Step 2: Restarting deployment $deployment_name...${NC}"
    kubectl rollout restart deployment "$deployment_name" -n understudy
    
    echo -e "${YELLOW}Step 3: Waiting for deployment to be ready...${NC}"
    kubectl rollout status deployment "$deployment_name" -n understudy
    
    echo -e "${GREEN}✓ $service rebuilt and redeployed successfully!${NC}"
}

# Function to rebuild all services
rebuild_all() {
    echo -e "${YELLOW}Rebuilding all debug services...${NC}"
    
    # Build all images
    "$SCRIPT_DIR/build-debug.sh" --rebuild
    
    # Restart all deployments
    echo -e "${YELLOW}Restarting all debug deployments...${NC}"
    kubectl rollout restart deployment -l environment=debug -n understudy
    
    # Wait for all to be ready
    echo -e "${YELLOW}Waiting for all deployments to be ready...${NC}"
    kubectl rollout status deployment -l environment=debug -n understudy
    
    echo -e "${GREEN}✓ All services rebuilt and redeployed successfully!${NC}"
}

# Function to setup port forwards if they're not running
ensure_port_forwards() {
    echo -e "${YELLOW}Checking debug port forwards...${NC}"
    
    # Check if any debug port forwards are running
    if ! pgrep -f "kubectl port-forward.*5678" > /dev/null; then
        echo -e "${YELLOW}Debug port forwards not running. Setting them up...${NC}"
        
        # Determine which services have debug enabled and setup port forwards
        setup_debug_port_forwards_smart
        
        sleep 2
        echo -e "${GREEN}✓ Debug port forwards established${NC}"
    else
        echo -e "${GREEN}✓ Debug port forwards already running${NC}"
    fi
}

# Function to intelligently setup port forwards based on current deployments
setup_debug_port_forwards_smart() {
    # Setup application port forwards (always needed)
    kubectl port-forward -n understudy service/backend-service 8000:8000 &
    kubectl port-forward -n understudy service/frontend-service 3000:3000 &
    kubectl port-forward -n understudy service/model-broker-service 8003:8003 &
    kubectl port-forward -n understudy service/postgres-service 5432:5432 &
    kubectl port-forward -n understudy service/redis-service 6379:6379 &
    
    # Check each service and setup debug ports if they have debug enabled
    local services=("backend" "evaluation-service" "training-service" "model-broker")
    local ports=(5678 5679 5680 5681)
    
    for i in "${!services[@]}"; do
        local service="${services[$i]}"
        local port="${ports[$i]}"
        
        # Check if deployment exists and has debug enabled
        if kubectl get deployment "$service" -n understudy &> /dev/null; then
            if kubectl get deployment "$service" -n understudy -o jsonpath='{.spec.template.metadata.labels.debug-enabled}' 2>/dev/null | grep -q "true"; then
                echo "Setting up debug port forward for $service on port $port"
                kubectl port-forward -n understudy deployment/"$service" "$port:$port" &
            fi
        elif kubectl get deployment "debug-$service" -n understudy &> /dev/null; then
            # Full debug mode
            echo "Setting up debug port forward for debug-$service on port $port"
            kubectl port-forward -n understudy deployment/"debug-$service" "$port:$port" &
        fi
    done
}

# Main logic
case $SERVICE_NAME in
    "backend")
        rebuild_service "backend"
        ;;
    "frontend") 
        rebuild_service "frontend"
        ;;
    "evaluation")
        rebuild_service "evaluation-service"
        ;;
    "training")
        rebuild_service "training-service"
        ;;
    "model-broker")
        rebuild_service "model-broker"
        ;;
    "all")
        rebuild_all
        ;;
    *)
        echo -e "${RED}Unknown service: $SERVICE_NAME${NC}"
        echo "Available services: backend, frontend, evaluation, training, model-broker, all"
        exit 1
        ;;
esac

# Ensure debug port forwards are running
ensure_port_forwards

echo ""
echo -e "${GREEN}============================================${NC}"
echo -e "${GREEN}    $SERVICE_NAME Ready for Debugging!     ${NC}"
echo -e "${GREEN}============================================${NC}"
echo ""
echo -e "${YELLOW}Next steps:${NC}"
echo "  1. Set breakpoints in your code"
echo "  2. In VSCode, go to Run & Debug"
echo "  3. Select the appropriate debug configuration"
echo "  4. Start debugging (F5)"
echo ""
echo -e "${BLUE}Debug Ports:${NC}"
echo "  Backend:      localhost:5678"
echo "  Evaluation:   localhost:5679"
echo "  Training:     localhost:5680"
echo "  Model Broker: localhost:5681"