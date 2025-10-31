#!/bin/bash

# Cleanup debug deployment script
# Removes all debug resources and port forwards

set -e

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
K8S_DIR="$( cd "$SCRIPT_DIR/.." && pwd )"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

echo -e "${BLUE}============================================${NC}"
echo -e "${BLUE}    Cleaning Up DEBUG Environment         ${NC}"
echo -e "${BLUE}============================================${NC}"
echo ""

# Function to stop port forwards
stop_port_forwards() {
    echo -e "${YELLOW}Stopping debug port forwards...${NC}"
    
    # Kill all kubectl port-forward processes for debug
    pkill -f "kubectl port-forward.*debug" || true
    
    # Wait a moment for processes to stop
    sleep 2
    
    echo -e "${GREEN}✓ Port forwards stopped${NC}"
}

# Function to delete debug resources
delete_debug_resources() {
    echo -e "${YELLOW}Deleting debug K8s resources...${NC}"
    
    # Delete debug resources using kustomization
    kubectl delete -k "$K8S_DIR/overlays/debug/" 2>/dev/null || true
    
    # Wait for cleanup
    sleep 5
    
    echo -e "${GREEN}✓ Debug resources deleted${NC}"
}

# Function to clean up debug images (optional)
cleanup_debug_images() {
    echo -e "${YELLOW}Cleaning up debug Docker images...${NC}"
    
    # Remove debug images
    docker images | grep "understudy.*debug" | awk '{print $1":"$2}' | xargs -r docker rmi || true
    
    echo -e "${GREEN}✓ Debug images removed${NC}"
}

# Function to show remaining resources
show_remaining() {
    echo ""
    echo -e "${BLUE}Checking for remaining debug resources...${NC}"
    
    # Check for any remaining debug pods
    REMAINING_PODS=$(kubectl get pods -n understudy -l environment=debug --no-headers 2>/dev/null | wc -l)
    if [ "$REMAINING_PODS" -gt 0 ]; then
        echo -e "${YELLOW}Warning: $REMAINING_PODS debug pods still exist:${NC}"
        kubectl get pods -n understudy -l environment=debug
    else
        echo -e "${GREEN}✓ No debug pods remaining${NC}"
    fi
    
    # Check for any remaining debug services
    REMAINING_SERVICES=$(kubectl get services -n understudy -l environment=debug --no-headers 2>/dev/null | wc -l)
    if [ "$REMAINING_SERVICES" -gt 0 ]; then
        echo -e "${YELLOW}Warning: $REMAINING_SERVICES debug services still exist:${NC}"
        kubectl get services -n understudy -l environment=debug
    else
        echo -e "${GREEN}✓ No debug services remaining${NC}"
    fi
}

# Main cleanup flow
main() {
    echo -e "${YELLOW}This will remove all debug resources and stop port forwards.${NC}"
    read -p "Do you want to continue? (y/N): " -n 1 -r
    echo
    
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        echo -e "${YELLOW}Cleanup cancelled.${NC}"
        exit 0
    fi
    
    stop_port_forwards
    delete_debug_resources
    
    # Ask if user wants to clean up images too
    echo ""
    read -p "Do you want to remove debug Docker images? (y/N): " -n 1 -r
    echo
    
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        cleanup_debug_images
    fi
    
    show_remaining
    
    echo ""
    echo -e "${GREEN}============================================${NC}"
    echo -e "${GREEN}    DEBUG Environment Cleaned Up!         ${NC}"
    echo -e "${GREEN}============================================${NC}"
    echo ""
    echo -e "${YELLOW}To deploy debug environment again:${NC}"
    echo "  ./k8s/scripts/deploy-debug.sh"
}

# Run main function
main "$@"