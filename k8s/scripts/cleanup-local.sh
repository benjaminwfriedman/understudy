#!/bin/bash

# Cleanup script for Understudy Kubernetes deployment

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

echo -e "${BLUE}============================================${NC}"
echo -e "${BLUE}   Understudy Kubernetes Cleanup            ${NC}"
echo -e "${BLUE}============================================${NC}"
echo ""

echo -e "${YELLOW}This will delete all Understudy resources from Kubernetes${NC}"
read -p "Are you sure you want to continue? (y/n): " -n 1 -r
echo
if [[ ! $REPLY =~ ^[Yy]$ ]]; then
    echo "Cleanup cancelled"
    exit 1
fi

# Kill port forwards
echo -e "${YELLOW}Stopping port forwards...${NC}"
pkill -f "kubectl port-forward" || true
echo -e "${GREEN}✓ Port forwards stopped${NC}"

# Delete namespace (this will delete all resources in the namespace)
echo -e "${YELLOW}Deleting understudy namespace and all resources...${NC}"
kubectl delete namespace understudy --ignore-not-found=true
echo -e "${GREEN}✓ Namespace and resources deleted${NC}"

echo ""
echo -e "${GREEN}Cleanup completed successfully!${NC}"