#!/bin/bash

# Monitoring script for Understudy Kubernetes deployment

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

clear

while true; do
    echo -e "${BLUE}============================================${NC}"
    echo -e "${BLUE}   Understudy Kubernetes Monitor            ${NC}"
    echo -e "${BLUE}============================================${NC}"
    echo -e "${YELLOW}$(date)${NC}"
    echo ""
    
    echo -e "${GREEN}PODS:${NC}"
    kubectl get pods -n understudy --no-headers | while read line; do
        POD_NAME=$(echo $line | awk '{print $1}')
        STATUS=$(echo $line | awk '{print $3}')
        READY=$(echo $line | awk '{print $2}')
        
        if [[ "$STATUS" == "Running" ]]; then
            echo -e "  ${GREEN}✓${NC} $POD_NAME ($READY)"
        elif [[ "$STATUS" == "Pending" ]] || [[ "$STATUS" == "ContainerCreating" ]]; then
            echo -e "  ${YELLOW}⟳${NC} $POD_NAME ($STATUS)"
        else
            echo -e "  ${RED}✗${NC} $POD_NAME ($STATUS)"
        fi
    done
    
    echo ""
    echo -e "${GREEN}SERVICES:${NC}"
    kubectl get services -n understudy --no-headers | while read line; do
        SERVICE_NAME=$(echo $line | awk '{print $1}')
        TYPE=$(echo $line | awk '{print $2}')
        CLUSTER_IP=$(echo $line | awk '{print $3}')
        echo -e "  $SERVICE_NAME ($TYPE) - $CLUSTER_IP"
    done
    
    echo ""
    echo -e "${GREEN}PERSISTENT VOLUMES:${NC}"
    kubectl get pvc -n understudy --no-headers | while read line; do
        PVC_NAME=$(echo $line | awk '{print $1}')
        STATUS=$(echo $line | awk '{print $2}')
        SIZE=$(echo $line | awk '{print $4}')
        
        if [[ "$STATUS" == "Bound" ]]; then
            echo -e "  ${GREEN}✓${NC} $PVC_NAME ($SIZE)"
        else
            echo -e "  ${YELLOW}⟳${NC} $PVC_NAME ($STATUS)"
        fi
    done
    
    echo ""
    echo -e "${GREEN}SLM DEPLOYMENTS:${NC}"
    SLM_COUNT=$(kubectl get deployments -n understudy -l app=slm-inference --no-headers 2>/dev/null | wc -l)
    if [ $SLM_COUNT -gt 0 ]; then
        kubectl get deployments -n understudy -l app=slm-inference --no-headers | while read line; do
            DEPLOYMENT_NAME=$(echo $line | awk '{print $1}')
            READY=$(echo $line | awk '{print $2}')
            echo -e "  $DEPLOYMENT_NAME ($READY)"
        done
    else
        echo -e "  No SLM deployments active"
    fi
    
    echo ""
    echo -e "${GREEN}JOBS:${NC}"
    JOB_COUNT=$(kubectl get jobs -n understudy --no-headers 2>/dev/null | wc -l)
    if [ $JOB_COUNT -gt 0 ]; then
        kubectl get jobs -n understudy --no-headers | while read line; do
            JOB_NAME=$(echo $line | awk '{print $1}')
            COMPLETIONS=$(echo $line | awk '{print $2}')
            echo -e "  $JOB_NAME ($COMPLETIONS)"
        done
    else
        echo -e "  No jobs running"
    fi
    
    echo ""
    echo -e "${YELLOW}Press Ctrl+C to exit${NC}"
    echo ""
    echo "Refreshing in 5 seconds..."
    sleep 5
    clear
done