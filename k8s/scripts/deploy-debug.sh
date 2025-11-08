#!/bin/bash

# Debug deployment script for Understudy
# This deploys a debug-enabled version with VSCode debugging support
# Usage: ./deploy-debug.sh [--services service1,service2,...]

set -e

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
K8S_DIR="$( cd "$SCRIPT_DIR/.." && pwd )"
PROJECT_ROOT="$( cd "$K8S_DIR/.." && pwd )"

# Available services
AVAILABLE_SERVICES=("backend" "frontend" "evaluation" "training" "model-broker")
DEBUG_SERVICES=()

# Parse arguments
SELECTIVE_DEBUG=false
AZURE_DEPLOYMENT=false
for arg in "$@"; do
    case $arg in
        --services=*)
            SELECTIVE_DEBUG=true
            IFS=',' read -ra DEBUG_SERVICES <<< "${arg#*=}"
            ;;
        --services)
            echo -e "${RED}Error: --services requires a value${NC}"
            echo "Usage: $0 --services=backend,evaluation"
            exit 1
            ;;
        --azure)
            AZURE_DEPLOYMENT=true
            ;;
        *)
            echo "Unknown argument: $arg"
            echo "Usage: $0 [--services=service1,service2,...] [--azure]"
            echo "Available services: ${AVAILABLE_SERVICES[*]}"
            echo "Flags:"
            echo "  --azure    Deploy for Azure AKS (uses Azure storage classes)"
            exit 1
            ;;
    esac
done

# If no services specified, debug all services
if [ "$SELECTIVE_DEBUG" = false ]; then
    DEBUG_SERVICES=("${AVAILABLE_SERVICES[@]}")
fi

# Validate debug services
for service in "${DEBUG_SERVICES[@]}"; do
    if [[ ! " ${AVAILABLE_SERVICES[@]} " =~ " ${service} " ]]; then
        echo -e "${RED}Error: Unknown service '$service'${NC}"
        echo "Available services: ${AVAILABLE_SERVICES[*]}"
        exit 1
    fi
done

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

echo -e "${BLUE}============================================${NC}"
echo -e "${BLUE}    Understudy DEBUG Deployment            ${NC}"
echo -e "${BLUE}============================================${NC}"
echo ""
if [ "$AZURE_DEPLOYMENT" = true ]; then
    echo -e "${BLUE}Cloud Platform: Azure AKS${NC}"
else
    echo -e "${BLUE}Cloud Platform: Local (Docker Desktop/Kind)${NC}"
fi
echo ""
if [ "$SELECTIVE_DEBUG" = true ]; then
    echo -e "${YELLOW}Services to deploy in DEBUG mode:${NC}"
    printf '  - %s\n' "${DEBUG_SERVICES[@]}"
    echo ""
else
    echo -e "${YELLOW}Deploying ALL services in debug mode${NC}"
    echo ""
fi

# Function to check prerequisites
check_prerequisites() {
    echo -e "${YELLOW}Checking prerequisites...${NC}"
    
    # Check if Docker is running
    if ! docker info > /dev/null 2>&1; then
        echo -e "${RED}Error: Docker is not running${NC}"
        exit 1
    fi
    
    # Check if kubectl is installed
    if ! command -v kubectl &> /dev/null; then
        echo -e "${RED}Error: kubectl is not installed${NC}"
        exit 1
    fi
    
    # Check if Kubernetes is running
    if ! kubectl cluster-info &> /dev/null; then
        echo -e "${RED}Error: Kubernetes is not running${NC}"
        exit 1
    fi
    
    # Debug images will be built per-service as needed
    
    echo -e "${GREEN}✓ Prerequisites checked${NC}"
}

# Function to clean up any existing debug deployment
cleanup_existing() {
    echo -e "${YELLOW}Cleaning up existing debug deployment...${NC}"
    
    # Kill any existing port forwards
    pkill -f "kubectl port-forward.*debug" || true
    
    # Delete existing debug resources
    kubectl delete -k "$K8S_DIR/overlays/debug/" 2>/dev/null || true
    
    # Wait for cleanup
    sleep 5
    
    echo -e "${GREEN}✓ Cleanup completed${NC}"
}

# Function to deploy debug environment
deploy_debug() {
    if [ "$SELECTIVE_DEBUG" = false ]; then
        echo -e "${YELLOW}Deploying full DEBUG environment...${NC}"
        
        # Deploy using debug kustomization
        kubectl apply -k "$K8S_DIR/overlays/debug/"
        
        echo -e "${GREEN}✓ Debug environment deployed${NC}"
    else
        echo -e "${YELLOW}Deploying selective DEBUG services...${NC}"
        
        # Ensure base infrastructure exists
        kubectl apply -f "$K8S_DIR/base/namespace.yaml"
        kubectl apply -f "$K8S_DIR/base/configmap.yaml"
        kubectl apply -f "$K8S_DIR/base/rbac.yaml"
        
        # Apply PVCs (Azure-specific or local)
        if [ "$AZURE_DEPLOYMENT" = true ]; then
            echo -e "${YELLOW}Applying Azure-specific PVCs...${NC}"
            
            # Check if PVCs exist with wrong storage class and delete them
            if kubectl get pvc model-weights-pvc -n understudy &>/dev/null; then
                EXISTING_SC=$(kubectl get pvc model-weights-pvc -n understudy -o jsonpath='{.spec.storageClassName}')
                if [ "$EXISTING_SC" != "default" ] && [ "$EXISTING_SC" != "azurefile" ]; then
                    echo -e "${YELLOW}Existing PVCs have incompatible storage class ($EXISTING_SC), recreating...${NC}"
                    kubectl delete pvc --all -n understudy
                    sleep 5
                fi
            fi
            
            kubectl apply -f "$K8S_DIR/overlays/azure/pvc-azure.yaml"
        else
            kubectl apply -f "$K8S_DIR/base/pvc.yaml"
        fi
        
        # Apply secrets
        if [ -f "$K8S_DIR/local/secrets-local.yaml" ]; then
            kubectl apply -f "$K8S_DIR/local/secrets-local.yaml"
        else
            kubectl apply -f "$K8S_DIR/base/secrets.yaml"
        fi
        
        # Deploy databases (always needed)
        if [ "$AZURE_DEPLOYMENT" = true ]; then
            echo -e "${YELLOW}Applying Azure-specific postgres configuration...${NC}"
            # Delete existing StatefulSet if it exists with wrong storage class
            if kubectl get statefulset postgres -n understudy &>/dev/null; then
                EXISTING_SC=$(kubectl get statefulset postgres -n understudy -o jsonpath='{.spec.volumeClaimTemplates[0].spec.storageClassName}')
                if [ "$EXISTING_SC" != "default" ] && [ "$EXISTING_SC" != "azurefile" ]; then
                    echo -e "${YELLOW}Existing postgres StatefulSet has incompatible storage class ($EXISTING_SC), recreating...${NC}"
                    kubectl delete statefulset postgres -n understudy
                    kubectl delete pvc postgres-storage-postgres-0 -n understudy 2>/dev/null || true
                    sleep 5
                fi
            fi
            # Apply postgres using Azure overlay
            # This will create ALL services, so we need to delete non-database services
            kubectl apply -k "$K8S_DIR/overlays/azure/"
            
            # Delete the service deployments that were just created (keep databases)
            # We'll recreate them properly based on debug/production mode
            kubectl delete deployment backend frontend evaluation-service training-service model-broker -n understudy --ignore-not-found=true
        else
            kubectl apply -f "$K8S_DIR/base/postgres.yaml"
        fi
        kubectl apply -f "$K8S_DIR/base/redis.yaml"
        
        # Build debug images for selected services
        for service in "${DEBUG_SERVICES[@]}"; do
            echo -e "${YELLOW}Building debug image for $service...${NC}"
            if [ "$AZURE_DEPLOYMENT" = true ]; then
                "$SCRIPT_DIR/build-debug.sh" --service "$service" --push --platform=amd64
            else
                "$SCRIPT_DIR/build-debug.sh" --service "$service" --push
            fi
        done
        
        # Deploy services in debug or production mode
        for service in "${AVAILABLE_SERVICES[@]}"; do
            if [[ " ${DEBUG_SERVICES[@]} " =~ " ${service} " ]]; then
                echo -e "${YELLOW}Deploying $service in DEBUG mode...${NC}"
                deploy_service_debug "$service"
            else
                echo -e "${YELLOW}Deploying $service in PRODUCTION mode...${NC}"
                deploy_service_production "$service"
            fi
        done
        
        # Stop existing port forwards before restarting services
        echo -e "${YELLOW}Stopping port forwards before restart...${NC}"
        pkill -f "kubectl port-forward" || true
        sleep 2
        
        # Restart debug services to ensure they pick up the latest images
        for service in "${DEBUG_SERVICES[@]}"; do
            local deployment_name
            case $service in
                "backend") deployment_name="backend" ;;
                "frontend") deployment_name="frontend" ;;
                "evaluation") deployment_name="evaluation-service" ;;
                "training") deployment_name="training-service" ;;
                "model-broker") deployment_name="model-broker" ;;
            esac
            echo -e "${YELLOW}Restarting $service deployment to pick up latest image...${NC}"
            kubectl rollout restart deployment "$deployment_name" -n understudy
        done
        
        # Wait for restarted pods to be ready before setting up port forwards
        echo -e "${YELLOW}Waiting for restarted pods to be ready...${NC}"
        for service in "${DEBUG_SERVICES[@]}"; do
            local deployment_name
            case $service in
                "backend") deployment_name="backend" ;;
                "frontend") deployment_name="frontend" ;;
                "evaluation") deployment_name="evaluation-service" ;;
                "training") deployment_name="training-service" ;;
                "model-broker") deployment_name="model-broker" ;;
            esac
            kubectl rollout status deployment "$deployment_name" -n understudy --timeout=120s
        done
        
        echo -e "${GREEN}✓ Selective debug deployment completed${NC}"
    fi
}

# Function to deploy a service in debug mode
deploy_service_debug() {
    local service=$1
    local temp_file="/tmp/debug-${service}-$$.yaml"
    
    # Determine target architecture tag based on deployment type
    local arch_tag=""
    if [ "$AZURE_DEPLOYMENT" = true ]; then
        arch_tag="amd64"
    else
        # Detect local architecture
        local arch=$(uname -m)
        case $arch in
            x86_64) arch_tag="amd64" ;;
            arm64|aarch64) arch_tag="arm64" ;;
            *) arch_tag="amd64" ;; # Default fallback
        esac
    fi
    
    case $service in
        "backend")
            cat > "$temp_file" <<EOF
apiVersion: apps/v1
kind: Deployment
metadata:
  name: backend
  namespace: understudy
spec:
  replicas: 1
  selector:
    matchLabels:
      app: backend
  template:
    metadata:
      labels:
        app: backend
        debug-enabled: "true"
    spec:
      serviceAccountName: backend-service
      containers:
      - name: backend
        image: bennyfriedman/understudy-backend-debug:latest-$arch_tag
        imagePullPolicy: Always
        ports:
        - containerPort: 8000
        - containerPort: 5678
          name: debug
        command: ["python", "-m", "debugpy", "--listen", "0.0.0.0:5678", "--wait-for-client", "-m", "uvicorn"]
        args: ["app.main:app", "--host", "0.0.0.0", "--port", "8000", "--reload"]
        env:
        - name: DATABASE_URL
          valueFrom:
            secretKeyRef:
              name: understudy-secrets
              key: DATABASE_URL
        - name: REDIS_URL
          valueFrom:
            secretKeyRef:
              name: understudy-secrets
              key: REDIS_URL
        - name: LOG_LEVEL
          value: "DEBUG"
        - name: DEBUG_MODE
          value: "true"
        - name: PYTHONUNBUFFERED
          value: "1"
        - name: K8S_IN_CLUSTER
          value: "true"
        envFrom:
        - configMapRef:
            name: understudy-config
        - secretRef:
            name: understudy-secrets
        volumeMounts:
        - name: training-data
          mountPath: /app/data
        resources:
          requests:
            cpu: "100m"
            memory: "256Mi"
          limits:
            cpu: "1000m"
            memory: "2Gi"
      volumes:
      - name: training-data
        persistentVolumeClaim:
          claimName: training-data-pvc
EOF
            ;;
        "evaluation")
            cat > "$temp_file" <<EOF
apiVersion: apps/v1
kind: Deployment
metadata:
  name: evaluation-service
  namespace: understudy
spec:
  replicas: 1
  selector:
    matchLabels:
      app: evaluation-service
  template:
    metadata:
      labels:
        app: evaluation-service
        debug-enabled: "true"
    spec:
      containers:
      - name: evaluation-service
        image: bennyfriedman/understudy-evaluation-debug:latest-$arch_tag
        imagePullPolicy: Always
        ports:
        - containerPort: 8001
        - containerPort: 5679
          name: debug
        command: ["python", "-m", "debugpy", "--listen", "0.0.0.0:5679", "--wait-for-client", "-m", "uvicorn"]
        args: ["main:app", "--host", "0.0.0.0", "--port", "8001", "--reload"]
        env:
        - name: LOG_LEVEL
          value: "DEBUG"
        - name: DEBUG_MODE
          value: "true"
        - name: PYTHONUNBUFFERED
          value: "1"
        resources:
          requests:
            cpu: "100m"
            memory: "768Mi"
          limits:
            cpu: "500m"
            memory: "1.5Gi"
EOF
            ;;
        "training")
            cat > "$temp_file" <<EOF
apiVersion: apps/v1
kind: Deployment
metadata:
  name: training-service
  namespace: understudy
spec:
  replicas: 1
  selector:
    matchLabels:
      app: training-service
  template:
    metadata:
      labels:
        app: training-service
        debug-enabled: "true"
    spec:
      containers:
      - name: training-service
        image: bennyfriedman/understudy-training-debug:latest-$arch_tag
        imagePullPolicy: Always
        ports:
        - containerPort: 8002
        - containerPort: 5680
          name: debug
        command: ["python", "-m", "debugpy", "--listen", "0.0.0.0:5680", "--wait-for-client", "-m", "uvicorn"]
        args: ["main:app", "--host", "0.0.0.0", "--port", "8002", "--reload"]
        env:
        - name: LOG_LEVEL
          value: "DEBUG"
        - name: DEBUG_MODE
          value: "true"
        - name: PYTHONUNBUFFERED
          value: "1"
        resources:
          requests:
            cpu: "100m"
            memory: "256Mi"
          limits:
            cpu: "500m"
            memory: "512Mi"
EOF
            ;;
        "model-broker")
            cat > "$temp_file" <<EOF
apiVersion: apps/v1
kind: Deployment
metadata:
  name: model-broker
  namespace: understudy
spec:
  replicas: 1
  selector:
    matchLabels:
      app: model-broker
  template:
    metadata:
      labels:
        app: model-broker
        debug-enabled: "true"
    spec:
      containers:
      - name: model-broker
        image: bennyfriedman/understudy-model-broker-debug:latest-$arch_tag
        imagePullPolicy: Always
        ports:
        - containerPort: 8003
        - containerPort: 5681
          name: debug
        env:
        - name: LOG_LEVEL
          value: "DEBUG"
        - name: DEBUG_MODE
          value: "true"
        - name: PYTHONUNBUFFERED
          value: "1"
        resources:
          requests:
            cpu: "100m"
            memory: "1Gi"
          limits:
            cpu: "500m"
            memory: "2Gi"
EOF
            ;;
        "frontend")
            cat > "$temp_file" <<EOF
apiVersion: apps/v1
kind: Deployment
metadata:
  name: frontend
  namespace: understudy
spec:
  replicas: 1
  selector:
    matchLabels:
      app: frontend
  template:
    metadata:
      labels:
        app: frontend
        debug-enabled: "true"
    spec:
      containers:
      - name: frontend
        image: bennyfriedman/understudy-frontend-debug:latest-$arch_tag
        imagePullPolicy: Always
        ports:
        - containerPort: 3000
        env:
        - name: NODE_ENV
          value: "development"
        - name: REACT_APP_DEBUG
          value: "true"
        resources:
          requests:
            cpu: "100m"
            memory: "256Mi"
          limits:
            cpu: "500m"
            memory: "1Gi"
EOF
            ;;
    esac
    
    kubectl apply -f "$temp_file"
    rm -f "$temp_file"
}

# Function to deploy a service in production mode
deploy_service_production() {
    local service=$1
    local deployment_name
    
    echo -e "${BLUE}DEBUG: In deploy_service_production - AZURE_DEPLOYMENT='$AZURE_DEPLOYMENT'${NC}"
    
    # Determine deployment name
    case $service in
        "backend") deployment_name="backend" ;;
        "frontend") deployment_name="frontend" ;;
        "evaluation") deployment_name="evaluation-service" ;;
        "training") deployment_name="training-service" ;;
        "model-broker") deployment_name="model-broker" ;;
    esac
    
    # Check if deployment already exists
    if kubectl get deployment "$deployment_name" -n understudy &> /dev/null; then
        echo -e "${GREEN}✓ $service already running in production mode${NC}"
        return 0
    fi
    
    echo -e "${YELLOW}Deploying $service in production mode...${NC}"
    
    # For Azure deployments, we need to patch the image to use amd64 tag
    if [ "$AZURE_DEPLOYMENT" = true ]; then
        # Apply the base YAML first
        case $service in
            "backend")
                kubectl apply -f "$K8S_DIR/base/backend.yaml"
                ;;
            "frontend")
                kubectl apply -f "$K8S_DIR/base/frontend.yaml"
                ;;
            "evaluation")
                kubectl apply -f "$K8S_DIR/base/evaluation-service.yaml"
                ;;
            "training")
                kubectl apply -f "$K8S_DIR/base/training-service.yaml"
                ;;
            "model-broker")
                kubectl apply -f "$K8S_DIR/base/model-broker.yaml"
                ;;
        esac
        
        # Wait for deployment to be created before patching
        sleep 2
        
        # Patch the deployment to use amd64 image
        echo -e "${YELLOW}Patching $service to use AMD64 image for Azure...${NC}"
        local image_name
        case $service in
            "backend") image_name="understudy-backend" ;;
            "frontend") image_name="understudy-frontend" ;;
            "evaluation") image_name="understudy-evaluation" ;;
            "training") image_name="understudy-training" ;;
            "model-broker") image_name="understudy-model-broker" ;;
        esac
        
        # Use kubectl set image instead of patch for reliability
        # Container name matches deployment name for most services
        local container_name="$deployment_name"
        kubectl set image deployment/"$deployment_name" \
            "$container_name=bennyfriedman/$image_name:latest-amd64" \
            -n understudy
        
        echo -e "${GREEN}✓ $service patched to use AMD64 image${NC}"
    else
        # Local deployment - use base YAMLs as-is
        case $service in
            "backend")
                kubectl apply -f "$K8S_DIR/base/backend.yaml"
                ;;
            "frontend")
                kubectl apply -f "$K8S_DIR/base/frontend.yaml"
                ;;
            "evaluation")
                kubectl apply -f "$K8S_DIR/base/evaluation-service.yaml"
                ;;
            "training")
                kubectl apply -f "$K8S_DIR/base/training-service.yaml"
                ;;
            "model-broker")
                kubectl apply -f "$K8S_DIR/base/model-broker.yaml"
                ;;
        esac
    fi
}

# Function to wait for pods to be ready
wait_for_pods() {
    echo -e "${YELLOW}Waiting for debug pods to be ready...${NC}"
    
    # Wait for key services
    kubectl wait --for=condition=ready pod -l app=debug-backend -n understudy --timeout=120s || true
    kubectl wait --for=condition=ready pod -l app=debug-postgres -n understudy --timeout=120s || true
    kubectl wait --for=condition=ready pod -l app=debug-redis -n understudy --timeout=120s || true
    
    echo -e "${GREEN}✓ Debug pods ready${NC}"
}

# Function to setup debug port forwards
setup_debug_port_forwards() {
    echo -e "${YELLOW}Setting up port forwards...${NC}"
    
    # Kill any existing port forwards
    pkill -f "kubectl port-forward" || true
    sleep 2
    
    # Start application port forwards (always needed)
    echo "Setting up application ports..."
    kubectl port-forward -n understudy service/backend-service 8000:8000 &
    kubectl port-forward -n understudy service/frontend-service 3000:3000 &
    kubectl port-forward -n understudy service/model-broker-service 8003:8003 &
    
    # Setup debug ports only for services in debug mode
    if [ "$SELECTIVE_DEBUG" = false ]; then
        echo "Setting up debug ports for all services..."
        kubectl port-forward -n understudy deployment/debug-backend 5678:5678 &
        kubectl port-forward -n understudy deployment/debug-evaluation-service 5679:5679 &
        kubectl port-forward -n understudy deployment/debug-training-service 5680:5680 &
        kubectl port-forward -n understudy deployment/debug-model-broker 5681:5681 &
    else
        echo "Setting up debug ports for selected services..."
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
    
    echo "Setting up database ports..."
    kubectl port-forward -n understudy service/postgres-service 5432:5432 &
    kubectl port-forward -n understudy service/redis-service 6379:6379 &
    
    # Wait for port forwards to establish
    sleep 3
    
    echo -e "${GREEN}✓ Port forwards established${NC}"
}


# Function to display debug status
display_debug_status() {
    echo ""
    echo -e "${BLUE}============================================${NC}"
    echo -e "${BLUE}         Deployment Status                 ${NC}"
    echo -e "${BLUE}============================================${NC}"
    
    kubectl get pods -n understudy
    
    echo ""
    if [ "$SELECTIVE_DEBUG" = true ]; then
        echo -e "${BLUE}Services in DEBUG mode:${NC}"
        printf '  🐛 %s\n' "${DEBUG_SERVICES[@]}"
        
        echo ""
        echo -e "${BLUE}Services in PRODUCTION mode:${NC}"
        for service in "${AVAILABLE_SERVICES[@]}"; do
            if [[ ! " ${DEBUG_SERVICES[@]} " =~ " ${service} " ]]; then
                echo "  🚀 $service"
            fi
        done
    else
        echo -e "${BLUE}All services in DEBUG mode${NC}"
    fi
    
    echo ""
    echo -e "${BLUE}============================================${NC}"
    echo -e "${BLUE}         Access Information                ${NC}"
    echo -e "${BLUE}============================================${NC}"
    echo ""
    echo -e "${GREEN}Application URLs:${NC}"
    echo -e "  Frontend:     http://localhost:3000"
    echo -e "  Backend API:  http://localhost:8000"
    echo -e "  API Docs:     http://localhost:8000/docs"
    echo -e "  Model Broker: http://localhost:8003"
    echo ""
    echo -e "${GREEN}Debug Ports (for VSCode):${NC}"
    
    local debug_services_to_show
    if [ "$SELECTIVE_DEBUG" = false ]; then
        debug_services_to_show=("${AVAILABLE_SERVICES[@]}")
    else
        debug_services_to_show=("${DEBUG_SERVICES[@]}")
    fi
    
    for service in "${debug_services_to_show[@]}"; do
        case $service in
            "backend") echo -e "  Backend:      localhost:5678" ;;
            "evaluation") echo -e "  Evaluation:   localhost:5679" ;;
            "training") echo -e "  Training:     localhost:5680" ;;
            "model-broker") echo -e "  Model Broker: localhost:5681" ;;
        esac
    done
    
    echo ""
    echo -e "${GREEN}Database Ports:${NC}"
    echo -e "  PostgreSQL:   localhost:5432"
    echo -e "  Redis:        localhost:6379"
    echo ""
    echo -e "${YELLOW}VSCode Debugging:${NC}"
    echo -e "  1. Open this project in VSCode"
    echo -e "  2. Go to Run & Debug (Ctrl+Shift+D)"
    echo -e "  3. Select debug configuration from dropdown"
    echo -e "  4. Set breakpoints in your code"
    echo -e "  5. Click Play button or press F5"
    echo ""
    echo -e "${YELLOW}Useful Commands:${NC}"
    echo -e "  View logs:        kubectl logs -n understudy -l app=backend"
    echo -e "  Stop port fwds:   pkill -f 'kubectl port-forward'"
    echo -e "  Cleanup:          ./k8s/scripts/cleanup-debug.sh"
    echo -e "  Rebuild service:  ./k8s/scripts/rebuild-debug.sh <service-name>"
}

# Main deployment flow
main() {
    check_prerequisites
    cleanup_existing
    deploy_debug
    wait_for_pods
    setup_debug_port_forwards
    display_debug_status
    
    echo ""
    echo -e "${GREEN}============================================${NC}"
    echo -e "${GREEN}    DEBUG Environment Ready for VSCode!    ${NC}"
    echo -e "${GREEN}============================================${NC}"
}

# Run main function
main "$@"