#!/bin/bash

# Parse command line arguments
WIPE_DATA=false

while [[ $# -gt 0 ]]; do
    case $1 in
        --wipe|--clean)
            WIPE_DATA=true
            shift
            ;;
        *)
            echo "Unknown option: $1"
            echo "Usage: $0 [--wipe|--clean]"
            exit 1
            ;;
    esac
done

echo "Starting Understudy in debug mode..."
echo "=================================="
echo ""

# Stop any existing containers
echo "Stopping existing containers..."
docker-compose down

# Wipe data if requested
if [ "$WIPE_DATA" = true ]; then
    echo ""
    echo "🧹 Wiping data..."
    echo "=================================="
    
    # Remove SQLite database file(s)
    if [ -f "data/understudy.db" ]; then
        echo "Removing SQLite database..."
        rm -f data/understudy.db
        echo "✓ SQLite database removed"
    fi
    
    # You can also add pattern matching for other possible DB files
    rm -f data/*.db data/*.sqlite data/*.sqlite3 2>/dev/null
    
    # Clear Redis data by removing the volume
    echo "Removing Redis volume..."
    docker volume rm understudy_redis_data 2>/dev/null || true
    echo "✓ Redis data volume removed"
    
    echo ""
    echo "✓ Data wipe complete!"
    echo ""
fi

echo "To debug in VS Code:"
echo "1. Wait for the containers to start (you'll see 'Waiting for debugger to attach...')"
echo "2. In VS Code, go to Run and Debug (Ctrl+Shift+D / Cmd+Shift+D)"
echo "3. Select 'Python: Attach to Docker Backend'"
echo "4. Click the green play button or press F5"
echo "5. The debugger will attach and continue execution"
echo ""
echo "Starting containers..."
echo ""

# Build and start with debug configuration
docker-compose -f docker-compose.yml -f docker-compose.debug.yml up --build