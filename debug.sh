#!/bin/bash

echo "Starting Understudy in debug mode..."
echo "=================================="
echo ""
echo "To debug in VS Code:"
echo "1. Wait for the containers to start (you'll see 'Waiting for debugger to attach...')"
echo "2. In VS Code, go to Run and Debug (Ctrl+Shift+D / Cmd+Shift+D)"
echo "3. Select 'Python: Attach to Docker Backend'"
echo "4. Click the green play button or press F5"
echo "5. The debugger will attach and continue execution"
echo ""
echo "Starting containers..."
echo ""

# Stop any existing containers
docker-compose down

# Build and start with debug configuration
docker-compose -f docker-compose.yml -f docker-compose.debug.yml up --build