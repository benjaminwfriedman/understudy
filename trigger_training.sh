#!/bin/bash

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# API base URL
API_URL="http://localhost:8000"

# Function to check if API is reachable
check_api() {
    if curl -s -o /dev/null -w "%{http_code}" "$API_URL/api/v1/health" | grep -q "200"; then
        return 0
    else
        return 1
    fi
}

# Check if API is running
echo "🔍 Checking API connection..."
if ! check_api; then
    echo -e "${RED}❌ Error: Cannot connect to API at $API_URL${NC}"
    echo "Make sure the Understudy backend is running."
    exit 1
fi
echo -e "${GREEN}✅ API is reachable${NC}"

# Prompt for model ID
echo ""
echo -e "${YELLOW}Enter the endpoint/model ID to trigger training:${NC}"
read -r MODEL_ID

# Validate input
if [ -z "$MODEL_ID" ]; then
    echo -e "${RED}❌ Error: Model ID cannot be empty${NC}"
    exit 1
fi

# Trigger training
echo ""
echo "🚀 Triggering training for endpoint: $MODEL_ID"
echo "Sending POST request to: $API_URL/api/v1/training/$MODEL_ID"
echo ""

# Create JSON payload
JSON_PAYLOAD='{
  "num_examples": 0,
  "force_retrain": false,
  "provider": "",
  "epochs": 3,
  "batch_size": 4,
  "learning_rate": 0.0002,
  "priority": 0
}'

# Make the API call and capture response
RESPONSE=$(curl -s -w "\n%{http_code}" -X POST \
    -H "Content-Type: application/json" \
    -d "$JSON_PAYLOAD" \
    "$API_URL/api/v1/training/$MODEL_ID")

# Extract HTTP status code (last line)
HTTP_CODE=$(echo "$RESPONSE" | tail -n1)
# Extract response body (everything except last line)
BODY=$(echo "$RESPONSE" | sed '$d')

# Check response
if [ "$HTTP_CODE" = "200" ] || [ "$HTTP_CODE" = "201" ] || [ "$HTTP_CODE" = "202" ]; then
    echo -e "${GREEN}✅ Training triggered successfully!${NC}"
    echo ""
    echo "Response:"
    echo "$BODY" | python3 -m json.tool 2>/dev/null || echo "$BODY"
else
    echo -e "${RED}❌ Failed to trigger training (HTTP $HTTP_CODE)${NC}"
    echo ""
    echo "Error response:"
    echo "$BODY" | python3 -m json.tool 2>/dev/null || echo "$BODY"
    exit 1
fi

echo ""
echo "📊 You can monitor training progress at: http://localhost:3000/endpoints/$MODEL_ID"