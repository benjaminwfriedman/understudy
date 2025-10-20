#!/bin/bash

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

echo -e "${BLUE}🚀 RunPod SSH Configuration Setup${NC}"
echo "=============================================="
echo ""

# Prompt for email
echo -e "${YELLOW}Enter your email address for SSH key generation:${NC}"
read -r USER_EMAIL

# Validate email input
if [ -z "$USER_EMAIL" ]; then
    echo -e "${RED}❌ Error: Email address cannot be empty${NC}"
    exit 1
fi

echo ""
echo -e "${BLUE}📧 Using email: $USER_EMAIL${NC}"
echo ""

# Create keys directory in current project
KEYS_DIR="./keys"
mkdir -p "$KEYS_DIR"

# Check if SSH key already exists
SSH_KEY_PATH="$KEYS_DIR/id_ed25519"
SSH_PUB_KEY_PATH="$KEYS_DIR/id_ed25519.pub"

if [ -f "$SSH_KEY_PATH" ]; then
    echo -e "${YELLOW}⚠️  SSH key already exists at $SSH_KEY_PATH${NC}"
    echo "Do you want to:"
    echo "1) Use existing key"
    echo "2) Generate new key (will overwrite existing)"
    read -p "Enter choice (1 or 2): " choice
    
    if [ "$choice" = "2" ]; then
        echo -e "${YELLOW}🔄 Generating new SSH key...${NC}"
        ssh-keygen -t ed25519 -C "$USER_EMAIL" -f "$SSH_KEY_PATH" -N ""
    else
        echo -e "${GREEN}✅ Using existing SSH key${NC}"
    fi
else
    echo -e "${YELLOW}🔑 Generating SSH key...${NC}"
    
    # Generate SSH key in keys directory
    ssh-keygen -t ed25519 -C "$USER_EMAIL" -f "$SSH_KEY_PATH" -N ""
    
    # Set proper permissions
    chmod 600 "$SSH_KEY_PATH"
    chmod 644 "$SSH_PUB_KEY_PATH"
fi

echo ""

# Display public key
if [ -f "$SSH_PUB_KEY_PATH" ]; then
    echo -e "${GREEN}✅ SSH key pair generated successfully!${NC}"
    echo ""
    echo -e "${BLUE}📋 Your PUBLIC SSH key (copy this to RunPod):${NC}"
    echo "=============================================="
    cat "$SSH_PUB_KEY_PATH"
    echo "=============================================="
    echo ""
    
    # Copy to clipboard if possible
    if command -v pbcopy >/dev/null 2>&1; then
        cat "$SSH_PUB_KEY_PATH" | pbcopy
        echo -e "${GREEN}✅ Public key copied to clipboard!${NC}"
    elif command -v xclip >/dev/null 2>&1; then
        cat "$SSH_PUB_KEY_PATH" | xclip -selection clipboard
        echo -e "${GREEN}✅ Public key copied to clipboard!${NC}"
    else
        echo -e "${YELLOW}💡 Manual copy required (clipboard tool not available)${NC}"
    fi
    
    echo ""
    echo -e "${BLUE}📝 Add SSH key to your RunPod account:${NC}"
    echo "1. Copy the public key above"
    echo "2. Go to: ${YELLOW}https://console.runpod.io/user/settings${NC}"
    echo "3. Scroll down to the 'SSH Public Keys' field"
    echo "4. Paste the public key and save your settings"
    echo ""
    echo -e "${YELLOW}⚠️  Important: Each SSH key must be on its own line if you add multiple keys${NC}"
    echo ""
    echo -e "${BLUE}🚀 Your SSH key is now configured!${NC}"
    echo ""
    echo -e "${GREEN}✅ What happens next:${NC}"
    echo "• Your SSH key is saved locally in: ${YELLOW}./keys/${NC}"
    echo "• Once added to RunPod settings, this key will be automatically"
    echo "  injected into every new pod you create"
    echo "• You can now create pods and SSH into them immediately"
    echo ""
    echo -e "${BLUE}💡 Next steps:${NC}"
    echo "1. Complete the RunPod account setup above"
    echo "2. Create a new pod from the RunPod console"
    echo "3. The SSH key will be automatically available in your pod"
    echo "4. Use this command format to connect:"
    echo "   ${YELLOW}ssh root@POD_IP -p SSH_PORT -i ./keys/id_ed25519${NC}"
    echo ""
    echo -e "${GREEN}🎉 Ready to create and connect to RunPod instances!${NC}"
    
else
    echo -e "${RED}❌ Error: Failed to generate SSH key${NC}"
    exit 1
fi