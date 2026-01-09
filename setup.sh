#!/bin/bash

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

echo -e "${BLUE}🚀 Setting up crosslayer-transcoder...${NC}"

# Function to check if command exists
command_exists() {
    command -v "$1" >/dev/null 2>&1
}

# Install uv if not present
if ! command_exists uv; then
    echo -e "${YELLOW}📦 Installing uv...${NC}"
    curl -LsSf https://astral.sh/uv/install.sh | sh
    export PATH="$HOME/.local/bin:$HOME/.cargo/bin:$PATH"
    
    if ! command_exists uv; then
        echo -e "${RED}❌ Failed to install uv. See: https://docs.astral.sh/uv/${NC}"
        exit 1
    fi
    echo -e "${GREEN}✅ uv installed${NC}"
else
    echo -e "${GREEN}✅ uv already installed${NC}"
fi

# Install dependencies
echo -e "${YELLOW}🔧 Installing dependencies...${NC}"
uv sync

if [ $? -eq 0 ]; then
    echo -e "${GREEN}✅ Setup complete!${NC}"
    echo ""
    echo -e "${BLUE}Usage:${NC}"
    echo -e "  ${YELLOW}uv run python script.py${NC}    - Run a Python script"
    echo -e "  ${YELLOW}uv add package_name${NC}        - Add a dependency"
    echo -e "  ${YELLOW}uv sync --dev${NC}              - Install dev tools (ruff, pytest, jupyter)"
else
    echo -e "${RED}❌ Setup failed${NC}"
    exit 1
fi
