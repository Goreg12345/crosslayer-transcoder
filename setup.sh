#!/bin/bash

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

echo -e "${BLUE}🚀 Setting up crosslayer-transcoder environment...${NC}"

# Function to check if command exists
command_exists() {
    command -v "$1" >/dev/null 2>&1
}

# Install uv if not present
if ! command_exists uv; then
    echo -e "${YELLOW}📦 Installing uv (fast Python package manager)...${NC}"
    curl -LsSf https://astral.sh/uv/install.sh | sh
    
    # Source the shell configuration to make uv available
    export PATH="$HOME/.local/bin:$HOME/.cargo/bin:$PATH"
    
    if ! command_exists uv; then
        echo -e "${RED}❌ Failed to install uv. Please install manually: https://docs.astral.sh/uv/getting-started/installation/${NC}"
        exit 1
    fi
    echo -e "${GREEN}✅ uv installed successfully!${NC}"
else
    echo -e "${GREEN}✅ uv is already installed${NC}"
fi

# Create .venv and install dependencies
echo -e "${YELLOW}🔧 Creating virtual environment and installing dependencies...${NC}"

# Remove existing .venv if it exists
if [ -d ".venv" ]; then
    echo -e "${YELLOW}🗑️  Removing existing .venv directory...${NC}"
    rm -rf .venv
fi

# Use uv to create venv and install dependencies
# This will automatically:
# - Install the correct Python version (3.12)
# - Create .venv directory
# - Install all dependencies from pyproject.toml
export PATH="$HOME/.local/bin:$PATH"
uv sync --dev --python 3.12

if [ $? -eq 0 ]; then
    echo -e "${YELLOW}🔧 Setting up permissions and PATH...${NC}"
    
    # Fix permissions for all executables in .venv/bin
    chmod +x .venv/bin/*
    
    # Add uv to PATH permanently by updating shell profile
    UV_PATH_EXPORT='export PATH="$HOME/.local/bin:$PATH"'
    
    # Check which shell profile to update
    UPDATED_PROFILE=false
    
    # Try .bashrc first (most common)
    if [ -f "$HOME/.bashrc" ] || [ "$SHELL" = "/bin/bash" ]; then
        if [ ! -f "$HOME/.bashrc" ]; then
            touch "$HOME/.bashrc"
        fi
        if ! grep -q "\.local/bin" "$HOME/.bashrc"; then
            echo "" >> "$HOME/.bashrc"
            echo "# Added by crosslayer-transcoder setup" >> "$HOME/.bashrc"
            echo "$UV_PATH_EXPORT" >> "$HOME/.bashrc"
            echo -e "${GREEN}✅ Added uv to PATH in ~/.bashrc${NC}"
            UPDATED_PROFILE=true
        fi
    fi
    
    # Try .zshrc for zsh users
    if [ -f "$HOME/.zshrc" ]; then
        if ! grep -q "\.local/bin" "$HOME/.zshrc"; then
            echo "" >> "$HOME/.zshrc"
            echo "# Added by crosslayer-transcoder setup" >> "$HOME/.zshrc"
            echo "$UV_PATH_EXPORT" >> "$HOME/.zshrc"
            echo -e "${GREEN}✅ Added uv to PATH in ~/.zshrc${NC}"
            UPDATED_PROFILE=true
        fi
    fi
    
    # Fallback to .profile (works for all POSIX shells)
    if [ "$UPDATED_PROFILE" = false ] && [ -f "$HOME/.profile" ]; then
        if ! grep -q "\.local/bin" "$HOME/.profile"; then
            echo "" >> "$HOME/.profile"
            echo "# Added by crosslayer-transcoder setup" >> "$HOME/.profile"
            echo "$UV_PATH_EXPORT" >> "$HOME/.profile"
            echo -e "${GREEN}✅ Added uv to PATH in ~/.profile${NC}"
            UPDATED_PROFILE=true
        fi
    fi
    
    # Create .bashrc as final fallback
    if [ "$UPDATED_PROFILE" = false ]; then
        echo "" >> "$HOME/.bashrc"
        echo "# Added by crosslayer-transcoder setup" >> "$HOME/.bashrc"
        echo "$UV_PATH_EXPORT" >> "$HOME/.bashrc"
        echo -e "${GREEN}✅ Created ~/.bashrc and added uv to PATH${NC}"
    fi
    
    echo -e "${GREEN}✅ Environment setup complete!${NC}"
    echo ""
    echo -e "${BLUE}📝 To activate the environment, run:${NC}"
    echo -e "${YELLOW}   source .venv/bin/activate${NC}"
    echo -e "${BLUE}   (or simply: .venv/bin/activate)${NC}"
    echo ""
    echo -e "${BLUE}📝 To run Jupyter Lab:${NC}"
    echo -e "${YELLOW}   .venv/bin/jupyter lab${NC}"
    echo ""
    echo -e "${BLUE}📝 To run Python scripts:${NC}"
    echo -e "${YELLOW}   .venv/bin/python your_script.py${NC}"
    echo ""
    echo -e "${BLUE}📝 To add new dependencies:${NC}"
    echo -e "${YELLOW}   uv add package_name${NC}"
    echo ""
    echo -e "${BLUE}📝 To make uv available globally, restart your terminal or run:${NC}"
    if [ -f "$HOME/.bashrc" ]; then
        echo -e "${YELLOW}   source ~/.bashrc${NC}"
    elif [ -f "$HOME/.zshrc" ]; then
        echo -e "${YELLOW}   source ~/.zshrc${NC}"
    elif [ -f "$HOME/.profile" ]; then
        echo -e "${YELLOW}   source ~/.profile${NC}"
    else
        echo -e "${YELLOW}   source ~/.bashrc${NC}"
    fi
else
    echo -e "${RED}❌ Setup failed. Please check the error messages above.${NC}"
    exit 1
fi 