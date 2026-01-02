#!/bin/bash
# AI-Gauge Setup Script
# Automates installation of AI-Gauge with Ollama backend

set -e

echo "=============================================="
echo "🌱 AI-Gauge Setup"
echo "=============================================="
echo ""

# Colors
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color

# Check for Ollama
check_ollama() {
    if command -v ollama &> /dev/null; then
        echo -e "${GREEN}✓${NC} Ollama found"
        return 0
    else
        echo -e "${YELLOW}!${NC} Ollama not found"
        return 1
    fi
}

# Check if Ollama is running
check_ollama_running() {
    if curl -s http://localhost:11434/api/tags > /dev/null 2>&1; then
        echo -e "${GREEN}✓${NC} Ollama is running"
        return 0
    else
        echo -e "${YELLOW}!${NC} Ollama is not running"
        return 1
    fi
}

# Check for Python
check_python() {
    if command -v python3 &> /dev/null; then
        PYTHON_VERSION=$(python3 --version 2>&1 | cut -d' ' -f2)
        echo -e "${GREEN}✓${NC} Python $PYTHON_VERSION found"
        return 0
    else
        echo -e "${RED}✗${NC} Python 3 not found"
        return 1
    fi
}

# Check for Node.js (optional, for IDE plugin)
check_node() {
    if command -v node &> /dev/null; then
        NODE_VERSION=$(node --version)
        echo -e "${GREEN}✓${NC} Node.js $NODE_VERSION found"
        return 0
    else
        echo -e "${YELLOW}!${NC} Node.js not found (optional, needed for VS Code extension)"
        return 1
    fi
}

# Main setup
main() {
    echo "Checking dependencies..."
    echo ""
    
    check_python || { echo -e "${RED}Error: Python 3 is required${NC}"; exit 1; }
    check_node
    OLLAMA_INSTALLED=$(check_ollama && echo "yes" || echo "no")
    
    echo ""
    echo "=============================================="
    echo "Step 1: Python Virtual Environment"
    echo "=============================================="
    
    # Create virtual environment if it doesn't exist
    if [ ! -d "venv" ]; then
        echo "Creating Python virtual environment..."
        python3 -m venv venv
        echo -e "${GREEN}✓${NC} Virtual environment created"
    else
        echo -e "${GREEN}✓${NC} Virtual environment already exists"
    fi
    
    # Activate virtual environment
    echo "Activating virtual environment..."
    source venv/bin/activate
    echo -e "${GREEN}✓${NC} Virtual environment activated"
    
    if [ -f "requirements.txt" ]; then
        echo "Installing Python packages..."
        pip install -r requirements.txt
        echo -e "${GREEN}✓${NC} Python packages installed"
    else
        echo -e "${RED}✗${NC} requirements.txt not found"
        exit 1
    fi
    
    echo ""
    echo "=============================================="
    echo "Step 2: Ollama Model Setup"
    echo "=============================================="
    
    if [ "$OLLAMA_INSTALLED" = "no" ]; then
        echo ""
        echo -e "${YELLOW}Ollama is not installed. Installing automatically...${NC}"
        echo ""
        
        # Detect OS and install Ollama
        if [[ "$OSTYPE" == "darwin"* ]]; then
            # macOS
            if command -v brew &> /dev/null; then
                echo "Installing Ollama via Homebrew..."
                brew install ollama
            else
                echo "Homebrew not found. Installing Ollama via official script..."
                curl -fsSL https://ollama.ai/install.sh | sh
            fi
        elif [[ "$OSTYPE" == "linux-gnu"* ]]; then
            # Linux
            echo "Installing Ollama for Linux..."
            curl -fsSL https://ollama.ai/install.sh | sh
        else
            echo -e "${RED}Unsupported OS: $OSTYPE${NC}"
            echo "Please install Ollama manually from: https://ollama.ai/download"
            exit 1
        fi
        
        # Verify installation
        if ! check_ollama; then
            echo -e "${RED}Failed to install Ollama${NC}"
            exit 1
        fi
    fi
    
    # Check if Ollama is running
    if ! check_ollama_running; then
        echo ""
        echo "Starting Ollama..."
        ollama serve &
        sleep 3
    fi
    
    # Check if model already exists
    if ollama list 2>/dev/null | grep -q "ai-gauge"; then
        echo -e "${GREEN}✓${NC} ai-gauge model already exists"
    else
        echo "Pulling ai-gauge model from Ollama registry..."
        ollama pull ajayvenki01/ai-gauge
        echo -e "${GREEN}✓${NC} Model pulled successfully"
    fi
    
    echo ""
    echo "=============================================="
    echo "Step 3: Verification"
    echo "=============================================="
    
    echo "Testing model..."
    RESPONSE=$(ollama run ai-gauge "Return JSON: {\"status\": \"ok\"}" 2>/dev/null | head -1)
    if echo "$RESPONSE" | grep -q "status"; then
        echo -e "${GREEN}✓${NC} Model is working"
    else
        echo -e "${YELLOW}!${NC} Model test returned unexpected output"
    fi
    
    echo ""
    echo "=============================================="
    echo "🎉 Setup Complete!"
    echo "=============================================="
    echo ""
    echo "Next steps:"
    echo ""
    echo "  1. Open VS Code in this folder"
    echo "  2. Install AI-Gauge extension from marketplace"
    echo "  3. Start coding - the extension will auto-detect this setup!"
    echo ""
    echo "Note: The virtual environment is at: $(pwd)/venv"
    echo "      The extension will automatically use it."
    echo ""
}

# Run
main "$@"
