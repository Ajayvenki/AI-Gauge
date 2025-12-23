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
    echo "Step 1: Python Dependencies"
    echo "=============================================="
    
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
        echo -e "${YELLOW}Ollama is not installed.${NC}"
        echo ""
        echo "Please install Ollama first:"
        echo "  macOS:   brew install ollama"
        echo "  Linux:   curl -fsSL https://ollama.ai/install.sh | sh"
        echo "  Windows: Download from https://ollama.ai/download"
        echo ""
        echo "Then run this script again."
        exit 1
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
        ollama pull ai-gauge
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
    echo "  1. Run a demo:"
    echo "     python test_samples/demo_single_test.py"
    echo ""
    echo "  2. Start the inference server:"
    echo "     python inference_server.py"
    echo ""
    echo "  3. (Optional) Build VS Code extension:"
    echo "     cd ide_plugin && npm install && npm run compile"
    echo ""
}

# Run
main "$@"
