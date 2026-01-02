# AI-Gauge Installation Guide

Complete setup instructions for end users.

---

## 🚀 Quick Start (Recommended)

**Total time: ~5 minutes**

### Step 1: Download Runtime Package

```bash
# Download the latest runtime package
curl -LO https://github.com/ajayvenki2910/ai-gauge/releases/latest/download/ai-gauge-runtime.tar.gz

# Extract it
tar -xzf ai-gauge-runtime.tar.gz
cd runtime
```

### Step 2: Run Setup

```bash
./setup.sh
```

This will automatically:
- ✅ Create a Python virtual environment
- ✅ Install required Python dependencies
- ✅ Install Ollama (if not present)
- ✅ Download the AI-Gauge model
- ✅ Verify everything works

### Step 3: Install VS Code Extension

1. Open VS Code
2. Go to Extensions (Cmd+Shift+X)
3. Search "AI-Gauge"
4. Click Install

**That's it!** The extension will automatically detect the runtime package and start analyzing your LLM calls.

---

## Alternative Installation Methods

### Option A: Install Extension First (Auto-Setup)

1. Install AI-Gauge extension from VS Code Marketplace
2. Open a workspace where you want to use AI-Gauge
3. The extension will prompt you to set up automatically
4. Follow the prompts

### Option B: Build from Source

```bash
# Clone the repository
git clone https://github.com/ajayvenki2910/ai-gauge.git
cd ai-gauge

# Run setup
cd runtime
./setup.sh

# Build extension (optional)
cd ../ide_plugin
npm install
npm run compile
npx vsce package
```

---

## Prerequisites

- **Python 3.9+** with pip
- **VS Code** (for the extension)
- **macOS or Linux** (Windows support coming soon)

---

## Configuration Options

### VS Code Extension Settings

| Setting | Default | Description |
|---------|---------|-------------|
| `aiGauge.enabled` | `true` | Enable/disable analysis |
| `aiGauge.modelServerUrl` | `http://localhost:8080` | Inference server URL |
| `aiGauge.showInlineHints` | `true` | Show inline cost hints |
| `aiGauge.costThreshold` | `20` | Minimum savings % to show hint |
| `aiGauge.serverHealthCheckInterval` | `30` | Health check interval (seconds) |

### Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `AI_GAUGE_PORT` | `8080` | Inference server port |
| `OLLAMA_URL` | `http://localhost:11434` | Ollama server URL (for agent analysis) |

---

## How It Works

AI-Gauge uses a **server-first architecture** with **agent orchestration**:

1. **VS Code Extension**: Automatically manages the inference server lifecycle
2. **Inference Server**: Runs the decision module with orchestrated agents
3. **Decision Module**: 3-agent pipeline (metadata extractor, analyzer with Ollama SLM, reporter)
4. **Model Cards**: Single source of truth for all model metadata (tiers, costs, carbon factors)
5. **Ollama Integration**: Local AI model used within the analyzer agent for task complexity assessment

**No manual configuration required** - the extension handles everything automatically!

---

## Troubleshooting

### Server Health Issues

```bash
# Check if inference server is running
curl http://localhost:8080/health

# Should return:
# {"status":"ok","agents":"ready","ollama":"connected"}
```

### Extension Not Starting Server Automatically

```bash
# Manually start the server
python src/inference_server.py

# Or check VS Code extension logs:
# Cmd+Shift+P → "Developer: Show Logs" → Extension Host
```

### Ollama Connection Issues (for Agent Analysis)

```bash
# Check if Ollama is running
ollama list

# If no models listed, start Ollama
ollama serve

# Pull required model (handled automatically by server)
ollama pull llama3.2:3b
```

### "Connection refused" errors

```bash
# Check inference server
curl http://localhost:8080/health

# Check Ollama (if needed)
curl http://localhost:11434/api/tags
```

### Agent Pipeline Issues

```bash
# Check server logs for agent orchestration errors
python src/inference_server.py  # Run with verbose logging

# Verify model cards are loaded
curl http://localhost:8080/models
```

### Performance Issues

- **First analysis slow?** Normal - agents are initializing and Ollama may be loading models
- **Memory usage high?** Expected - agent orchestration and local AI analysis require resources
- **Server not responding?** Check health endpoint and restart if needed

---

## Architecture Overview

```
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐
│   VS Code       │    │  Inference       │    │  Decision       │
│   Extension     │◄──►│  Server          │◄──►│  Module         │
│                 │    │  (Flask)         │    │  (LangGraph)    │
└─────────────────┘    └──────────────────┘    └─────────────────┘
         ▲                       ▲                       ▲
         │                       │                       │
         └───────────────────────┼───────────────────────┘
                                 ▼
                    ┌─────────────────┐    ┌─────────────────┐
                    │   Model Cards   │    │     Ollama      │
                    │   (Database)    │    │   (Local AI)    │
                    └─────────────────┘    └─────────────────┘
```

**Agent Pipeline:**
1. **Metadata Extractor**: Analyzes LLM call patterns and context
2. **Analyzer Agent**: Uses Ollama SLM to assess task complexity and requirements  
3. **Reporter Agent**: Generates cost-saving recommendations based on model tiers

---

## Model Files

| Component | Description |
|-----------|-------------|
| `src/decision_module.py` | Agent orchestration pipeline |
| `src/model_cards.py` | Model metadata database |
| `src/local_inference.py` | Ollama integration for agents |
| `training_data/models/*.gguf` | Fallback quantized models |

---

## Uninstall

```bash
# Stop any running servers
pkill -f inference_server.py

# Remove VS Code extension
# Cmd+Shift+P → "Extensions: Uninstall Extension" → AI-Gauge

# Remove Python packages (optional)
pip uninstall flask langgraph ollama
```
