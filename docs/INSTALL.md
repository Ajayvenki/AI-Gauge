# AI-Gauge Installation Guide

Complete setup instructions for end users.

---

## 🚀 Quick Start (Automatic Setup)

**Total time: ~5 minutes. Everything handled automatically.**

### Step 1: Install VS Code Extension

**Option A: Install from Marketplace (Recommended)**
```bash
# In VS Code: Cmd+Shift+P → Search "AI-Gauge" → Install
# Or visit: https://marketplace.visualstudio.com/items?itemName=Ajayvenki2910.ai-gauge
```

**Option B: Install from VSIX file**
```bash
# Download ai-gauge-0.4.0.vsix from the project releases
# Then in VS Code: Cmd+Shift+P → "Extensions: Install from VSIX..."
```

### Step 2: Enable AI-Gauge
The extension automatically:
- ✅ Copies bundled Python code to your local storage
- ✅ Installs required Python dependencies
- ✅ Sets up Ollama for agent analysis (if needed)
- ✅ Starts the inference server
- ✅ Configures all necessary components

**That's it!** AI-Gauge will now analyze your LLM calls automatically using its agent orchestration pipeline.
The extension will automatically:
- ✅ Start the AI-Gauge inference server
- ✅ Install required Python dependencies
- ✅ Set up Ollama (if needed for local AI analysis)
- ✅ Configure all necessary components

**That's it!** AI-Gauge will now analyze your LLM calls automatically using its agent orchestration pipeline.

**Option C: Build from Source**
```bash
cd ide_plugin
npm install  # (Requires Node.js 20)
npm run compile
npx vsce package
# Then install the generated .vsix file
```

### Step 3: Test It!

```bash
# Install Python dependencies (if not done automatically)
pip install -r requirements.txt

# Run demo
python test_samples/demo_single_test.py
```

**Done!** AI-Gauge will analyze your LLM calls and recommend cost-saving alternatives using its sophisticated agent pipeline.

**Note**: The HuggingFace model is processing (usually 5-30 minutes). For now, AI-Gauge will use the local model and automatically switch to cloud inference when ready.

### Step 4: Test It!

```bash
# Install Python dependencies
pip install -r requirements.txt

# Set API key as env var
export HF_API_KEY="hf_your_token_here"

# Run demo
python test_samples/demo_single_test.py
```

**Done!** AI-Gauge will analyze your LLM calls and recommend cost-saving alternatives.

---

## Prerequisites

- **Python 3.9+** with pip
- **VS Code** (for the extension)
- **Ollama** (automatically managed by the extension for local AI analysis)

---

## Configuration Options

### VS Code Extension Settings

| Setting | Default | Description |
|---------|---------|-------------|
| `aiGauge.enabled` | `true` | Enable/disable analysis |
| `aiGauge.modelServerUrl` | `http://localhost:8080` | Inference server URL (managed automatically) |
| `aiGauge.showInlineHints` | `true` | Show inline cost hints |
| `aiGauge.costThreshold` | `20` | Minimum savings % to show hint |
| `aiGauge.serverAutoStart` | `true` | Automatically start inference server |
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
