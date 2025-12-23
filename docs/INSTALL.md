# AI-Gauge Installation Guide

Complete setup instructions for end users.

---

## 🚀 Quick Start (Ollama Local - Simplest)

**Total time: ~3 minutes. Local inference, no API keys needed.**

### Step 1: Install Ollama

```bash
# macOS
brew install ollama

# Linux
curl -fsSL https://ollama.ai/install.sh | sh

# Windows: Download from https://ollama.ai/download
```

### Step 2: Run Setup Script

```bash
# Clone the repository
git clone https://github.com/your-repo/ai-gauge.git
cd ai-gauge

# Run automated setup
./setup.sh
```

This will:
- Install Python dependencies
- Pull the AI-Gauge model from Ollama registry
- Verify everything works

### Step 3: Install VS Code Extension

**Option A: Install from Marketplace (Recommended)**
```bash
# In VS Code: Cmd+Shift+P → Search "AI-Gauge" → Install
# Or visit: https://marketplace.visualstudio.com/items?itemName=Ajayvenki2910.ai-gauge
```

**Option B: Install from VSIX file**
```bash
# Download ai-gauge-0.1.0.vsix from the project releases
# Then in VS Code: Cmd+Shift+P → "Extensions: Install from VSIX..."
```

**Option C: Build from Source**
```bash
cd ide_plugin
npm install  # (Requires Node.js 20)
npm run compile
npx vsce package
# Then install the generated .vsix file
```

### Step 4: Test It!

```bash
# Run demo
python test_samples/demo_single_test.py
```

**Done!** AI-Gauge will analyze your LLM calls and recommend cost-saving alternatives.

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
- **Ollama** (for local inference)
- **VS Code** (for the extension)

---

## Configuration Options

### VS Code Extension Settings

| Setting | Default | Description |
|---------|---------|-------------|
| `aiGauge.enabled` | `true` | Enable/disable analysis |
| `aiGauge.useOllama` | `true` | **Use local Ollama (recommended)** |
| `aiGauge.ollamaModel` | `ai-gauge` | **Model name in Ollama** |
| `aiGauge.ollamaUrl` | `http://localhost:11434` | **Ollama server URL** |
| `aiGauge.useHuggingFace` | `false` | Use HuggingFace cloud (fallback) |
| `aiGauge.huggingFaceApiKey` | `""` | Your HuggingFace API token |
| `aiGauge.huggingFaceModel` | `ajayvenkatesan/ai-gauge` | HuggingFace model repo |
| `aiGauge.modelServerUrl` | `http://localhost:8080` | Inference server URL |
| `aiGauge.showInlineHints` | `true` | Show inline cost hints |
| `aiGauge.costThreshold` | `20` | Minimum savings % to show hint |

### Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `AI_GAUGE_BACKEND` | `ollama` | **Backend: `ollama`, `huggingface`, or `llama_cpp`** |
| `OLLAMA_URL` | `http://localhost:11434` | **Ollama server URL** |
| `AI_GAUGE_MODEL` | `ai-gauge` | **Model name** |
| `HF_API_KEY` | `""` | HuggingFace API token (if using cloud) |
| `HF_MODEL_ID` | `ajayvenkatesan/ai-gauge` | HuggingFace model repo |
| `AI_GAUGE_PORT` | `8080` | Inference server port |

---

## Three Ways to Use

### 1. Local Ollama (Recommended) ⭐ RECOMMENDED

Just run the setup script:
```bash
./setup.sh
python test_samples/demo_single_test.py
```

**Pros**: Offline, private, no API costs, fast after first run  
**Cons**: 2.2GB download, requires Ollama setup

### 2. HuggingFace Cloud (Internet Required)

```bash
export HF_API_KEY="hf_your_token_here"
export AI_GAUGE_BACKEND="huggingface"
python test_samples/demo_single_test.py
```

In VS Code settings:
```json
{
  "aiGauge.useOllama": false,
  "aiGauge.useHuggingFace": true,
  "aiGauge.huggingFaceApiKey": "hf_your_token"
}
```

**Pros**: No downloads, works everywhere  
**Cons**: Requires internet, API costs, slower first call

### 3. Inference Server (Teams)

```bash
# Terminal 1: Start server
python inference_server.py

# The VS Code extension connects to http://localhost:8080
```

**Pros**: Centralized, full analysis pipeline  
**Cons**: Requires server management

---

## Troubleshooting

### Ollama Issues

```bash
# Check if Ollama is running
ollama list

# If no models listed, start Ollama
ollama serve

# Pull model manually if needed
ollama pull ai-gauge
```

### "Model not found" in Ollama

```bash
# Verify model exists
ollama list

# If missing, pull it
ollama pull ai-gauge
```

### "Connection refused" errors

```bash
# Check if Ollama is running
curl http://localhost:11434/api/tags

# Start Ollama if needed
ollama serve
```

### HuggingFace Issues (if using cloud fallback)

```bash
# Model loading (first call takes 20-60 seconds)
# This is normal - HuggingFace loads the model on first request

# Check API key is valid
curl -H "Authorization: Bearer hf_your_token" \
  https://api-inference.huggingface.co/models/ajayvenkatesan/ai-gauge

# Rate limit: Free tier = 1,000 calls/hour
```

### Inference server not working

```bash
# Check health endpoint
curl http://localhost:8080/health

# Should return:
# {"status":"ok","backend":"OLLAMA|HUGGINGFACE|LLAMA_CPP",...}
```

---

## Model Files

| File | Size | Description |
|------|------|-------------|
| `training_data/models/ai-gauge-q4_k_m.gguf` | ~2.2GB | Quantized model (local only) |
| `Modelfile` | 1KB | Ollama model definition |

---

## Uninstall

```bash
# Remove Ollama model
ollama rm ai-gauge

# Remove VS Code extension
# Cmd+Shift+P → "Extensions: Uninstall Extension" → AI-Gauge

# Remove Python package
pip uninstall ai-gauge
```
